use anyhow::Result;
use log::{debug, info};
use std::time::Instant;

use super::config::Config;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use crate::weight_loader::{
    load_shard_info, load_tensor_1d, load_tensor_2d, mmap_shards, precompute_rope,
};

#[derive(Clone, Copy, Debug)]
pub struct ModelRuntimeConfig {
    pub enable_cuda_graph: bool,
}

impl Default for ModelRuntimeConfig {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

/// Attention layer weights.
/// QKV stored as a single concatenated matrix [q_dim + 2*kv_dim, hidden_size].
/// Individual projections accessed via row offsets (zero extra memory).
pub(super) struct Attention {
    /// Fused [q_proj; k_proj; v_proj] row-major
    pub(super) qkv_proj: DeviceMatrix,
    pub(super) o_proj: DeviceMatrix,
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
    pub(super) q_dim: usize,
    pub(super) kv_dim: usize,
}

/// MLP layer weights.
/// Gate+Up stored as a single concatenated matrix [2*intermediate_size, hidden_size].
#[allow(clippy::upper_case_acronyms, clippy::struct_field_names)]
pub(super) struct MLP {
    /// Fused [gate_proj; up_proj] row-major
    pub(super) gate_up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

/// Transformer block
pub(super) struct TransformerBlock {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attention: Attention,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP,
}

/// Qwen3 model — weights and config only. Mutable state lives in `Qwen3State`.
pub struct Qwen3Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) lm_head: Option<DeviceMatrix>,
    pub(super) layers: Vec<TransformerBlock>,
    pub(super) norm: DeviceVec,
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
    pub(super) kv_pool: crate::kv_pool::KvPool,
}

impl Qwen3Model {
    pub fn from_safetensors_with_runtime(
        model_path: &str,
        _runtime: ModelRuntimeConfig,
    ) -> Result<Self> {
        info!("Loading model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config::from_file(model_path)?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        debug!("Loading {} safetensor shard(s)", shard_paths.len());
        let mmaps = mmap_shards(&shard_paths)?;
        let shards: Vec<safetensors::SafeTensors> = mmaps
            .iter()
            .map(|m| {
                safetensors::SafeTensors::deserialize(m)
                    .map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))
            })
            .collect::<Result<_>>()?;

        let t_gpu = Instant::now();
        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(&ctx, &shards, &weight_map, "model.embed_tokens.weight")?;
        let lm_head = if config.tie_word_embeddings {
            debug!("Using tied input/output embeddings");
            None
        } else {
            debug!("Loading untied LM head to GPU");
            Some(load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                config.lm_head_tensor_name(),
            )?)
        };

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{}", i);

            let q_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.self_attn.q_proj.weight", prefix),
            )?;
            let k_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.self_attn.k_proj.weight", prefix),
            )?;
            let v_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.self_attn.v_proj.weight", prefix),
            )?;
            let q_dim = q_proj.rows;
            let kv_dim = k_proj.rows;
            let qkv_proj = DeviceMatrix::vstack(&ctx, &[&q_proj, &k_proj, &v_proj])?;
            drop(q_proj);
            drop(k_proj);
            drop(v_proj);

            let gate_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.gate_proj.weight", prefix),
            )?;
            let up_proj = load_tensor_2d(
                &ctx,
                &shards,
                &weight_map,
                &format!("{}.mlp.up_proj.weight", prefix),
            )?;
            let gate_up_proj = DeviceMatrix::vstack(&ctx, &[&gate_proj, &up_proj])?;
            drop(gate_proj);
            drop(up_proj);

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: Attention {
                    qkv_proj,
                    o_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.o_proj.weight", prefix),
                    )?,
                    q_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_norm.weight", prefix),
                    )?,
                    k_norm: load_tensor_1d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_norm.weight", prefix),
                    )?,
                    q_dim,
                    kv_dim,
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP {
                    gate_up_proj,
                    down_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.down_proj.weight", prefix),
                    )?,
                },
            };
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, "model.norm.weight")?;

        debug!("Precomputing RoPE cache on GPU");
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.head_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("GPU model loaded successfully");

        let page_size = 16;
        let layout = crate::kv_pool::KvLayout::new(
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            page_size,
        );
        let bytes_per_page = layout.page_stride * std::mem::size_of::<half::bf16>();
        let (free_bytes, _total_bytes) = cudarc::driver::result::mem_get_info()
            .map_err(|e| anyhow::anyhow!("cuMemGetInfo failed: {e}"))?;
        let kv_budget = (free_bytes as f64 * 0.85) as usize;
        let num_pages = (kv_budget / bytes_per_page).max(64);
        let kv_mb = num_pages * bytes_per_page / (1024 * 1024);
        info!(
            "KV cache: {num_pages} pages ({kv_mb} MB, {:.0}% of {:.0} MB free)",
            kv_budget as f64 / free_bytes as f64 * 100.0,
            free_bytes as f64 / 1024.0 / 1024.0
        );
        let kv_pool = crate::kv_pool::KvPool::new(
            &ctx,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim,
            page_size,
            num_pages,
        )?;

        let model = Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: _runtime.enable_cuda_graph,
            kv_pool,
        };

        if model.enable_cuda_graph {
            debug!("Decode path CUDA Graph is enabled (captures on first decode step)");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }

    pub(super) fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn device_ctx(&self) -> &crate::tensor::DeviceContext {
        &self.ctx
    }

    /// Allocate a fresh (empty) per-request KV state from the shared pool.
    pub(crate) fn alloc_kv(&self) -> crate::kv_pool::KvState {
        self.kv_pool.alloc()
    }

    /// Create pre-allocated batch decode buffers.
    pub(crate) fn create_batch_decode_bufs(
        &self,
        max_batch_size: usize,
    ) -> anyhow::Result<super::batch_decode_buffers::BatchDecodeBuffers> {
        super::batch_decode_buffers::BatchDecodeBuffers::new(
            &self.ctx,
            &self.config,
            max_batch_size,
            self.kv_pool.capacity_pages(),
            self.kv_pool.padding_page_id(),
        )
    }
}
