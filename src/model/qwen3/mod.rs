//! Qwen3 model: weights, forward pass.

use anyhow::Result;
use log::{debug, info};
use std::time::Instant;

use rand::RngExt;
use rand::rngs::StdRng;

use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

mod config;
mod decode_buffers;

pub use config::Config;

use self::decode_buffers::DecodeBuffers;
use super::kv_cache::KVCache;
use super::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::{self, SamplingParams};
use crate::tensor::*;
use crate::weight_loader::*;

/// CUDA Graph state for decode path.
/// First decode call captures the graph; subsequent calls replay it.
struct CudaGraphState {
    graph: Option<CudaGraph>,
}

// SAFETY: CudaGraph contains raw CUDA pointers that are not Send by default.
// We only access the graph from the single inference thread that owns the model.
unsafe impl Send for CudaGraphState {}

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

/// Attention layer weights
struct Attention {
    q_proj: DeviceMatrix,
    k_proj: DeviceMatrix,
    v_proj: DeviceMatrix,
    o_proj: DeviceMatrix,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
}

/// MLP layer weights
struct MLP {
    gate_proj: DeviceMatrix,
    up_proj: DeviceMatrix,
    down_proj: DeviceMatrix,
}

/// Transformer block
struct TransformerBlock {
    input_layernorm: DeviceVec,
    attention: Attention,
    post_attention_layernorm: DeviceVec,
    mlp: MLP,
}

/// Qwen3 model — weights and config only. Mutable state lives in `Qwen3State`.
pub struct Qwen3Model {
    ctx: DeviceContext,
    config: Config,
    embed_tokens: DeviceMatrix,
    lm_head: Option<DeviceMatrix>,
    layers: Vec<TransformerBlock>,
    norm: DeviceVec,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    enable_cuda_graph: bool,
}

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
    decode_bufs: DecodeBuffers,
    kv_cache: KVCache,
    graph_state: CudaGraphState,
    /// Logits from multi-token prefill (None after decode path — logits are in decode_bufs).
    prefill_logits: Option<DeviceVec>,
}

// SAFETY: Qwen3State contains CudaGraph (raw CUDA pointers) that are not Send by default.
// We only access state from the single inference thread.
unsafe impl Send for Qwen3State {}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.prefill_logits
            .as_ref()
            .unwrap_or(&self.decode_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_cache.reset();
        self.prefill_logits = None;
        Ok(())
    }
}

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill in `process_all_layers_batch`, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    hidden_out: HiddenStates, // hidden_dim × seq_len
    normed: HiddenStates,      // hidden_dim × seq_len (reused for normed2)
    q_batch: HiddenStates,     // q_dim × seq_len
    k_batch: HiddenStates,     // kv_dim × seq_len
    v_batch: HiddenStates,     // kv_dim × seq_len
    o_buf: HiddenStates,       // hidden_dim × seq_len (reused for mlp_out)
    gate_out: HiddenStates,    // inter_dim × seq_len
    up_out: HiddenStates,      // inter_dim × seq_len
    act_out: HiddenStates,     // inter_dim × seq_len
    attn_output: HiddenStates, // q_dim × seq_len
}

impl PrefillBuffers {
    fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }
}

impl Qwen3Model {
    pub fn from_safetensors_with_runtime(
        model_path: &str,
        runtime: ModelRuntimeConfig,
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

            let block = TransformerBlock {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attention: Attention {
                    q_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.q_proj.weight", prefix),
                    )?,
                    k_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.k_proj.weight", prefix),
                    )?,
                    v_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.self_attn.v_proj.weight", prefix),
                    )?,
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
                },
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP {
                    gate_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.gate_proj.weight", prefix),
                    )?,
                    up_proj: load_tensor_2d(
                        &ctx,
                        &shards,
                        &weight_map,
                        &format!("{}.mlp.up_proj.weight", prefix),
                    )?,
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

        let model = Self {
            ctx,
            config,
            embed_tokens,
            lm_head,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph: runtime.enable_cuda_graph,
        };

        if model.enable_cuda_graph {
            debug!("Preloading decode-path Triton kernels before CUDA Graph capture");
            model.preload_decode_triton_kernels()?;
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(model)
    }

    fn preload_decode_triton_kernels(&self) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let cache_len = self.config.num_key_value_heads * 4096 * self.config.head_dim;
        let dummy_token_id = 0_i32;
        let dummy_pos = 0_i32;
        let dummy_seq_len = 1_i32;

        let decode_meta = self
            .ctx
            .stream
            .clone_htod(&[dummy_token_id, dummy_pos, dummy_seq_len])
            .map_err(|e| anyhow::anyhow!("Preload decode_meta H2D failed: {}", e))?;
        let mut embed_out = DeviceVec::zeros(&self.ctx, hidden_size)?;
        ops::embedding_decode_into(&self.ctx, &self.embed_tokens, &decode_meta, &mut embed_out)?;

        let layer0 = &self.layers[0];
        let q = DeviceVec::zeros(&self.ctx, q_dim)?;
        let k = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let v = DeviceVec::zeros(&self.ctx, kv_dim)?;
        let mut k_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut v_cache = DeviceVec::zeros(&self.ctx, cache_len)?;
        let mut out = DeviceVec::zeros(&self.ctx, q_dim)?;

        let num_qheads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim;
        let num_kv_splits = 4usize;
        let mut partial_out = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits * head_dim)
            .map_err(|e| anyhow::anyhow!("Alloc partial_out failed: {}", e))?;
        let mut partial_m = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits)
            .map_err(|e| anyhow::anyhow!("Alloc partial_m failed: {}", e))?;
        let mut partial_l = self
            .ctx
            .stream
            .alloc_zeros::<f32>(num_qheads * num_kv_splits)
            .map_err(|e| anyhow::anyhow!("Alloc partial_l failed: {}", e))?;

        ops::fused_attention_decode_into(
            &self.ctx,
            &q,
            &k,
            &v,
            &layer0.attention.q_norm,
            &layer0.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &decode_meta,
            &mut k_cache,
            &mut v_cache,
            &mut out,
            &mut partial_out,
            &mut partial_m,
            &mut partial_l,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )?;

        self.ctx.sync()?;
        Ok(())
    }

    // ============================================================
    // Batched forward path (prefill)
    // ============================================================

    #[fastrace::trace(name = "get_embeddings_batch")]
    fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_size;

        // Copy token IDs to GPU
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut out = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut out)?;

        Ok(out)
    }

    #[fastrace::trace(name = "process_all_layers_batch")]
    fn process_all_layers_batch(
        &self,
        mut hidden: HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<HiddenStates> {
        let seq_len = hidden.seq_len;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Allocate all intermediates once — eliminates ~11k cuMemAllocAsync calls.
        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            seq_len,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_layer_batch(
                layer_idx,
                layer,
                &mut hidden,
                start_pos,
                kv_cache,
                &mut bufs,
            )?;
        }

        // Increment sequence length AFTER all layers processed
        for _ in 0..seq_len {
            kv_cache.increment_seq_len();
        }

        Ok(hidden)
    }

    fn compute_logits_batch(&self, hidden: &HiddenStates) -> Result<DeviceVec> {
        let last_hidden = ops::extract_vec(&self.ctx, hidden, hidden.seq_len - 1)?;
        let normed = ops::rms_norm(
            &self.ctx,
            &last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
        )?;
        ops::linear(&self.ctx, &normed, self.output_projection())
    }

    fn output_projection(&self) -> &DeviceMatrix {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    fn forward_layer_batch(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        start_pos: usize,
        kv_cache: &mut KVCache,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim;

        kv_cache.init_if_needed(&self.ctx, head_dim)?;

        // 1. RMSNorm → bufs.normed
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 2. QKV projections → bufs.q_batch, bufs.k_batch, bufs.v_batch
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q_batch,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k_batch,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v_batch,
        )?;

        // 3. FlashAttention-2 (Triton) → bufs.attn_output
        let (k_cache_layer, v_cache_layer) = kv_cache.get_cache_mut(&self.ctx, layer_idx)?;
        ops::prefill_attention_batch(
            &self.ctx,
            &mut bufs.q_batch,
            &mut bufs.k_batch,
            &bufs.v_batch,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            k_cache_layer,
            v_cache_layer,
            &mut bufs.attn_output,
            num_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            self.config.rms_norm_eps,
        )?;

        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        )?;

        // 5. Residual add: hidden_in + o_batch → bufs.hidden_out
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = attn_residual, bufs.hidden_out = old hidden_in (now free)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        // 6. MLP RMSNorm → bufs.normed (reused for normed2; steps 1-4 are done)
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 7. MLP: gate + up → act → down → bufs.o_buf (reused for mlp_out; step 5 is done)
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        )?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        )?;
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        )?;

        // 8. Residual add: attn_residual + mlp_out → bufs.hidden_out (old hidden_in, free to overwrite)
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = layer output, bufs.hidden_out = attn_residual (free next layer)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }

    // ============================================================
    // Zero-allocation decode path (pre-allocated buffers)
    // ============================================================

    /// Single decode step using pre-allocated buffers. Zero GPU allocation.
    /// With CUDA Graph: first call captures, subsequent calls replay.
    fn decode_one_token(
        &self,
        token_id: u32,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
        graph_state: &mut CudaGraphState,
    ) -> Result<()> {
        let pos = kv_cache.len();
        let seq_len = pos + 1;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        self.ctx
            .stream
            .memcpy_htod(
                &[token_id as i32, pos as i32, seq_len as i32],
                &mut bufs.decode_meta,
            )
            .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {}", e))?;

        if !self.enable_cuda_graph {
            self.decode_kernels(kv_cache, bufs)?;
            kv_cache.increment_seq_len();
            return Ok(());
        }

        match &graph_state.graph {
            Some(graph) => {
                graph
                    .launch()
                    .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {}", e))?;
            }
            None => {
                debug!("Capturing CUDA Graph for decode path...");
                self.ctx
                    .stream
                    .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .map_err(|e| anyhow::anyhow!("begin_capture failed: {}", e))?;

                self.decode_kernels(kv_cache, bufs)?;

                graph_state.graph = self
                    .ctx
                    .stream
                    .end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                    .map_err(|e| anyhow::anyhow!("end_capture failed: {}", e))?;
                debug!("CUDA Graph captured successfully");

                if let Some(ref graph) = graph_state.graph {
                    graph
                        .launch()
                        .map_err(|e| anyhow::anyhow!("CUDA Graph first launch failed: {}", e))?;
                }
            }
        }

        kv_cache.increment_seq_len();
        Ok(())
    }

    fn decode_kernels(&self, kv_cache: &mut KVCache, bufs: &mut DecodeBuffers) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_layers = self.layers.len();

        ops::embedding_decode_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.decode_meta,
            &mut bufs.hidden,
        )?;

        ops::rms_norm_into(
            &self.ctx,
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.decode_layer_inner(layer_idx, layer, kv_cache, bufs)?;

            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            ops::fused_add_rms_norm_into(
                &self.ctx,
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                eps,
                &mut bufs.normed,
            )?;
        }

        ops::gemv(
            &self.ctx,
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        )?;

        ops::argmax_into(&self.ctx, &bufs.logits, &mut bufs.argmax_out);

        Ok(())
    }

    fn decode_layer_inner(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        ops::gemv(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q,
        )?;
        ops::gemv(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k,
        )?;
        ops::gemv(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v,
        )?;

        let (k_cache, v_cache) = kv_cache.get_cache_mut(&self.ctx, layer_idx)?;
        ops::fused_attention_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.decode_meta,
            k_cache,
            v_cache,
            &mut bufs.attn_out,
            &mut bufs.partial_out,
            &mut bufs.partial_m,
            &mut bufs.partial_l,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
        )?;

        ops::gemv(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        )?;

        ops::fused_add_rms_norm_into(
            &self.ctx,
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        ops::fused_mlp_into(
            &self.ctx,
            &bufs.normed,
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
            &mut bufs.mlp_act,
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }
}

// ============================================================================
// ModelForward implementation
// ============================================================================

impl ModelForward for Qwen3Model {
    type State = Qwen3State;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen3State {
            decode_bufs: DecodeBuffers::new(&self.ctx, &self.config)?,
            kv_cache: KVCache::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
            graph_state: CudaGraphState { graph: None },
            prefill_logits: None,
        })
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.decode_one_token(
                tokens[0],
                &mut state.kv_cache,
                &mut state.decode_bufs,
                &mut state.graph_state,
            )?;
            state.prefill_logits = None;
        } else {
            let start_pos = state.kv_cache.len();
            let hidden = self.get_embeddings_batch(tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.prefill_logits = Some(logits);
        }
        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        if let Some(ref logits) = state.prefill_logits {
            if params.is_greedy() {
                ops::argmax(&self.ctx, logits)
            } else {
                let logits_f32 = logits.to_host(&self.ctx)?;
                Ok(sampler::sample(&logits_f32, params, rng))
            }
        } else if params.is_greedy() {
            ops::read_argmax(&self.ctx, &state.decode_bufs.argmax_out)
        } else {
            let random_val: f32 = rng.random();
            ops::gpu_sample(
                &self.ctx,
                &state.decode_bufs.logits,
                &mut state.decode_bufs.sample_probs,
                params,
                random_val,
            )
        }
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }
}
