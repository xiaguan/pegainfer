use anyhow::Result;
use cudarc::driver::CudaSlice;
use log::{debug, info};
use std::time::Instant;

use super::config::{Config35, LayerType};
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec};
use crate::weight_loader::{
    load_shard_info_fixed, load_tensor_1d, load_tensor_1d_f32, load_tensor_2d, mmap_shards,
    precompute_rope,
};

/// Full attention layer weights (8 layers in Qwen3.5-4B).
pub(super) struct FullAttentionLayer {
    /// Q projection including gate: [num_heads * head_dim * 2, hidden_size]
    pub(super) q_proj: DeviceMatrix,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub(super) k_proj: DeviceMatrix,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub(super) v_proj: DeviceMatrix,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub(super) o_proj: DeviceMatrix,
    /// QK norm weights: [head_dim] (broadcast to all heads)
    pub(super) q_norm: DeviceVec,
    pub(super) k_norm: DeviceVec,
}

/// Linear attention layer weights (24 layers in Qwen3.5-4B).
pub(super) struct LinearAttentionLayer {
    /// Fused QKV projection: [q_dim + k_dim + v_dim, hidden_size]
    pub(super) in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [z_dim, hidden_size]
    pub(super) in_proj_z: DeviceMatrix,
    /// Beta projection: [num_value_heads, hidden_size]
    pub(super) in_proj_b: DeviceMatrix,
    /// Alpha projection: [num_value_heads, hidden_size]
    pub(super) in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [qkv_dim * conv_kernel_dim] (flattened from [qkv_dim, 1, 4])
    pub(super) conv1d_weight: DeviceVec,
    /// dt_bias: [num_value_heads] bf16
    pub(super) dt_bias: DeviceVec,
    /// A_log: [num_value_heads] f32
    pub(super) a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32
    pub(super) norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, z_dim]
    pub(super) out_proj: DeviceMatrix,
}

/// Attention layer — either full or linear.
pub(super) enum LayerKind {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

/// MLP layer weights (shared between both layer types).
#[allow(clippy::struct_field_names)]
pub(super) struct MLP35 {
    pub(super) gate_proj: DeviceMatrix,
    pub(super) up_proj: DeviceMatrix,
    pub(super) down_proj: DeviceMatrix,
}

/// Transformer block for Qwen3.5.
pub(super) struct TransformerBlock35 {
    pub(super) input_layernorm: DeviceVec,
    pub(super) attn: LayerKind,
    pub(super) post_attention_layernorm: DeviceVec,
    pub(super) mlp: MLP35,
}

/// Qwen3.5 model (text-only).
pub struct Qwen35Model {
    pub(super) ctx: DeviceContext,
    pub(super) config: Config35,
    pub(super) embed_tokens: DeviceMatrix,
    pub(super) layers: Vec<TransformerBlock35>,
    pub(super) norm: DeviceVec,
    // Partial RoPE cache: [max_seq_len * rotary_dim]
    pub(super) cos_cache: DeviceVec,
    pub(super) sin_cache: DeviceVec,
    pub(super) enable_cuda_graph: bool,
}

impl Qwen35Model {
    pub fn from_safetensors_with_options(
        model_path: &str,
        enable_cuda_graph: bool,
    ) -> Result<Self> {
        info!("Loading Qwen3.5 model from: {}", model_path);
        debug!("Initializing GPU");
        let ctx = DeviceContext::new()?;

        let config = Config35::from_file(model_path)?;
        debug!(
            "Config: hidden_size={}, num_layers={}, full_attn={}, linear_attn={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_full_attention_layers(),
            config.num_hidden_layers - config.num_full_attention_layers()
        );

        let (shard_paths, weight_map) = load_shard_info_fixed(model_path)?;
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
        // Weight prefix for Qwen3.5 text model
        let wp = "model.language_model";

        debug!("Loading embeddings to GPU");
        let embed_tokens = load_tensor_2d(
            &ctx,
            &shards,
            &weight_map,
            &format!("{}.embed_tokens.weight", wp),
        )?;
        debug!(
            "embed_tokens: [{}, {}]",
            embed_tokens.rows, embed_tokens.cols
        );

        debug!(
            "Loading layers to GPU: num_layers={}",
            config.num_hidden_layers
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("{}.layers.{}", wp, i);
            let layer_type = config.layer_types[i];

            let attn = match layer_type {
                LayerType::FullAttention => {
                    let attn_prefix = format!("{}.self_attn", prefix);
                    LayerKind::FullAttention(FullAttentionLayer {
                        q_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.q_proj.weight", attn_prefix),
                        )?,
                        k_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_proj.weight", attn_prefix),
                        )?,
                        v_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.v_proj.weight", attn_prefix),
                        )?,
                        o_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.o_proj.weight", attn_prefix),
                        )?,
                        q_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.q_norm.weight", attn_prefix),
                        )?,
                        k_norm: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.k_norm.weight", attn_prefix),
                        )?,
                    })
                }
                LayerType::LinearAttention => {
                    let attn_prefix = format!("{}.linear_attn", prefix);
                    LayerKind::LinearAttention(LinearAttentionLayer {
                        in_proj_qkv: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_qkv.weight", attn_prefix),
                        )?,
                        in_proj_z: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_z.weight", attn_prefix),
                        )?,
                        in_proj_b: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_b.weight", attn_prefix),
                        )?,
                        in_proj_a: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.in_proj_a.weight", attn_prefix),
                        )?,
                        conv1d_weight: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.conv1d.weight", attn_prefix),
                        )?,
                        dt_bias: load_tensor_1d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.dt_bias", attn_prefix),
                        )?,
                        a_log: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.A_log", attn_prefix),
                        )?,
                        norm_weight: load_tensor_1d_f32(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.norm.weight", attn_prefix),
                        )?,
                        out_proj: load_tensor_2d(
                            &ctx,
                            &shards,
                            &weight_map,
                            &format!("{}.out_proj.weight", attn_prefix),
                        )?,
                    })
                }
            };

            let block = TransformerBlock35 {
                input_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.input_layernorm.weight", prefix),
                )?,
                attn,
                post_attention_layernorm: load_tensor_1d(
                    &ctx,
                    &shards,
                    &weight_map,
                    &format!("{}.post_attention_layernorm.weight", prefix),
                )?,
                mlp: MLP35 {
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

            debug!(
                "Loaded layer {}/{}: {:?}",
                i + 1,
                config.num_hidden_layers,
                layer_type
            );
            layers.push(block);
        }

        let norm = load_tensor_1d(&ctx, &shards, &weight_map, &format!("{}.norm.weight", wp))?;

        debug!(
            "Precomputing partial RoPE cache (rotary_dim={})",
            config.rotary_dim
        );
        let (cos_cache, sin_cache) =
            precompute_rope(&ctx, config.rotary_dim, 4096, config.rope_theta)?;

        ctx.sync()?;
        info!(
            "GPU transfer complete in {:.0}ms",
            t_gpu.elapsed().as_secs_f64() * 1e3
        );
        info!("Qwen3.5 GPU model loaded successfully");
        if enable_cuda_graph {
            debug!("Decode path CUDA Graph is enabled");
        } else {
            debug!("Decode path CUDA Graph is disabled");
        }

        Ok(Self {
            ctx,
            config,
            embed_tokens,
            layers,
            norm,
            cos_cache,
            sin_cache,
            enable_cuda_graph,
        })
    }
}
