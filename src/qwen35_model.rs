//! Qwen3.5 model: weights, forward pass, generation (text-only).
//!
//! Separate from Qwen3Model due to fundamental architectural differences:
//! mixed layer types (full attention + linear attention), partial RoPE,
//! output gating, different head configurations.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
use log::{debug, info};
use rand::RngExt;
use rand::rngs::StdRng;
use std::time::Instant;

use crate::decode_buffers35::DecodeBuffers35;
use crate::kv_cache::KVCache;
use crate::model::StreamingStats;
use crate::ops;
use crate::qwen35_config::{Config35, LayerType};
use crate::recurrent_state::RecurrentState;
use crate::sampler::{self, SamplingParams};
use crate::tensor::*;
use crate::weight_loader::*;

/// CUDA Graph state for decode path.
struct CudaGraphState35 {
    graph: Option<CudaGraph>,
}

unsafe impl Send for CudaGraphState35 {}

/// Full attention layer weights (8 layers in Qwen3.5-4B).
pub struct FullAttentionLayer {
    /// Q projection including gate: [num_heads * head_dim * 2, hidden_size]
    pub q_proj: DeviceMatrix,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub k_proj: DeviceMatrix,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    pub v_proj: DeviceMatrix,
    /// Output projection: [hidden_size, num_heads * head_dim]
    pub o_proj: DeviceMatrix,
    /// QK norm weights: [head_dim] (broadcast to all heads)
    pub q_norm: DeviceVec,
    pub k_norm: DeviceVec,
}

/// Linear attention layer weights (24 layers in Qwen3.5-4B).
pub struct LinearAttentionLayer {
    /// Fused QKV projection: [q_dim + k_dim + v_dim, hidden_size]
    pub in_proj_qkv: DeviceMatrix,
    /// Z projection (for output gating): [z_dim, hidden_size]
    pub in_proj_z: DeviceMatrix,
    /// Beta projection: [num_value_heads, hidden_size]
    pub in_proj_b: DeviceMatrix,
    /// Alpha projection: [num_value_heads, hidden_size]
    pub in_proj_a: DeviceMatrix,
    /// Depthwise conv1d weight: [qkv_dim * conv_kernel_dim] (flattened from [qkv_dim, 1, 4])
    pub conv1d_weight: DeviceVec,
    /// dt_bias: [num_value_heads] bf16
    pub dt_bias: DeviceVec,
    /// A_log: [num_value_heads] f32
    pub a_log: CudaSlice<f32>,
    /// RMSNorm weight for output normalization: [value_head_dim] f32
    pub norm_weight: CudaSlice<f32>,
    /// Output projection: [hidden_size, z_dim]
    pub out_proj: DeviceMatrix,
}

/// Attention layer — either full or linear.
pub enum LayerKind {
    FullAttention(FullAttentionLayer),
    LinearAttention(LinearAttentionLayer),
}

/// MLP layer weights (shared between both layer types).
pub struct MLP35 {
    pub gate_proj: DeviceMatrix,
    pub up_proj: DeviceMatrix,
    pub down_proj: DeviceMatrix,
}

/// Transformer block for Qwen3.5.
pub struct TransformerBlock35 {
    pub input_layernorm: DeviceVec,
    pub attn: LayerKind,
    pub post_attention_layernorm: DeviceVec,
    pub mlp: MLP35,
}

/// Qwen3.5 model (text-only).
pub struct Qwen35Model {
    pub ctx: DeviceContext,
    pub config: Config35,
    pub embed_tokens: DeviceMatrix,
    pub layers: Vec<TransformerBlock35>,
    pub norm: DeviceVec,
    // Partial RoPE cache: [max_seq_len * rotary_dim]
    pub cos_cache: DeviceVec,
    pub sin_cache: DeviceVec,
    // Persistent decode state
    decode_bufs: Option<DecodeBuffers35>,
    kv_cache: Option<KVCache>,
    recurrent_state: Option<RecurrentState>,
    graph_state: CudaGraphState35,
    enable_cuda_graph: bool,
}

impl Qwen35Model {
    pub fn from_safetensors(model_path: &str) -> Result<Self> {
        Self::from_safetensors_with_options(model_path, true)
    }

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
            decode_bufs: None,
            kv_cache: None,
            recurrent_state: None,
            graph_state: CudaGraphState35 { graph: None },
            enable_cuda_graph,
        })
    }

    // ========================================================================
    // Forward pass — decode (single token, pre-allocated buffers)
    // ========================================================================

    /// Single decode step using pre-allocated buffers. Zero GPU allocation.
    fn decode_one_token(
        &self,
        token_id: u32,
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
        bufs: &mut DecodeBuffers35,
        graph_state: &mut CudaGraphState35,
    ) -> Result<()> {
        let pos = kv_cache.len(); // full attention seq len = total tokens processed
        let seq_len = pos + 1;

        kv_cache.init_if_needed(&self.ctx, self.config.head_dim)?;

        // Upload decode metadata (outside graph)
        self.ctx
            .stream
            .memcpy_htod(
                &[token_id as i32, pos as i32, seq_len as i32],
                &mut bufs.decode_meta,
            )
            .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {}", e))?;

        if !self.enable_cuda_graph {
            self.decode_kernels(kv_cache, recurrent, bufs)?;
            kv_cache.increment_seq_len();
            recurrent.seq_len += 1;
            return Ok(());
        }

        match &graph_state.graph {
            Some(graph) => {
                graph
                    .launch()
                    .map_err(|e| anyhow::anyhow!("CUDA Graph launch failed: {}", e))?;
            }
            None => {
                debug!("Capturing CUDA Graph for Qwen3.5 decode path...");
                self.ctx
                    .stream
                    .begin_capture(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .map_err(|e| anyhow::anyhow!("begin_capture failed: {}", e))?;

                self.decode_kernels(kv_cache, recurrent, bufs)?;

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
        recurrent.seq_len += 1;
        Ok(())
    }

    /// Pure kernel sequence for decode — no CPU-GPU sync, no allocation.
    fn decode_kernels(
        &self,
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
        bufs: &mut DecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_layers = self.layers.len();

        // 1. Embedding (reads token_id from decode_meta[0])
        ops::embedding_decode_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.decode_meta,
            &mut bufs.hidden,
        )?;

        // 2. First layer input norm (1+weight offset style)
        ops::rms_norm_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // 3. All transformer layers
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    self.decode_full_attention_layer(attn, full_idx, kv_cache, bufs)?;
                    full_idx += 1;
                }
                LayerKind::LinearAttention(attn) => {
                    self.decode_linear_attention_layer(attn, linear_idx, recurrent, bufs)?;
                    linear_idx += 1;
                }
            }

            // Post-attention: hidden += attn_proj; normed = rms_norm_offset(hidden, post_attn_ln)
            ops::fused_add_rms_norm_offset_into(
                &self.ctx,
                &mut bufs.hidden,
                &bufs.attn_proj,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            // MLP
            ops::fused_mlp_into(
                &self.ctx,
                &bufs.normed,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &layer.mlp.down_proj,
                &mut bufs.mlp_act,
                &mut bufs.mlp_out,
            )?;

            // Post-MLP: fused residual + next norm
            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            ops::fused_add_rms_norm_offset_into(
                &self.ctx,
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                eps,
                &mut bufs.normed,
            )?;
        }

        // 4. LM Head (normed already computed by last fused_add_rms_norm_offset)
        ops::gemv(
            &self.ctx,
            &self.embed_tokens,
            &bufs.normed,
            &mut bufs.logits,
        )?;

        Ok(())
    }

    /// Decode a full attention layer.
    fn decode_full_attention_layer(
        &self,
        attn: &FullAttentionLayer,
        kv_layer_idx: usize,
        kv_cache: &mut KVCache,
        bufs: &mut DecodeBuffers35,
    ) -> Result<()> {
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();
        let eps = self.config.rms_norm_eps;

        // QKV projection
        ops::gemv(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.proj_8192)?;
        ops::gemv(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_full)?;
        ops::gemv(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_full)?;

        // Fused attention HD256 (includes QK norm, partial RoPE, tiled attention, output gating)
        let (k_cache, v_cache) = kv_cache.get_cache_mut(&self.ctx, kv_layer_idx)?;
        ops::fused_attention_hd256_decode_into(
            &self.ctx,
            &bufs.proj_8192, // q_full (interleaved query+gate)
            &bufs.k_full,
            &bufs.v_full,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.decode_meta,
            k_cache,
            v_cache,
            &mut bufs.attn_out,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rotary_dim,
            scale,
            eps,
        )?;

        // O projection
        ops::gemv(&self.ctx, &attn.o_proj, &bufs.attn_out, &mut bufs.attn_proj)?;

        Ok(())
    }

    /// Decode a linear attention layer.
    fn decode_linear_attention_layer(
        &self,
        attn: &LinearAttentionLayer,
        recurrent_idx: usize,
        recurrent: &mut RecurrentState,
        bufs: &mut DecodeBuffers35,
    ) -> Result<()> {
        let c = &self.config;

        // Projections
        ops::gemv(
            &self.ctx,
            &attn.in_proj_qkv,
            &bufs.normed,
            &mut bufs.proj_8192,
        )?;
        ops::gemv(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.proj_z)?;
        ops::gemv(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.proj_b)?;
        ops::gemv(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.proj_a)?;

        // Conv1d decode (updates conv_state, applies SiLU)
        let layer_state = &mut recurrent.layers[recurrent_idx];
        ops::conv1d_decode_into(
            &self.ctx,
            &bufs.proj_8192, // qkv_raw input
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut bufs.qkv_conv, // conv output
            c.linear_conv_kernel_dim,
        )?;

        // Gated delta rule decode (updates recurrent state)
        ops::gated_delta_rule_decode_into(
            &self.ctx,
            &bufs.qkv_conv,
            &bufs.proj_b,
            &bufs.proj_a,
            &attn.dt_bias,
            &attn.a_log,
            &mut layer_state.state,
            &mut bufs.attn_out,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )?;

        // Gated RMSNorm: out = rms_norm(attn_out, norm_weight) * silu(z)
        ops::rms_norm_gated_into(
            &self.ctx,
            &bufs.attn_out,
            &attn.norm_weight,
            &bufs.proj_z,
            &mut bufs.norm_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        )?;

        // Output projection
        ops::gemv(
            &self.ctx,
            &attn.out_proj,
            &bufs.norm_gated,
            &mut bufs.attn_proj,
        )?;

        Ok(())
    }

    // ========================================================================
    // Forward pass — prefill (multi-token, naive sequential)
    // ========================================================================

    fn prefill_forward(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
    ) -> Result<DeviceVec> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "prefill_forward requires at least one token");
        let c = &self.config;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;

        // Get embeddings for all tokens
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let hidden_dim = c.hidden_size;
        let mut hidden_batch = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        // Process layers
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_batch = self.prefill_layer(
                layer_idx,
                layer,
                hidden_batch,
                &mut linear_idx,
                &mut full_idx,
                kv_cache,
                recurrent,
            )?;
        }

        // All layers processed. Advance seq_len counters once for the entire prefill.
        kv_cache.advance_seq_len(seq_len);
        recurrent.seq_len += seq_len;

        // Extract last token's hidden state
        let last_hidden = ops::extract_vec(&self.ctx, &hidden_batch, seq_len - 1)?;

        // Final norm (1+weight offset)
        let normed = {
            let mut out = DeviceVec::zeros(&self.ctx, hidden_dim)?;
            ops::rms_norm_offset_into(
                &self.ctx,
                &last_hidden,
                &self.norm,
                c.rms_norm_eps,
                &mut out,
            )?;
            out
        };

        // LM head (tied embeddings)
        ops::linear(&self.ctx, &normed, &self.embed_tokens)
    }

    /// Process one layer during prefill. Returns updated hidden_batch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer(
        &self,
        _layer_idx: usize,
        layer: &TransformerBlock35,
        hidden_batch: HiddenStates,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        recurrent: &mut RecurrentState,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let seq_len = hidden_batch.seq_len;

        // 1. Input layernorm — per-token (no batched offset norm kernel yet)
        // Use standard batched norm and add the offset correction manually
        // Actually we need the (1+w) variant. Process token by token for now.
        let mut normed_batch =
            self.batched_rms_norm_offset(&hidden_batch, &layer.input_layernorm, eps)?;

        // 2. Attention / Linear attention — per-token for correctness
        let attn_out_dim = match &layer.attn {
            LayerKind::FullAttention(_) => c.full_attn_q_dim(),
            LayerKind::LinearAttention(_) => c.linear_attn_z_dim(),
        };

        // Batch project, then per-token attention/recurrent
        let attn_results = match &layer.attn {
            LayerKind::FullAttention(attn) => {
                let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, &normed_batch)?;
                let k_batch = ops::gemm(&self.ctx, &attn.k_proj, &normed_batch)?;
                let v_batch = ops::gemm(&self.ctx, &attn.v_proj, &normed_batch)?;
                let mut attn_out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;

                let base_pos = kv_cache.len();
                let (kc, vc) = kv_cache.get_cache_mut(&self.ctx, *full_idx)?;
                ops::prefill_attention_hd256_batch(
                    &self.ctx,
                    &q_full_batch,
                    &k_batch,
                    &v_batch,
                    &attn.q_norm,
                    &attn.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    kc,
                    vc,
                    &mut attn_out_batch,
                    c.num_attention_heads,
                    c.num_key_value_heads,
                    base_pos,
                    c.rotary_dim,
                    eps,
                )?;

                *full_idx += 1;

                // O projection (batched)
                ops::gemm(&self.ctx, &attn.o_proj, &attn_out_batch)?
            }
            LayerKind::LinearAttention(attn) => {
                // Batch projections
                let qkv_batch = ops::gemm(&self.ctx, &attn.in_proj_qkv, &normed_batch)?;
                let z_batch = ops::gemm(&self.ctx, &attn.in_proj_z, &normed_batch)?;
                let b_batch = ops::gemm(&self.ctx, &attn.in_proj_b, &normed_batch)?;
                let a_batch = ops::gemm(&self.ctx, &attn.in_proj_a, &normed_batch)?;

                let qkv_dim = c.linear_attn_qkv_dim();
                let z_dim = c.linear_attn_z_dim();
                let layer_state = &mut recurrent.layers[*linear_idx];

                let mut out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;

                // Sequential: conv1d + gated delta rule per token
                for t in 0..seq_len {
                    let qkv_raw = ops::extract_vec(&self.ctx, &qkv_batch, t)?;
                    let z = ops::extract_vec(&self.ctx, &z_batch, t)?;
                    let b = ops::extract_vec(&self.ctx, &b_batch, t)?;
                    let a = ops::extract_vec(&self.ctx, &a_batch, t)?;

                    // Conv1d
                    let mut qkv_conv = DeviceVec::zeros(&self.ctx, qkv_dim)?;
                    ops::conv1d_decode_into(
                        &self.ctx,
                        &qkv_raw,
                        &attn.conv1d_weight,
                        &mut layer_state.conv_state,
                        &mut qkv_conv,
                        c.linear_conv_kernel_dim,
                    )?;

                    // GDR
                    let mut gdr_out = DeviceVec::zeros(&self.ctx, z_dim)?;
                    ops::gated_delta_rule_decode_into(
                        &self.ctx,
                        &qkv_conv,
                        &b,
                        &a,
                        &attn.dt_bias,
                        &attn.a_log,
                        &mut layer_state.state,
                        &mut gdr_out,
                        c.linear_num_key_heads,
                        c.linear_num_value_heads,
                        c.linear_key_head_dim,
                        c.linear_value_head_dim,
                    )?;

                    // Gated norm
                    let mut normed_out = DeviceVec::zeros(&self.ctx, z_dim)?;
                    ops::rms_norm_gated_into(
                        &self.ctx,
                        &gdr_out,
                        &attn.norm_weight,
                        &z,
                        &mut normed_out,
                        c.linear_num_value_heads,
                        c.linear_value_head_dim,
                        c.rms_norm_eps,
                    )?;

                    ops::write_vec(&self.ctx, &mut out_batch, t, &normed_out)?;
                }

                *linear_idx += 1;

                // Output projection (batched)
                ops::gemm(&self.ctx, &attn.out_proj, &out_batch)?
            }
        };

        // 3. Residual + post-attention layernorm
        let hidden_plus_attn = ops::add_batch(&self.ctx, &hidden_batch, &attn_results)?;

        // Post-attention layernorm (1+weight offset, batched per-token)
        normed_batch =
            self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

        // 4. MLP (batched)
        let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &normed_batch)?;
        let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &normed_batch)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;

        // 5. Residual
        ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)
    }

    fn batched_rms_norm_offset(
        &self,
        x: &HiddenStates,
        weight: &DeviceVec,
        eps: f32,
    ) -> Result<HiddenStates> {
        let mut out = HiddenStates::zeros(&self.ctx, x.hidden_dim, x.seq_len)?;
        ops::rms_norm_batch_offset_into(&self.ctx, x, weight, eps, &mut out)?;
        Ok(out)
    }

    // ========================================================================
    // Token selection
    // ========================================================================

    fn select_token(
        &self,
        logits: &DeviceVec,
        params: &SamplingParams,
        rng: &mut StdRng,
        sample_probs: Option<&mut CudaSlice<f32>>,
    ) -> Result<u32> {
        if params.is_greedy() {
            ops::argmax(&self.ctx, logits)
        } else if let Some(probs) = sample_probs {
            let random_val: f32 = rng.random();
            ops::gpu_sample(&self.ctx, logits, probs, params, random_val)
        } else {
            let logits_f32 = logits.to_host(&self.ctx)?;
            Ok(sampler::sample(&logits_f32, params, rng))
        }
    }

    // ========================================================================
    // State management
    // ========================================================================

    fn take_decode_bufs(&mut self) -> Result<DecodeBuffers35> {
        match self.decode_bufs.take() {
            Some(bufs) => Ok(bufs),
            None => DecodeBuffers35::new(&self.ctx, &self.config),
        }
    }

    fn take_kv_cache(&mut self) -> KVCache {
        match self.kv_cache.take() {
            Some(mut kv) => {
                kv.reset();
                kv
            }
            None => KVCache::new(
                self.config.num_full_attention_layers(),
                self.config.num_key_value_heads,
            ),
        }
    }

    fn take_recurrent_state(&mut self) -> Result<RecurrentState> {
        match self.recurrent_state.take() {
            Some(mut rs) => {
                rs.reset(&self.ctx)?;
                Ok(rs)
            }
            None => RecurrentState::new(&self.ctx, &self.config),
        }
    }

    fn take_graph_state(&mut self) -> CudaGraphState35 {
        std::mem::replace(&mut self.graph_state, CudaGraphState35 { graph: None })
    }

    // ========================================================================
    // Generation
    // ========================================================================

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        let ttft_start = Instant::now();
        let mut tokens = prompt_tokens.to_vec();

        let mut kv_cache = self.take_kv_cache();
        let mut recurrent = self.take_recurrent_state()?;
        let mut bufs = self.take_decode_bufs()?;
        let mut graph_state = self.take_graph_state();

        // Prefill
        let next_token = if prompt_tokens.len() == 1 {
            // Single token: use decode path (CUDA Graph eligible)
            self.decode_one_token(
                prompt_tokens[0],
                &mut kv_cache,
                &mut recurrent,
                &mut bufs,
                &mut graph_state,
            )?;
            self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?
        } else {
            // Multi-token: batch prefill
            let logits = self.prefill_forward(prompt_tokens, &mut kv_cache, &mut recurrent)?;
            self.select_token(&logits, params, rng, None)?
        };

        let ttft = ttft_start.elapsed();
        tokens.push(next_token);

        debug!(
            "TTFT: {:.2}ms (prompt_len={})",
            ttft.as_secs_f64() * 1000.0,
            prompt_tokens.len()
        );

        // Decode
        let tpot_start = Instant::now();
        let mut generated_count = 0;
        for _i in 1..max_new_tokens {
            self.decode_one_token(
                *tokens.last().unwrap(),
                &mut kv_cache,
                &mut recurrent,
                &mut bufs,
                &mut graph_state,
            )?;
            let next_token =
                self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?;

            if !params.ignore_eos && next_token == self.config.eos_token_id {
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
        }

        // Return state
        self.decode_bufs = Some(bufs);
        self.kv_cache = Some(kv_cache);
        self.recurrent_state = Some(recurrent);
        self.graph_state = graph_state;

        if generated_count > 0 {
            let tpot_total = tpot_start.elapsed();
            let tpot_avg = tpot_total.as_secs_f64() / generated_count as f64;
            debug!(
                "TPOT: {:.2}ms/tok (generated {} tokens in {:.2}ms, {:.1} tok/s)",
                tpot_avg * 1000.0,
                generated_count,
                tpot_total.as_secs_f64() * 1000.0,
                generated_count as f64 / tpot_total.as_secs_f64()
            );
        }

        Ok(tokens)
    }

    pub fn generate_streaming_with_callback<F>(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        mut on_token: F,
    ) -> Result<StreamingStats>
    where
        F: FnMut(u32) -> bool,
    {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt_tokens must not be empty");
        let mut tokens = prompt_tokens.to_vec();

        let mut kv_cache = self.take_kv_cache();
        let mut recurrent = self.take_recurrent_state()?;
        let mut bufs = self.take_decode_bufs()?;
        let mut graph_state = self.take_graph_state();

        // Prefill
        let ttft_start = Instant::now();
        let next_token = if prompt_tokens.len() == 1 {
            self.decode_one_token(
                prompt_tokens[0],
                &mut kv_cache,
                &mut recurrent,
                &mut bufs,
                &mut graph_state,
            )?;
            self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?
        } else {
            let logits = self.prefill_forward(prompt_tokens, &mut kv_cache, &mut recurrent)?;
            self.select_token(&logits, params, rng, None)?
        };

        let ttft = ttft_start.elapsed();
        debug!(
            "TTFT: {:.2}ms (prompt_len={})",
            ttft.as_secs_f64() * 1000.0,
            prompt_tokens.len()
        );

        tokens.push(next_token);
        let mut emitted_tokens = 1usize;
        if !on_token(next_token) {
            self.decode_bufs = Some(bufs);
            self.kv_cache = Some(kv_cache);
            self.recurrent_state = Some(recurrent);
            self.graph_state = graph_state;
            return Ok(StreamingStats {
                emitted_tokens,
                hit_eos: false,
                consumer_dropped: true,
            });
        }

        // Decode loop
        let tpot_start = Instant::now();
        let mut generated_count = 0;
        let mut hit_eos = false;
        for _i in 1..max_new_tokens {
            self.decode_one_token(
                *tokens.last().unwrap(),
                &mut kv_cache,
                &mut recurrent,
                &mut bufs,
                &mut graph_state,
            )?;
            let next_token =
                self.select_token(&bufs.logits, params, rng, Some(&mut bufs.sample_probs))?;

            if !params.ignore_eos && next_token == self.config.eos_token_id {
                hit_eos = true;
                break;
            }

            tokens.push(next_token);
            generated_count += 1;
            emitted_tokens += 1;

            if !on_token(next_token) {
                self.decode_bufs = Some(bufs);
                self.kv_cache = Some(kv_cache);
                self.recurrent_state = Some(recurrent);
                self.graph_state = graph_state;
                return Ok(StreamingStats {
                    emitted_tokens,
                    hit_eos: false,
                    consumer_dropped: true,
                });
            }
        }

        self.decode_bufs = Some(bufs);
        self.kv_cache = Some(kv_cache);
        self.recurrent_state = Some(recurrent);
        self.graph_state = graph_state;

        if generated_count > 0 {
            let tpot_total = tpot_start.elapsed();
            let tpot_avg = tpot_total.as_secs_f64() / generated_count as f64;
            debug!(
                "TPOT: {:.2}ms/tok (generated {} tokens in {:.2}ms, {:.1} tok/s)",
                tpot_avg * 1000.0,
                generated_count,
                tpot_total.as_secs_f64() * 1000.0,
                generated_count as f64 / tpot_total.as_secs_f64()
            );
        }

        Ok(StreamingStats {
            emitted_tokens,
            hit_eos,
            consumer_dropped: false,
        })
    }

    pub fn generate_streaming(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        tx: tokio::sync::mpsc::UnboundedSender<u32>,
    ) -> Result<()> {
        let _ = self.generate_streaming_with_callback(
            prompt_tokens,
            max_new_tokens,
            params,
            rng,
            |token_id| tx.send(token_id).is_ok(),
        )?;
        Ok(())
    }

    // ========================================================================
    // Shape verification
    // ========================================================================

    pub fn verify_shapes(&self) -> Result<()> {
        let c = &self.config;

        assert_shape(
            "embed_tokens",
            &self.embed_tokens,
            c.vocab_size,
            c.hidden_size,
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            let prefix = format!("layer.{}", i);

            assert_vec_len(
                &format!("{}.input_layernorm", prefix),
                &layer.input_layernorm,
                c.hidden_size,
            )?;
            assert_vec_len(
                &format!("{}.post_attn_layernorm", prefix),
                &layer.post_attention_layernorm,
                c.hidden_size,
            )?;

            assert_shape(
                &format!("{}.mlp.gate_proj", prefix),
                &layer.mlp.gate_proj,
                c.intermediate_size,
                c.hidden_size,
            )?;
            assert_shape(
                &format!("{}.mlp.up_proj", prefix),
                &layer.mlp.up_proj,
                c.intermediate_size,
                c.hidden_size,
            )?;
            assert_shape(
                &format!("{}.mlp.down_proj", prefix),
                &layer.mlp.down_proj,
                c.hidden_size,
                c.intermediate_size,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    let q_proj_dim = c.full_attn_q_proj_dim();
                    let kv_dim = c.full_attn_kv_dim();
                    let q_dim = c.full_attn_q_dim();

                    assert_shape(
                        &format!("{}.q_proj", prefix),
                        &attn.q_proj,
                        q_proj_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.k_proj", prefix),
                        &attn.k_proj,
                        kv_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.v_proj", prefix),
                        &attn.v_proj,
                        kv_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.o_proj", prefix),
                        &attn.o_proj,
                        c.hidden_size,
                        q_dim,
                    )?;
                    assert_vec_len(&format!("{}.q_norm", prefix), &attn.q_norm, c.head_dim)?;
                    assert_vec_len(&format!("{}.k_norm", prefix), &attn.k_norm, c.head_dim)?;
                }
                LayerKind::LinearAttention(attn) => {
                    let qkv_dim = c.linear_attn_qkv_dim();
                    let z_dim = c.linear_attn_z_dim();
                    let num_v_heads = c.linear_num_value_heads;

                    assert_shape(
                        &format!("{}.in_proj_qkv", prefix),
                        &attn.in_proj_qkv,
                        qkv_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.in_proj_z", prefix),
                        &attn.in_proj_z,
                        z_dim,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.in_proj_b", prefix),
                        &attn.in_proj_b,
                        num_v_heads,
                        c.hidden_size,
                    )?;
                    assert_shape(
                        &format!("{}.in_proj_a", prefix),
                        &attn.in_proj_a,
                        num_v_heads,
                        c.hidden_size,
                    )?;
                    assert_vec_len(
                        &format!("{}.conv1d_weight", prefix),
                        &attn.conv1d_weight,
                        qkv_dim * c.linear_conv_kernel_dim,
                    )?;
                    assert_vec_len(&format!("{}.dt_bias", prefix), &attn.dt_bias, num_v_heads)?;
                    assert_shape(
                        &format!("{}.out_proj", prefix),
                        &attn.out_proj,
                        c.hidden_size,
                        z_dim,
                    )?;
                }
            }
        }

        assert_vec_len("norm", &self.norm, c.hidden_size)?;

        info!("All weight shapes verified successfully");
        Ok(())
    }
}

fn assert_shape(name: &str, m: &DeviceMatrix, rows: usize, cols: usize) -> Result<()> {
    anyhow::ensure!(
        m.rows == rows && m.cols == cols,
        "{}: expected [{}, {}], got [{}, {}]",
        name,
        rows,
        cols,
        m.rows,
        m.cols
    );
    Ok(())
}

fn assert_vec_len(name: &str, v: &DeviceVec, expected: usize) -> Result<()> {
    anyhow::ensure!(
        v.len == expected,
        "{}: expected len {}, got {}",
        name,
        expected,
        v.len
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    #[test]
    fn test_load_qwen35_model() {
        let model = Qwen35Model::from_safetensors(MODEL_PATH).unwrap();

        assert_eq!(model.layers.len(), 32);
        assert_eq!(model.config.num_hidden_layers, 32);

        let full_count = model
            .layers
            .iter()
            .filter(|l| matches!(l.attn, LayerKind::FullAttention(_)))
            .count();
        let linear_count = model
            .layers
            .iter()
            .filter(|l| matches!(l.attn, LayerKind::LinearAttention(_)))
            .count();
        assert_eq!(full_count, 8);
        assert_eq!(linear_count, 24);

        model.verify_shapes().unwrap();
    }
}
