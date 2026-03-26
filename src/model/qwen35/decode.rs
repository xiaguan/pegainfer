use anyhow::Result;
use log::debug;

use cudarc::driver::safe::CudaGraph;
use cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
use cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

use super::decode_buffers::DecodeBuffers35;
use super::recurrent_state::RecurrentState;
use super::weights::{FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model};
use crate::model::kv_cache::KVCache;
use crate::ops;

/// CUDA Graph state for decode path.
pub(super) struct CudaGraphState35 {
    pub(super) graph: Option<CudaGraph>,
}

unsafe impl Send for CudaGraphState35 {}

impl Qwen35Model {
    /// Single decode step using pre-allocated buffers. Zero GPU allocation.
    pub(super) fn decode_one_token(
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

        // 5. Argmax (pre-allocated, captured inside CUDA Graph)
        ops::argmax_into(&self.ctx, &bufs.logits, &mut bufs.argmax_out);

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
}
