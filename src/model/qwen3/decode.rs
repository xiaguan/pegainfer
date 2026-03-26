use anyhow::Result;

use super::decode_buffers::DecodeBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::KVCache;
use crate::ops;

impl Qwen3Model {
    /// Single decode step using pre-allocated buffers. Zero GPU allocation.
    pub(super) fn decode_one_token(
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

        if self.enable_cuda_graph {
            graph_state.run_or_capture(&self.ctx, || self.decode_kernels(kv_cache, bufs))?;
        } else {
            self.decode_kernels(kv_cache, bufs)?;
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
