use anyhow::Result;

use super::decode_buffers::DecodeBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::kv_pool::KvState;
use crate::ops;

impl Qwen3Model {
    /// Single decode step using pre-allocated buffers.
    pub(super) fn decode_one_token(
        &self,
        token_id: u32,
        kv_state: &mut KvState,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let pos = kv_state.seq_len();

        // Grow pages if needed, then advance seq_len so desc() reflects the
        // new token. FlashInfer's AppendPagedKVCacheDecode writes at position
        // (seq_len - 1), so desc must already include the new token.
        kv_state.ensure_capacity(pos + 1)?;
        kv_state.advance(1);

        // Upload GPU metadata (embedding kernel reads token_id from here)
        self.ctx
            .stream
            .memcpy_htod(
                &[token_id as i32, pos as i32, (pos + 1) as i32],
                &mut bufs.decode_meta,
            )
            .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {}", e))?;

        self.decode_kernels_paged(kv_state, bufs)?;

        Ok(())
    }

    fn decode_kernels_paged(&self, kv_state: &KvState, bufs: &mut DecodeBuffers) -> Result<()> {
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

        let desc = kv_state.desc();
        // RoPE position = index of the new token = seq_len - 1 (after advance)
        let position = kv_state.seq_len() - 1;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.decode_layer_paged(layer_idx, layer, &desc, position, bufs)?;

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

    fn decode_layer_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        desc: &crate::kv_pool::KvDesc<'_>,
        position: usize,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Q/K/V projections
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

        // QK RMSNorm + RoPE (in-place)
        ops::qk_norm_rope_into(
            &self.ctx,
            &mut bufs.q,
            &mut bufs.k,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            position,
            eps,
        );

        // KV append + paged attention (FlashInfer)
        ops::paged_attention_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            desc,
            layer_idx,
            &mut bufs.attn_out,
            self.config.num_attention_heads,
        )?;

        // O projection
        ops::gemv(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        )?;

        // Residual + LayerNorm
        ops::fused_add_rms_norm_into(
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

        Ok(())
    }
}
