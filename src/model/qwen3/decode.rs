use anyhow::Result;

use super::decode_buffers::DecodeBuffers;
use super::weights::{Qwen3Model, TransformerBlock};
use crate::kv_pool::{KvLayout, KvState};
use crate::model::cuda_graph::CudaGraphState;
use crate::ops;

impl Qwen3Model {
    /// Single decode step using pre-allocated buffers.
    pub(super) fn decode_one_token(
        &self,
        token_id: u32,
        kv_state: &mut KvState,
        bufs: &mut DecodeBuffers,
        graph_state: &mut CudaGraphState,
    ) -> Result<()> {
        let pos = kv_state.seq_len();

        // Grow pages if needed, then advance seq_len so desc() reflects the
        // new token. FlashInfer's AppendPagedKVCacheDecode writes at position
        // (seq_len - 1), so desc must already include the new token.
        kv_state.ensure_capacity(pos + 1)?;
        kv_state.advance(1);

        // --- Sync metadata to GPU (before graph — stable pointers, varying data) ---

        // decode_meta: [token_id, position, seq_len_after]
        self.ctx
            .stream
            .memcpy_htod(
                &[token_id as i32, pos as i32, (pos + 1) as i32],
                &mut bufs.decode_meta,
            )
            .map_err(|e| anyhow::anyhow!("H2D decode_meta failed: {}", e))?;

        // Paged attention metadata
        bufs.sync_paged_meta(&self.ctx, kv_state)?;

        // --- GPU kernels (captured into CUDA Graph on first call) ---
        let layout = *kv_state.layout();
        if self.enable_cuda_graph {
            let kv_buffer = kv_state.buffer();
            graph_state.run_or_capture(&self.ctx, || {
                self.decode_kernels_paged(kv_buffer, &layout, bufs)
            })?;
        } else {
            let kv_buffer = kv_state.buffer();
            self.decode_kernels_paged(kv_buffer, &layout, bufs)?;
        }

        Ok(())
    }

    fn decode_kernels_paged(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
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
            self.decode_layer_paged(layer_idx, layer, kv_buffer, layout, bufs)?;

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
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bufs: &mut DecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Q/K/V projections from fused qkv_proj
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        ops::gemv_rows(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim,
            &bufs.normed,
            &mut bufs.q,
        )?;
        ops::gemv_rows(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k,
        )?;
        ops::gemv_rows(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v,
        )?;

        // QK RMSNorm + RoPE (reads position from decode_meta[1])
        ops::qk_norm_rope_into(
            &self.ctx,
            &mut bufs.q,
            &mut bufs.k,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.decode_meta,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            eps,
        );

        // KV append + paged attention (FlashInfer, pre-allocated metadata)
        ops::paged_attention_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            kv_buffer,
            layout,
            layer_idx,
            &bufs.page_indices_d,
            &bufs.page_indptr_d,
            &bufs.last_page_len_d,
            &bufs.request_indices_d,
            &bufs.kv_tile_indices_d,
            &bufs.kv_chunk_size_d,
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

        // MLP (gate+up from fused weight matrix)
        ops::fused_mlp_gate_up_into(
            &self.ctx,
            &bufs.normed,
            &layer.mlp.gate_up_proj,
            layer.mlp.intermediate_size,
            &layer.mlp.down_proj,
            &mut bufs.mlp_act,
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }
}
