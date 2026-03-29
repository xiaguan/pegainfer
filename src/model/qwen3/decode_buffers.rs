//! Pre-allocated GPU buffers for zero-allocation decode steps.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config;
use crate::tensor::{DeviceContext, DeviceVec};

/// Pre-allocated temporary buffers for the single-token decode path.
///
/// All buffer dimensions are determined by the model config and remain fixed
/// for the entire generation. Reusing these across decode steps eliminates
/// ~292 cudaMalloc/cudaFree calls per token.
pub(crate) struct DecodeBuffers {
    /// RMSNorm output / general scratch (hidden_size)
    pub(crate) normed: DeviceVec,
    /// Q projection output (num_attention_heads * head_dim)
    pub(crate) q: DeviceVec,
    /// K projection output (num_key_value_heads * head_dim)
    pub(crate) k: DeviceVec,
    /// V projection output (num_key_value_heads * head_dim)
    pub(crate) v: DeviceVec,
    /// Fused attention output (num_attention_heads * head_dim)
    pub(crate) attn_out: DeviceVec,
    /// O projection output (hidden_size)
    pub(crate) attn_proj: DeviceVec,
    /// Fused MLP intermediate activation (intermediate_size)
    pub(crate) mlp_act: DeviceVec,
    /// Fused MLP output (hidden_size)
    pub(crate) mlp_out: DeviceVec,
    /// Current hidden state, persists across layers (hidden_size)
    pub(crate) hidden: DeviceVec,
    /// LM head logits (vocab_size)
    pub(crate) logits: DeviceVec,
    /// Decode metadata on GPU: [token_id, current_pos, seq_len] as i32
    pub(crate) decode_meta: CudaSlice<i32>,
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size)
    pub(crate) sample_probs: CudaSlice<f32>,
    /// Pre-allocated sampling output (1 element, token id)
    pub(crate) sample_out: CudaSlice<i32>,

    // -- Paged attention metadata (CUDA Graph safe) --
    // Updated via memcpy_htod before each decode step.
    /// GPU page indices for this request [max_pages capacity]
    pub(crate) page_indices_d: CudaSlice<i32>,
    /// GPU page indptr [2]: [0, num_pages] for batch_size=1
    pub(crate) page_indptr_d: CudaSlice<i32>,
    /// GPU last page occupancy [1]
    pub(crate) last_page_len_d: CudaSlice<i32>,
    /// GPU request indices [1]: constant [0] for batch_size=1
    pub(crate) request_indices_d: CudaSlice<i32>,
    /// GPU KV tile indices [1]: constant [0] for non-partition
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    /// GPU KV chunk size [1]: [seq_len], updated per step
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,
}

impl DecodeBuffers {
    pub(crate) fn new(ctx: &DeviceContext, config: &Config, max_pages: usize) -> Result<Self> {
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        Ok(Self {
            normed: DeviceVec::zeros(ctx, h)?,
            q: DeviceVec::zeros(ctx, q_dim)?,
            k: DeviceVec::zeros(ctx, kv_dim)?,
            v: DeviceVec::zeros(ctx, kv_dim)?,
            attn_out: DeviceVec::zeros(ctx, q_dim)?,
            attn_proj: DeviceVec::zeros(ctx, h)?,
            mlp_act: DeviceVec::zeros(ctx, config.intermediate_size)?,
            mlp_out: DeviceVec::zeros(ctx, h)?,
            hidden: DeviceVec::zeros(ctx, h)?,
            logits: DeviceVec::zeros(ctx, config.vocab_size)?,
            decode_meta: ctx
                .stream
                .alloc_zeros(3)
                .map_err(|e| anyhow::anyhow!("Alloc decode_meta failed: {}", e))?,
            sample_probs: ctx
                .stream
                .alloc_zeros(config.vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {}", e))?,
            sample_out: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc sample_out failed: {}", e))?,
            // Paged attention metadata — pre-allocated for CUDA Graph pointer stability
            page_indices_d: ctx
                .stream
                .alloc_zeros(max_pages)
                .map_err(|e| anyhow::anyhow!("Alloc page_indices_d failed: {}", e))?,
            page_indptr_d: ctx
                .stream
                .alloc_zeros(2)
                .map_err(|e| anyhow::anyhow!("Alloc page_indptr_d failed: {}", e))?,
            last_page_len_d: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc last_page_len_d failed: {}", e))?,
            request_indices_d: ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("Alloc request_indices_d failed: {}", e))?,
            kv_tile_indices_d: ctx
                .stream
                .clone_htod(&[0i32])
                .map_err(|e| anyhow::anyhow!("Alloc kv_tile_indices_d failed: {}", e))?,
            kv_chunk_size_d: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc kv_chunk_size_d failed: {}", e))?,
        })
    }

    /// Sync paged attention metadata from KvState to pre-allocated GPU buffers.
    /// Must be called BEFORE CUDA Graph capture/replay.
    pub(crate) fn sync_paged_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_state: &crate::kv_pool::KvState,
    ) -> Result<()> {
        let page_indices = kv_state.page_indices_i32();
        let num_pages = kv_state.num_pages() as i32;
        let last_page_len = kv_state.last_page_len() as i32;
        let seq_len = kv_state.seq_len() as i32;

        ctx.stream
            .memcpy_htod(&page_indices, &mut self.page_indices_d)
            .map_err(|e| anyhow::anyhow!("sync page_indices failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&[0i32, num_pages], &mut self.page_indptr_d)
            .map_err(|e| anyhow::anyhow!("sync page_indptr failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&[last_page_len], &mut self.last_page_len_d)
            .map_err(|e| anyhow::anyhow!("sync last_page_len failed: {e}"))?;
        ctx.stream
            .memcpy_htod(&[seq_len], &mut self.kv_chunk_size_d)
            .map_err(|e| anyhow::anyhow!("sync kv_chunk_size failed: {e}"))?;

        Ok(())
    }
}
