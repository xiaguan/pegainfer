//! Pre-allocated GPU buffers for batched decode (multiple requests, 1 token each).

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config;
use crate::kv_pool::KvState;
use crate::tensor::{DeviceContext, HiddenStates};

/// Pre-allocated buffers for batch decode. All tensors are sized for `max_batch_size`.
///
/// Uses `HiddenStates` (2D) instead of `DeviceVec` (1D) — the "seq_len" dimension
/// is actually the batch dimension (one token per request).
pub(crate) struct BatchDecodeBuffers {
    pub(crate) max_batch_size: usize,

    // Per-layer intermediates [dim, max_batch_size]
    pub(crate) normed: HiddenStates,
    pub(crate) q: HiddenStates,
    pub(crate) k: HiddenStates,
    pub(crate) v: HiddenStates,
    pub(crate) attn_out: HiddenStates,
    pub(crate) attn_proj: HiddenStates,
    pub(crate) gate_out: HiddenStates,
    pub(crate) up_out: HiddenStates,
    pub(crate) mlp_act: HiddenStates,
    pub(crate) mlp_out: HiddenStates,
    pub(crate) hidden: HiddenStates,
    pub(crate) logits: HiddenStates,

    // GPU metadata
    pub(crate) token_ids_d: CudaSlice<i32>,
    pub(crate) positions_d: CudaSlice<i32>,

    // Paged attention metadata (concatenated across requests, CSR format)
    pub(crate) page_indices_d: CudaSlice<i32>,
    pub(crate) page_indptr_d: CudaSlice<i32>,
    pub(crate) last_page_len_d: CudaSlice<i32>,
    pub(crate) request_indices_d: CudaSlice<i32>,
    pub(crate) kv_tile_indices_d: CudaSlice<i32>,
    pub(crate) kv_chunk_size_d: CudaSlice<i32>,

    // Per-request sampling scratch (reused across requests in a loop)
    pub(crate) sample_probs: CudaSlice<f32>,
    pub(crate) sample_out: CudaSlice<i32>,
}

impl BatchDecodeBuffers {
    pub(crate) fn new(
        ctx: &DeviceContext,
        config: &Config,
        max_batch_size: usize,
        max_total_pages: usize,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let bs = max_batch_size;

        Ok(Self {
            max_batch_size: bs,
            normed: HiddenStates::zeros(ctx, h, bs)?,
            q: HiddenStates::zeros(ctx, q_dim, bs)?,
            k: HiddenStates::zeros(ctx, kv_dim, bs)?,
            v: HiddenStates::zeros(ctx, kv_dim, bs)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, bs)?,
            attn_proj: HiddenStates::zeros(ctx, h, bs)?,
            gate_out: HiddenStates::zeros(ctx, config.intermediate_size, bs)?,
            up_out: HiddenStates::zeros(ctx, config.intermediate_size, bs)?,
            mlp_act: HiddenStates::zeros(ctx, config.intermediate_size, bs)?,
            mlp_out: HiddenStates::zeros(ctx, h, bs)?,
            hidden: HiddenStates::zeros(ctx, h, bs)?,
            logits: HiddenStates::zeros(ctx, config.vocab_size, bs)?,
            token_ids_d: ctx.stream.alloc_zeros(bs)?,
            positions_d: ctx.stream.alloc_zeros(bs)?,
            // Paged attention: worst case all requests use max_total_pages
            page_indices_d: ctx.stream.alloc_zeros(max_total_pages)?,
            page_indptr_d: ctx.stream.alloc_zeros(bs + 1)?,
            last_page_len_d: ctx.stream.alloc_zeros(bs)?,
            request_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_tile_indices_d: ctx.stream.alloc_zeros(bs)?,
            kv_chunk_size_d: ctx.stream.alloc_zeros(bs)?,
            sample_probs: ctx.stream.alloc_zeros(config.vocab_size)?,
            sample_out: ctx.stream.alloc_zeros(1)?,
        })
    }

    /// Set actual batch size for this step. Adjusts the seq_len field on all HiddenStates.
    pub(crate) fn set_batch_size(&mut self, bs: usize) {
        assert!(bs <= self.max_batch_size);
        self.normed.seq_len = bs;
        self.q.seq_len = bs;
        self.k.seq_len = bs;
        self.v.seq_len = bs;
        self.attn_out.seq_len = bs;
        self.attn_proj.seq_len = bs;
        self.gate_out.seq_len = bs;
        self.up_out.seq_len = bs;
        self.mlp_act.seq_len = bs;
        self.mlp_out.seq_len = bs;
        self.hidden.seq_len = bs;
        self.logits.seq_len = bs;
    }

    /// Sync paged attention metadata from multiple KvStates to GPU buffers.
    pub(crate) fn sync_paged_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_states: &[&KvState],
    ) -> Result<()> {
        let bs = kv_states.len();

        // Build concatenated page_indices and CSR indptr
        let mut all_page_indices = Vec::new();
        let mut indptr = vec![0i32];
        let mut last_page_lens = Vec::with_capacity(bs);
        let mut chunk_sizes = Vec::with_capacity(bs);

        for kv in kv_states {
            let pages = kv.page_indices_i32();
            all_page_indices.extend_from_slice(&pages);
            indptr.push(all_page_indices.len() as i32);
            last_page_lens.push(kv.last_page_len() as i32);
            chunk_sizes.push(kv.seq_len() as i32);
        }

        // Non-partition: request_indices = [0, 1, ..., bs-1], kv_tile_indices = [0, 0, ..., 0]
        let request_indices: Vec<i32> = (0..bs as i32).collect();
        let kv_tile_indices = vec![0i32; bs];

        ctx.stream
            .memcpy_htod(&all_page_indices, &mut self.page_indices_d)?;
        ctx.stream.memcpy_htod(&indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_lens, &mut self.last_page_len_d)?;
        ctx.stream
            .memcpy_htod(&chunk_sizes, &mut self.kv_chunk_size_d)?;
        ctx.stream
            .memcpy_htod(&request_indices, &mut self.request_indices_d)?;
        ctx.stream
            .memcpy_htod(&kv_tile_indices, &mut self.kv_tile_indices_d)?;

        Ok(())
    }
}
