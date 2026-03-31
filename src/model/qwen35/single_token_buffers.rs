//! Pre-allocated buffers for single-token prefill (decode-as-prefill).
//!
//! Eliminates per-step HiddenStates allocation overhead by caching
//! all intermediate buffers for seq_len=1 processing.

use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::config::Config35;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// All HiddenStates(seq_len=1) buffers needed for one prefill step.
/// Allocated once, reused every decode step. Eliminates ~500 alloc/free pairs per token.
pub(super) struct SingleTokenBuffers {
    // ── Main hidden state ───────────────────────────────────────────
    pub hidden_a: HiddenStates,

    // ── Norm output ─────────────────────────────────────────────────
    pub normed: HiddenStates,

    // ── Attention outputs (reused across layers) ────────────────────
    /// Full attention: q_full [q_proj_dim, 1]
    pub q_full: HiddenStates,
    /// Full attention: k [kv_dim, 1]
    pub k_attn: HiddenStates,
    /// Full attention: v [kv_dim, 1]
    pub v_attn: HiddenStates,
    /// Full attention: prepared q after norm+rope [q_dim, 1]
    pub q_prepped: HiddenStates,
    /// Full attention: attention output [q_dim, 1]
    pub attn_out_full: HiddenStates,

    // ── Linear attention outputs (reused across layers) ─────────────
    /// Linear: qkv projection [qkv_dim, 1]
    pub qkv: HiddenStates,
    /// Linear: z projection [z_dim, 1]
    pub z: HiddenStates,
    /// Linear: b projection [b_dim, 1]
    pub b_proj: HiddenStates,
    /// Linear: a projection [a_dim, 1]
    pub a_proj: HiddenStates,
    /// Linear: conv1d output [qkv_dim, 1]
    pub qkv_conv: HiddenStates,
    /// Linear: GDR output [z_dim, 1]
    pub gdr_out: HiddenStates,
    /// Linear: gated norm output [z_dim, 1]
    pub normed_gated: HiddenStates,

    // ── Compact K/V for paged_kv_append (one token, NHD) ───────────
    /// Compact K scratch: [kv_dim] = num_kv_heads * head_dim.
    pub kv_k_compact: DeviceVec,
    /// Compact V scratch: [kv_dim] = num_kv_heads * head_dim.
    pub kv_v_compact: DeviceVec,

    // ── Attention results (after O/out projection) [hidden_size, 1] ─
    pub attn_results: HiddenStates,

    // ── MLP outputs (reused across layers) ──────────────────────────
    pub gate_out: HiddenStates,
    pub up_out: HiddenStates,
    pub act_out: HiddenStates,
    pub mlp_out: HiddenStates,

    // ── Residual intermediate [hidden_size, 1] ──────────────────────
    pub hidden_mid: HiddenStates,

    // ── Final outputs ───────────────────────────────────────────────
    pub last_normed: DeviceVec,
    pub normed_out: DeviceVec,
    pub logits: DeviceVec,

    // ── CUDA Graph support ──────────────────────────────────────────
    /// Pre-allocated token_id buffer for embedding lookup (avoids clone_htod)
    pub token_id_gpu: CudaSlice<i32>,
    /// GPU-resident start_pos for CUDA Graph-safe attention and scatter
    pub start_pos_buf: CudaSlice<i32>,

    // ── Paged KV decode metadata (stable GPU addresses for CUDA Graph) ───
    /// Page index list for the single request (capacity_pages+1 slots).
    pub page_indices_d: CudaSlice<i32>,
    /// CSR indptr for the single request: [0, num_pages] (2 elements).
    pub page_indptr_d: CudaSlice<i32>,
    /// Number of occupied slots in the last page (1 element).
    pub last_page_len_d: CudaSlice<i32>,
    /// Request index for BatchDecode: always [0] (1 element, constant).
    pub request_indices_d: CudaSlice<i32>,
    /// KV tile index for BatchDecode: always [0] (1 element, constant).
    pub kv_tile_indices_d: CudaSlice<i32>,
    /// Sequence length for BatchDecode: seq_len after advance (1 element).
    pub kv_chunk_size_d: CudaSlice<i32>,
}

impl SingleTokenBuffers {
    pub(super) fn new(
        ctx: &DeviceContext,
        c: &Config35,
        pool_capacity: usize,
        _padding_page_id: i32,
    ) -> Result<Self> {
        let h = c.hidden_size;
        let q_proj_dim = c.full_attn_q_proj_dim();
        let q_dim = c.full_attn_q_dim();
        let kv_dim = c.full_attn_kv_dim();
        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let inter = c.intermediate_size;

        // b and a projections output [num_value_heads] (one scalar per value head)
        let b_dim = c.linear_num_value_heads;
        let a_dim = b_dim;

        Ok(Self {
            hidden_a: HiddenStates::zeros(ctx, h, 1)?,
            normed: HiddenStates::zeros(ctx, h, 1)?,

            q_full: HiddenStates::zeros(ctx, q_proj_dim, 1)?,
            k_attn: HiddenStates::zeros(ctx, kv_dim, 1)?,
            v_attn: HiddenStates::zeros(ctx, kv_dim, 1)?,
            q_prepped: HiddenStates::zeros(ctx, q_dim, 1)?,
            attn_out_full: HiddenStates::zeros(ctx, q_dim, 1)?,

            qkv: HiddenStates::zeros(ctx, qkv_dim, 1)?,
            z: HiddenStates::zeros(ctx, z_dim, 1)?,
            b_proj: HiddenStates::zeros(ctx, b_dim, 1)?,
            a_proj: HiddenStates::zeros(ctx, a_dim, 1)?,
            qkv_conv: HiddenStates::zeros(ctx, qkv_dim, 1)?,
            gdr_out: HiddenStates::zeros(ctx, z_dim, 1)?,
            normed_gated: HiddenStates::zeros(ctx, z_dim, 1)?,

            kv_k_compact: DeviceVec::zeros(ctx, kv_dim)?,
            kv_v_compact: DeviceVec::zeros(ctx, kv_dim)?,

            attn_results: HiddenStates::zeros(ctx, h, 1)?,

            gate_out: HiddenStates::zeros(ctx, inter, 1)?,
            up_out: HiddenStates::zeros(ctx, inter, 1)?,
            act_out: HiddenStates::zeros(ctx, inter, 1)?,
            mlp_out: HiddenStates::zeros(ctx, h, 1)?,

            hidden_mid: HiddenStates::zeros(ctx, h, 1)?,

            last_normed: DeviceVec::zeros(ctx, h)?,
            normed_out: DeviceVec::zeros(ctx, h)?,
            logits: DeviceVec::zeros(ctx, c.vocab_size)?,
            token_id_gpu: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc token_id failed: {}", e))?,
            start_pos_buf: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc start_pos failed: {}", e))?,

            // Paged decode metadata (stable GPU addresses, values updated each step)
            page_indices_d: ctx
                .stream
                .alloc_zeros(pool_capacity + 1)
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

    /// Sync paged decode metadata from `kv_state` to GPU arrays.
    /// Must be called OUTSIDE the CUDA Graph, before each decode step.
    pub(super) fn sync_paged_decode_meta(
        &mut self,
        ctx: &DeviceContext,
        kv_state: &crate::kv_pool::KvState,
        padding_page_id: i32,
    ) -> Result<()> {
        let pages = kv_state.page_indices_i32();
        let mut page_indices = pages.clone();
        // Pad remaining slots with the padding page so GPU sees valid page IDs.
        page_indices.push(padding_page_id);

        let indptr = [0i32, pages.len() as i32];
        let last_page_len = [kv_state.last_page_len() as i32];
        let chunk_size = [kv_state.seq_len() as i32];

        ctx.stream
            .memcpy_htod(&page_indices, &mut self.page_indices_d)?;
        ctx.stream
            .memcpy_htod(&indptr, &mut self.page_indptr_d)?;
        ctx.stream
            .memcpy_htod(&last_page_len, &mut self.last_page_len_d)?;
        ctx.stream
            .memcpy_htod(&chunk_size, &mut self.kv_chunk_size_d)?;
        Ok(())
    }
}
