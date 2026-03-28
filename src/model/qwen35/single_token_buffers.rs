//! Pre-allocated buffers for single-token prefill (decode-as-prefill).
//!
//! Eliminates per-step HiddenStates allocation overhead by caching
//! all intermediate buffers for seq_len=1 processing.

use anyhow::Result;
use cudarc::driver::CudaSlice;

use super::config::Config35;
use super::prefill_buffers::GdrChunkwiseScratch35;
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

    // ── Attention results (after O/out projection) [hidden_size, 1] ─
    pub attn_results: HiddenStates,

    // ── MLP outputs (reused across layers) ──────────────────────────
    pub gate_out: HiddenStates,
    pub up_out: HiddenStates,
    pub act_out: HiddenStates,
    pub mlp_out: HiddenStates,

    // ── Residual intermediate [hidden_size, 1] ──────────────────────
    pub hidden_mid: HiddenStates,

    // ── GDR chunkwise scratch (seq_len=1, reused across layers) ─────
    pub gdr_scratch: GdrChunkwiseScratch35,

    // ── Final outputs ───────────────────────────────────────────────
    pub last_normed: DeviceVec,
    pub normed_out: DeviceVec,
    pub logits: DeviceVec,

    // ── CUDA Graph support ──────────────────────────────────────────
    /// Pre-allocated token_id buffer for embedding lookup (avoids clone_htod)
    pub token_id_gpu: CudaSlice<i32>,
    /// GPU-resident start_pos for CUDA Graph-safe attention
    pub start_pos_buf: CudaSlice<i32>,
}

impl SingleTokenBuffers {
    pub(super) fn new(ctx: &DeviceContext, c: &Config35) -> Result<Self> {
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

            attn_results: HiddenStates::zeros(ctx, h, 1)?,

            gate_out: HiddenStates::zeros(ctx, inter, 1)?,
            up_out: HiddenStates::zeros(ctx, inter, 1)?,
            act_out: HiddenStates::zeros(ctx, inter, 1)?,
            mlp_out: HiddenStates::zeros(ctx, h, 1)?,

            hidden_mid: HiddenStates::zeros(ctx, h, 1)?,

            gdr_scratch: GdrChunkwiseScratch35::new(ctx, c, 1)?,

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
        })
    }
}
