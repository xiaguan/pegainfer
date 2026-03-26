//! Pre-allocated GPU buffers for zero-allocation Qwen3.5 decode steps.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use crate::qwen35_config::Config35;
use crate::tensor::{DeviceContext, DeviceVec};

/// Pre-allocated temporary buffers for the single-token decode path.
///
/// All buffer dimensions are determined by the model config and remain fixed
/// for the entire generation. Shared between full and linear attention layers
/// where buffer sizes match (only one layer type runs at a time).
pub(crate) struct DecodeBuffers35 {
    /// Current hidden state, persists across layers (hidden_size=2560)
    pub(crate) hidden: DeviceVec,
    /// RMSNorm output / general scratch (hidden_size=2560)
    pub(crate) normed: DeviceVec,

    // Shared large projection buffer: q_full [8192] for full attn, qkv_raw [8192] for linear
    pub(crate) proj_8192: DeviceVec,

    // Full attention specific
    /// K projection output (num_kv_heads * head_dim = 1024)
    pub(crate) k_full: DeviceVec,
    /// V projection output (num_kv_heads * head_dim = 1024)
    pub(crate) v_full: DeviceVec,

    // Linear attention specific
    /// Conv1d output (qkv_dim = 8192)
    pub(crate) qkv_conv: DeviceVec,
    /// Z projection output (z_dim = 4096)
    pub(crate) proj_z: DeviceVec,
    /// B projection output (num_value_heads = 32)
    pub(crate) proj_b: DeviceVec,
    /// A projection output (num_value_heads = 32)
    pub(crate) proj_a: DeviceVec,

    // Shared attention / GDR output (num_qheads * head_dim = 4096 or num_vheads * v_dim = 4096)
    pub(crate) attn_out: DeviceVec,
    // Gated norm output — linear attention only (4096)
    pub(crate) norm_gated: DeviceVec,

    /// O/out projection output (hidden_size = 2560)
    pub(crate) attn_proj: DeviceVec,

    // MLP
    pub(crate) mlp_act: DeviceVec,
    pub(crate) mlp_out: DeviceVec,

    // Output
    pub(crate) logits: DeviceVec,

    /// Decode metadata on GPU: [token_id, current_pos, seq_len] as i32
    pub(crate) decode_meta: CudaSlice<i32>,
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size)
    pub(crate) sample_probs: CudaSlice<f32>,
    /// Pre-allocated argmax output (1 element) — lives inside CUDA Graph
    pub(crate) argmax_out: CudaSlice<i32>,
}

impl DecodeBuffers35 {
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35) -> Result<Self> {
        let h = config.hidden_size;
        let q_proj_dim = config.full_attn_q_proj_dim(); // 8192 (includes gate)
        let kv_dim = config.full_attn_kv_dim(); // 1024
        let q_dim = config.full_attn_q_dim(); // 4096
        let qkv_dim = config.linear_attn_qkv_dim(); // 8192
        let z_dim = config.linear_attn_z_dim(); // 4096
        let num_v_heads = config.linear_num_value_heads; // 32

        // proj_8192 must fit both q_full and qkv_raw
        assert_eq!(q_proj_dim, qkv_dim);

        Ok(Self {
            hidden: DeviceVec::zeros(ctx, h)?,
            normed: DeviceVec::zeros(ctx, h)?,
            proj_8192: DeviceVec::zeros(ctx, q_proj_dim)?,
            k_full: DeviceVec::zeros(ctx, kv_dim)?,
            v_full: DeviceVec::zeros(ctx, kv_dim)?,
            qkv_conv: DeviceVec::zeros(ctx, qkv_dim)?,
            proj_z: DeviceVec::zeros(ctx, z_dim)?,
            proj_b: DeviceVec::zeros(ctx, num_v_heads)?,
            proj_a: DeviceVec::zeros(ctx, num_v_heads)?,
            attn_out: DeviceVec::zeros(ctx, q_dim)?,
            norm_gated: DeviceVec::zeros(ctx, z_dim)?,
            attn_proj: DeviceVec::zeros(ctx, h)?,
            mlp_act: DeviceVec::zeros(ctx, config.intermediate_size)?,
            mlp_out: DeviceVec::zeros(ctx, h)?,
            logits: DeviceVec::zeros(ctx, config.vocab_size)?,
            decode_meta: ctx
                .stream
                .alloc_zeros(3)
                .map_err(|e| anyhow::anyhow!("Alloc decode_meta failed: {}", e))?,
            sample_probs: ctx
                .stream
                .alloc_zeros(config.vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {}", e))?,
            argmax_out: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc argmax_out failed: {}", e))?,
        })
    }
}
