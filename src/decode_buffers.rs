//! Pre-allocated GPU buffers for zero-allocation decode steps.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use crate::qwen3_config::Config;
use crate::tensor::{DeviceContext, DeviceVec};

/// Pre-allocated temporary buffers for the single-token decode path.
///
/// All buffer dimensions are determined by the model config and remain fixed
/// for the entire generation. Reusing these across decode steps eliminates
/// ~292 cudaMalloc/cudaFree calls per token.
pub struct DecodeBuffers {
    /// RMSNorm output / general scratch (hidden_size)
    pub normed: DeviceVec,
    /// Q projection output (num_attention_heads * head_dim)
    pub q: DeviceVec,
    /// K projection output (num_key_value_heads * head_dim)
    pub k: DeviceVec,
    /// V projection output (num_key_value_heads * head_dim)
    pub v: DeviceVec,
    /// Fused attention output (num_attention_heads * head_dim)
    pub attn_out: DeviceVec,
    /// O projection output (hidden_size)
    pub attn_proj: DeviceVec,
    /// Fused MLP intermediate activation (intermediate_size)
    pub mlp_act: DeviceVec,
    /// Fused MLP output (hidden_size)
    pub mlp_out: DeviceVec,
    /// Current hidden state, persists across layers (hidden_size)
    pub hidden: DeviceVec,
    /// LM head logits (vocab_size)
    pub logits: DeviceVec,
    /// Decode metadata on GPU: [token_id, current_pos, seq_len] as i32
    pub decode_meta: CudaSlice<i32>,
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size)
    pub sample_probs: CudaSlice<f32>,
    /// Split-KV partial output accumulator: [num_qheads * NUM_KV_SPLITS * HEAD_DIM] f32
    pub partial_out: CudaSlice<f32>,
    /// Split-KV partial max: [num_qheads * NUM_KV_SPLITS] f32
    pub partial_m: CudaSlice<f32>,
    /// Split-KV partial sum: [num_qheads * NUM_KV_SPLITS] f32
    pub partial_l: CudaSlice<f32>,
}

impl DecodeBuffers {
    /// NUM_KV_SPLITS must match the Triton AOT compile-time constant.
    const NUM_KV_SPLITS: usize = 4;

    pub fn new(ctx: &DeviceContext, config: &Config) -> Result<Self> {
        let h = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let num_qheads = config.num_attention_heads;

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
            partial_out: ctx
                .stream
                .alloc_zeros(num_qheads * Self::NUM_KV_SPLITS * config.head_dim)
                .map_err(|e| anyhow::anyhow!("Alloc partial_out failed: {}", e))?,
            partial_m: ctx
                .stream
                .alloc_zeros(num_qheads * Self::NUM_KV_SPLITS)
                .map_err(|e| anyhow::anyhow!("Alloc partial_m failed: {}", e))?,
            partial_l: ctx
                .stream
                .alloc_zeros(num_qheads * Self::NUM_KV_SPLITS)
                .map_err(|e| anyhow::anyhow!("Alloc partial_l failed: {}", e))?,
        })
    }
}
