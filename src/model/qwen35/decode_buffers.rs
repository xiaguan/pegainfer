//! Sampling buffers for Qwen3.5 token selection.

use anyhow::Result;

use cudarc::driver::CudaSlice;

use super::config::Config35;
use crate::tensor::DeviceContext;

/// Pre-allocated GPU buffers for token sampling (softmax + multinomial).
pub(crate) struct DecodeBuffers35 {
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size)
    pub(crate) sample_probs: CudaSlice<f32>,
    /// Pre-allocated sampling output (1 element, token id)
    pub(crate) sample_out: CudaSlice<i32>,
}

impl DecodeBuffers35 {
    pub(crate) fn new(ctx: &DeviceContext, config: &Config35) -> Result<Self> {
        Ok(Self {
            sample_probs: ctx
                .stream
                .alloc_zeros(config.vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {}", e))?,
            sample_out: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc sample_out failed: {}", e))?,
        })
    }
}
