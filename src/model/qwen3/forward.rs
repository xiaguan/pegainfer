use anyhow::Result;
use cudarc::driver::CudaSlice;
use rand::RngExt;
use rand::rngs::StdRng;

use super::weights::Qwen3Model;
use crate::kv_pool::KvState;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec};

/// Per-request mutable state for Qwen3.
///
/// Used by the `ModelForward` trait (tests, bench_serving). The production
/// scheduler uses `BatchDecodeBuffers` directly and never creates this struct.
pub struct Qwen3State {
    /// Paged KV state — used by the prefill path (FlashInfer).
    pub(super) kv_state: KvState,
    /// Logits from the last forward pass.
    logits: Option<DeviceVec>,
    /// FP32 scratch buffer for GPU sampling softmax (vocab_size).
    sample_probs: CudaSlice<f32>,
    /// Pre-allocated sampling output (1 element).
    sample_out: CudaSlice<i32>,
}

// SAFETY: Contains raw CUDA pointers (CudaSlice, etc.) that are not Send by default.
// We only access state from the single inference thread.
unsafe impl Send for Qwen3State {}

impl Qwen3State {
    fn new(ctx: &DeviceContext, kv_state: KvState, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            kv_state,
            logits: None,
            sample_probs: ctx
                .stream
                .alloc_zeros(vocab_size)
                .map_err(|e| anyhow::anyhow!("Alloc sample_probs failed: {e}"))?,
            sample_out: ctx
                .stream
                .alloc_zeros(1)
                .map_err(|e| anyhow::anyhow!("Alloc sample_out failed: {e}"))?,
        })
    }
}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.logits.as_ref().expect("forward() not called yet")
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_state.reset();
        self.logits = None;
        Ok(())
    }
}

impl ModelForward for Qwen3Model {
    type State = Qwen3State;

    fn create_state(&self) -> Result<Self::State> {
        Qwen3State::new(&self.ctx, self.kv_pool.alloc(), self.config.vocab_size)
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        let start_pos = state.kv_state.seq_len();
        let hidden = self.get_embeddings_batch(tokens)?;
        let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_state)?;
        let logits = self.compute_logits_batch(&hidden)?;
        state.logits = Some(logits);
        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let logits = state.logits.as_ref().expect("forward() not called yet");
        ops::gpu_sample_into(
            &self.ctx,
            logits,
            &mut state.sample_probs,
            &mut state.sample_out,
            params,
            random_val,
        )
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }
}
