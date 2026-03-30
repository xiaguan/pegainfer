use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers;
use super::weights::Qwen3Model;
use crate::kv_pool::KvState;
use crate::model::cuda_graph::CudaGraphState;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::DeviceVec;

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
    pub(super) decode_bufs: DecodeBuffers,
    /// Paged KV state — used by both prefill and decode paths (FlashInfer).
    pub(super) kv_state: KvState,
    /// CUDA Graph state for decode path — captures on first token, replays after.
    pub(super) graph_state: CudaGraphState,
    /// Logits from multi-token prefill (None after decode path — logits are in decode_bufs).
    pub(super) prefill_logits: Option<DeviceVec>,
}

// SAFETY: Contains raw CUDA pointers (CudaSlice, etc.) that are not Send by default.
// We only access state from the single inference thread.
unsafe impl Send for Qwen3State {}

impl Qwen3State {
    /// Swap out the KV state, replacing it with `replacement`.
    /// Returns the previous KV state (e.g. prefilled pages for the active set).
    pub(crate) fn take_kv_state(&mut self, replacement: KvState) -> KvState {
        std::mem::replace(&mut self.kv_state, replacement)
    }
}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.prefill_logits
            .as_ref()
            .unwrap_or(&self.decode_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_state.reset();
        // graph_state is intentionally kept — topology is identical across
        // requests (same kernels, same buffer pointers). Only metadata values
        // change, and those are updated via memcpy_htod before each launch.
        self.prefill_logits = None;
        Ok(())
    }
}

impl ModelForward for Qwen3Model {
    type State = Qwen3State;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen3State {
            decode_bufs: DecodeBuffers::new(
                &self.ctx,
                &self.config,
                self.kv_pool.capacity_pages(),
            )?,
            kv_state: self.kv_pool.alloc(),
            graph_state: CudaGraphState::new(),
            prefill_logits: None,
        })
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.decode_one_token(
                tokens[0],
                &mut state.kv_state,
                &mut state.decode_bufs,
                &mut state.graph_state,
            )?;
            state.prefill_logits = None;
        } else {
            // Prefill writes directly to paged KV — no contiguous cache or scatter.
            let start_pos = state.kv_state.seq_len();
            let hidden = self.get_embeddings_batch(tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_state)?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.prefill_logits = Some(logits);
            // Graph is kept — page metadata is updated via memcpy_htod before
            // each decode launch, so replay reads the correct post-prefill state.
        }
        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        let random_val: f32 = rng.random();
        let logits = state
            .prefill_logits
            .as_ref()
            .unwrap_or(&state.decode_bufs.logits);
        ops::gpu_sample_into(
            &self.ctx,
            logits,
            &mut state.decode_bufs.sample_probs,
            &mut state.decode_bufs.sample_out,
            params,
            random_val,
        )
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }
}
