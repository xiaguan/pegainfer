use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers;
use super::weights::Qwen3Model;
use crate::kv_pool::KvState;
use crate::model::kv_cache::KVCache;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::DeviceVec;

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
    pub(super) decode_bufs: DecodeBuffers,
    /// Contiguous KV cache — used by prefill path only.
    pub(super) kv_cache: KVCache,
    /// Paged KV state — used by decode path (FlashInfer).
    pub(super) kv_state: KvState,
    /// Logits from multi-token prefill (None after decode path — logits are in decode_bufs).
    pub(super) prefill_logits: Option<DeviceVec>,
}

// SAFETY: Contains raw CUDA pointers (CudaSlice, etc.) that are not Send by default.
// We only access state from the single inference thread.
unsafe impl Send for Qwen3State {}

impl GenerationState for Qwen3State {
    fn logits(&self) -> &DeviceVec {
        self.prefill_logits
            .as_ref()
            .unwrap_or(&self.decode_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_cache.reset();
        self.kv_state.reset();
        self.prefill_logits = None;
        Ok(())
    }
}

impl ModelForward for Qwen3Model {
    type State = Qwen3State;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen3State {
            decode_bufs: DecodeBuffers::new(&self.ctx, &self.config)?,
            kv_cache: KVCache::new(
                self.config.num_hidden_layers,
                self.config.num_key_value_heads,
            ),
            kv_state: self.kv_pool.alloc(),
            prefill_logits: None,
        })
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.decode_one_token(tokens[0], &mut state.kv_state, &mut state.decode_bufs)?;
            state.prefill_logits = None;
        } else {
            // Prefill uses contiguous KV cache, then scatters into paged layout.
            let start_pos = state.kv_cache.len();
            let hidden = self.get_embeddings_batch(tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.prefill_logits = Some(logits);

            // Scatter contiguous KV → paged cache so decode can see prompt context.
            let seq_len = state.kv_cache.len();
            state.kv_state.ensure_capacity(seq_len)?;
            state.kv_state.advance(seq_len);
            let desc = state.kv_state.desc();
            ops::scatter_kv_to_paged(&self.ctx, &state.kv_cache, &desc)?;
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
