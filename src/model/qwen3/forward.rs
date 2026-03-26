use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode::CudaGraphState;
use super::decode_buffers::DecodeBuffers;
use super::weights::Qwen3Model;
use crate::model::kv_cache::KVCache;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::{self, SamplingParams};
use crate::tensor::DeviceVec;

/// Per-request mutable state for Qwen3.
pub struct Qwen3State {
    pub(super) decode_bufs: DecodeBuffers,
    pub(super) kv_cache: KVCache,
    pub(super) graph_state: CudaGraphState,
    /// Logits from multi-token prefill (None after decode path — logits are in decode_bufs).
    pub(super) prefill_logits: Option<DeviceVec>,
}

// SAFETY: Qwen3State contains CudaGraph (raw CUDA pointers) that are not Send by default.
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
            graph_state: CudaGraphState { graph: None },
            prefill_logits: None,
        })
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.decode_one_token(
                tokens[0],
                &mut state.kv_cache,
                &mut state.decode_bufs,
                &mut state.graph_state,
            )?;
            state.prefill_logits = None;
        } else {
            let start_pos = state.kv_cache.len();
            let hidden = self.get_embeddings_batch(tokens)?;
            let hidden = self.process_all_layers_batch(hidden, start_pos, &mut state.kv_cache)?;
            let logits = self.compute_logits_batch(&hidden)?;
            state.prefill_logits = Some(logits);
        }
        Ok(())
    }

    fn select_token(
        &self,
        state: &mut Self::State,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<u32> {
        if let Some(ref logits) = state.prefill_logits {
            if params.is_greedy() {
                ops::argmax(&self.ctx, logits)
            } else {
                let logits_f32 = logits.to_host(&self.ctx)?;
                Ok(sampler::sample(&logits_f32, params, rng))
            }
        } else if params.is_greedy() {
            ops::read_argmax(&self.ctx, &state.decode_bufs.argmax_out)
        } else {
            let random_val: f32 = rng.random();
            ops::gpu_sample(
                &self.ctx,
                &state.decode_bufs.logits,
                &mut state.decode_bufs.sample_probs,
                params,
                random_val,
            )
        }
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.config.is_stop_token(token_id)
    }
}
