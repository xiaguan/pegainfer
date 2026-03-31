use anyhow::Result;
use rand::RngExt;
use rand::rngs::StdRng;

use super::decode_buffers::DecodeBuffers35;
use super::recurrent_state::RecurrentState;
use super::single_token_buffers::SingleTokenBuffers;
use super::weights::Qwen35Model;
use crate::model::cuda_graph::CudaGraphState;
use crate::model::kv_cache::KVCache;
use crate::model::{GenerationState, ModelForward};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec};

pub struct Qwen35State {
    pub(super) ctx: DeviceContext,
    pub(super) decode_bufs: DecodeBuffers35,
    pub(super) single_token_bufs: SingleTokenBuffers,
    pub(super) kv_cache: KVCache,
    /// Paged KV state tracking for full-attention layers.
    pub(super) kv_state: crate::kv_pool::KvState,
    pub(super) recurrent_state: RecurrentState,
    pub(super) graph_state: CudaGraphState,
    pub(super) prefill_logits: Option<DeviceVec>,
}

unsafe impl Send for Qwen35State {}

impl GenerationState for Qwen35State {
    fn logits(&self) -> &DeviceVec {
        self.prefill_logits
            .as_ref()
            .unwrap_or(&self.single_token_bufs.logits)
    }

    fn reset(&mut self) -> Result<()> {
        self.kv_cache.reset();
        self.kv_state.reset();
        self.recurrent_state.reset(&self.ctx)?;
        self.graph_state = CudaGraphState::new();
        self.prefill_logits = None;
        Ok(())
    }
}

impl ModelForward for Qwen35Model {
    type State = Qwen35State;

    fn create_state(&self) -> Result<Self::State> {
        Ok(Qwen35State {
            ctx: self.ctx.clone(),
            decode_bufs: DecodeBuffers35::new(&self.ctx, &self.config)?,
            single_token_bufs: SingleTokenBuffers::new(
                &self.ctx,
                &self.config,
                self.kv_pool().capacity_pages(),
                self.kv_pool().padding_page_id(),
            )?,
            kv_cache: KVCache::new(
                self.config.num_full_attention_layers(),
                self.config.num_key_value_heads,
            ),
            kv_state: self.alloc_kv(),
            recurrent_state: RecurrentState::new(&self.ctx, &self.config)?,
            graph_state: CudaGraphState::new(),
            prefill_logits: None,
        })
    }

    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()> {
        if tokens.len() == 1 {
            self.prefill_forward_single_token(
                tokens[0],
                &mut state.kv_cache,
                &mut state.kv_state,
                &mut state.recurrent_state,
                &mut state.single_token_bufs,
                &mut state.graph_state,
            )?;
            state.prefill_logits = None;
        } else {
            let logits = self.prefill_forward(
                tokens,
                &mut state.kv_cache,
                &mut state.kv_state,
                &mut state.recurrent_state,
            )?;
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
        let random_val: f32 = rng.random();
        let logits = state
            .prefill_logits
            .as_ref()
            .unwrap_or(&state.single_token_bufs.logits);
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
        token_id == self.config.eos_token_id
    }
}
