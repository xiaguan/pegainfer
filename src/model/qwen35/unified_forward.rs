//! Qwen3.5 batch prefill and unified step (prefill + decode combined).
//!
//! Linear attention (GDR chunkwise) does not have an efficient batched prefill
//! kernel, so `batch_prefill` runs each request's prefill serially. Full-attention
//! layers also run per-request to reuse the existing paged prefill path.
//!
//! `unified_step` combines:
//!   1. Serial `batch_prefill` for new requests entering the batch.
//!   2. CUDA Graph `batch_decode_graph` for existing decode requests.

use anyhow::Result;

use super::batch_decode_graph::BatchDecodeGraphState;
use super::recurrent_state::RecurrentState;
use super::weights::Qwen35Model;
use crate::kv_pool::KvState;
use crate::model::kv_cache::KVCache;
use crate::ops;
use crate::tensor::DeviceVec;

impl Qwen35Model {
    /// Prefill `n` prompts sequentially, updating each request's KV and recurrent state.
    ///
    /// Returns one `DeviceVec` of logits (vocab_size) per request (last-token logits).
    /// Requests are independent — there is no cross-request batching in the prefill pass.
    pub(crate) fn batch_prefill(
        &self,
        prompts: &[&[u32]],
        kv_states: &mut [KvState],
        recurrent_states: &mut [&mut RecurrentState],
    ) -> Result<Vec<DeviceVec>> {
        let n = prompts.len();
        anyhow::ensure!(n > 0, "batch_prefill requires at least one prompt");
        anyhow::ensure!(n == kv_states.len(), "prompts / kv_states len mismatch");
        anyhow::ensure!(
            n == recurrent_states.len(),
            "prompts / recurrent_states len mismatch"
        );

        let mut logits = Vec::with_capacity(n);
        for i in 0..n {
            // Each request needs its own KVCache staging buffer for the HND→paged scatter.
            let mut kv_cache = KVCache::new(
                self.config.num_full_attention_layers(),
                self.config.num_key_value_heads,
            );
            let logit = self.prefill_forward(
                prompts[i],
                &mut kv_cache,
                &mut kv_states[i],
                recurrent_states[i],
            )?;
            logits.push(logit);
        }
        Ok(logits)
    }

    /// Unified step: prefill new requests and decode existing requests in one call.
    ///
    /// Prefill is run serially per-request (GDR chunkwise per request). Decode runs
    /// via CUDA Graph on the pre-allocated `graph_state` for the decode batch.
    ///
    /// Either `prefill_prompts` or `decode_tokens` may be empty (but not both).
    ///
    /// Returns `(prefill_logits, decode_logits)` — one `DeviceVec` per request.
    /// Decode logits are extracted from `graph_state.buffers.logits` after the decode
    /// forward pass; callers can use them for sampling.
    pub(crate) fn unified_step(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_kv_states: &mut [KvState],
        prefill_recurrent_states: &mut [&mut RecurrentState],
        decode_tokens: &[u32],
        decode_kv_states: &mut [&mut KvState],
        graph_state: &mut BatchDecodeGraphState,
    ) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
        anyhow::ensure!(
            !prefill_prompts.is_empty() || !decode_tokens.is_empty(),
            "unified_step: both prefill and decode are empty"
        );

        // ── Prefill phase ─────────────────────────────────────────────────────
        let prefill_logits = if !prefill_prompts.is_empty() {
            self.batch_prefill(prefill_prompts, prefill_kv_states, prefill_recurrent_states)?
        } else {
            Vec::new()
        };

        // ── Decode phase ──────────────────────────────────────────────────────
        let decode_logits = if !decode_tokens.is_empty() {
            self.batch_decode_graph(decode_tokens, decode_kv_states, graph_state)?;

            // Extract per-request logits from the batch logits buffer.
            let num_decode = decode_tokens.len();
            let mut dlogits = Vec::with_capacity(num_decode);
            for i in 0..num_decode {
                let logit =
                    ops::extract_vec(&self.ctx, &graph_state.buffers.logits, i)?;
                dlogits.push(logit);
            }
            dlogits
        } else {
            Vec::new()
        };

        Ok((prefill_logits, decode_logits))
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::kv_pool::KvState;
    use crate::model::ModelForward;
    use crate::sampler::SamplingParams;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    fn get_model_path() -> String {
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
    }

    /// Sample a token from a DeviceVec logits using greedy (argmax).
    fn greedy_sample(model: &Qwen35Model, logits: &DeviceVec, rng: &mut StdRng) -> u32 {
        let params = SamplingParams::default();
        let mut probs: cudarc::driver::CudaSlice<f32> = model
            .ctx
            .stream
            .alloc_zeros(model.config.vocab_size)
            .unwrap();
        let mut out: cudarc::driver::CudaSlice<i32> =
            model.ctx.stream.alloc_zeros(1).unwrap();
        let random_val: f32 = rand::RngExt::random(rng);
        crate::ops::gpu_sample_into(&model.ctx, logits, &mut probs, &mut out, &params, random_val)
            .unwrap()
    }

    /// Run a full decode sequence using ModelForward::forward (reference path).
    fn sequential_forward(
        model: &crate::model::qwen35::weights::Qwen35Model,
        prompt: &[u32],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<u32> {
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state = model.create_state().unwrap();
        model.forward(prompt, &mut state).unwrap();
        let first = model.select_token(&mut state, &params, &mut rng).unwrap();
        let mut tokens = vec![first];
        for _ in 1..num_decode_steps {
            let last = *tokens.last().unwrap();
            if model.is_stop_token(last) {
                break;
            }
            model.forward(&[last], &mut state).unwrap();
            tokens.push(model.select_token(&mut state, &params, &mut rng).unwrap());
        }
        tokens
    }

    /// Full integration: batch_prefill followed by decode steps matches sequential.
    #[test]
    fn batch_prefill_then_decode_matches_sequential() {
        let model_path = get_model_path();
        let model =
            Qwen35Model::from_safetensors_with_options(&model_path, false).unwrap();

        let prompt_a: Vec<u32> = vec![9707];
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
        let num_steps = 6;
        let seed = 42;

        // Reference: sequential forward for each
        let seq_a = sequential_forward(&model, &prompt_a, num_steps, seed);
        let seq_b = sequential_forward(&model, &prompt_b, num_steps, seed);

        // batch_prefill → batch_decode
        let batch_tokens = {
            let prompts_ref: Vec<&[u32]> = vec![&prompt_a, &prompt_b];
            let mut kv_states: Vec<KvState> =
                (0..2).map(|_| model.alloc_kv()).collect();
            let mut rec_vec: Vec<RecurrentState> = (0..2)
                .map(|_| RecurrentState::new(&model.ctx, &model.config).unwrap())
                .collect();
            let mut rec_refs: Vec<&mut RecurrentState> = rec_vec.iter_mut().collect();

            let prefill_logits = model
                .batch_prefill(&prompts_ref, &mut kv_states, &mut rec_refs)
                .unwrap();

            let mut rng = StdRng::seed_from_u64(seed);
            let first_a = greedy_sample(&model, &prefill_logits[0], &mut rng);
            let first_b = greedy_sample(&model, &prefill_logits[1], &mut rng);

            let mut bufs = model.create_batch_decode_bufs(2).unwrap();
            let mut tokens_a = vec![first_a];
            let mut tokens_b = vec![first_b];

            for _ in 1..num_steps {
                let tids = [*tokens_a.last().unwrap(), *tokens_b.last().unwrap()];
                let (kv_left, kv_right) = kv_states.split_at_mut(1);
                let mut kv_refs: Vec<&mut KvState> = vec![&mut kv_left[0], &mut kv_right[0]];
                let (rec_left, rec_right) = rec_vec.split_at_mut(1);
                let mut rec_refs2: Vec<&mut RecurrentState> =
                    vec![&mut rec_left[0], &mut rec_right[0]];
                model
                    .batch_decode(&tids, &mut kv_refs, &mut rec_refs2, &mut bufs)
                    .unwrap();
                let la = ops::extract_vec(&model.ctx, &bufs.logits, 0).unwrap();
                let lb = ops::extract_vec(&model.ctx, &bufs.logits, 1).unwrap();
                tokens_a.push(greedy_sample(&model, &la, &mut rng));
                tokens_b.push(greedy_sample(&model, &lb, &mut rng));
            }
            (tokens_a, tokens_b)
        };

        assert_eq!(
            batch_tokens.0, seq_a,
            "batch prompt_a mismatch:\n  batch: {:?}\n  seq:   {:?}",
            batch_tokens.0, seq_a
        );
        assert_eq!(
            batch_tokens.1, seq_b,
            "batch prompt_b mismatch:\n  batch: {:?}\n  seq:   {:?}",
            batch_tokens.1, seq_b
        );
    }
}
