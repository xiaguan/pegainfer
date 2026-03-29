//! Batched decode: process N requests' tokens in one forward pass.

use anyhow::Result;

use super::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers, bucket_for};
use super::weights::{Qwen3Model, TransformerBlock};
use crate::kv_pool::{KvLayout, KvState};
use crate::ops;

impl Qwen3Model {
    /// Sample one token per request, each with its own sampling params.
    pub(crate) fn select_tokens_batch_varied(
        &self,
        bufs: &mut BatchDecodeBuffers,
        params: &[&crate::sampler::SamplingParams],
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<u32>> {
        let batch_size = params.len();
        let mut tokens = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let logits_i = ops::extract_vec(&self.ctx, &bufs.logits, i)?;
            let random_val: f32 = rand::RngExt::random(rng);
            let token = ops::gpu_sample_into(
                &self.ctx,
                &logits_i,
                &mut bufs.sample_probs,
                &mut bufs.sample_out,
                params[i],
                random_val,
            )?;
            tokens.push(token);
        }
        Ok(tokens)
    }

    /// Batch decode step: N requests, 1 new token each, one forward pass.
    ///
    /// When `enable_cuda_graph` is set, pads to the nearest bucket size and
    /// uses per-bucket CUDA Graph capture/replay.
    pub(crate) fn batch_decode(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let bs = token_ids.len();
        assert_eq!(bs, kv_states.len());
        assert!(bs > 0);

        // Grow pages and advance seq_len for each request
        let mut positions = Vec::with_capacity(bs);
        for kv in kv_states.iter_mut() {
            let pos = kv.seq_len();
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            positions.push(pos as i32);
        }

        // Pad to bucket size for CUDA Graph stability
        let padded_bs = if self.enable_cuda_graph {
            bucket_for(bs)
        } else {
            bs
        };

        // Set batch size on all buffers (padded — kernels run at bucket width)
        bufs.set_batch_size(padded_bs);

        // Sync metadata to GPU (pad token_ids/positions with 0 for padding slots)
        let mut token_ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        token_ids_i32.resize(padded_bs, 0);
        positions.resize(padded_bs, 0);

        self.ctx
            .stream
            .memcpy_htod(&token_ids_i32, &mut bufs.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut bufs.positions_d)?;

        let kv_refs: Vec<&KvState> = kv_states.iter().map(|s| &**s).collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;

        // Forward pass — with or without CUDA Graph
        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        if self.enable_cuda_graph {
            let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();
            // Take graphs out of bufs to avoid split-borrow conflict with closure
            let mut graphs = std::mem::take(&mut bufs.graphs);
            let result = graphs[bucket_idx].run_or_capture(&self.ctx, || {
                self.batch_decode_kernels(kv_buffer, &layout, padded_bs, bufs)
            });
            bufs.graphs = graphs;
            result?;
        } else {
            self.batch_decode_kernels(kv_buffer, &layout, padded_bs, bufs)?;
        }

        Ok(())
    }

    fn batch_decode_kernels(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bs: usize,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let num_layers = self.layers.len();

        // Embedding: N token_ids → hidden [hidden_dim, bs]
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        // First layer norm
        ops::rms_norm_batch_into(
            &self.ctx,
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            eps,
            &mut bufs.normed,
        );

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.batch_decode_layer(layer_idx, layer, kv_buffer, layout, bs, bufs)?;

            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            ops::fused_add_rms_norm_batch_into(
                &self.ctx,
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                eps,
                &mut bufs.normed,
            )?;
        }

        // Output projection: logits [vocab_size, bs]
        ops::gemm_into(
            &self.ctx,
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_decode_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bs: usize,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        // Q/K/V projections via GEMM (batched)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.q_proj,
            &bufs.normed,
            &mut bufs.q,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.k_proj,
            &bufs.normed,
            &mut bufs.k,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.attention.v_proj,
            &bufs.normed,
            &mut bufs.v,
        );

        // QK norm + RoPE (batched, per-request positions)
        ops::qk_norm_rope_batch_decode_into(
            &self.ctx,
            &mut bufs.q,
            &mut bufs.k,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_d,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            eps,
        );

        // KV append + paged attention decode (FlashInfer, batched)
        ops::paged_attention_batch_decode_into(
            &self.ctx,
            &bufs.q,
            &bufs.k,
            &bufs.v,
            kv_buffer,
            layout,
            layer_idx,
            &bufs.page_indices_d,
            &bufs.page_indptr_d,
            &bufs.last_page_len_d,
            &bufs.request_indices_d,
            &bufs.kv_tile_indices_d,
            &bufs.kv_chunk_size_d,
            &mut bufs.attn_out,
            self.config.num_attention_heads,
            bs,
        )?;

        // O projection (GEMM)
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        );

        // Residual + LayerNorm
        ops::fused_add_rms_norm_batch_into(
            &self.ctx,
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        // MLP: decomposed into GEMM + SiLU-mul + GEMM
        // gate_out, up_out: [intermediate, bs]
        // mlp_act: [intermediate, bs] (silu_mul output)
        // mlp_out: [hidden, bs] (down_proj output)
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.mlp_act)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.mlp_act,
            &mut bufs.mlp_out,
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelForward;
    use crate::model::qwen3::batch_decode_buffers::BatchDecodeBuffers;
    use crate::model::qwen3::weights::ModelRuntimeConfig;
    use crate::sampler::SamplingParams;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

    fn get_model_path() -> String {
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
    }

    /// Run single-request sequential decode for a prompt, return generated token IDs.
    fn sequential_decode(
        model: &Qwen3Model,
        prompt_tokens: &[u32],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<u32> {
        let mut state = model.create_state().unwrap();
        let params = SamplingParams::default(); // greedy
        let mut rng = StdRng::seed_from_u64(seed);

        // Prefill
        model.forward(prompt_tokens, &mut state).unwrap();
        let first_token = model.select_token(&mut state, &params, &mut rng).unwrap();

        let mut tokens = vec![first_token];
        for _ in 1..num_decode_steps {
            model
                .forward(&[*tokens.last().unwrap()], &mut state)
                .unwrap();
            let token = model.select_token(&mut state, &params, &mut rng).unwrap();
            tokens.push(token);
        }
        tokens
    }

    /// Run batch decode for multiple prompts, return per-request generated tokens.
    fn batch_decode_run(
        model: &Qwen3Model,
        prompts: &[&[u32]],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<Vec<u32>> {
        let bs = prompts.len();
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);

        // Create per-request states and prefill each independently
        let mut kv_states: Vec<KvState> = (0..bs).map(|_| model.kv_pool.alloc()).collect();
        let mut first_tokens = Vec::with_capacity(bs);

        for (i, prompt) in prompts.iter().enumerate() {
            let mut state = model.create_state().unwrap();
            model.forward(*prompt, &mut state).unwrap();
            let token = model.select_token(&mut state, &params, &mut rng).unwrap();
            first_tokens.push(token);
            std::mem::swap(&mut kv_states[i], &mut state.kv_state);
        }

        let mut all_tokens: Vec<Vec<u32>> = first_tokens.iter().map(|&t| vec![t]).collect();

        // Allocate buffers at bucket size (graph padding may exceed actual bs)
        let max_bs = if model.enable_cuda_graph {
            bucket_for(bs)
        } else {
            bs
        };
        let mut bufs = BatchDecodeBuffers::new(
            &model.ctx,
            &model.config,
            max_bs,
            model.kv_pool.capacity_pages(),
            model.kv_pool.padding_page_id(),
        )
        .unwrap();

        for _ in 1..num_decode_steps {
            let token_ids: Vec<u32> = all_tokens.iter().map(|t| *t.last().unwrap()).collect();
            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();
            model
                .batch_decode(&token_ids, &mut kv_refs, &mut bufs)
                .unwrap();
            let params_refs: Vec<&SamplingParams> = (0..bs).map(|_| &params).collect();
            let tokens = model
                .select_tokens_batch_varied(&mut bufs, &params_refs, &mut rng)
                .unwrap();
            for (i, &tok) in tokens.iter().enumerate() {
                all_tokens[i].push(tok);
            }
        }

        all_tokens
    }

    #[test]
    fn batch_decode_matches_sequential() {
        let model_path = get_model_path();
        let model = Qwen3Model::from_safetensors_with_runtime(
            &model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: false,
            },
        )
        .unwrap();

        // Two different prompts at different lengths to test varied contexts
        let prompt_a: Vec<u32> = vec![9707]; // "Hello"
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17]; // "What is 2+2"

        let num_steps = 10;
        let seed = 42;

        // Sequential reference
        let seq_a = sequential_decode(&model, &prompt_a, num_steps, seed);
        let seq_b = sequential_decode(&model, &prompt_b, num_steps, seed);

        // Batch
        let batch_results = batch_decode_run(&model, &[&prompt_a, &prompt_b], num_steps, seed);

        // Compare
        assert_eq!(
            batch_results[0], seq_a,
            "Request A mismatch:\n  batch: {:?}\n  seq:   {:?}",
            batch_results[0], seq_a
        );
        assert_eq!(
            batch_results[1], seq_b,
            "Request B mismatch:\n  batch: {:?}\n  seq:   {:?}",
            batch_results[1], seq_b
        );
    }

    #[test]
    fn batch_decode_graph_matches_sequential() {
        let model_path = get_model_path();
        let model = Qwen3Model::from_safetensors_with_runtime(
            &model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: true,
            },
        )
        .unwrap();

        let prompt_a: Vec<u32> = vec![9707]; // "Hello"
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17]; // "What is 2+2"

        let num_steps = 10;
        let seed = 42;

        // Sequential reference (uses single-request CUDA Graph)
        let seq_a = sequential_decode(&model, &prompt_a, num_steps, seed);
        let seq_b = sequential_decode(&model, &prompt_b, num_steps, seed);

        // Batch with bucket CUDA Graph
        let batch_results = batch_decode_run(&model, &[&prompt_a, &prompt_b], num_steps, seed);

        assert_eq!(
            batch_results[0], seq_a,
            "Graph request A mismatch:\n  batch: {:?}\n  seq:   {:?}",
            batch_results[0], seq_a
        );
        assert_eq!(
            batch_results[1], seq_b,
            "Graph request B mismatch:\n  batch: {:?}\n  seq:   {:?}",
            batch_results[1], seq_b
        );
    }
}
