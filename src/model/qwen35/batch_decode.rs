//! Batched decode for Qwen3.5: N requests, 1 token each, shared full-attn kernels
//! and per-request recurrent-state updates for linear attention.

use anyhow::Result;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state::RecurrentState;
use super::weights::{FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model};
use crate::kv_pool::{KvLayout, KvState};
use crate::ops;

#[allow(dead_code)]
impl Qwen35Model {
    pub(crate) fn select_tokens_batch_varied(
        &self,
        bufs: &mut BatchDecodeBuffers35,
        params: &[&crate::sampler::SamplingParams],
        rng: &mut rand::rngs::StdRng,
    ) -> Result<Vec<u32>> {
        let batch_size = params.len();

        if params.iter().all(|p| p.is_greedy()) {
            return ops::argmax_batched(&self.ctx, &bufs.logits, &mut bufs.sample_out, batch_size);
        }

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

    pub(crate) fn batch_decode(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        recurrent_states: &mut [&mut RecurrentState],
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let bs = token_ids.len();
        anyhow::ensure!(bs > 0, "batch_decode requires at least one request");
        anyhow::ensure!(bs == kv_states.len(), "token_ids / kv_states len mismatch");
        anyhow::ensure!(
            bs == recurrent_states.len(),
            "token_ids / recurrent_states len mismatch"
        );

        let mut positions = Vec::with_capacity(bs);
        for (kv, recurrent) in kv_states.iter_mut().zip(recurrent_states.iter_mut()) {
            let pos = kv.seq_len();
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            recurrent.seq_len += 1;
            positions.push(pos as i32);
        }

        bufs.set_batch_size(bs);

        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        self.ctx
            .stream
            .memcpy_htod(&token_ids_i32, &mut bufs.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut bufs.positions_d)?;

        let kv_refs: Vec<&KvState> = kv_states.iter().map(|s| &**s).collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs)?;

        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        self.batch_decode_kernels(kv_buffer, &layout, recurrent_states, bs, bufs)
    }

    fn batch_decode_kernels(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        recurrent_states: &mut [&mut RecurrentState],
        bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for layer in &self.layers {
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    self.batch_decode_full_attention(attn, kv_buffer, layout, full_idx, bs, bufs)?;
                    full_idx += 1;
                }
                LayerKind::LinearAttention(attn) => {
                    self.batch_decode_linear_attention(
                        attn,
                        recurrent_states,
                        linear_idx,
                        bs,
                        bufs,
                    )?;
                    linear_idx += 1;
                }
            }

            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden,
                &bufs.attn_results,
                &mut bufs.hidden_mid,
            )?;

            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;

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
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.mlp_out,
            );

            ops::add_batch_into(&self.ctx, &bufs.hidden_mid, &bufs.mlp_out, &mut bufs.hidden)?;
        }

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            eps,
            &mut bufs.normed,
        )?;
        ops::gemm_into(
            &self.ctx,
            &self.embed_tokens,
            &bufs.normed,
            &mut bufs.logits,
        );
        debug_assert_eq!(bufs.logits.seq_len, bs);

        Ok(())
    }

    fn batch_decode_full_attention(
        &self,
        attn: &FullAttentionLayer,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        layer_idx: usize,
        bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::gemm_into(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full);
        ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_attn);
        ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_attn);

        ops::qk_norm_partial_rope_batched_decode_hd256_into(
            &self.ctx,
            &bufs.q_full,
            &mut bufs.q_attn,
            &mut bufs.k_attn,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_d,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rotary_dim,
            eps,
        );

        ops::paged_attention_batch_decode_hd256_into(
            &self.ctx,
            &bufs.q_attn,
            &bufs.k_attn,
            &bufs.v_attn,
            kv_buffer,
            layout,
            layer_idx,
            &bufs.page_indices_d,
            &bufs.page_indptr_d,
            &bufs.last_page_len_d,
            &bufs.request_indices_d,
            &bufs.kv_tile_indices_d,
            &bufs.kv_chunk_size_d,
            &mut bufs.attn_out_full,
            self.config.num_attention_heads,
            bs,
        )?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            crate::ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const crate::ffi::Half,
                out_ptr as *mut crate::ffi::Half,
                self.config.num_attention_heads as i32,
                bs as i32,
                self.ctx.stream.cu_stream(),
            );
        }

        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
        );
        Ok(())
    }

    fn batch_decode_linear_attention(
        &self,
        attn: &LinearAttentionLayer,
        recurrent_states: &mut [&mut RecurrentState],
        layer_idx: usize,
        bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        ops::gemm_into(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv);
        ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z);
        ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj);
        ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj);

        for (req_idx, recurrent) in recurrent_states.iter_mut().enumerate().take(bs) {
            let layer_state = &mut recurrent.layers[layer_idx];

            ops::extract_vec_into(&self.ctx, &bufs.qkv, req_idx, &mut bufs.qkv_tmp)?;
            ops::conv1d_decode_into(
                &self.ctx,
                &bufs.qkv_tmp,
                &attn.conv1d_weight,
                &mut layer_state.conv_state,
                &mut bufs.qkv_conv_tmp,
                self.config.linear_conv_kernel_dim,
            );
            ops::extract_vec_into(&self.ctx, &bufs.b_proj, req_idx, &mut bufs.b_tmp)?;
            ops::extract_vec_into(&self.ctx, &bufs.a_proj, req_idx, &mut bufs.a_tmp)?;

            ops::gated_delta_rule_decode_vec_into(
                &self.ctx,
                &bufs.qkv_conv_tmp,
                &bufs.b_tmp,
                &bufs.a_tmp,
                &attn.dt_bias,
                &attn.a_log,
                &mut layer_state.state,
                &mut bufs.gdr_tmp,
                self.config.linear_num_key_heads,
                self.config.linear_num_value_heads,
                self.config.linear_key_head_dim,
                self.config.linear_value_head_dim,
            )?;
            ops::write_vec_into(&self.ctx, &bufs.gdr_tmp, &mut bufs.gdr_out, req_idx)?;
        }

        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            self.config.linear_num_value_heads,
            self.config.linear_value_head_dim,
            self.config.rms_norm_eps,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelForward;
    use crate::sampler::SamplingParams;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

    fn get_model_path() -> String {
        std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
    }

    fn sequential_decode(
        model: &Qwen35Model,
        prompt_tokens: &[u32],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<u32> {
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state = model.create_state().unwrap();

        model.forward(prompt_tokens, &mut state).unwrap();
        let first_token = model.select_token(&mut state, &params, &mut rng).unwrap();

        let mut tokens = vec![first_token];
        for _ in 1..num_decode_steps {
            let last = *tokens.last().unwrap();
            if model.is_stop_token(last) {
                break;
            }
            model.forward(&[last], &mut state).unwrap();
            let next = model.select_token(&mut state, &params, &mut rng).unwrap();
            tokens.push(next);
        }
        tokens
    }

    fn batch_decode_run(
        model: &Qwen35Model,
        prompts: &[&[u32]],
        num_decode_steps: usize,
        seed: u64,
    ) -> Vec<Vec<u32>> {
        let bs = prompts.len();
        let params = SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(seed);

        let mut kv_states = Vec::with_capacity(bs);
        let mut recurrent_states = Vec::with_capacity(bs);
        let mut first_tokens = Vec::with_capacity(bs);

        for prompt in prompts {
            let mut state = model.create_state().unwrap();
            model.forward(prompt, &mut state).unwrap();
            let token = model.select_token(&mut state, &params, &mut rng).unwrap();
            first_tokens.push(token);
            kv_states.push(std::mem::replace(&mut state.kv_state, model.alloc_kv()));
            recurrent_states.push(std::mem::replace(
                &mut state.recurrent_state,
                RecurrentState::new(&model.ctx, &model.config).unwrap(),
            ));
        }

        let mut all_tokens: Vec<Vec<u32>> = first_tokens.iter().map(|&t| vec![t]).collect();
        let mut bufs = model.create_batch_decode_bufs(bs).unwrap();

        for _ in 1..num_decode_steps {
            let token_ids: Vec<u32> = all_tokens.iter().map(|t| *t.last().unwrap()).collect();
            let mut kv_refs: Vec<&mut KvState> = kv_states.iter_mut().collect();
            let mut recurrent_refs: Vec<&mut RecurrentState> =
                recurrent_states.iter_mut().collect();
            model
                .batch_decode(&token_ids, &mut kv_refs, &mut recurrent_refs, &mut bufs)
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
        let model = Qwen35Model::from_safetensors_with_options(&model_path, false).unwrap();
        let prompt_a: Vec<u32> = vec![9707];
        let prompt_b: Vec<u32> = vec![3838, 374, 220, 17, 10, 17];
        let num_steps = 8;
        let seed = 42;

        let seq_a = sequential_decode(&model, &prompt_a, num_steps, seed);
        let seq_b = sequential_decode(&model, &prompt_b, num_steps, seed);
        let batch = batch_decode_run(&model, &[&prompt_a, &prompt_b], num_steps, seed);

        assert_eq!(batch[0], seq_a, "batch decode mismatch for prompt_a");
        assert_eq!(batch[1], seq_b, "batch decode mismatch for prompt_b");
    }
}
