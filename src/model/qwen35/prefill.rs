use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// Maximum prefill sequence length for Qwen3.5 full-attention paged kernels.
pub(crate) const MAX_SEQ: usize = 20_000;

use super::prefill_buffers::GdrChunkwiseScratch35;
use super::recurrent_state::RecurrentState;
use super::weights::{
    FullAttentionLayer, LayerKind, LinearAttentionLayer, Qwen35Model, TransformerBlock35,
};
use crate::ffi;
use crate::kv_pool::KvState;
use crate::model::kv_cache::KVCache;
use crate::ops;
use crate::ops::PrefillPagedPlan;
use crate::tensor::{DeviceVec, HiddenStates};

impl Qwen35Model {
    #[allow(clippy::type_complexity)]
    pub fn debug_prefill_full_attention_internal_last_hidden(
        &self,
        token_ids: &[u32],
        target_layer_idx: usize,
    ) -> Result<(
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> {
        let seq_len = token_ids.len();
        anyhow::ensure!(
            seq_len > 0,
            "debug_prefill_full_attention_internal_last_hidden requires non-empty input"
        );
        anyhow::ensure!(
            target_layer_idx < self.layers.len(),
            "target_layer_idx {} out of range {}",
            target_layer_idx,
            self.layers.len()
        );
        let c = &self.config;

        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut hidden_batch = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        let mut kv_cache = KVCache::new(c.num_full_attention_layers(), c.num_key_value_heads);
        let mut kv_state = self.alloc_kv();
        let mut recurrent = RecurrentState::new(&self.ctx, &self.config)?;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;
        let base_pos = kv_state.seq_len();
        kv_state.ensure_capacity(base_pos + seq_len)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let prefill_plan = PrefillPagedPlan::new(
            &self.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer_idx != target_layer_idx {
                hidden_batch = self.prefill_layer(
                    layer_idx,
                    layer,
                    &hidden_batch,
                    &mut gdr_chunkwise_scratch,
                    &mut linear_idx,
                    &mut full_idx,
                    &mut kv_cache,
                    &kv_state,
                    &prefill_plan,
                    &mut recurrent,
                )?;
                continue;
            }

            let attn = match &layer.attn {
                LayerKind::FullAttention(attn) => attn,
                LayerKind::LinearAttention(_) => {
                    anyhow::bail!("target layer {} is not full_attention", target_layer_idx)
                }
            };

            let eps = c.rms_norm_eps;
            const HEAD_DIM: usize = 256;
            let input_normed =
                self.batched_rms_norm_offset(&hidden_batch, &layer.input_layernorm, eps)?;
            let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, &input_normed)?;
            let k_batch = ops::gemm(&self.ctx, &attn.k_proj, &input_normed)?;
            let v_batch = ops::gemm(&self.ctx, &attn.v_proj, &input_normed)?;
            let attn_out_dim = c.full_attn_q_dim();

            let base_pos = kv_cache.len();
            let (kc, vc) = kv_cache.get_cache_mut(&self.ctx, full_idx)?;
            let mut q_prepped = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
            let mut attn_pre_gate_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
            let start_pos_cpu: CudaSlice<i32> = self
                .ctx
                .stream
                .clone_htod(&[base_pos as i32])
                .map_err(|e| anyhow::anyhow!("H2D start_pos failed: {e}"))?;

            unsafe {
                let (qf_ptr, _) = q_full_batch.data.device_ptr(&self.ctx.stream);
                let (k_ptr, _) = k_batch.data.device_ptr(&self.ctx.stream);
                let (v_ptr, _) = v_batch.data.device_ptr(&self.ctx.stream);
                let (qn_ptr, _) = attn.q_norm.data.device_ptr(&self.ctx.stream);
                let (kn_ptr, _) = attn.k_norm.data.device_ptr(&self.ctx.stream);
                let (cos_ptr, _) = self.cos_cache.data.device_ptr(&self.ctx.stream);
                let (sin_ptr, _) = self.sin_cache.data.device_ptr(&self.ctx.stream);
                let (qp_ptr, _) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
                let (kc_ptr, _) = kc.data.device_ptr_mut(&self.ctx.stream);
                let (vc_ptr, _) = vc.data.device_ptr_mut(&self.ctx.stream);
                let (sp_ptr, _) = start_pos_cpu.device_ptr(&self.ctx.stream);
                ffi::prefill_attention_hd256_prep_cuda(
                    qf_ptr as *const ffi::Half,
                    k_ptr as *const ffi::Half,
                    v_ptr as *const ffi::Half,
                    qn_ptr as *const ffi::Half,
                    kn_ptr as *const ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    qp_ptr as *mut ffi::Half,
                    kc_ptr as *mut ffi::Half,
                    vc_ptr as *mut ffi::Half,
                    c.num_attention_heads as i32,
                    c.num_key_value_heads as i32,
                    seq_len as i32,
                    sp_ptr as *const i32,
                    c.rotary_dim as i32,
                    eps,
                    MAX_SEQ as i32,
                    self.ctx.stream.cu_stream(),
                );
            }

            let layout = kv_state.layout();
            let layer_k_off = (full_idx * layout.layer_stride) as i64;
            let layer_v_off = layer_k_off + layout.kv_block_len as i64;
            let stride_page = layout.page_stride as i64;
            let src_stride_n = HEAD_DIM as i64;
            let src_stride_h = (MAX_SEQ * HEAD_DIM) as i64;
            {
                let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
                let (kc_ptr, _gkc) = kc.data.device_ptr(&self.ctx.stream);
                let (vc_ptr, _gvc) = vc.data.device_ptr(&self.ctx.stream);
                let elem_off = base_pos as u64 * HEAD_DIM as u64 * 2;
                let kc_scatter = (kc_ptr + elem_off) as *const ffi::Half;
                let vc_scatter = (vc_ptr + elem_off) as *const ffi::Half;
                let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
                let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
                let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
                let (bi_ptr, _gbi) = prefill_plan.batch_indices_d().device_ptr(&self.ctx.stream);
                let (pos_ptr, _gpos) = prefill_plan.positions_d().device_ptr(&self.ctx.stream);
                let result = unsafe {
                    ffi::paged_kv_scatter_cuda(
                        buf_ptr as *const ffi::Half,
                        layer_k_off,
                        layer_v_off,
                        pi_ptr as *const i32,
                        pip_ptr as *const i32,
                        lpl_ptr as *const i32,
                        kc_scatter,
                        vc_scatter,
                        bi_ptr as *const i32,
                        pos_ptr as *const i32,
                        seq_len as i32,
                        c.num_key_value_heads as i32,
                        HEAD_DIM as i32,
                        layout.page_size as i32,
                        stride_page,
                        src_stride_n,
                        src_stride_h,
                        self.ctx.stream.cu_stream(),
                    )
                };
                anyhow::ensure!(
                    result == 0,
                    "paged_kv_scatter_cuda (debug prefill) failed: {result}"
                );
            }

            let sm_scale = 1.0f32 / f32::sqrt(HEAD_DIM as f32);
            {
                let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
                let (qp_ptr, _gqp) = q_prepped.data.device_ptr(&self.ctx.stream);
                let (out_ptr, _go) = attn_pre_gate_batch.data.device_ptr_mut(&self.ctx.stream);
                let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
                let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
                let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
                let (qi_ptr, _gqi) = prefill_plan.q_indptr_d().device_ptr(&self.ctx.stream);
                let (ri_ptr, _gri) = prefill_plan
                    .request_indices_d()
                    .device_ptr(&self.ctx.stream);
                let (qti_ptr, _gqti) = prefill_plan
                    .qo_tile_indices_d()
                    .device_ptr(&self.ctx.stream);
                let (kti_ptr, _gkti) = prefill_plan
                    .kv_tile_indices_d()
                    .device_ptr(&self.ctx.stream);
                let (kcs_ptr, _gkcs) = prefill_plan.kv_chunk_size_d().device_ptr(&self.ctx.stream);
                let (tnr_ptr, _gtnr) = prefill_plan.total_num_rows_d().device_ptr(&self.ctx.stream);
                let result = unsafe {
                    ffi::batch_prefill_paged_cuda_hd256(
                        qp_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        buf_ptr as *const ffi::Half,
                        layer_k_off,
                        layer_v_off,
                        pi_ptr as *const i32,
                        pip_ptr as *const i32,
                        lpl_ptr as *const i32,
                        qi_ptr as *const i32,
                        ri_ptr as *const i32,
                        qti_ptr as *const i32,
                        kti_ptr as *const i32,
                        kcs_ptr as *const i32,
                        tnr_ptr as *const u32,
                        c.num_attention_heads as i32,
                    c.num_key_value_heads as i32,
                    HEAD_DIM as i32,
                    layout.page_size as i32,
                    seq_len as i32,
                    prefill_plan.batch_size(),
                    prefill_plan.num_tiles(),
                    stride_page,
                    sm_scale,
                    self.ctx.stream.cu_stream(),
                )
            };
                anyhow::ensure!(
                    result == 0,
                    "batch_prefill_paged_cuda_hd256 (debug) failed: {result}"
                );
            }

            let attn_pre_gate_last =
                ops::extract_vec(&self.ctx, &attn_pre_gate_batch, seq_len - 1)?.to_host(&self.ctx)?;

            {
                let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&self.ctx.stream);
                let (out_ptr, _go) = attn_pre_gate_batch.data.device_ptr_mut(&self.ctx.stream);
                unsafe {
                    ffi::attention_gate_batch_hd256_cuda(
                        qf_ptr as *const ffi::Half,
                        out_ptr as *mut ffi::Half,
                        c.num_attention_heads as i32,
                        seq_len as i32,
                        self.ctx.stream.cu_stream(),
                    );
                }
            }

            let gated_attn_last =
                ops::extract_vec(&self.ctx, &attn_pre_gate_batch, seq_len - 1)?.to_host(&self.ctx)?;
            let o_proj_last = ops::extract_vec(
                &self.ctx,
                &ops::gemm(&self.ctx, &attn.o_proj, &attn_pre_gate_batch)?,
                seq_len - 1,
            )?
            .to_host(&self.ctx)?;

            return Ok((
                ops::extract_vec(&self.ctx, &input_normed, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &q_full_batch, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &k_batch, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &v_batch, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &q_prepped, seq_len - 1)?.to_host(&self.ctx)?,
                attn_pre_gate_last,
                gated_attn_last,
                o_proj_last,
            ));
        }

        unreachable!("target layer should have returned");
    }

    pub fn debug_prefill_layer_stage_last_hidden(
        &self,
        token_ids: &[u32],
        target_layer_idx: usize,
    ) -> Result<(String, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "debug_prefill_layer_stage_last_hidden requires non-empty input");
        anyhow::ensure!(
            target_layer_idx < self.layers.len(),
            "target_layer_idx {} out of range {}",
            target_layer_idx,
            self.layers.len()
        );
        let c = &self.config;

        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut hidden_batch = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        let mut kv_cache = KVCache::new(c.num_full_attention_layers(), c.num_key_value_heads);
        let mut kv_state = self.alloc_kv();
        let mut recurrent = RecurrentState::new(&self.ctx, &self.config)?;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;
        let base_pos = kv_state.seq_len();
        kv_state.ensure_capacity(base_pos + seq_len)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let prefill_plan = PrefillPagedPlan::new(
            &self.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer_idx != target_layer_idx {
                hidden_batch = self.prefill_layer(
                    layer_idx,
                    layer,
                    &hidden_batch,
                    &mut gdr_chunkwise_scratch,
                    &mut linear_idx,
                    &mut full_idx,
                    &mut kv_cache,
                    &kv_state,
                    &prefill_plan,
                    &mut recurrent,
                )?;
                continue;
            }

            let eps = c.rms_norm_eps;
            let input_normed =
                self.batched_rms_norm_offset(&hidden_batch, &layer.input_layernorm, eps)?;

            let attn_results = match &layer.attn {
                LayerKind::FullAttention(attn) => self.prefill_full_attention(
                    attn,
                    &input_normed,
                    &mut full_idx,
                    &mut kv_cache,
                    &kv_state,
                    &prefill_plan,
                    c.full_attn_q_dim(),
                    seq_len,
                )?,
                LayerKind::LinearAttention(attn) => self.prefill_linear_attention(
                    attn,
                    &input_normed,
                    &mut linear_idx,
                    &mut recurrent,
                    &mut gdr_chunkwise_scratch,
                    seq_len,
                )?,
            };

            let hidden_plus_attn = ops::add_batch(&self.ctx, &hidden_batch, &attn_results)?;
            let post_attn_normed =
                self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

            let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &post_attn_normed)?;
            let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &post_attn_normed)?;
            let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
            let mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;
            let layer_out = ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)?;

            let layer_type = match &layer.attn {
                LayerKind::FullAttention(_) => "full_attention".to_string(),
                LayerKind::LinearAttention(_) => "linear_attention".to_string(),
            };
            return Ok((
                layer_type,
                ops::extract_vec(&self.ctx, &input_normed, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &attn_results, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &hidden_plus_attn, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &post_attn_normed, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &mlp_out, seq_len - 1)?.to_host(&self.ctx)?,
                ops::extract_vec(&self.ctx, &layer_out, seq_len - 1)?.to_host(&self.ctx)?,
            ));
        }

        unreachable!("target layer should have returned");
    }

    pub fn debug_prefill_last_hidden_by_layer(
        &self,
        token_ids: &[u32],
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<String>)> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "debug_prefill_last_hidden_by_layer requires non-empty input");
        let c = &self.config;

        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut hidden_batch = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;
        let embedding_last = ops::extract_vec(&self.ctx, &hidden_batch, seq_len - 1)?.to_host(&self.ctx)?;

        let mut kv_cache = KVCache::new(c.num_full_attention_layers(), c.num_key_value_heads);
        let mut kv_state = self.alloc_kv();
        let mut recurrent = RecurrentState::new(&self.ctx, &self.config)?;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;
        let base_pos = kv_state.seq_len();
        kv_state.ensure_capacity(base_pos + seq_len)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let prefill_plan = PrefillPagedPlan::new(
            &self.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;
        let mut layer_hiddens = Vec::with_capacity(self.layers.len());
        let mut layer_types = Vec::with_capacity(self.layers.len());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_batch = self.prefill_layer(
                layer_idx,
                layer,
                &hidden_batch,
                &mut gdr_chunkwise_scratch,
                &mut linear_idx,
                &mut full_idx,
                &mut kv_cache,
                &kv_state,
                &prefill_plan,
                &mut recurrent,
            )?;
            layer_hiddens.push(
                ops::extract_vec(&self.ctx, &hidden_batch, seq_len - 1)?.to_host(&self.ctx)?,
            );
            layer_types.push(match &layer.attn {
                LayerKind::FullAttention(_) => "full_attention".to_string(),
                LayerKind::LinearAttention(_) => "linear_attention".to_string(),
            });
        }

        Ok((embedding_last, layer_hiddens, layer_types))
    }

    pub(super) fn prefill_forward(
        &self,
        token_ids: &[u32],
        kv_cache: &mut KVCache,
        kv_state: &mut KvState,
        recurrent: &mut RecurrentState,
    ) -> Result<DeviceVec> {
        let seq_len = token_ids.len();
        anyhow::ensure!(seq_len > 0, "prefill_forward requires at least one token");
        let c = &self.config;

        kv_cache.init_if_needed(&self.ctx, c.head_dim)?;

        // Get embeddings for all tokens
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(&token_ids_i32)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let hidden_dim = c.hidden_size;
        let mut hidden_batch = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        // Advance paged KV state and build prefill plan (shared across all layers).
        let base_pos = kv_state.seq_len();
        kv_state.ensure_capacity(base_pos + seq_len)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let prefill_plan = PrefillPagedPlan::new(
            &self.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;

        // Process layers
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_batch = self.prefill_layer(
                layer_idx,
                layer,
                &hidden_batch,
                &mut gdr_chunkwise_scratch,
                &mut linear_idx,
                &mut full_idx,
                kv_cache,
                kv_state,
                &prefill_plan,
                recurrent,
            )?;
        }

        // All layers processed. Advance write-buffer seq_len for next prefill call.
        kv_cache.advance_seq_len(seq_len);
        recurrent.seq_len += seq_len;
        debug_assert_eq!(
            kv_cache.len(),
            kv_state.seq_len(),
            "kv_cache and kv_state seq_len diverged"
        );

        // Extract last token's hidden state
        let last_hidden = ops::extract_vec(&self.ctx, &hidden_batch, seq_len - 1)?;

        // Final norm (1+weight offset)
        let normed = {
            let mut out = DeviceVec::zeros(&self.ctx, hidden_dim)?;
            ops::rms_norm_offset_into(
                &self.ctx,
                &last_hidden,
                &self.norm,
                c.rms_norm_eps,
                &mut out,
            )?;
            out
        };

        // LM head (tied embeddings)
        ops::linear(&self.ctx, &normed, &self.embed_tokens)
    }

    /// Process one layer during prefill. Returns updated hidden_batch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer(
        &self,
        _layer_idx: usize,
        layer: &TransformerBlock35,
        hidden_batch: &HiddenStates,
        gdr_chunkwise_scratch: &mut GdrChunkwiseScratch35,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        kv_state: &KvState,
        prefill_plan: &PrefillPagedPlan,
        recurrent: &mut RecurrentState,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let seq_len = hidden_batch.seq_len;

        // 1. Input layernorm — per-token (no batched offset norm kernel yet)
        // Use standard batched norm and add the offset correction manually
        // Actually we need the (1+w) variant. Process token by token for now.
        let mut normed_batch =
            self.batched_rms_norm_offset(hidden_batch, &layer.input_layernorm, eps)?;

        // 2. Attention / Linear attention — per-token for correctness
        let attn_out_dim = match &layer.attn {
            LayerKind::FullAttention(_) => c.full_attn_q_dim(),
            LayerKind::LinearAttention(_) => c.linear_attn_z_dim(),
        };

        // Batch project, then per-token attention/recurrent
        let attn_results = match &layer.attn {
            LayerKind::FullAttention(attn) => self.prefill_full_attention(
                attn,
                &normed_batch,
                full_idx,
                kv_cache,
                kv_state,
                prefill_plan,
                attn_out_dim,
                seq_len,
            )?,
            LayerKind::LinearAttention(attn) => self.prefill_linear_attention(
                attn,
                &normed_batch,
                linear_idx,
                recurrent,
                gdr_chunkwise_scratch,
                seq_len,
            )?,
        };

        // 3. Residual + post-attention layernorm
        let hidden_plus_attn = ops::add_batch(&self.ctx, hidden_batch, &attn_results)?;

        // Post-attention layernorm (1+weight offset, batched per-token)
        normed_batch =
            self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

        // 4. MLP (batched)
        let gate_out = ops::gemm(&self.ctx, &layer.mlp.gate_proj, &normed_batch)?;
        let up_out = ops::gemm(&self.ctx, &layer.mlp.up_proj, &normed_batch)?;
        let act_out = ops::silu_mul_batch(&self.ctx, &gate_out, &up_out)?;
        let mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;

        // 5. Residual
        ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_full_attention(
        &self,
        attn: &FullAttentionLayer,
        normed_batch: &HiddenStates,
        full_idx: &mut usize,
        kv_cache: &mut KVCache,
        kv_state: &KvState,
        prefill_plan: &PrefillPagedPlan,
        _attn_out_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let attn_out_dim = c.full_attn_q_dim();
        let eps = c.rms_norm_eps;
        const HEAD_DIM: usize = 256;

        let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, normed_batch)?;
        let k_batch = ops::gemm(&self.ctx, &attn.k_proj, normed_batch)?;
        let v_batch = ops::gemm(&self.ctx, &attn.v_proj, normed_batch)?;
        let mut attn_out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;

        let base_pos = kv_cache.len();
        let (kc, vc) = kv_cache.get_cache_mut(&self.ctx, *full_idx)?;
        let mut q_prepped = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
        let start_pos_cpu: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&[base_pos as i32])
            .map_err(|e| anyhow::anyhow!("H2D start_pos failed: {e}"))?;

        // Step 1: QK norm + partial RoPE + write processed K/V to HND write buffers.
        unsafe {
            let (qf_ptr, _) = q_full_batch.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (kc_ptr, _) = kc.data.device_ptr_mut(&self.ctx.stream);
            let (vc_ptr, _) = vc.data.device_ptr_mut(&self.ctx.stream);
            let (sp_ptr, _) = start_pos_cpu.device_ptr(&self.ctx.stream);
            ffi::prefill_attention_hd256_prep_cuda(
                qf_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                qp_ptr as *mut ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                c.num_attention_heads as i32,
                c.num_key_value_heads as i32,
                seq_len as i32,
                sp_ptr as *const i32,
                c.rotary_dim as i32,
                eps,
                MAX_SEQ as i32,
                self.ctx.stream.cu_stream(),
            );
        }

        // Step 2: Scatter processed K/V from HND write buffers to paged pool.
        // Offset src by base_pos so nnz-index 0 maps to HND position base_pos.
        let layout = kv_state.layout();
        let layer_k_off = (*full_idx * layout.layer_stride) as i64;
        let layer_v_off = layer_k_off + layout.kv_block_len as i64;
        let stride_page = layout.page_stride as i64;
        let src_stride_n = HEAD_DIM as i64;
        let src_stride_h = (MAX_SEQ * HEAD_DIM) as i64;
        {
            let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
            let (kc_ptr, _gkc) = kc.data.device_ptr(&self.ctx.stream);
            let (vc_ptr, _gvc) = vc.data.device_ptr(&self.ctx.stream);
            // Offset by base_pos so scatter nnz-index i reads from HND position (base_pos + i)
            let elem_off = base_pos as u64 * HEAD_DIM as u64 * 2; // bf16 = 2 bytes
            let kc_scatter = (kc_ptr + elem_off) as *const ffi::Half;
            let vc_scatter = (vc_ptr + elem_off) as *const ffi::Half;
            let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (bi_ptr, _gbi) = prefill_plan.batch_indices_d().device_ptr(&self.ctx.stream);
            let (pos_ptr, _gpos) = prefill_plan.positions_d().device_ptr(&self.ctx.stream);
            let result = unsafe {
                ffi::paged_kv_scatter_cuda(
                    buf_ptr as *const ffi::Half,
                    layer_k_off,
                    layer_v_off,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    kc_scatter,
                    vc_scatter,
                    bi_ptr as *const i32,
                    pos_ptr as *const i32,
                    seq_len as i32,
                    c.num_key_value_heads as i32,
                    HEAD_DIM as i32,
                    layout.page_size as i32,
                    stride_page,
                    src_stride_n,
                    src_stride_h,
                    self.ctx.stream.cu_stream(),
                )
            };
            anyhow::ensure!(
                result == 0,
                "paged_kv_scatter_cuda (prefill) failed: {result}"
            );
        }

        // Step 3: Batch prefill paged attention (HD=256).
        let sm_scale = 1.0f32 / f32::sqrt(HEAD_DIM as f32);
        {
            let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
            let (qp_ptr, _gqp) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = attn_out_batch.data.device_ptr_mut(&self.ctx.stream);
            let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (qi_ptr, _gqi) = prefill_plan.q_indptr_d().device_ptr(&self.ctx.stream);
            let (ri_ptr, _gri) = prefill_plan
                .request_indices_d()
                .device_ptr(&self.ctx.stream);
            let (qti_ptr, _gqti) = prefill_plan
                .qo_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kti_ptr, _gkti) = prefill_plan
                .kv_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kcs_ptr, _gkcs) = prefill_plan.kv_chunk_size_d().device_ptr(&self.ctx.stream);
            let (tnr_ptr, _gtnr) = prefill_plan.total_num_rows_d().device_ptr(&self.ctx.stream);
            let result = unsafe {
                ffi::batch_prefill_paged_cuda_hd256(
                    qp_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    buf_ptr as *const ffi::Half,
                    layer_k_off,
                    layer_v_off,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    qi_ptr as *const i32,
                    ri_ptr as *const i32,
                    qti_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    tnr_ptr as *const u32,
                    c.num_attention_heads as i32,
                        c.num_key_value_heads as i32,
                        HEAD_DIM as i32,
                        layout.page_size as i32,
                        seq_len as i32,
                        prefill_plan.batch_size(),
                        prefill_plan.num_tiles(),
                        stride_page,
                        sm_scale,
                        self.ctx.stream.cu_stream(),
                    )
                };
            anyhow::ensure!(
                result == 0,
                "batch_prefill_paged_cuda_hd256 failed: {result}"
            );
        }

        // Step 4: Apply gate from q_full_batch.
        {
            let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = attn_out_batch.data.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::attention_gate_batch_hd256_cuda(
                    qf_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    c.num_attention_heads as i32,
                    seq_len as i32,
                    self.ctx.stream.cu_stream(),
                );
            }
        }

        *full_idx += 1;

        // O projection (batched)
        ops::gemm(&self.ctx, &attn.o_proj, &attn_out_batch)
    }

    fn prefill_linear_attention(
        &self,
        attn: &LinearAttentionLayer,
        normed_batch: &HiddenStates,
        linear_idx: &mut usize,
        recurrent: &mut RecurrentState,
        gdr_chunkwise_scratch: &mut GdrChunkwiseScratch35,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;

        // Batch projections
        let qkv_batch = ops::gemm(&self.ctx, &attn.in_proj_qkv, normed_batch)?;
        let z_batch = ops::gemm(&self.ctx, &attn.in_proj_z, normed_batch)?;
        let b_batch = ops::gemm(&self.ctx, &attn.in_proj_b, normed_batch)?;
        let a_batch = ops::gemm(&self.ctx, &attn.in_proj_a, normed_batch)?;

        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let layer_state = &mut recurrent.layers[*linear_idx];

        let mut qkv_conv_batch = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &qkv_batch,
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut qkv_conv_batch,
            c.linear_conv_kernel_dim,
        );

        let mut gdr_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::gated_delta_rule_prefill_chunkwise_into(
            &self.ctx,
            &qkv_conv_batch,
            &b_batch,
            &a_batch,
            &attn.dt_bias,
            &attn.a_log,
            &mut layer_state.state,
            gdr_chunkwise_scratch,
            &mut gdr_out_batch,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )?;

        let mut normed_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &gdr_out_batch,
            &attn.norm_weight,
            &z_batch,
            &mut normed_out_batch,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );

        *linear_idx += 1;

        // Output projection (batched)
        ops::gemm(&self.ctx, &attn.out_proj, &normed_out_batch)
    }

    fn batched_rms_norm_offset(
        &self,
        x: &HiddenStates,
        weight: &DeviceVec,
        eps: f32,
    ) -> Result<HiddenStates> {
        let mut out = HiddenStates::zeros(&self.ctx, x.hidden_dim, x.seq_len)?;
        ops::rms_norm_batch_offset_into(&self.ctx, x, weight, eps, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn get_model_path() -> String {
        std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B").into())
    }

    fn run_prefill_layers(
        model: &Qwen35Model,
        token_ids: &[u32],
        layer_limit: usize,
        kv_cache: &mut KVCache,
        kv_state: &mut KvState,
        recurrent: &mut RecurrentState,
    ) -> Result<Vec<f32>> {
        let seq_len = token_ids.len();
        let c = &model.config;

        kv_cache.init_if_needed(&model.ctx, c.head_dim)?;

        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&x| x as i32).collect();
        let token_ids_gpu = model.ctx.stream.clone_htod(&token_ids_i32)?;

        let mut hidden_batch = HiddenStates::zeros(&model.ctx, c.hidden_size, seq_len)?;
        ops::embedding_batch(&model.ctx, &model.embed_tokens, &token_ids_gpu, &mut hidden_batch)?;

        let base_pos = kv_state.seq_len();
        kv_state.ensure_capacity(base_pos + seq_len)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let prefill_plan = PrefillPagedPlan::new(
            &model.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut gdr_chunkwise_scratch = GdrChunkwiseScratch35::new(&model.ctx, c, seq_len)?;

        for (layer_idx, layer) in model.layers.iter().enumerate().take(layer_limit) {
            hidden_batch = model.prefill_layer(
                layer_idx,
                layer,
                &hidden_batch,
                &mut gdr_chunkwise_scratch,
                &mut linear_idx,
                &mut full_idx,
                kv_cache,
                kv_state,
                &prefill_plan,
                recurrent,
            )?;
        }

        ops::extract_vec(&model.ctx, &hidden_batch, seq_len - 1)?.to_host(&model.ctx)
    }

    fn make_hidden_batch(model: &Qwen35Model, start_token: usize, seq_len: usize) -> HiddenStates {
        let hidden = model.config.hidden_size;
        let start = start_token * hidden;
        let end = start + hidden * seq_len;
        let host: Vec<bf16> = (start..end)
            .map(|i| bf16::from_f32(((i % 113) as f32 - 56.0) * 0.03125))
            .collect();
        HiddenStates {
            data: model.ctx.stream.clone_htod(&host).unwrap(),
            hidden_dim: hidden,
            seq_len,
        }
    }

    #[test]
    #[ignore = "debug helper for split-prefill layer localization"]
    fn debug_prefill_split_layer_diffs() {
        let model_path = get_model_path();
        let model = Qwen35Model::from_safetensors_with_options(&model_path, true).unwrap();
        let prompt: Vec<u32> = (0..128).map(|i| ((i % 1000) + 100) as u32).collect();
        let split = 64;

        for layer_limit in 1..=8 {
            let full = {
                let mut kv = model.alloc_kv();
                let mut rec = RecurrentState::new(&model.ctx, &model.config).unwrap();
                let mut kv_cache = KVCache::new(
                    model.config.num_full_attention_layers(),
                    model.config.num_key_value_heads,
                );
                run_prefill_layers(&model, &prompt, layer_limit, &mut kv_cache, &mut kv, &mut rec)
                    .unwrap()
            };

            let split_out = {
                let mut kv = model.alloc_kv();
                let mut rec = RecurrentState::new(&model.ctx, &model.config).unwrap();
                let mut kv_cache = KVCache::new(
                    model.config.num_full_attention_layers(),
                    model.config.num_key_value_heads,
                );
                run_prefill_layers(
                    &model,
                    &prompt[..split],
                    layer_limit,
                    &mut kv_cache,
                    &mut kv,
                    &mut rec,
                )
                .unwrap();
                run_prefill_layers(
                    &model,
                    &prompt[split..],
                    layer_limit,
                    &mut kv_cache,
                    &mut kv,
                    &mut rec,
                )
                .unwrap()
            };

            let max_diff = full
                .iter()
                .zip(split_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!("layer_limit={layer_limit} max_diff={max_diff}");
        }
    }

    #[test]
    #[ignore = "debug helper to isolate full-attention split-prefill drift"]
    fn debug_full_attention_split_vs_single() {
        let model_path = get_model_path();
        let model = Qwen35Model::from_safetensors_with_options(&model_path, true).unwrap();
        let attn = model
            .layers
            .iter()
            .find_map(|layer| match &layer.attn {
                LayerKind::FullAttention(attn) => Some(attn),
                LayerKind::LinearAttention(_) => None,
            })
            .unwrap();

        let full_hidden = make_hidden_batch(&model, 0, 128);
        let single_last = {
            let mut kv = model.alloc_kv();
            let mut kv_cache = KVCache::new(
                model.config.num_full_attention_layers(),
                model.config.num_key_value_heads,
            );
            kv_cache.init_if_needed(&model.ctx, model.config.head_dim).unwrap();
            kv.ensure_capacity(128).unwrap();
            kv.advance(128);
            let plan = PrefillPagedPlan::new(
                &model.ctx,
                &kv.desc(),
                0,
                128,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                model.config.head_dim,
            )
            .unwrap();
            let mut full_idx = 0usize;
            let out = model
                .prefill_full_attention(
                    attn,
                    &full_hidden,
                    &mut full_idx,
                    &mut kv_cache,
                    &kv,
                    &plan,
                    model.config.full_attn_q_dim(),
                    128,
                )
                .unwrap();
            ops::extract_vec(&model.ctx, &out, 127)
                .unwrap()
                .to_host(&model.ctx)
                .unwrap()
        };

        let split_last = {
            let prefix_hidden = make_hidden_batch(&model, 0, 64);
            let suffix_hidden = make_hidden_batch(&model, 64, 64);

            let mut kv = model.alloc_kv();
            let mut kv_cache = KVCache::new(
                model.config.num_full_attention_layers(),
                model.config.num_key_value_heads,
            );
            kv_cache.init_if_needed(&model.ctx, model.config.head_dim).unwrap();

            kv.ensure_capacity(64).unwrap();
            kv.advance(64);
            let plan0 = PrefillPagedPlan::new(
                &model.ctx,
                &kv.desc(),
                0,
                64,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                model.config.head_dim,
            )
            .unwrap();
            let mut full_idx = 0usize;
            model.prefill_full_attention(
                attn,
                &prefix_hidden,
                &mut full_idx,
                &mut kv_cache,
                &kv,
                &plan0,
                model.config.full_attn_q_dim(),
                64,
            )
            .unwrap();

            kv.ensure_capacity(128).unwrap();
            kv.advance(64);
            let plan1 = PrefillPagedPlan::new(
                &model.ctx,
                &kv.desc(),
                64,
                64,
                model.config.num_attention_heads,
                model.config.num_key_value_heads,
                model.config.head_dim,
            )
            .unwrap();
            let mut full_idx = 0usize;
            let out = model
                .prefill_full_attention(
                    attn,
                    &suffix_hidden,
                    &mut full_idx,
                    &mut kv_cache,
                    &kv,
                    &plan1,
                    model.config.full_attn_q_dim(),
                    64,
                )
                .unwrap();
            ops::extract_vec(&model.ctx, &out, 63)
                .unwrap()
                .to_host(&model.ctx)
                .unwrap()
        };

        let max_diff = single_last
            .iter()
            .zip(split_last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("full_attention_split_vs_single max_diff={max_diff}");
    }

    #[test]
    #[ignore = "debug helper to isolate linear-attention split-prefill drift"]
    fn debug_linear_attention_split_vs_single() {
        let model_path = get_model_path();
        let model = Qwen35Model::from_safetensors_with_options(&model_path, true).unwrap();
        let attn = model
            .layers
            .iter()
            .find_map(|layer| match &layer.attn {
                LayerKind::LinearAttention(attn) => Some(attn),
                LayerKind::FullAttention(_) => None,
            })
            .unwrap();

        let full_hidden = make_hidden_batch(&model, 0, 128);
        let single_last = {
            let mut rec = RecurrentState::new(&model.ctx, &model.config).unwrap();
            let mut scratch = GdrChunkwiseScratch35::new(&model.ctx, &model.config, 128).unwrap();
            let mut linear_idx = 0usize;
            let out = model
                .prefill_linear_attention(attn, &full_hidden, &mut linear_idx, &mut rec, &mut scratch, 128)
                .unwrap();
            ops::extract_vec(&model.ctx, &out, 127)
                .unwrap()
                .to_host(&model.ctx)
                .unwrap()
        };

        let split_last = {
            let prefix_hidden = make_hidden_batch(&model, 0, 64);
            let suffix_hidden = make_hidden_batch(&model, 64, 64);

            let mut rec = RecurrentState::new(&model.ctx, &model.config).unwrap();
            let mut linear_idx = 0usize;
            let mut scratch0 = GdrChunkwiseScratch35::new(&model.ctx, &model.config, 64).unwrap();
            model.prefill_linear_attention(
                attn,
                &prefix_hidden,
                &mut linear_idx,
                &mut rec,
                &mut scratch0,
                64,
            )
            .unwrap();

            let mut linear_idx = 0usize;
            let mut scratch1 = GdrChunkwiseScratch35::new(&model.ctx, &model.config, 64).unwrap();
            let out = model
                .prefill_linear_attention(
                    attn,
                    &suffix_hidden,
                    &mut linear_idx,
                    &mut rec,
                    &mut scratch1,
                    64,
                )
                .unwrap();
            ops::extract_vec(&model.ctx, &out, 63)
                .unwrap()
                .to_host(&model.ctx)
                .unwrap()
        };

        let max_diff = single_last
            .iter()
            .zip(split_last.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("linear_attention_split_vs_single max_diff={max_diff}");
    }

    #[test]
    #[ignore = "debug helper to localize linear-attention split-prefill drift by stage"]
    fn debug_linear_attention_split_stage_diffs() {
        let model_path = get_model_path();
        let model = Qwen35Model::from_safetensors_with_options(&model_path, true).unwrap();
        let attn = model
            .layers
            .iter()
            .find_map(|layer| match &layer.attn {
                LayerKind::LinearAttention(attn) => Some(attn),
                LayerKind::FullAttention(_) => None,
            })
            .unwrap();
        let c = &model.config;
        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
        let seq_full = 128usize;
        let seq_half = 64usize;
        let hidden_full = make_hidden_batch(&model, 0, seq_full);
        let hidden_a = make_hidden_batch(&model, 0, seq_half);
        let hidden_b = make_hidden_batch(&model, seq_half, seq_half);

        let compare_stage =
            |name: &str, full: &[f32], suffix_offset: usize, split: &[f32]| -> () {
                let suffix = &full[suffix_offset..suffix_offset + split.len()];
                let max_diff = suffix
                    .iter()
                    .zip(split.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                println!("{name} max_diff={max_diff}");
            };

        // Single 128-token run.
        let qkv_full = ops::gemm(&model.ctx, &attn.in_proj_qkv, &hidden_full).unwrap();
        let z_full = ops::gemm(&model.ctx, &attn.in_proj_z, &hidden_full).unwrap();
        let b_full = ops::gemm(&model.ctx, &attn.in_proj_b, &hidden_full).unwrap();
        let a_full = ops::gemm(&model.ctx, &attn.in_proj_a, &hidden_full).unwrap();
        let mut rec_full = RecurrentState::new(&model.ctx, &model.config).unwrap();
        let mut qkv_conv_full = HiddenStates::zeros(&model.ctx, qkv_dim, seq_full).unwrap();
        ops::conv1d_prefill_batch_into(
            &model.ctx,
            &qkv_full,
            &attn.conv1d_weight,
            &mut rec_full.layers[0].conv_state,
            &mut qkv_conv_full,
            c.linear_conv_kernel_dim,
        );
        let mut scratch_full = GdrChunkwiseScratch35::new(&model.ctx, &model.config, seq_full).unwrap();
        let mut gdr_out_full = HiddenStates::zeros(&model.ctx, z_dim, seq_full).unwrap();
        ops::gated_delta_rule_prefill_chunkwise_into(
            &model.ctx,
            &qkv_conv_full,
            &b_full,
            &a_full,
            &attn.dt_bias,
            &attn.a_log,
            &mut rec_full.layers[0].state,
            &mut scratch_full,
            &mut gdr_out_full,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )
        .unwrap();
        let mut normed_full = HiddenStates::zeros(&model.ctx, z_dim, seq_full).unwrap();
        ops::rms_norm_gated_batch_into(
            &model.ctx,
            &gdr_out_full,
            &attn.norm_weight,
            &z_full,
            &mut normed_full,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );
        let out_full = ops::gemm(&model.ctx, &attn.out_proj, &normed_full).unwrap();

        // Split 64 + 64 run with carried states.
        let qkv_a = ops::gemm(&model.ctx, &attn.in_proj_qkv, &hidden_a).unwrap();
        let _z_a = ops::gemm(&model.ctx, &attn.in_proj_z, &hidden_a).unwrap();
        let b_a = ops::gemm(&model.ctx, &attn.in_proj_b, &hidden_a).unwrap();
        let a_a = ops::gemm(&model.ctx, &attn.in_proj_a, &hidden_a).unwrap();
        let qkv_b = ops::gemm(&model.ctx, &attn.in_proj_qkv, &hidden_b).unwrap();
        let z_b = ops::gemm(&model.ctx, &attn.in_proj_z, &hidden_b).unwrap();
        let b_b = ops::gemm(&model.ctx, &attn.in_proj_b, &hidden_b).unwrap();
        let a_b = ops::gemm(&model.ctx, &attn.in_proj_a, &hidden_b).unwrap();

        let mut rec_split = RecurrentState::new(&model.ctx, &model.config).unwrap();
        let mut qkv_conv_a = HiddenStates::zeros(&model.ctx, qkv_dim, seq_half).unwrap();
        ops::conv1d_prefill_batch_into(
            &model.ctx,
            &qkv_a,
            &attn.conv1d_weight,
            &mut rec_split.layers[0].conv_state,
            &mut qkv_conv_a,
            c.linear_conv_kernel_dim,
        );
        let mut qkv_conv_b = HiddenStates::zeros(&model.ctx, qkv_dim, seq_half).unwrap();
        ops::conv1d_prefill_batch_into(
            &model.ctx,
            &qkv_b,
            &attn.conv1d_weight,
            &mut rec_split.layers[0].conv_state,
            &mut qkv_conv_b,
            c.linear_conv_kernel_dim,
        );
        let mut scratch_a = GdrChunkwiseScratch35::new(&model.ctx, &model.config, seq_half).unwrap();
        let mut gdr_out_a = HiddenStates::zeros(&model.ctx, z_dim, seq_half).unwrap();
        ops::gated_delta_rule_prefill_chunkwise_into(
            &model.ctx,
            &qkv_conv_a,
            &b_a,
            &a_a,
            &attn.dt_bias,
            &attn.a_log,
            &mut rec_split.layers[0].state,
            &mut scratch_a,
            &mut gdr_out_a,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )
        .unwrap();
        let mut scratch_b = GdrChunkwiseScratch35::new(&model.ctx, &model.config, seq_half).unwrap();
        let mut gdr_out_b = HiddenStates::zeros(&model.ctx, z_dim, seq_half).unwrap();
        ops::gated_delta_rule_prefill_chunkwise_into(
            &model.ctx,
            &qkv_conv_b,
            &b_b,
            &a_b,
            &attn.dt_bias,
            &attn.a_log,
            &mut rec_split.layers[0].state,
            &mut scratch_b,
            &mut gdr_out_b,
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        )
        .unwrap();
        let mut normed_b = HiddenStates::zeros(&model.ctx, z_dim, seq_half).unwrap();
        ops::rms_norm_gated_batch_into(
            &model.ctx,
            &gdr_out_b,
            &attn.norm_weight,
            &z_b,
            &mut normed_b,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );
        let out_b = ops::gemm(&model.ctx, &attn.out_proj, &normed_b).unwrap();

        let qkv_full_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&qkv_full.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let qkv_b_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&qkv_b.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let z_full_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&z_full.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let z_b_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&z_b.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let qkv_conv_full_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&qkv_conv_full.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let qkv_conv_b_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&qkv_conv_b.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let gdr_out_full_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&gdr_out_full.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let gdr_out_b_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&gdr_out_b.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let normed_full_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&normed_full.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let normed_b_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&normed_b.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let out_full_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&out_full.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let out_b_host: Vec<f32> = model
            .ctx
            .stream
            .clone_dtoh(&out_b.data)
            .unwrap()
            .into_iter()
            .map(|x| x.to_f32())
            .collect();
        let rec_state_full = model
            .ctx
            .stream
            .clone_dtoh(&rec_full.layers[0].state)
            .unwrap();
        let rec_state_split = model
            .ctx
            .stream
            .clone_dtoh(&rec_split.layers[0].state)
            .unwrap();
        let conv_state_full = rec_full.layers[0].conv_state.to_host(&model.ctx).unwrap();
        let conv_state_split = rec_split.layers[0].conv_state.to_host(&model.ctx).unwrap();
        model.ctx.sync().unwrap();

        compare_stage("qkv_proj", &qkv_full_host, seq_half * qkv_dim, &qkv_b_host);
        compare_stage("z_proj", &z_full_host, seq_half * z_dim, &z_b_host);
        compare_stage("conv1d", &qkv_conv_full_host, seq_half * qkv_dim, &qkv_conv_b_host);
        compare_stage("gdr_out", &gdr_out_full_host, seq_half * z_dim, &gdr_out_b_host);
        compare_stage("normed", &normed_full_host, seq_half * z_dim, &normed_b_host);
        compare_stage("out_proj", &out_full_host, seq_half * c.hidden_size, &out_b_host);
        let rec_state_max = rec_state_full
            .iter()
            .zip(rec_state_split.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let conv_state_max = conv_state_full
            .iter()
            .zip(conv_state_split.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("recurrent_state max_diff={rec_state_max}");
        println!("conv_state max_diff={conv_state_max}");
    }
}
