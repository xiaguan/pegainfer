//! DeepSeek-V3.2 MLA + MoE forward pass.
//!
//! MLA (Multi-head Latent Attention) with absorption:
//!   Q path:  hidden → q_a_proj → q_a_norm → q_b_proj → Q absorption + RoPE
//!   KV path: hidden → kv_a_proj → split → kv_a_norm(c_kv) + RoPE(k_rope) → KV cache
//!   Attention: FlashMLA decode (3-phase)
//!   Output: V de-absorption → o_proj → residual
//!
//! MoE (Mixture of Experts):
//!   Routing: gate GEMM → sigmoid + group-limited TopK-8
//!   Shared expert FFN + per-expert routed FFN → weighted accumulation

use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use log::debug;

use super::config::DsV32Config;
use super::mla_kv::{MlaKvPool, MlaKvState};
use super::weights::{DsV32Model, FfnWeights, MoeFfnWeights};
use crate::ffi;
use crate::ops;
use crate::ops::fp8::{Fp8Scratch, fp8_gemm_into, fp8_linear_into, fp8_quantize_into};
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Pre-allocated scratch buffers for MLA forward.
pub(crate) struct MlaForwardBuffers {
    /// FP8 scratch for hidden_size input projections (q_a, kv_a, shared quantize).
    fp8_hidden: Fp8Scratch,
    /// FP8 scratch for q_compressed (q_lora_rank) → q_b_proj.
    fp8_q_compressed: Fp8Scratch,
    /// FP8 scratch for intermediate activations (ffn, o_proj).
    fp8_intermediate: Fp8Scratch,
    /// q_compressed: [q_lora_rank, bs]
    q_compressed: HiddenStates,
    /// q_full: [q_b_proj_dim, bs] = [24576, bs]
    q_full: HiddenStates,
    /// kv_a: [kv_a_proj_dim, bs] = [576, bs]
    kv_a: HiddenStates,
    /// FlashMLA Q buffer: [bs * num_heads * kv_a_proj_dim] bf16
    /// Layout: [bs, num_heads, kv_a_proj_dim]
    q_mla: CudaSlice<bf16>,
    /// FlashMLA output: [bs * num_heads * kv_lora_rank] bf16
    /// Layout: [bs, 1, num_heads, kv_lora_rank]
    attn_out: CudaSlice<bf16>,
    /// V de-absorption output: [o_proj_input_dim, bs] = [16384, bs]
    v_deabsorbed: HiddenStates,
    /// Attention output after o_proj: [hidden_size, bs]
    attn_proj_out: HiddenStates,
    /// Normed hidden for current layer.
    normed: HiddenStates,
    /// FFN gate output: [intermediate_size, bs] (also used for shared/routed expert FFN)
    ffn_gate: HiddenStates,
    /// FFN up output: [intermediate_size, bs]
    ffn_up: HiddenStates,
    /// FFN act (silu(gate) * up): [intermediate_size, bs]
    ffn_act: HiddenStates,
    /// FFN down output: [hidden_size, bs]
    ffn_out: HiddenStates,
    /// FlashMLA metadata buffers
    tile_scheduler_metadata: CudaSlice<i32>,
    num_splits: CudaSlice<i32>,
    lse: CudaSlice<f32>,
    lse_accum: CudaSlice<f32>,
    o_accum: CudaSlice<f32>,
    /// Positions buffer for RoPE
    positions_d: CudaSlice<i32>,
    /// seqlens_k for FlashMLA: [bs]
    seqlens_k_d: CudaSlice<i32>,
    /// Block table for FlashMLA: [bs, max_blocks_per_seq]
    block_table_d: CudaSlice<i32>,
    /// Page indices for KV cache write: [max_pages]
    page_indices_d: CudaSlice<i32>,

    // ---- MoE routing & expert FFN buffers ----
    /// Router logits: [n_routed_experts, bs] bf16
    router_logits: HiddenStates,
    /// TopK expert indices: [bs * num_experts_per_tok] i32
    topk_indices: CudaSlice<i32>,
    /// TopK expert weights: [bs * num_experts_per_tok] f32
    topk_weights: CudaSlice<f32>,
    /// FP8 scratch for moe_intermediate_size inputs (expert down_proj)
    fp8_moe_intermediate: Fp8Scratch,
    /// MoE expert gate output: [moe_intermediate_size, bs]
    moe_gate: HiddenStates,
    /// MoE expert up output: [moe_intermediate_size, bs]
    moe_up: HiddenStates,
    /// MoE expert act: [moe_intermediate_size, bs]
    moe_act: HiddenStates,
    /// MoE expert down output: [hidden_size, bs]
    moe_down: HiddenStates,

    // ---- DeepEP dispatch/combine buffers (EP > 1) ----
    /// TopK indices as i64 (DeepEP uses TOPK_IDX_BITS=64): [bs * topk]
    topk_indices_i64: CudaSlice<i64>,
    /// num_tokens_per_rank: [num_ranks] i32
    num_tokens_per_rank: CudaSlice<i32>,
    /// num_tokens_per_expert: [n_routed_experts] i32
    num_tokens_per_expert: CudaSlice<i32>,
    /// is_token_in_rank: [bs * num_ranks] bool
    is_token_in_rank: CudaSlice<u8>,
    /// channel_prefix_matrix: [num_ranks * num_channels] i32
    channel_prefix_matrix: CudaSlice<i32>,
    /// rank_prefix_matrix_copy: [num_ranks * num_ranks] i32
    rank_prefix_matrix_copy: CudaSlice<i32>,
    /// recv_x: received tokens after dispatch [max_recv_tokens * hidden_size] bf16
    ep_recv_x: CudaSlice<bf16>,
    /// recv_src_idx: [max_recv_tokens] i32
    ep_recv_src_idx: CudaSlice<i32>,
    /// recv_topk_idx: [max_recv_tokens * topk] i64
    /// DeepEP rewrites these into rank-local expert ids for the receiving rank;
    /// entries with no local expert are set to -1.
    ep_recv_topk_idx: CudaSlice<i64>,
    /// recv_topk_weights: [max_recv_tokens * topk] f32
    ep_recv_topk_weights: CudaSlice<f32>,
    /// recv_channel_prefix_matrix: [num_ranks * num_channels] i32 — recv-side channel prefix matrix from dispatch
    ep_recv_channel_prefix_matrix: CudaSlice<i32>,
    /// send_head: [max_tokens * num_ranks] i32
    ep_send_head: CudaSlice<i32>,
    /// combined_x output: [bs * hidden_size] bf16
    ep_combined_x: CudaSlice<bf16>,
    /// combined_topk_weights: [bs * topk] f32
    ep_combined_topk_weights: CudaSlice<f32>,

    // ---- NSA indexer buffers ----
    /// Indexer Q: [max_bs * index_n_heads * index_head_dim] bf16
    /// Layout: [T, H_idx, D_idx]
    indexer_q: CudaSlice<bf16>,
    /// Indexer K: [max_bs * index_head_dim] bf16
    /// Layout: [T, D_idx]
    indexer_k: CudaSlice<bf16>,
    /// Indexer weights (from weights_proj GEMM): [max_bs * index_n_heads] bf16
    /// Layout: [T, H_idx]
    indexer_weights: CudaSlice<bf16>,
    /// Sparse attention indices: [max_bs * index_topk] i32
    indexer_indices: CudaSlice<i32>,
    /// FP8 scratch for indexer wq_b projection (input dim = q_lora_rank)
    fp8_indexer_q: Fp8Scratch,
    /// FP8 scratch for indexer wk projection (input dim = hidden_size)
    fp8_indexer_k: Fp8Scratch,

    // ---- Sparse prefill buffers ----
    /// Contiguous KV for sparse prefill: [max_bs, 1, kv_a_proj_dim] bf16
    /// Layout: [s_kv, h_kv=1, d_qk=576]
    prefill_kv: CudaSlice<bf16>,
    /// Sparse prefill output: [max_bs * num_heads * kv_lora_rank] bf16
    /// Layout: [s_q, h_q=128, d_v=512]
    prefill_attn_out: CudaSlice<bf16>,
    /// Sparse prefill max_logits: [max_bs * num_heads] f32
    prefill_max_logits: CudaSlice<f32>,
    /// Sparse prefill LSE: [max_bs * num_heads] f32
    prefill_lse: CudaSlice<f32>,

    // ---- Final output ----
    /// Output logits: [vocab_size, bs] bf16
    logits: HiddenStates,
}

// Number of SM partitions for FlashMLA.
const FLASH_MLA_NUM_SM_PARTS: i32 = 72;

impl MlaForwardBuffers {
    pub(crate) fn new(ctx: &DeviceContext, config: &DsV32Config, max_bs: usize) -> Result<Self> {
        let hidden = config.hidden_size;
        let q_lora = config.q_lora_rank;
        let q_b_dim = config.q_b_proj_dim();
        let kv_a_dim = config.kv_a_proj_dim();
        let num_heads = config.num_attention_heads;
        let kv_lora = config.kv_lora_rank;
        let o_proj_in = config.o_proj_input_dim();
        let intermediate = config.intermediate_size;
        let moe_intermediate = config.moe_intermediate_size;
        let n_routed_experts = config.n_routed_experts;
        let topk = config.num_experts_per_tok;

        // FP8 scratch: max over all projection input dims
        let fp8_hidden = Fp8Scratch::new(ctx, max_bs, hidden);
        let fp8_q_compressed = Fp8Scratch::new(ctx, max_bs, q_lora);
        let fp8_intermediate = Fp8Scratch::new(ctx, max_bs, intermediate.max(o_proj_in));

        let q_compressed = HiddenStates::zeros(ctx, q_lora, max_bs)?;
        let q_full = HiddenStates::zeros(ctx, q_b_dim, max_bs)?;
        let kv_a = HiddenStates::zeros(ctx, kv_a_dim, max_bs)?;
        let q_mla: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * num_heads * kv_a_dim)?;
        let attn_out: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * num_heads * kv_lora)?;
        let v_deabsorbed = HiddenStates::zeros(ctx, o_proj_in, max_bs)?;
        let attn_proj_out = HiddenStates::zeros(ctx, hidden, max_bs)?;
        let normed = HiddenStates::zeros(ctx, hidden, max_bs)?;
        let ffn_gate = HiddenStates::zeros(ctx, intermediate, max_bs)?;
        let ffn_up = HiddenStates::zeros(ctx, intermediate, max_bs)?;
        let ffn_act = HiddenStates::zeros(ctx, intermediate, max_bs)?;
        let ffn_out = HiddenStates::zeros(ctx, hidden, max_bs)?;

        // FlashMLA metadata
        let tile_scheduler_metadata: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(FLASH_MLA_NUM_SM_PARTS as usize * 8)?;
        let num_splits: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs + 1)?;
        let lse: CudaSlice<f32> = ctx.stream.alloc_zeros(max_bs * num_heads)?;
        let lse_accum: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(FLASH_MLA_NUM_SM_PARTS as usize * num_heads * max_bs)?;
        let o_accum: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(FLASH_MLA_NUM_SM_PARTS as usize * max_bs * num_heads * kv_lora)?;

        let positions_d: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs)?;
        let seqlens_k_d: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs)?;
        // Generous max blocks per seq
        let max_blocks_per_seq = 1024;
        let block_table_d: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs * max_blocks_per_seq)?;
        let page_indices_d: CudaSlice<i32> = ctx.stream.alloc_zeros(max_blocks_per_seq)?;

        // MoE buffers
        let router_logits = HiddenStates::zeros(ctx, n_routed_experts, max_bs)?;
        let topk_indices: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs * topk)?;
        let topk_weights: CudaSlice<f32> = ctx.stream.alloc_zeros(max_bs * topk)?;
        let fp8_moe_intermediate = Fp8Scratch::new(ctx, max_bs, moe_intermediate);
        let moe_gate = HiddenStates::zeros(ctx, moe_intermediate, max_bs)?;
        let moe_up = HiddenStates::zeros(ctx, moe_intermediate, max_bs)?;
        let moe_act = HiddenStates::zeros(ctx, moe_intermediate, max_bs)?;
        let moe_down = HiddenStates::zeros(ctx, hidden, max_bs)?;

        // DeepEP dispatch/combine buffers
        // Max recv tokens: conservative upper bound — each rank can receive up to
        // max_bs * topk tokens in the worst case (all tokens route to one rank).
        let num_ranks = 8usize; // EP8
        let num_channels = 10usize; // num_sms/2 default
        let max_recv_tokens = max_bs * topk * num_ranks; // generous upper bound
        let topk_indices_i64: CudaSlice<i64> = ctx.stream.alloc_zeros(max_bs * topk)?;
        let num_tokens_per_rank: CudaSlice<i32> = ctx.stream.alloc_zeros(num_ranks)?;
        let num_tokens_per_expert: CudaSlice<i32> = ctx.stream.alloc_zeros(n_routed_experts)?;
        let is_token_in_rank: CudaSlice<u8> = ctx.stream.alloc_zeros(max_bs * num_ranks)?;
        let channel_prefix_matrix: CudaSlice<i32> =
            ctx.stream.alloc_zeros(num_ranks * num_channels)?;
        let rank_prefix_matrix_copy: CudaSlice<i32> =
            ctx.stream.alloc_zeros(num_ranks * num_ranks)?;
        let ep_recv_x: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_recv_tokens * hidden)?;
        let ep_recv_src_idx: CudaSlice<i32> = ctx.stream.alloc_zeros(max_recv_tokens)?;
        let ep_recv_topk_idx: CudaSlice<i64> = ctx.stream.alloc_zeros(max_recv_tokens * topk)?;
        let ep_recv_topk_weights: CudaSlice<f32> =
            ctx.stream.alloc_zeros(max_recv_tokens * topk)?;
        let ep_recv_channel_prefix_matrix: CudaSlice<i32> =
            ctx.stream.alloc_zeros(num_ranks * num_channels)?;
        let ep_send_head: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs * num_ranks)?;
        let ep_combined_x: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * hidden)?;
        let ep_combined_topk_weights: CudaSlice<f32> = ctx.stream.alloc_zeros(max_bs * topk)?;

        // NSA indexer buffers
        let idx_h = config.index_n_heads.unwrap_or(0);
        let idx_d = config.index_head_dim.unwrap_or(0);
        let idx_topk = config.index_topk.unwrap_or(0);
        let indexer_q: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * idx_h * idx_d.max(1))?;
        let indexer_k: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * idx_d.max(1))?;
        let indexer_weights: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * idx_h.max(1))?;
        let indexer_indices: CudaSlice<i32> = ctx.stream.alloc_zeros(max_bs * idx_topk.max(1))?;
        let fp8_indexer_q = Fp8Scratch::new(ctx, max_bs, q_lora);
        let fp8_indexer_k = Fp8Scratch::new(ctx, max_bs, hidden);

        // Sparse prefill buffers
        let prefill_kv: CudaSlice<bf16> = ctx.stream.alloc_zeros(max_bs * kv_a_dim)?;
        let prefill_attn_out: CudaSlice<bf16> =
            ctx.stream.alloc_zeros(max_bs * num_heads * kv_lora)?;
        let prefill_max_logits: CudaSlice<f32> = ctx.stream.alloc_zeros(max_bs * num_heads)?;
        let prefill_lse: CudaSlice<f32> = ctx.stream.alloc_zeros(max_bs * num_heads)?;

        // Final output logits
        let logits = HiddenStates::zeros(ctx, config.vocab_size, max_bs)?;

        Ok(Self {
            fp8_hidden,
            fp8_q_compressed,
            fp8_intermediate,
            q_compressed,
            q_full,
            kv_a,
            q_mla,
            attn_out,
            v_deabsorbed,
            attn_proj_out,
            normed,
            ffn_gate,
            ffn_up,
            ffn_act,
            ffn_out,
            tile_scheduler_metadata,
            num_splits,
            lse,
            lse_accum,
            o_accum,
            positions_d,
            seqlens_k_d,
            block_table_d,
            page_indices_d,
            router_logits,
            topk_indices,
            topk_weights,
            fp8_moe_intermediate,
            moe_gate,
            moe_up,
            moe_act,
            moe_down,
            topk_indices_i64,
            num_tokens_per_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            channel_prefix_matrix,
            rank_prefix_matrix_copy,
            ep_recv_x,
            ep_recv_src_idx,
            ep_recv_topk_idx,
            ep_recv_topk_weights,
            ep_recv_channel_prefix_matrix,
            ep_send_head,
            ep_combined_x,
            ep_combined_topk_weights,
            indexer_q,
            indexer_k,
            indexer_weights,
            indexer_indices,
            fp8_indexer_q,
            fp8_indexer_k,
            prefill_kv,
            prefill_attn_out,
            prefill_max_logits,
            prefill_lse,
            logits,
        })
    }
}

impl DsV32Model {
    /// Full forward pass: embedding → all layers → final norm + lm_head.
    ///
    /// Processes tokens sequentially (token-by-token decode) since `forward_layer`
    /// is decode-only. Each token is processed as bs=1 through all 61 layers,
    /// building up the KV cache. Logits are returned for the last token only.
    ///
    /// `token_ids`: token IDs for this request (len = seq_len).
    /// `positions`: absolute positions for each token (len = seq_len).
    /// Returns a reference to the logits buffer `[vocab_size, 1]` inside `bufs`.
    pub(crate) fn forward_prefill<'a>(
        &self,
        token_ids: &[u32],
        positions: &[i32],
        kv_state: &mut MlaKvState,
        bufs: &'a mut MlaForwardBuffers,
        kv_pool: &MlaKvPool,
    ) -> Result<&'a HiddenStates> {
        let ctx = &self.ctx;
        let config = &self.config;
        let seq_len = token_ids.len();
        assert_eq!(seq_len, positions.len());

        let num_layers = self.layers.len();

        // Process each token sequentially through all layers (token-by-token).
        // This avoids the need for a prefill attention kernel.
        for t in 0..seq_len {
            // Embedding for this single token
            let token_ids_gpu = ctx.stream.clone_htod(&[token_ids[t]])?;
            let mut hidden = HiddenStates::zeros(ctx, config.hidden_size, 1)?;
            hidden.seq_len = 1;
            crate::ops::embedding_batch(ctx, &self.embed_tokens, &token_ids_gpu, &mut hidden)?;

            let pos = [positions[t]];

            // Forward through all layers for this token
            let mut kv_refs: Vec<&mut MlaKvState> = vec![kv_state];
            for layer_idx in 0..num_layers {
                self.forward_layer(
                    layer_idx,
                    &mut hidden,
                    &mut kv_refs,
                    &pos,
                    bufs,
                    &self.cos_cache,
                    &self.sin_cache,
                    kv_pool,
                )?;
            }

            // Only compute logits for the last token
            if t == seq_len - 1 {
                return Ok(self.forward_final(&hidden, bufs));
            }
        }

        unreachable!("seq_len must be > 0")
    }

    /// Final projection: RMSNorm → lm_head GEMM → logits.
    ///
    /// Returns a reference to the logits buffer `[vocab_size, bs]` inside `bufs`.
    pub(crate) fn forward_final<'a>(
        &self,
        hidden: &HiddenStates,
        bufs: &'a mut MlaForwardBuffers,
    ) -> &'a HiddenStates {
        let ctx = &self.ctx;
        let bs = hidden.seq_len;

        // Final RMSNorm
        bufs.normed.seq_len = bs;
        ops::rms_norm_batch_into(
            ctx,
            hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // lm_head GEMM: logits = output_projection @ normed
        bufs.logits.seq_len = bs;
        ops::gemm_into(
            ctx,
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        );

        &bufs.logits
    }

    /// Forward pass for dense layers (layers 0..first_k_dense_replace).
    ///
    /// Processes all tokens in `hidden` through the attention + FFN pipeline for one layer.
    /// `hidden` is modified in-place (residual connections).
    pub(crate) fn forward_layer(
        &self,
        layer_idx: usize,
        hidden: &mut HiddenStates,
        kv_states: &mut [&mut MlaKvState],
        positions: &[i32],
        bufs: &mut MlaForwardBuffers,
        cos_cache: &DeviceVec,
        sin_cache: &DeviceVec,
        kv_pool: &MlaKvPool,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let layer = &self.layers[layer_idx];
        let bs = hidden.seq_len;

        // Upload positions
        let positions_slice = ctx.stream.clone_htod(positions)?;
        ctx.stream.memcpy_dtod(
            &positions_slice,
            &mut bufs.positions_d.slice_mut(..positions.len()),
        )?;

        // ====================================================================
        // 1. Input LayerNorm
        // ====================================================================
        bufs.normed.seq_len = bs;
        ops::rms_norm_batch_into(
            ctx,
            hidden,
            &layer.input_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );

        // ====================================================================
        // 2. Shared FP8 quantization of normed hidden (for q_a_proj + kv_a_proj)
        // ====================================================================
        fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);

        // ====================================================================
        // 3. Q path: q_a_proj → q_a_norm → q_b_proj
        // ====================================================================
        bufs.q_compressed.seq_len = bs;
        fp8_gemm_into(
            ctx,
            bs,
            config.hidden_size,
            &layer.mla.q_a_proj,
            &bufs.fp8_hidden,
            &mut bufs.q_compressed,
        );

        // q_a_layernorm on q_compressed
        let mut q_normed = HiddenStates::zeros(ctx, config.q_lora_rank, bs)?;
        ops::rms_norm_batch_into(
            ctx,
            &bufs.q_compressed,
            &layer.mla.q_a_layernorm,
            config.rms_norm_eps,
            &mut q_normed,
        );

        // FP8 quantize q_normed → q_b_proj
        bufs.q_full.seq_len = bs;
        fp8_linear_into(
            ctx,
            &q_normed,
            &layer.mla.q_b_proj,
            &mut bufs.fp8_q_compressed,
            &mut bufs.q_full,
        );

        // ====================================================================
        // 4. KV path: kv_a_proj → split → kv_a_layernorm(c_kv) + RoPE(k_rope)
        // ====================================================================
        bufs.kv_a.seq_len = bs;
        fp8_gemm_into(
            ctx,
            bs,
            config.hidden_size,
            &layer.mla.kv_a_proj_with_mqa,
            &bufs.fp8_hidden,
            &mut bufs.kv_a,
        );

        // kv_a_layernorm on first kv_lora_rank dims (in-place)
        {
            let (kv_a_ptr, _g) = bufs.kv_a.data.device_ptr_mut(&ctx.stream);
            let (norm_w_ptr, _gw) = layer.mla.kv_a_layernorm.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::rms_norm_partial_cuda(
                    kv_a_ptr as *mut ffi::Half,
                    norm_w_ptr as *const ffi::Half,
                    config.kv_a_proj_dim() as i32,
                    config.kv_lora_rank as i32,
                    bs as i32,
                    config.rms_norm_eps,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // RoPE on k_rope (in-place on kv_a)
        {
            let (kv_a_ptr, _g) = bufs.kv_a.data.device_ptr_mut(&ctx.stream);
            let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
            let (pos_ptr, _gp) = bufs.positions_d.device_ptr(&ctx.stream);
            unsafe {
                ffi::mla_rope_kv_cuda(
                    kv_a_ptr as *mut ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    config.kv_a_proj_dim() as i32,
                    config.kv_lora_rank as i32,
                    config.qk_rope_head_dim as i32,
                    bs as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 5. Write KV to paged cache
        // ====================================================================
        for (req_idx, kv) in kv_states.iter_mut().enumerate() {
            let token_pos = positions[req_idx] as usize;
            kv.ensure_capacity(token_pos + 1)?;

            let page_indices = kv.page_indices_i32();
            let page_indices_d = ctx.stream.clone_htod(&page_indices)?;

            let (kv_a_ptr, _g) = bufs.kv_a.data.device_ptr(&ctx.stream);
            let kv_a_token_ptr =
                kv_a_ptr + (req_idx * config.kv_a_proj_dim() * std::mem::size_of::<bf16>()) as u64;

            let layer_offset = kv_pool.layer_offset(layer_idx);
            let (buf_ptr, _gb) = kv_pool.buffer().device_ptr(&ctx.stream);
            let kv_buf_ptr = buf_ptr + (layer_offset * std::mem::size_of::<bf16>()) as u64;

            let (pi_ptr, _gpi) = page_indices_d.device_ptr(&ctx.stream);

            unsafe {
                ffi::mla_kv_cache_write_cuda(
                    kv_a_token_ptr as *const ffi::Half,
                    kv_buf_ptr as *mut ffi::Half,
                    pi_ptr as *const i32,
                    config.kv_a_proj_dim() as i32,
                    kv_pool.layout().page_size as i32,
                    token_pos as i32,
                    1, // one token per request for decode
                    ctx.stream.cu_stream(),
                );
            }

            kv.advance(1);
        }

        // ====================================================================
        // 6. Q absorption: q_nope @ W_UK → q_absorbed, write to q_mla buffer
        // ====================================================================
        // cublasGemmStridedBatchedEx: C_h = W_UK_h^T @ q_nope_h for all heads
        // A = W_UK_h [128, 512] row-major = [512, 128] col-major → CUBLAS_OP_N
        // B = q_nope_h from q_full [128, bs] with ldb=q_b_proj_dim (stride between tokens)
        // C = q_absorbed in q_mla [512, bs] with ldc=num_heads*kv_a_proj_dim
        {
            let num_heads = config.num_attention_heads;
            let nope = config.qk_nope_head_dim;
            let kv_lora = config.kv_lora_rank;
            let q_head_dim = config.q_head_dim();
            let kv_a_dim = config.kv_a_proj_dim();
            let q_b_dim = config.q_b_proj_dim();

            let (w_uk_ptr, _gwu) = layer.absorbed.w_uk.device_ptr(&ctx.stream);
            let (q_full_ptr, _gq) = bufs.q_full.data.device_ptr(&ctx.stream);
            let (q_mla_ptr, _gm) = bufs.q_mla.device_ptr_mut(&ctx.stream);

            // m=kv_lora_rank(512), n=bs, k=nope(128), batch=num_heads(128)
            unsafe {
                ffi::gemm_strided_batched_cuda(
                    0,              // transa = N (A col-major [kv_lora, nope] = W_UK row-major [nope, kv_lora])
                    0,              // transb = N
                    kv_lora as i32, // m
                    bs as i32,      // n
                    nope as i32,    // k
                    w_uk_ptr as *const ffi::Half,
                    kv_lora as i32,          // lda = kv_lora_rank (512)
                    (nope * kv_lora) as i64, // strideA = 128*512 per head
                    q_full_ptr as *const ffi::Half,
                    q_b_dim as i32, // ldb = q_b_proj_dim (24576) — stride between tokens
                    q_head_dim as i64, // strideB = 192 — head stride in interleaved layout
                    q_mla_ptr as *mut ffi::Half,
                    (num_heads * kv_a_dim) as i32, // ldc — stride between tokens in q_mla
                    kv_a_dim as i64,               // strideC = 576 — head stride in q_mla
                    num_heads as i32,              // batch_count
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 7. Q RoPE + copy: q_rope from q_full → q_mla buffer
        // ====================================================================
        {
            let (q_full_ptr, _gq) = bufs.q_full.data.device_ptr(&ctx.stream);
            let (q_mla_ptr, _gm) = bufs.q_mla.device_ptr_mut(&ctx.stream);
            let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
            let (pos_ptr, _gp) = bufs.positions_d.device_ptr(&ctx.stream);

            unsafe {
                ffi::mla_rope_q_copy_cuda(
                    q_full_ptr as *const ffi::Half,
                    q_mla_ptr as *mut ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    config.q_b_proj_dim() as i32,
                    config.q_head_dim() as i32,
                    config.qk_nope_head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    config.num_attention_heads as i32,
                    config.kv_a_proj_dim() as i32,
                    config.kv_lora_rank as i32,
                    bs as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 8. FlashMLA decode attention
        // ====================================================================
        self.flash_mla_decode(layer_idx, bs, kv_states, bufs, kv_pool)?;

        // ====================================================================
        // 9. V de-absorption: attn_out @ W_UV → v_deabsorbed
        // ====================================================================
        // C_h = W_UV_h @ attn_out_h
        // W_UV_h: [v_head_dim(128), kv_lora(512)] row-major = [kv_lora, v_head_dim] col-major
        // → opA = CUBLAS_OP_T to get [v_head_dim, kv_lora]
        // attn_out_h: [kv_lora(512), bs] with ldb = num_heads * kv_lora
        // C_h: [v_head_dim(128), bs] with ldc = o_proj_input_dim
        {
            let num_heads = config.num_attention_heads;
            let v_dim = config.v_head_dim;
            let kv_lora = config.kv_lora_rank;
            let o_proj_in = config.o_proj_input_dim();

            let (w_uv_ptr, _gwv) = layer.absorbed.w_uv.device_ptr(&ctx.stream);
            let (attn_ptr, _ga) = bufs.attn_out.device_ptr(&ctx.stream);
            let (v_ptr, _gv) = bufs.v_deabsorbed.data.device_ptr_mut(&ctx.stream);

            bufs.v_deabsorbed.seq_len = bs;

            unsafe {
                ffi::gemm_strided_batched_cuda(
                    1,              // transa = T (A col-major [kv_lora, v_dim] transposed → [v_dim, kv_lora])
                    0,              // transb = N
                    v_dim as i32,   // m
                    bs as i32,      // n
                    kv_lora as i32, // k
                    w_uv_ptr as *const ffi::Half,
                    kv_lora as i32,           // lda = kv_lora (col-major leading dim)
                    (v_dim * kv_lora) as i64, // strideA per head
                    attn_ptr as *const ffi::Half,
                    (num_heads * kv_lora) as i32, // ldb — stride between tokens in attn_out
                    kv_lora as i64,               // strideB = kv_lora — head stride
                    v_ptr as *mut ffi::Half,
                    o_proj_in as i32, // ldc — stride between tokens
                    v_dim as i64,     // strideC = v_head_dim — head stride
                    num_heads as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 10. O projection + residual: hidden += o_proj(v_deabsorbed)
        // ====================================================================
        bufs.attn_proj_out.seq_len = bs;
        fp8_linear_into(
            ctx,
            &bufs.v_deabsorbed,
            &layer.mla.o_proj,
            &mut bufs.fp8_intermediate,
            &mut bufs.attn_proj_out,
        );

        // Fused residual + post_attention_layernorm
        // hidden += attn_proj_out; normed = rms_norm(hidden)
        bufs.normed.seq_len = bs;
        ops::fused_add_rms_norm_batch_into(
            ctx,
            hidden,
            &bufs.attn_proj_out,
            &layer.post_attention_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );

        // ====================================================================
        // 11. Dense FFN: gate_proj, up_proj, silu*up, down_proj + residual
        // ====================================================================
        match &layer.ffn {
            FfnWeights::Dense(ffn) => {
                // Shared FP8 quantize normed for gate + up
                fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);

                bufs.ffn_gate.seq_len = bs;
                fp8_gemm_into(
                    ctx,
                    bs,
                    config.hidden_size,
                    &ffn.gate_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.ffn_gate,
                );

                bufs.ffn_up.seq_len = bs;
                fp8_gemm_into(
                    ctx,
                    bs,
                    config.hidden_size,
                    &ffn.up_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.ffn_up,
                );

                // silu(gate) * up → act
                bufs.ffn_act.seq_len = bs;
                ops::silu_mul_batch_into(ctx, &bufs.ffn_gate, &bufs.ffn_up, &mut bufs.ffn_act)?;

                // FP8 quantize act → down_proj
                bufs.ffn_out.seq_len = bs;
                fp8_linear_into(
                    ctx,
                    &bufs.ffn_act,
                    &ffn.down_proj,
                    &mut bufs.fp8_intermediate,
                    &mut bufs.ffn_out,
                );

                // Residual: hidden += ffn_out (in-place)
                {
                    let n = (hidden.hidden_dim * bs) as i32;
                    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
                    let (f_ptr, _gf) = bufs.ffn_out.data.device_ptr(&ctx.stream);
                    unsafe {
                        ffi::add_cuda(
                            h_ptr as *const ffi::Half,
                            f_ptr as *const ffi::Half,
                            h_ptr as *mut ffi::Half,
                            n,
                            ctx.stream.cu_stream(),
                        );
                    }
                }
            }
            FfnWeights::MoE(moe) => {
                self.forward_moe(hidden, moe, bufs)?;
            }
        }

        Ok(())
    }

    /// Sparse prefill forward: embed → all layers → final norm + lm_head.
    ///
    /// Processes all tokens simultaneously using the NSA indexer to select
    /// sparse KV positions, then FlashMLA sparse prefill attention.
    /// No paged KV cache is used — KV is contiguous per layer.
    ///
    /// Returns a reference to the logits buffer `[vocab_size, 1]` (last token only).
    pub(crate) fn forward_prefill_sparse<'a>(
        &self,
        token_ids: &[u32],
        positions: &[i32],
        bufs: &'a mut MlaForwardBuffers,
    ) -> Result<&'a HiddenStates> {
        let ctx = &self.ctx;
        let config = &self.config;
        let seq_len = token_ids.len();
        assert_eq!(seq_len, positions.len());
        assert!(config.has_indexer(), "sparse prefill requires NSA indexer");

        let num_layers = self.layers.len();

        // Batch embedding: all tokens at once → hidden [hidden_size, seq_len]
        let token_ids_gpu = ctx
            .stream
            .clone_htod(&token_ids.iter().map(|&t| t).collect::<Vec<u32>>())?;
        let mut hidden = HiddenStates::zeros(ctx, config.hidden_size, seq_len)?;
        hidden.seq_len = seq_len;
        crate::ops::embedding_batch(ctx, &self.embed_tokens, &token_ids_gpu, &mut hidden)?;

        // Upload positions for all tokens
        let positions_d = ctx.stream.clone_htod(positions)?;
        ctx.stream
            .memcpy_dtod(&positions_d, &mut bufs.positions_d.slice_mut(..seq_len))?;

        // Forward through all layers
        for layer_idx in 0..num_layers {
            self.forward_layer_prefill_sparse(layer_idx, &mut hidden, seq_len, bufs)?;
        }

        // Final projection on last token only
        // Extract last token from hidden [hidden_size, seq_len]
        let last_token_offset = (seq_len - 1) * config.hidden_size;
        let last_hidden = HiddenStates {
            data: ctx.stream.clone_dtod(
                &hidden
                    .data
                    .slice(last_token_offset..last_token_offset + config.hidden_size),
            )?,
            hidden_dim: config.hidden_size,
            seq_len: 1,
        };

        Ok(self.forward_final(&last_hidden, bufs))
    }

    /// Single layer for sparse prefill: attention + FFN with all tokens batched.
    ///
    /// Uses NSA indexer for sparse KV selection and FlashMLA sparse prefill kernel.
    /// KV is contiguous in `kv_a` (no paged cache).
    fn forward_layer_prefill_sparse(
        &self,
        layer_idx: usize,
        hidden: &mut HiddenStates,
        bs: usize,
        bufs: &mut MlaForwardBuffers,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let layer = &self.layers[layer_idx];

        // ====================================================================
        // 1. Input LayerNorm
        // ====================================================================
        bufs.normed.seq_len = bs;
        ops::rms_norm_batch_into(
            ctx,
            hidden,
            &layer.input_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );

        // ====================================================================
        // 2. Shared FP8 quantization of normed hidden
        // ====================================================================
        fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);

        // ====================================================================
        // 3. Q path: q_a_proj → q_a_norm → q_b_proj
        // ====================================================================
        bufs.q_compressed.seq_len = bs;
        fp8_gemm_into(
            ctx,
            bs,
            config.hidden_size,
            &layer.mla.q_a_proj,
            &bufs.fp8_hidden,
            &mut bufs.q_compressed,
        );

        // q_a_layernorm: norm in-place into q_compressed so forward_indexer
        // reads the post-norm value (same tensor used for q_b_proj below).
        {
            let mut q_normed = HiddenStates::zeros(ctx, config.q_lora_rank, bs)?;
            ops::rms_norm_batch_into(
                ctx,
                &bufs.q_compressed,
                &layer.mla.q_a_layernorm,
                config.rms_norm_eps,
                &mut q_normed,
            );
            // Copy normed result back to q_compressed for indexer access.
            ctx.stream
                .memcpy_dtod(&q_normed.data, &mut bufs.q_compressed.data)?;
        }

        bufs.q_full.seq_len = bs;
        fp8_linear_into(
            ctx,
            &bufs.q_compressed,
            &layer.mla.q_b_proj,
            &mut bufs.fp8_q_compressed,
            &mut bufs.q_full,
        );

        // ====================================================================
        // 4. KV path: kv_a_proj → kv_a_norm → RoPE
        //    kv_a [kv_a_proj_dim, bs] is the contiguous KV for this layer.
        // ====================================================================
        bufs.kv_a.seq_len = bs;
        fp8_gemm_into(
            ctx,
            bs,
            config.hidden_size,
            &layer.mla.kv_a_proj_with_mqa,
            &bufs.fp8_hidden,
            &mut bufs.kv_a,
        );

        // kv_a_layernorm on first kv_lora_rank dims (in-place)
        {
            let (kv_a_ptr, _g) = bufs.kv_a.data.device_ptr_mut(&ctx.stream);
            let (norm_w_ptr, _gw) = layer.mla.kv_a_layernorm.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::rms_norm_partial_cuda(
                    kv_a_ptr as *mut ffi::Half,
                    norm_w_ptr as *const ffi::Half,
                    config.kv_a_proj_dim() as i32,
                    config.kv_lora_rank as i32,
                    bs as i32,
                    config.rms_norm_eps,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // RoPE on k_rope (in-place on kv_a)
        {
            let (kv_a_ptr, _g) = bufs.kv_a.data.device_ptr_mut(&ctx.stream);
            let (cos_ptr, _gc) = self.cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gs) = self.sin_cache.data.device_ptr(&ctx.stream);
            let (pos_ptr, _gp) = bufs.positions_d.device_ptr(&ctx.stream);
            unsafe {
                ffi::mla_rope_kv_cuda(
                    kv_a_ptr as *mut ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    config.kv_a_proj_dim() as i32,
                    config.kv_lora_rank as i32,
                    config.qk_rope_head_dim as i32,
                    bs as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 5. NSA Indexer → sparse indices [T, topk]
        // ====================================================================
        self.forward_indexer(layer_idx, bufs);

        // ====================================================================
        // 6. Q absorption: q_nope @ W_UK → q_absorbed, write to q_mla buffer
        // ====================================================================
        {
            let num_heads = config.num_attention_heads;
            let nope = config.qk_nope_head_dim;
            let kv_lora = config.kv_lora_rank;
            let q_head_dim = config.q_head_dim();
            let kv_a_dim = config.kv_a_proj_dim();
            let q_b_dim = config.q_b_proj_dim();

            let (w_uk_ptr, _gwu) = layer.absorbed.w_uk.device_ptr(&ctx.stream);
            let (q_full_ptr, _gq) = bufs.q_full.data.device_ptr(&ctx.stream);
            let (q_mla_ptr, _gm) = bufs.q_mla.device_ptr_mut(&ctx.stream);

            unsafe {
                ffi::gemm_strided_batched_cuda(
                    0,              // transa = N
                    0,              // transb = N
                    kv_lora as i32, // m
                    bs as i32,      // n
                    nope as i32,    // k
                    w_uk_ptr as *const ffi::Half,
                    kv_lora as i32,          // lda
                    (nope * kv_lora) as i64, // strideA
                    q_full_ptr as *const ffi::Half,
                    q_b_dim as i32,    // ldb
                    q_head_dim as i64, // strideB
                    q_mla_ptr as *mut ffi::Half,
                    (num_heads * kv_a_dim) as i32, // ldc
                    kv_a_dim as i64,               // strideC
                    num_heads as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 7. Q RoPE + copy: q_rope from q_full → q_mla buffer
        // ====================================================================
        {
            let (q_full_ptr, _gq) = bufs.q_full.data.device_ptr(&ctx.stream);
            let (q_mla_ptr, _gm) = bufs.q_mla.device_ptr_mut(&ctx.stream);
            let (cos_ptr, _gc) = self.cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gs) = self.sin_cache.data.device_ptr(&ctx.stream);
            let (pos_ptr, _gp) = bufs.positions_d.device_ptr(&ctx.stream);

            unsafe {
                ffi::mla_rope_q_copy_cuda(
                    q_full_ptr as *const ffi::Half,
                    q_mla_ptr as *mut ffi::Half,
                    cos_ptr as *const ffi::Half,
                    sin_ptr as *const ffi::Half,
                    pos_ptr as *const i32,
                    config.q_b_proj_dim() as i32,
                    config.q_head_dim() as i32,
                    config.qk_nope_head_dim as i32,
                    config.qk_rope_head_dim as i32,
                    config.num_attention_heads as i32,
                    config.kv_a_proj_dim() as i32,
                    config.kv_lora_rank as i32,
                    bs as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 8. FlashMLA sparse prefill attention
        //
        //    q:       [s_q, h_q, d_qk] — q_mla [bs, num_heads, kv_a_proj_dim]
        //    kv:      [s_kv, h_kv, d_qk] — kv_a [bs, 1, kv_a_proj_dim]
        //    indices: [s_q, h_kv, topk] — indexer_indices [bs, 1, topk] = [bs, topk]
        //    out:     [s_q, h_q, d_v] — prefill_attn_out [bs, num_heads, kv_lora_rank]
        // ====================================================================
        {
            let num_heads = config.num_attention_heads;
            let kv_lora = config.kv_lora_rank;
            let kv_a_dim = config.kv_a_proj_dim();
            let idx_topk = config.index_topk.unwrap();

            let (q_ptr, _gq) = bufs.q_mla.device_ptr(&ctx.stream);
            let (kv_ptr, _gkv) = bufs.kv_a.data.device_ptr(&ctx.stream);
            let (idx_ptr, _gi) = bufs.indexer_indices.device_ptr(&ctx.stream);
            let (out_ptr, _go) = bufs.prefill_attn_out.device_ptr_mut(&ctx.stream);
            let (ml_ptr, _gml) = bufs.prefill_max_logits.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _gl) = bufs.prefill_lse.device_ptr_mut(&ctx.stream);

            let softmax_scale = config.base_softmax_scale();

            // H100 SXM has 132 SMs
            const NUM_SM: i32 = 132;

            unsafe {
                ffi::flash_mla_sparse_prefill(
                    q_ptr as *const std::ffi::c_void,
                    kv_ptr as *const std::ffi::c_void,
                    idx_ptr as *const i32,
                    out_ptr as *mut std::ffi::c_void,
                    ml_ptr as *mut f32,
                    lse_ptr as *mut f32,
                    bs as i32,        // s_q
                    bs as i32,        // s_kv (same: all tokens are both queries and keys)
                    num_heads as i32, // h_q
                    1,                // h_kv (MLA: single KV head)
                    kv_a_dim as i32,  // d_qk = 576
                    kv_lora as i32,   // d_v = 512
                    idx_topk as i32,
                    softmax_scale,
                    NUM_SM,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 9. V de-absorption: prefill_attn_out @ W_UV → v_deabsorbed
        //
        //    attn_out layout: [bs, num_heads, kv_lora] row-major
        //    = [num_heads * kv_lora, bs] col-major with interleaved heads
        // ====================================================================
        {
            let num_heads = config.num_attention_heads;
            let v_dim = config.v_head_dim;
            let kv_lora = config.kv_lora_rank;
            let o_proj_in = config.o_proj_input_dim();

            let (w_uv_ptr, _gwv) = layer.absorbed.w_uv.device_ptr(&ctx.stream);
            let (attn_ptr, _ga) = bufs.prefill_attn_out.device_ptr(&ctx.stream);
            let (v_ptr, _gv) = bufs.v_deabsorbed.data.device_ptr_mut(&ctx.stream);

            bufs.v_deabsorbed.seq_len = bs;

            unsafe {
                ffi::gemm_strided_batched_cuda(
                    1,              // transa = T
                    0,              // transb = N
                    v_dim as i32,   // m
                    bs as i32,      // n
                    kv_lora as i32, // k
                    w_uv_ptr as *const ffi::Half,
                    kv_lora as i32,           // lda
                    (v_dim * kv_lora) as i64, // strideA
                    attn_ptr as *const ffi::Half,
                    (num_heads * kv_lora) as i32, // ldb
                    kv_lora as i64,               // strideB
                    v_ptr as *mut ffi::Half,
                    o_proj_in as i32, // ldc
                    v_dim as i64,     // strideC
                    num_heads as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ====================================================================
        // 10. O projection + residual: hidden += o_proj(v_deabsorbed)
        // ====================================================================
        bufs.attn_proj_out.seq_len = bs;
        fp8_linear_into(
            ctx,
            &bufs.v_deabsorbed,
            &layer.mla.o_proj,
            &mut bufs.fp8_intermediate,
            &mut bufs.attn_proj_out,
        );

        // Fused residual + post_attention_layernorm
        bufs.normed.seq_len = bs;
        ops::fused_add_rms_norm_batch_into(
            ctx,
            hidden,
            &bufs.attn_proj_out,
            &layer.post_attention_layernorm,
            config.rms_norm_eps,
            &mut bufs.normed,
        );

        // ====================================================================
        // 11. FFN (Dense or MoE)
        // ====================================================================
        match &layer.ffn {
            FfnWeights::Dense(ffn) => {
                fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);

                bufs.ffn_gate.seq_len = bs;
                fp8_gemm_into(
                    ctx,
                    bs,
                    config.hidden_size,
                    &ffn.gate_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.ffn_gate,
                );

                bufs.ffn_up.seq_len = bs;
                fp8_gemm_into(
                    ctx,
                    bs,
                    config.hidden_size,
                    &ffn.up_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.ffn_up,
                );

                bufs.ffn_act.seq_len = bs;
                ops::silu_mul_batch_into(ctx, &bufs.ffn_gate, &bufs.ffn_up, &mut bufs.ffn_act)?;

                bufs.ffn_out.seq_len = bs;
                fp8_linear_into(
                    ctx,
                    &bufs.ffn_act,
                    &ffn.down_proj,
                    &mut bufs.fp8_intermediate,
                    &mut bufs.ffn_out,
                );

                // Residual: hidden += ffn_out
                {
                    let n = (hidden.hidden_dim * bs) as i32;
                    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
                    let (f_ptr, _gf) = bufs.ffn_out.data.device_ptr(&ctx.stream);
                    unsafe {
                        ffi::add_cuda(
                            h_ptr as *const ffi::Half,
                            f_ptr as *const ffi::Half,
                            h_ptr as *mut ffi::Half,
                            n,
                            ctx.stream.cu_stream(),
                        );
                    }
                }
            }
            FfnWeights::MoE(moe) => {
                self.forward_moe(hidden, moe, bufs)?;
            }
        }

        Ok(())
    }

    /// NSA indexer forward: compute sparse attention indices for one layer.
    ///
    /// q_compressed, normed, and fp8_hidden must already be computed for this layer.
    /// Reads bufs.normed for weights_proj GEMM and bufs.fp8_hidden for wk GEMM.
    ///
    /// Output: bufs.indexer_indices [T, topk] i32 — token-level KV positions.
    fn forward_indexer(&self, layer_idx: usize, bufs: &mut MlaForwardBuffers) {
        let ctx = &self.ctx;
        let config = &self.config;
        let layer = &self.layers[layer_idx];
        let indexer = layer.indexer.as_ref().expect("indexer weights required");
        let idx_h = config.index_n_heads.unwrap();
        let idx_d = config.index_head_dim.unwrap();
        let idx_topk = config.index_topk.unwrap();
        let rope_dim = config.qk_rope_head_dim;
        let bs = bufs.q_compressed.seq_len;

        // 1. wq_b: q_compressed [q_lora_rank, T] → indexer_q [idx_h * idx_d, T]
        //    FP8 quantize q_compressed, then GEMM: wq_b @ q_compressed_fp8
        {
            let (q_ptr, _g) = bufs.q_compressed.data.device_ptr(&ctx.stream);
            let (fp8_ptr, _gf) = bufs.fp8_indexer_q.fp8_act.device_ptr_mut(&ctx.stream);
            let (scale_ptr, _gs) = bufs.fp8_indexer_q.scale_a.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::fp8_quantize_1x128_cuda(
                    q_ptr as *const ffi::Half,
                    fp8_ptr as *mut u8,
                    scale_ptr as *mut f32,
                    bs as i32,
                    config.q_lora_rank as i32,
                    ctx.stream.cu_stream(),
                );
            }

            let (w_ptr, _gw) = indexer.wq_b.data.device_ptr(&ctx.stream);
            let (ws_ptr, _gsb) = indexer.wq_b.scale_inv.device_ptr(&ctx.stream);
            let (iq_ptr, _go) = bufs.indexer_q.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::fp8_gemm_cuda(
                    fp8_ptr as *const u8,
                    scale_ptr as *const f32,
                    w_ptr as *const u8,
                    ws_ptr as *const f32,
                    iq_ptr as *mut ffi::Half,
                    bs as i32,
                    (idx_h * idx_d) as i32,
                    config.q_lora_rank as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // 2. wk: hidden_normed → indexer_k [idx_d, T]
        //    Uses already-quantized fp8_hidden from main MLA path.
        {
            let (fp8_ptr, _gf) = bufs.fp8_hidden.fp8_act.device_ptr(&ctx.stream);
            let (scale_ptr, _gs) = bufs.fp8_hidden.scale_a.device_ptr(&ctx.stream);
            let (w_ptr, _gw) = indexer.wk.data.device_ptr(&ctx.stream);
            let (ws_ptr, _gsb) = indexer.wk.scale_inv.device_ptr(&ctx.stream);
            let (ik_ptr, _go) = bufs.indexer_k.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::fp8_gemm_cuda(
                    fp8_ptr as *const u8,
                    scale_ptr as *const f32,
                    w_ptr as *const u8,
                    ws_ptr as *const f32,
                    ik_ptr as *mut ffi::Half,
                    bs as i32,
                    idx_d as i32,
                    config.hidden_size as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // 3. k_norm: LayerNorm with bias on indexer_k [idx_d, T] (in-place)
        {
            let (ik_ptr, _g) = bufs.indexer_k.device_ptr_mut(&ctx.stream);
            let (w_ptr, _gw) = indexer.k_norm_weight.data.device_ptr(&ctx.stream);
            let (b_ptr, _gb) = indexer.k_norm_bias.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::nsa_layernorm_bias_cuda(
                    ik_ptr as *const std::ffi::c_void,
                    w_ptr as *const std::ffi::c_void,
                    b_ptr as *const std::ffi::c_void,
                    ik_ptr as *mut std::ffi::c_void,
                    idx_d as i32,
                    bs as i32,
                    1e-5,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // 4. weights_proj: normed [hidden, T] → weights [idx_h, T]
        //    bf16 GEMM: weights_proj [idx_h, hidden] @ normed
        //    cuBLAS col-major output: [idx_h, T] = [T, idx_h] row-major
        {
            let (h_ptr, _gh) = bufs.normed.data.device_ptr(&ctx.stream);
            let (wp_ptr, _gwp) = indexer.weights_proj.data.device_ptr(&ctx.stream);
            let (iw_ptr, _gw) = bufs.indexer_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::gemm_cuda(
                    wp_ptr as *const ffi::Half,
                    h_ptr as *const ffi::Half,
                    iw_ptr as *mut ffi::Half,
                    idx_h as i32,
                    bs as i32,
                    config.hidden_size as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // 5. RoPE on indexer q and k.
        //    Indexer layout: rope(64) first, nope(64) second.
        //    indexer_q: [idx_h * idx_d, T] col-major = [T, idx_h, idx_d] row-major
        //    indexer_k: [idx_d, T] col-major = [T, idx_d] row-major
        {
            let (iq_ptr, _g) = bufs.indexer_q.device_ptr_mut(&ctx.stream);
            let (ik_ptr, _gk) = bufs.indexer_k.device_ptr_mut(&ctx.stream);
            let (cos_ptr, _gc) = self.cos_cache.data.device_ptr(&ctx.stream);
            let (sin_ptr, _gs) = self.sin_cache.data.device_ptr(&ctx.stream);
            let (pos_ptr, _gp) = bufs.positions_d.device_ptr(&ctx.stream);
            unsafe {
                ffi::nsa_indexer_rope_cuda(
                    iq_ptr as *mut std::ffi::c_void,
                    ik_ptr as *mut std::ffi::c_void,
                    cos_ptr as *const std::ffi::c_void,
                    sin_ptr as *const std::ffi::c_void,
                    pos_ptr as *const i32,
                    bs as i32,
                    idx_h as i32,
                    idx_d as i32,
                    rope_dim as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // 6. Fused score + causal topk
        //    score[m][n] = sum_h relu(q[m,h,:] · k[n,:]) * weights[m,h] * scale
        //    scale = D^{-0.5} * H^{-0.5}
        //    → indices [T, topk]
        {
            let weight_scale = (idx_d as f32).powf(-0.5) * (idx_h as f32).powf(-0.5);
            let (iq_ptr, _g) = bufs.indexer_q.device_ptr(&ctx.stream);
            let (ik_ptr, _gk) = bufs.indexer_k.device_ptr(&ctx.stream);
            let (iw_ptr, _gw) = bufs.indexer_weights.device_ptr(&ctx.stream);
            let (idx_ptr, _gi) = bufs.indexer_indices.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::nsa_indexer_fused_score_topk_cuda(
                    iq_ptr as *const std::ffi::c_void,
                    ik_ptr as *const std::ffi::c_void,
                    iw_ptr as *const std::ffi::c_void,
                    idx_ptr as *mut i32,
                    bs as i32,
                    idx_h as i32,
                    idx_d as i32,
                    idx_topk as i32,
                    weight_scale,
                    ctx.stream.cu_stream(),
                );
            }
        }
    }

    /// Run FlashMLA 3-phase decode attention.
    fn flash_mla_decode(
        &self,
        layer_idx: usize,
        bs: usize,
        kv_states: &[&mut MlaKvState],
        bufs: &mut MlaForwardBuffers,
        kv_pool: &MlaKvPool,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let num_heads = config.num_attention_heads;
        let kv_lora = config.kv_lora_rank;
        let kv_a_dim = config.kv_a_proj_dim();

        // Build FlashMLA metadata on CPU
        let mut seqlens_k = Vec::with_capacity(bs);
        let mut block_table_flat = Vec::new();
        let mut max_blocks_per_seq = 0usize;

        for kv in kv_states.iter() {
            seqlens_k.push(kv.seq_len() as i32);
            let pages = kv.page_indices_i32();
            max_blocks_per_seq = max_blocks_per_seq.max(pages.len());
            block_table_flat.extend_from_slice(&pages);
        }

        // Pad block_table to rectangular [bs, max_blocks_per_seq]
        let padding_page = kv_pool.padding_page_id();
        let mut block_table_rect = vec![padding_page; bs * max_blocks_per_seq];
        for (i, kv) in kv_states.iter().enumerate() {
            let pages = kv.page_indices_i32();
            for (j, &p) in pages.iter().enumerate() {
                block_table_rect[i * max_blocks_per_seq + j] = p;
            }
        }

        // Upload metadata
        let seqlens_k_d = ctx.stream.clone_htod(&seqlens_k)?;
        let block_table_d = ctx.stream.clone_htod(&block_table_rect)?;

        let num_sm_parts = FLASH_MLA_NUM_SM_PARTS;

        // Phase 1: get metadata
        {
            let (seqlens_ptr, _gs) = seqlens_k_d.device_ptr(&ctx.stream);
            let (meta_ptr, _gm) = bufs.tile_scheduler_metadata.device_ptr_mut(&ctx.stream);
            let (splits_ptr, _gsp) = bufs.num_splits.device_ptr_mut(&ctx.stream);

            unsafe {
                ffi::flash_mla_get_metadata(
                    bs as i32,
                    1, // seqlen_q = 1 for decode
                    seqlens_ptr as *const i32,
                    meta_ptr as *mut i32,
                    splits_ptr as *mut i32,
                    num_sm_parts,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // Read total_num_splits from GPU
        let num_splits_host: Vec<i32> = ctx.stream.clone_dtoh(&bufs.num_splits.slice(..bs + 1))?;
        ctx.sync()?;
        let total_num_splits = num_splits_host[bs];

        // Phase 2: decode attention
        let layer_offset = kv_pool.layer_offset(layer_idx);
        {
            let (q_ptr, _gq) = bufs.q_mla.device_ptr(&ctx.stream);
            let (buf_ptr, _gb) = kv_pool.buffer().device_ptr(&ctx.stream);
            let kcache_ptr = buf_ptr + (layer_offset * std::mem::size_of::<bf16>()) as u64;
            let (o_ptr, _go) = bufs.attn_out.device_ptr_mut(&ctx.stream);
            let (lse_ptr, _gl) = bufs.lse.device_ptr_mut(&ctx.stream);
            let (lse_acc_ptr, _gla) = bufs.lse_accum.device_ptr_mut(&ctx.stream);
            let (o_acc_ptr, _goa) = bufs.o_accum.device_ptr_mut(&ctx.stream);
            let (bt_ptr, _gbt) = block_table_d.device_ptr(&ctx.stream);
            let (seqlens_ptr, _gs) = seqlens_k_d.device_ptr(&ctx.stream);
            let (meta_ptr, _gm) = bufs.tile_scheduler_metadata.device_ptr(&ctx.stream);
            let (splits_ptr, _gsp) = bufs.num_splits.device_ptr(&ctx.stream);

            let softmax_scale = config.base_softmax_scale();
            let q_seq_per_hk = num_heads; // h_q / h_k = 128 / 1

            unsafe {
                ffi::flash_mla_decode(
                    q_ptr as *const ffi::Half,
                    kcache_ptr as *const ffi::Half,
                    o_ptr as *mut ffi::Half,
                    lse_ptr as *mut f32,
                    lse_acc_ptr as *mut f32,
                    o_acc_ptr as *mut f32,
                    bt_ptr as *const i32,
                    seqlens_ptr as *const i32,
                    meta_ptr as *const i32,
                    splits_ptr as *const i32,
                    bs as i32,
                    1, // seqlen_q
                    q_seq_per_hk as i32,
                    num_heads as i32, // h_q
                    1,                // h_k
                    kv_a_dim as i32,  // d_k = 576
                    kv_lora as i32,   // d_v = 512
                    kv_pool.num_pages() as i32,
                    max_blocks_per_seq as i32,
                    num_sm_parts,
                    total_num_splits,
                    softmax_scale,
                    0, // is_causal = 0 for decode
                    ctx.stream.cu_stream(),
                );
            }
        }

        // Phase 3: combine
        {
            let (lse_ptr, _gl) = bufs.lse.device_ptr_mut(&ctx.stream);
            let (o_ptr, _go) = bufs.attn_out.device_ptr_mut(&ctx.stream);
            let (lse_acc_ptr, _gla) = bufs.lse_accum.device_ptr_mut(&ctx.stream);
            let (o_acc_ptr, _goa) = bufs.o_accum.device_ptr_mut(&ctx.stream);
            let (meta_ptr, _gm) = bufs.tile_scheduler_metadata.device_ptr(&ctx.stream);
            let (splits_ptr, _gsp) = bufs.num_splits.device_ptr(&ctx.stream);

            unsafe {
                ffi::flash_mla_combine(
                    lse_ptr as *mut f32,
                    o_ptr as *mut ffi::Half,
                    lse_acc_ptr as *mut f32,
                    o_acc_ptr as *mut f32,
                    meta_ptr as *const i32,
                    splits_ptr as *const i32,
                    bs as i32,
                    1, // seqlen_q
                    num_heads as i32,
                    kv_lora as i32, // d_v
                    num_sm_parts,
                    ctx.stream.cu_stream(),
                );
            }
        }

        Ok(())
    }

    /// MoE forward: routing + shared expert FFN + routed expert FFN.
    ///
    /// For decode (bs=1), each token selects `num_experts_per_tok` experts.
    /// Expert FFN is run sequentially per-expert; outputs are weighted and
    /// accumulated into `hidden`.
    fn forward_moe(
        &self,
        hidden: &mut HiddenStates,
        moe: &MoeFfnWeights,
        bufs: &mut MlaForwardBuffers,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let bs = hidden.seq_len;
        let topk = config.num_experts_per_tok;
        // bufs.normed is already populated by the caller (fused_add_rms_norm).

        // ==================================================================
        // 1. Router: gate_weight @ normed → router_logits [n_routed_experts, bs]
        // ==================================================================
        bufs.router_logits.seq_len = bs;
        {
            let (w_ptr, _gw) = moe.gate.weight.data.device_ptr(&ctx.stream);
            let (x_ptr, _gx) = bufs.normed.data.device_ptr(&ctx.stream);
            let (y_ptr, _gy) = bufs.router_logits.data.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::gemm_cuda(
                    w_ptr as *const ffi::Half,
                    x_ptr as *const ffi::Half,
                    y_ptr as *mut ffi::Half,
                    config.n_routed_experts as i32,
                    bs as i32,
                    config.hidden_size as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ==================================================================
        // 2. Routing kernel: sigmoid + group-limited TopK
        // ==================================================================
        {
            let (logits_ptr, _gl) = bufs.router_logits.data.device_ptr(&ctx.stream);
            let (bias_ptr, _gb) = moe.gate.e_score_correction_bias.device_ptr(&ctx.stream);
            let (idx_ptr, _gi) = bufs.topk_indices.device_ptr_mut(&ctx.stream);
            let (wt_ptr, _gw) = bufs.topk_weights.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::moe_routing_cuda(
                    logits_ptr as *const ffi::Half,
                    bias_ptr as *const f32,
                    idx_ptr as *mut i32,
                    wt_ptr as *mut f32,
                    config.n_routed_experts as i32,
                    bs as i32,
                    topk as i32,
                    config.n_group as i32,
                    config.topk_group as i32,
                    config.norm_topk_prob as i32,
                    config.routed_scaling_factor,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ==================================================================
        // 3. Shared expert FFN (runs on all tokens, same as dense FFN)
        // ==================================================================
        {
            let expert = &moe.shared_expert;

            // Shared FP8 quantize normed (reuse fp8_hidden from attention path)
            fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);

            bufs.moe_gate.seq_len = bs;
            fp8_gemm_into(
                ctx,
                bs,
                config.hidden_size,
                &expert.gate_proj,
                &bufs.fp8_hidden,
                &mut bufs.moe_gate,
            );

            bufs.moe_up.seq_len = bs;
            fp8_gemm_into(
                ctx,
                bs,
                config.hidden_size,
                &expert.up_proj,
                &bufs.fp8_hidden,
                &mut bufs.moe_up,
            );

            bufs.moe_act.seq_len = bs;
            ops::silu_mul_batch_into(ctx, &bufs.moe_gate, &bufs.moe_up, &mut bufs.moe_act)?;

            bufs.moe_down.seq_len = bs;
            fp8_linear_into(
                ctx,
                &bufs.moe_act,
                &expert.down_proj,
                &mut bufs.fp8_moe_intermediate,
                &mut bufs.moe_down,
            );

            // hidden += shared_expert_output
            let n = (config.hidden_size * bs) as i32;
            let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
            let (s_ptr, _gs) = bufs.moe_down.data.device_ptr(&ctx.stream);
            unsafe {
                ffi::add_cuda(
                    h_ptr as *const ffi::Half,
                    s_ptr as *const ffi::Half,
                    h_ptr as *mut ffi::Half,
                    n,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ==================================================================
        // 4. Routed expert FFN
        // ==================================================================
        if self.parallel.is_ep_sharded() {
            self.forward_moe_ep(hidden, moe, bufs)?;
        } else {
            self.forward_moe_local(hidden, moe, bufs)?;
        }

        Ok(())
    }

    /// EP1 routed expert FFN: per-token, per-expert sequential.
    /// All experts are local — no cross-GPU communication needed.
    fn forward_moe_local(
        &self,
        hidden: &mut HiddenStates,
        moe: &MoeFfnWeights,
        bufs: &mut MlaForwardBuffers,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let bs = hidden.seq_len;
        let topk = config.num_experts_per_tok;

        // Read routing results to CPU
        let topk_idx_host: Vec<i32> = ctx
            .stream
            .clone_dtoh(&bufs.topk_indices.slice(..bs * topk))?;
        let topk_wt_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(&bufs.topk_weights.slice(..bs * topk))?;
        ctx.sync()?;

        let (ep_start, _ep_count) = self.parallel.local_expert_range(config.n_routed_experts);

        for token_idx in 0..bs {
            let n_tok = 1usize;
            if bs == 1 {
                fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);
            } else {
                let offset_elems = token_idx * config.hidden_size;
                let token_view = HiddenStates {
                    data: ctx.stream.clone_dtod(
                        &bufs
                            .normed
                            .data
                            .slice(offset_elems..offset_elems + config.hidden_size),
                    )?,
                    hidden_dim: config.hidden_size,
                    seq_len: 1,
                };
                fp8_quantize_into(ctx, &token_view, &mut bufs.fp8_hidden);
            }

            for k in 0..topk {
                let expert_idx = topk_idx_host[token_idx * topk + k] as usize;
                let weight = topk_wt_host[token_idx * topk + k];

                let local_idx = expert_idx - ep_start;
                let expert = &moe.experts[local_idx];

                bufs.moe_gate.seq_len = n_tok;
                fp8_gemm_into(
                    ctx,
                    n_tok,
                    config.hidden_size,
                    &expert.gate_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.moe_gate,
                );

                bufs.moe_up.seq_len = n_tok;
                fp8_gemm_into(
                    ctx,
                    n_tok,
                    config.hidden_size,
                    &expert.up_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.moe_up,
                );

                bufs.moe_act.seq_len = n_tok;
                ops::silu_mul_batch_into(ctx, &bufs.moe_gate, &bufs.moe_up, &mut bufs.moe_act)?;

                bufs.moe_down.seq_len = n_tok;
                fp8_linear_into(
                    ctx,
                    &bufs.moe_act,
                    &expert.down_proj,
                    &mut bufs.fp8_moe_intermediate,
                    &mut bufs.moe_down,
                );

                let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
                let h_token_ptr = (h_ptr
                    + (token_idx * config.hidden_size * std::mem::size_of::<bf16>()) as u64)
                    as *mut ffi::Half;
                let (e_ptr, _ge) = bufs.moe_down.data.device_ptr(&ctx.stream);
                unsafe {
                    ffi::moe_weighted_add_cuda(
                        h_token_ptr,
                        e_ptr as *const ffi::Half,
                        weight,
                        config.hidden_size as i32,
                        ctx.stream.cu_stream(),
                    );
                }
            }
        }

        Ok(())
    }

    /// EP>1 routed expert FFN: DeepEP dispatch → local expert compute → combine.
    fn forward_moe_ep(
        &self,
        hidden: &mut HiddenStates,
        moe: &MoeFfnWeights,
        bufs: &mut MlaForwardBuffers,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let config = &self.config;
        let bs = hidden.seq_len;
        let topk = config.num_experts_per_tok;
        let hidden_size = config.hidden_size;

        let ep_buf = self
            .deep_ep_buffer
            .as_ref()
            .expect("DeepEP buffer must be initialized for EP > 1");
        let ep_config = &ep_buf.config;
        let rank = ep_buf.rank();
        let num_ranks = ep_buf.num_ranks();
        let num_channels = ep_config.num_channels();
        let (_, ep_count) = self.parallel.local_expert_range(config.n_routed_experts);

        // hidden_int4 = hidden_size * sizeof(bf16) / sizeof(int4) = hidden_size * 2 / 16
        let hidden_int4 = (hidden_size * 2 / 16) as i32;

        // ==================================================================
        // Step 1: Cast topk_indices i32 → i64 for DeepEP
        // ==================================================================
        {
            let (idx_ptr, _gi) = bufs.topk_indices.device_ptr(&ctx.stream);
            let (idx64_ptr, _gi64) = bufs.topk_indices_i64.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::cast_i32_to_i64_cuda(
                    idx_ptr as *const i32,
                    idx64_ptr as *mut i64,
                    (bs * topk) as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ==================================================================
        // Step 2: get_dispatch_layout — compute routing metadata
        // ==================================================================
        {
            let (idx64_ptr, _gi) = bufs.topk_indices_i64.device_ptr(&ctx.stream);
            let (ntpr_ptr, _g1) = bufs.num_tokens_per_rank.device_ptr_mut(&ctx.stream);
            let (ntpe_ptr, _g2) = bufs.num_tokens_per_expert.device_ptr_mut(&ctx.stream);
            let (itr_ptr, _g3) = bufs.is_token_in_rank.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::deep_ep_get_dispatch_layout(
                    idx64_ptr as *const i64,
                    ntpr_ptr as *mut i32,
                    ntpe_ptr as *mut i32,
                    itr_ptr as *mut bool,
                    bs as i32,
                    topk as i32,
                    num_ranks,
                    config.n_routed_experts as i32,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ==================================================================
        // Step 3: notify_dispatch — exchange token counts via NVLink
        // ==================================================================
        ep_buf.reset_recv_counters();
        let num_memset_int = {
            // Number of ints to memset in NVLink buffer for this dispatch.
            // This covers the rank prefix matrix + channel metadata.
            let prefix_matrix_ints = num_ranks * num_ranks;
            let expert_prefix_ints = num_ranks * (config.n_routed_experts as i32 / num_ranks);
            let channel_ints = num_channels * num_ranks * 5; // 5 control arrays
            (prefix_matrix_ints + expert_prefix_ints + channel_ints) as i32
        };

        {
            let (ntpr_ptr, _g1) = bufs.num_tokens_per_rank.device_ptr(&ctx.stream);
            let (ntpe_ptr, _g2) = bufs.num_tokens_per_expert.device_ptr(&ctx.stream);
            let (itr_ptr, _g3) = bufs.is_token_in_rank.device_ptr(&ctx.stream);
            let (cpm_ptr, _g4) = bufs.channel_prefix_matrix.device_ptr_mut(&ctx.stream);
            let (rpm_ptr, _g5) = bufs.rank_prefix_matrix_copy.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::deep_ep_notify_dispatch(
                    ntpr_ptr as *const i32,
                    ep_buf.moe_recv_counter_mapped(),
                    num_ranks,
                    ntpe_ptr as *const i32,
                    ep_buf.moe_recv_expert_counter_mapped(),
                    config.n_routed_experts as i32,
                    bs as i32,
                    itr_ptr as *const bool,
                    cpm_ptr as *mut i32,
                    rpm_ptr as *mut i32,
                    num_memset_int,
                    1, // expert_alignment
                    ep_buf.buffer_ptrs_gpu(),
                    ep_buf.barrier_signal_ptrs_gpu(),
                    rank,
                    ctx.stream.cu_stream(),
                    num_channels, // notify_dispatch expects num_channels, not num_sms
                );
            }
        }

        // Sync to read recv count from host-mapped memory
        ctx.sync()?;
        let num_recv_tokens = ep_buf.read_recv_count();

        // ==================================================================
        // Step 4: dispatch — send tokens to target ranks via NVLink
        // ==================================================================
        {
            let (normed_ptr, _gn) = bufs.normed.data.device_ptr(&ctx.stream);
            let (idx64_ptr, _gi) = bufs.topk_indices_i64.device_ptr(&ctx.stream);
            let (wt_ptr, _gw) = bufs.topk_weights.device_ptr(&ctx.stream);
            let (itr_ptr, _g3) = bufs.is_token_in_rank.device_ptr(&ctx.stream);
            let (cpm_ptr, _g4) = bufs.channel_prefix_matrix.device_ptr(&ctx.stream);

            let (recv_x_ptr, _grx) = bufs.ep_recv_x.device_ptr_mut(&ctx.stream);
            let (recv_si_ptr, _grs) = bufs.ep_recv_src_idx.device_ptr_mut(&ctx.stream);
            let (recv_ti_ptr, _grt) = bufs.ep_recv_topk_idx.device_ptr_mut(&ctx.stream);
            let (recv_tw_ptr, _grw) = bufs.ep_recv_topk_weights.device_ptr_mut(&ctx.stream);
            let (recv_co_ptr, _grc) = bufs
                .ep_recv_channel_prefix_matrix
                .device_ptr_mut(&ctx.stream);
            let (sh_ptr, _gsh) = bufs.ep_send_head.device_ptr_mut(&ctx.stream);

            unsafe {
                ffi::deep_ep_intranode_dispatch(
                    recv_x_ptr as *mut std::ffi::c_void,
                    std::ptr::null_mut(), // recv_x_scales — no FP8 for dispatch (bf16 tokens)
                    recv_si_ptr as *mut i32,
                    recv_ti_ptr as *mut i64,
                    recv_tw_ptr as *mut f32,
                    recv_co_ptr as *mut i32,
                    sh_ptr as *mut i32,
                    normed_ptr as *const std::ffi::c_void, // send normed hidden (bf16)
                    std::ptr::null(),                      // x_scales — no FP8
                    idx64_ptr as *const i64,
                    wt_ptr as *const f32,
                    itr_ptr as *const bool,
                    cpm_ptr as *const i32,
                    bs as i32,
                    0, // num_worst_tokens
                    hidden_int4,
                    topk as i32,
                    config.n_routed_experts as i32,
                    0, // num_scales — no FP8
                    0, // scale_token_stride
                    0, // scale_hidden_stride
                    ep_buf.buffer_ptrs_gpu(),
                    rank,
                    num_ranks,
                    ctx.stream.cu_stream(),
                    ep_config.num_sms,
                    ep_config.dispatch_max_send_tokens,
                    ep_config.num_max_nvl_chunked_recv_tokens,
                );
            }
        }

        debug!(
            "[rank {}] dispatch done: num_recv_tokens={}, bs={}",
            rank, num_recv_tokens, bs
        );

        // ==================================================================
        // Step 5: Run local expert FFN on received tokens
        // ==================================================================
        // recv_x: [num_recv_tokens, hidden_size] bf16 (row-major from dispatch)
        // We need to transpose to column-major [hidden_size, num_recv_tokens] for our GEMM.
        // For decode (small num_recv_tokens), we process per-token sequentially.
        //
        // Read recv_topk_idx to know which local expert each recv'd token goes to.
        if num_recv_tokens > 0 {
            let recv_topk_idx_host: Vec<i64> = ctx.stream.clone_dtoh(
                &bufs
                    .ep_recv_topk_idx
                    .slice(..num_recv_tokens as usize * topk),
            )?;
            let recv_topk_wt_host: Vec<f32> = ctx.stream.clone_dtoh(
                &bufs
                    .ep_recv_topk_weights
                    .slice(..num_recv_tokens as usize * topk),
            )?;
            ctx.sync()?;

            // Process each received token through its designated local experts.
            // DeepEP dispatch rewrites recv_topk_idx into rank-local expert ids
            // for this receiving rank and fills non-local entries with -1.
            for tok in 0..num_recv_tokens as usize {
                // Create a view of this token in recv_x (row-major from dispatch).
                // DeepEP dispatch outputs recv_x in row-major: [num_recv_tokens, hidden].
                // Our GEMM expects column-major [hidden, bs].
                // For bs=1 per token, row-major [1, hidden] == column-major [hidden, 1].
                let offset = tok * hidden_size;
                let token_view = HiddenStates {
                    data: ctx
                        .stream
                        .clone_dtod(&bufs.ep_recv_x.slice(offset..offset + hidden_size))?,
                    hidden_dim: hidden_size,
                    seq_len: 1,
                };
                fp8_quantize_into(ctx, &token_view, &mut bufs.fp8_hidden);

                // This recv slot is the accumulation buffer for local expert outputs.
                // Zero it once up front so the first valid local expert does not
                // depend on its original top-k position or on stale input activations.
                let (recv_x_ptr, _grx) = bufs.ep_recv_x.device_ptr_mut(&ctx.stream);
                let recv_tok_ptr =
                    (recv_x_ptr + (offset * std::mem::size_of::<bf16>()) as u64) as *mut ffi::Half;
                unsafe {
                    ffi::cudaMemsetAsync(
                        recv_tok_ptr as *mut std::ffi::c_void,
                        0,
                        hidden_size * std::mem::size_of::<bf16>(),
                        ctx.stream.cu_stream(),
                    );
                }

                for k in 0..topk {
                    let expert_idx = recv_topk_idx_host[tok * topk + k];
                    let weight = recv_topk_wt_host[tok * topk + k];

                    // DeepEP recv_topk_idx is rank-local expert space on the
                    // receiving rank. Non-local experts are encoded as -1.
                    if expert_idx < 0 {
                        continue;
                    }
                    let local_idx = expert_idx as usize;
                    if local_idx >= ep_count {
                        continue;
                    }
                    let expert = &moe.experts[local_idx];

                    bufs.moe_gate.seq_len = 1;
                    fp8_gemm_into(
                        ctx,
                        1,
                        hidden_size,
                        &expert.gate_proj,
                        &bufs.fp8_hidden,
                        &mut bufs.moe_gate,
                    );

                    bufs.moe_up.seq_len = 1;
                    fp8_gemm_into(
                        ctx,
                        1,
                        hidden_size,
                        &expert.up_proj,
                        &bufs.fp8_hidden,
                        &mut bufs.moe_up,
                    );

                    bufs.moe_act.seq_len = 1;
                    ops::silu_mul_batch_into(ctx, &bufs.moe_gate, &bufs.moe_up, &mut bufs.moe_act)?;

                    bufs.moe_down.seq_len = 1;
                    fp8_linear_into(
                        ctx,
                        &bufs.moe_act,
                        &expert.down_proj,
                        &mut bufs.fp8_moe_intermediate,
                        &mut bufs.moe_down,
                    );

                    // Accumulate weighted expert output into the recv_x slot
                    // for this token. Normal DeepEP combine only sums activations
                    // across ranks, so routed weights must not be passed in again.
                    let (e_ptr, _ge) = bufs.moe_down.data.device_ptr(&ctx.stream);
                    unsafe {
                        ffi::moe_weighted_add_cuda(
                            recv_tok_ptr,
                            e_ptr as *const ffi::Half,
                            weight,
                            hidden_size as i32,
                            ctx.stream.cu_stream(),
                        );
                    }
                }
            }
        }

        // ==================================================================
        // Step 5.5: cached_notify_combine — barrier + zero NVL buffer for combine
        // ==================================================================
        // The combine kernel uses a different buffer layout than dispatch:
        //   dispatch starts at buffer + kNumRanks² ints (skips rank_prefix_matrix)
        //   combine starts at buffer + 0
        // After dispatch, the buffer beginning contains the rank_prefix_matrix
        // which overlaps combine's channel_head_idx. We must zero the combine
        // channel control area (head_idx + tail_idx) and preprocess send_head.
        //
        // num_recv_tokens here = send_head.size(0) = original bs (combine output tokens),
        // NOT the dispatch recv count. See deep_ep.cpp:798,839.
        {
            let (sh_ptr, _gsh) = bufs.ep_send_head.device_ptr_mut(&ctx.stream);
            // combine channel control: 2 arrays of (num_channels * num_ranks) ints
            let combine_memset_int = 2 * num_channels * num_ranks;
            unsafe {
                ffi::deep_ep_cached_notify_combine(
                    ep_buf.buffer_ptrs_gpu(),
                    sh_ptr as *mut i32,
                    num_channels,
                    bs as i32,
                    combine_memset_int,
                    ep_buf.barrier_signal_ptrs_gpu(),
                    rank,
                    num_ranks,
                    ctx.stream.cu_stream(),
                );
            }
        }

        // ==================================================================
        // Step 6: combine — gather expert outputs back to source ranks
        // ==================================================================
        // Python handle from dispatch: (rank_prefix_matrix, channel_prefix_matrix,
        //   recv_channel_prefix_matrix, recv_src_idx, is_recv_token_in_rank, send_head)
        // Combine unpacks: rank_prefix_matrix, _, channel_prefix_matrix(=recv_channel_prefix_matrix),
        //   src_idx, ..., send_head
        // So combine uses:
        //   topk_weights = nullptr because this path has already applied routed
        //     weights locally into recv_x; normal DeepEP combine only reduces x.
        //   channel_prefix_matrix = ep_recv_channel_prefix_matrix (handle[2], recv-side matrix)
        //   num_tokens = dispatch recv count (x.size(0) in Python, deep_ep.cpp:797)
        //   num_recv_tokens = original bs (send_head.size(0) in Python, deep_ep.cpp:798)
        {
            let dispatch_recv_tokens = num_recv_tokens; // from dispatch: tokens this rank received
            let combine_output_tokens = bs as i32; // original batch size: tokens to reconstruct

            let (recv_x_ptr, _grx) = bufs.ep_recv_x.device_ptr(&ctx.stream);
            let (si_ptr, _gs) = bufs.ep_recv_src_idx.device_ptr(&ctx.stream);
            let (rpm_ptr, _g5) = bufs.rank_prefix_matrix_copy.device_ptr(&ctx.stream);
            let (cpm_ptr, _g4) = bufs.ep_recv_channel_prefix_matrix.device_ptr(&ctx.stream);
            let (sh_ptr, _gsh) = bufs.ep_send_head.device_ptr_mut(&ctx.stream);

            let (combined_ptr, _gc) = bufs.ep_combined_x.device_ptr_mut(&ctx.stream);

            debug!(
                "[rank {}] combine: dispatch_recv_tokens={}, combine_output_tokens(bs)={}, hidden={}, topk={}, num_sms={}, max_send={}, max_recv_buf={}",
                rank,
                dispatch_recv_tokens,
                combine_output_tokens,
                hidden_size,
                topk,
                ep_config.num_sms,
                ep_config.combine_max_send_tokens,
                ep_config.num_max_nvl_chunked_recv_tokens,
            );

            unsafe {
                ffi::deep_ep_intranode_combine(
                    combined_ptr as *mut std::ffi::c_void,
                    std::ptr::null_mut(),
                    recv_x_ptr as *const std::ffi::c_void,
                    std::ptr::null(),
                    si_ptr as *const i32,
                    rpm_ptr as *const i32,
                    cpm_ptr as *const i32,
                    sh_ptr as *mut i32,
                    dispatch_recv_tokens, // num_tokens: combine input x dimension
                    combine_output_tokens, // num_recv_tokens: combine output dimension (original bs)
                    hidden_size as i32,
                    0,
                    ep_buf.buffer_ptrs_gpu(),
                    rank,
                    num_ranks,
                    ctx.stream.cu_stream(),
                    ep_config.num_sms,
                    ep_config.combine_max_send_tokens,
                    ep_config.num_max_nvl_chunked_recv_tokens,
                );
            }
        }

        // ==================================================================
        // Step 7: Add combined routed expert output to hidden
        // ==================================================================
        // combined_x is [bs, hidden_size] row-major from combine.
        // For bs=1: [1, hidden] row == [hidden, 1] col, direct add works.
        {
            let n = (hidden_size * bs) as i32;
            let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
            let (c_ptr, _gc) = bufs.ep_combined_x.device_ptr(&ctx.stream);
            unsafe {
                ffi::add_cuda(
                    h_ptr as *const ffi::Half,
                    c_ptr as *const ffi::Half,
                    h_ptr as *mut ffi::Half,
                    n,
                    ctx.stream.cu_stream(),
                );
            }
        }

        Ok(())
    }
}
