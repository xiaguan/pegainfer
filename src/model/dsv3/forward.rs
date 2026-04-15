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

use super::config::DsV3Config;
use super::mla_kv::{MlaKvPool, MlaKvState};
use super::weights::{DsV3Model, FfnWeights, MoeFfnWeights};
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
}

// Number of SM partitions for FlashMLA.
const FLASH_MLA_NUM_SM_PARTS: i32 = 72;

impl MlaForwardBuffers {
    pub(crate) fn new(ctx: &DeviceContext, config: &DsV3Config, max_bs: usize) -> Result<Self> {
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
        })
    }
}

impl DsV3Model {
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
        let mut offset = 0;
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
        // 3. Read routing results to CPU (small: bs * topk ints + floats)
        // ==================================================================
        let topk_idx_host: Vec<i32> = ctx
            .stream
            .clone_dtoh(&bufs.topk_indices.slice(..bs * topk))?;
        let topk_wt_host: Vec<f32> = ctx
            .stream
            .clone_dtoh(&bufs.topk_weights.slice(..bs * topk))?;
        ctx.sync()?;

        // ==================================================================
        // 4. Shared expert FFN (runs on all tokens, same as dense FFN)
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
        // 5. Routed expert FFN (per-expert sequential, decode-optimized)
        // ==================================================================
        // For each token, run its topk selected experts and accumulate
        // weighted outputs into hidden.
        //
        // Current implementation: per-token, per-expert sequential FFN.
        // Decode bs=1 → 8 expert FFN calls (one per selected expert).
        // TODO: for prefill bs>1, use grouped GEMM (DeepGEMM GroupedContiguous).
        let (ep_start, _ep_count) = self.parallel.local_expert_range(config.n_routed_experts);

        for token_idx in 0..bs {
            // FP8 quantize this token's normed hidden.
            // For bs=1 normed is already the right shape; for bs>1 we need to
            // point fp8_quantize at the right column.
            let n_tok = 1usize;
            if bs == 1 {
                fp8_quantize_into(ctx, &bufs.normed, &mut bufs.fp8_hidden);
            } else {
                // Create a single-token view by offsetting into normed's buffer.
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

                // Map global expert index to local (EP-sharded) index
                let local_idx = expert_idx - ep_start;
                let expert = &moe.experts[local_idx];

                // gate_proj: [moe_intermediate, hidden] @ [hidden, 1] → [moe_intermediate, 1]
                bufs.moe_gate.seq_len = n_tok;
                fp8_gemm_into(
                    ctx,
                    n_tok,
                    config.hidden_size,
                    &expert.gate_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.moe_gate,
                );

                // up_proj
                bufs.moe_up.seq_len = n_tok;
                fp8_gemm_into(
                    ctx,
                    n_tok,
                    config.hidden_size,
                    &expert.up_proj,
                    &bufs.fp8_hidden,
                    &mut bufs.moe_up,
                );

                // silu(gate) * up → act
                bufs.moe_act.seq_len = n_tok;
                ops::silu_mul_batch_into(ctx, &bufs.moe_gate, &bufs.moe_up, &mut bufs.moe_act)?;

                // down_proj: [hidden, moe_intermediate] @ [moe_intermediate, 1] → [hidden, 1]
                bufs.moe_down.seq_len = n_tok;
                fp8_linear_into(
                    ctx,
                    &bufs.moe_act,
                    &expert.down_proj,
                    &mut bufs.fp8_moe_intermediate,
                    &mut bufs.moe_down,
                );

                // hidden[:, token_idx] += weight * expert_output
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
}
