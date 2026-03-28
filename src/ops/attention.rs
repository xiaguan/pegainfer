use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Batched prefill attention with FlashAttention-2.
///
/// Pipeline:
///   1. QK norm + RoPE (CUDA kernel, in-place on q_batch/k_batch)
///   2. KV cache write (CUDA kernel)
///   3. FlashAttention-2 (Triton kernel — fused QK + causal softmax + V)
///
/// No O(n²) scratch buffers needed — FlashAttention uses online softmax.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_batch(
    ctx: &DeviceContext,
    q_batch: &mut HiddenStates,
    k_batch: &mut HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let q_dim = num_q_heads * head_dim;
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    let gqa_ratio = num_q_heads / num_kv_heads;

    {
        let (q_ptr, _gq) = q_batch.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin_cache.data.device_ptr(&ctx.stream);
        let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
        let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

        unsafe {
            // Steps 1-2: QK norm + RoPE, KV cache write
            ffi::prefill_attention_prep_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                start_pos as i32,
                rms_eps,
                ctx.stream.cu_stream(),
            );

            // Step 3: FlashAttention-2 (Triton) — reads normed Q and KV cache
            ffi::flash_attention_prefill_cuda(
                q_ptr as *const ffi::Half,
                kc_ptr as *const ffi::Half,
                vc_ptr as *const ffi::Half,
                o_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                gqa_ratio as i32,
                seq_len as i32,
                start_pos as i32,
                q_dim as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(())
}

/// FlashAttention-2 prefill for HEAD_DIM=256 with precomputed Q and KV cache.
/// Q / output layout: HiddenStates [q_dim, seq_len] in column-major token-major storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flash_attention_prefill_hd256_into(
    ctx: &DeviceContext,
    q_batch: &HiddenStates,
    k_cache: &DeviceVec,
    v_cache: &DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos_buf: &CudaSlice<i32>,
) -> Result<()> {
    let seq_len = q_batch.seq_len;
    let q_dim = q_batch.hidden_dim;
    let head_dim = q_dim / num_q_heads;
    assert_eq!(head_dim, 256, "HD256 kernel requires head_dim=256");
    assert_eq!(q_dim, output.hidden_dim, "output hidden_dim mismatch");
    assert_eq!(seq_len, output.seq_len, "output seq_len mismatch");
    assert!(num_kv_heads > 0, "num_kv_heads must be > 0");
    let gqa_ratio = num_q_heads / num_kv_heads;

    let (q_ptr, _gq) = q_batch.data.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (sp_ptr, _gsp) = start_pos_buf.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::flash_attention_prefill_hd256_cuda(
            q_ptr as *const ffi::Half,
            kc_ptr as *const ffi::Half,
            vc_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            num_kv_heads as i32,
            gqa_ratio as i32,
            seq_len as i32,
            sp_ptr as *const i32,
            q_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Qwen3.5 full-attention prefill: prep Q/K/cache, run HD256 FlashAttention-2, then apply gate.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_batch(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos: usize,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let q_dim = num_q_heads * 256;
    let mut q_prepped = HiddenStates::zeros(ctx, q_dim, q_full_batch.seq_len)?;
    // Allocate temporary GPU scalar for start_pos
    let start_pos_buf: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&[start_pos as i32])
        .map_err(|e| anyhow::anyhow!("start_pos H2D failed: {e}"))?;
    prefill_attention_hd256_batch_with_scratch(
        ctx,
        q_full_batch,
        k_batch,
        v_batch,
        q_norm,
        k_norm,
        cos_cache,
        sin_cache,
        k_cache,
        v_cache,
        output,
        &mut q_prepped,
        num_q_heads,
        num_kv_heads,
        &start_pos_buf,
        rotary_dim,
        rms_eps,
    )
}

/// Same as `prefill_attention_hd256_batch` but uses pre-allocated scratch buffers.
/// `start_pos_buf` is a GPU-resident `i32` for CUDA Graph safety.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_hd256_batch_with_scratch(
    ctx: &DeviceContext,
    q_full_batch: &HiddenStates,
    k_batch: &HiddenStates,
    v_batch: &HiddenStates,
    q_norm: &DeviceVec,
    k_norm: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut HiddenStates,
    q_prepped: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    start_pos_buf: &CudaSlice<i32>,
    rotary_dim: usize,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q_full_batch.seq_len;
    let q_dim = num_q_heads * 256;
    let kv_dim = num_kv_heads * 256;

    assert_eq!(q_full_batch.hidden_dim, q_dim * 2);
    assert_eq!(k_batch.hidden_dim, kv_dim);
    assert_eq!(v_batch.hidden_dim, kv_dim);
    assert_eq!(k_batch.seq_len, seq_len);
    assert_eq!(v_batch.seq_len, seq_len);
    assert_eq!(output.hidden_dim, q_dim);
    assert_eq!(output.seq_len, seq_len);
    assert_eq!(q_prepped.hidden_dim, q_dim);

    unsafe {
        let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&ctx.stream);
        let (k_ptr, _gk) = k_batch.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_batch.data.device_ptr(&ctx.stream);
        let (qn_ptr, _gqn) = q_norm.data.device_ptr(&ctx.stream);
        let (kn_ptr, _gkn) = k_norm.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
        let (qp_ptr, _gqp) = q_prepped.data.device_ptr_mut(&ctx.stream);
        let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
        let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
        let (sp_ptr, _gsp) = start_pos_buf.device_ptr(&ctx.stream);

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
            num_q_heads as i32,
            num_kv_heads as i32,
            seq_len as i32,
            sp_ptr as *const i32,
            rotary_dim as i32,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }

    flash_attention_prefill_hd256_into(
        ctx,
        q_prepped,
        k_cache,
        v_cache,
        output,
        num_q_heads,
        num_kv_heads,
        start_pos_buf,
    )?;

    unsafe {
        let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
        ffi::attention_gate_batch_hd256_cuda(
            qf_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_q_heads as i32,
            seq_len as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Fused GQA Attention for decode (Triton AOT, split-KV, HEAD_DIM=128).
/// Reads pos/seq_len from decode_meta — CUDA Graph safe.
/// cos_cache_base/sin_cache_base: full RoPE buffers [max_seq_len * head_dim].
/// decode_meta: [token_id, current_pos, seq_len] on GPU.
/// partial_out/m/l: pre-allocated FP32 scratch for split-KV intermediates.
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_decode_into(
    ctx: &DeviceContext,
    q_full: &DeviceVec,
    k_full: &DeviceVec,
    v_full: &DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache_base: &DeviceVec,
    sin_cache_base: &DeviceVec,
    decode_meta: &CudaSlice<i32>,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut DeviceVec,
    partial_out: &mut CudaSlice<f32>,
    partial_m: &mut CudaSlice<f32>,
    partial_l: &mut CudaSlice<f32>,
    num_qheads: usize,
    num_kvheads: usize,
) -> Result<()> {
    let (q_ptr, _gq) = q_full.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_full.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_full.data.device_ptr(&ctx.stream);
    let (q_norm_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (k_norm_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gcos) = cos_cache_base.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gsin) = sin_cache_base.data.device_ptr(&ctx.stream);
    let (meta_ptr, _gm) = decode_meta.device_ptr(&ctx.stream);
    let (k_cache_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
    let (v_cache_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);
    let (partial_out_ptr, _gpo) = partial_out.device_ptr_mut(&ctx.stream);
    let (partial_m_ptr, _gpm) = partial_m.device_ptr_mut(&ctx.stream);
    let (partial_l_ptr, _gpl) = partial_l.device_ptr_mut(&ctx.stream);

    // Phase 1: split-KV attention (writes partials)
    let result = unsafe {
        ffi::fused_gqa_attention_decode(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            meta_ptr as *const i32,
            k_cache_ptr as *mut ffi::Half,
            v_cache_ptr as *mut ffi::Half,
            partial_out_ptr as *mut f32,
            partial_m_ptr as *mut f32,
            partial_l_ptr as *mut f32,
            num_qheads as i32,
            num_kvheads as i32,
            (num_qheads / num_kvheads) as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    // Phase 2: reduce partials → final bf16 output
    let result = unsafe {
        ffi::attention_decode_reduce(
            partial_out_ptr as *mut f32,
            partial_m_ptr as *mut f32,
            partial_l_ptr as *mut f32,
            out_ptr as *mut ffi::Half,
            num_qheads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}
