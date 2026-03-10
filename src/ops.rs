//! GPU operations on device tensors.

use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::*;

/// Matrix-vector multiplication: y = A @ x
/// A: (M, K) row-major, x: (K,), y: (M,)
pub fn gemv(ctx: &DeviceContext, a: &DeviceMatrix, x: &DeviceVec, y: &mut DeviceVec) -> Result<()> {
    assert_eq!(a.cols, x.len, "A cols {} != x len {}", a.cols, x.len);
    assert_eq!(a.rows, y.len, "A rows {} != y len {}", a.rows, y.len);

    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gemv_cuda(
            a_ptr as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            a.rows as i32,
            a.cols as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Linear layer: y = weight @ x
pub fn linear(ctx: &DeviceContext, x: &DeviceVec, weight: &DeviceMatrix) -> Result<DeviceVec> {
    let mut y = DeviceVec::zeros(ctx, weight.rows)?;
    gemv(ctx, weight, x, &mut y)?;
    Ok(y)
}

/// RMSNorm into pre-allocated output buffer
pub fn rms_norm_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::rms_norm_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// RMSNorm (allocating)
pub fn rms_norm(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, x.len)?;
    rms_norm_into(ctx, x, weight, eps, &mut out)?;
    Ok(out)
}

/// Fully fused MLP into pre-allocated output buffer
pub fn fused_mlp_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
    act: &mut DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(gate_proj.cols, x.len, "gate_proj cols != x len");
    assert_eq!(up_proj.cols, x.len, "up_proj cols != x len");
    assert_eq!(
        gate_proj.rows, up_proj.rows,
        "gate and up must have same output dim"
    );
    assert_eq!(
        down_proj.cols, gate_proj.rows,
        "down_proj cols != intermediate_size"
    );
    assert_eq!(down_proj.rows, out.len, "down_proj rows != out len");
    assert_eq!(act.len, gate_proj.rows, "act len != intermediate_size");

    let hidden_size = x.len;
    let intermediate_size = gate_proj.rows;

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (gate_ptr, _gg) = gate_proj.data.device_ptr(&ctx.stream);
    let (up_ptr, _gu) = up_proj.data.device_ptr(&ctx.stream);
    let (down_ptr, _gd) = down_proj.data.device_ptr(&ctx.stream);
    let (act_ptr, _ga) = act.data.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_mlp_cuda(
            x_ptr as *const ffi::Half,
            gate_ptr as *const ffi::Half,
            up_ptr as *const ffi::Half,
            down_ptr as *const ffi::Half,
            act_ptr as *mut ffi::Half,
            out_ptr as *mut ffi::Half,
            hidden_size as i32,
            intermediate_size as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Fully fused MLP (allocating)
pub fn fused_mlp(
    ctx: &DeviceContext,
    x: &DeviceVec,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
) -> Result<DeviceVec> {
    let mut act = DeviceVec::zeros(ctx, gate_proj.rows)?;
    let mut out = DeviceVec::zeros(ctx, down_proj.rows)?;
    fused_mlp_into(ctx, x, gate_proj, up_proj, down_proj, &mut act, &mut out)?;
    Ok(out)
}

/// RoPE
pub fn rope(
    ctx: &DeviceContext,
    x: &DeviceVec,
    cos: &DeviceVec,
    sin: &DeviceVec,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, x.len)?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gc) = cos.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gs) = sin.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::rope_cuda(
                x_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(out)
}

/// Element-wise add in-place: a += b
pub fn add_inplace(ctx: &DeviceContext, a: &mut DeviceVec, b: &DeviceVec) -> Result<()> {
    assert_eq!(a.len, b.len);

    let (a_ptr, _ga) = a.data.device_ptr_mut(&ctx.stream);
    let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::add_cuda(
            a_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *mut ffi::Half,
            a.len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Fused add + RMSNorm: hidden += residual; out = rms_norm(hidden, weight)
/// Saves one global read of hidden compared to separate add + rms_norm.
pub fn fused_add_rms_norm_into(
    ctx: &DeviceContext,
    hidden: &mut DeviceVec,
    residual: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(hidden.len, residual.len);
    assert_eq!(hidden.len, out.len);

    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_add_rms_norm_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.len as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Element-wise add (allocating)
pub fn add(ctx: &DeviceContext, a: &DeviceVec, b: &DeviceVec) -> Result<DeviceVec> {
    assert_eq!(a.len, b.len);
    let mut out = DeviceVec::zeros(ctx, a.len)?;

    {
        let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                a.len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(out)
}

/// Embedding lookup into pre-allocated output buffer
pub fn embedding_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: u32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(embed.cols, out.len);

    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_cuda(
            embed_ptr as *const ffi::Half,
            token_id as i32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Embedding lookup (allocating)
pub fn embedding(ctx: &DeviceContext, embed: &DeviceMatrix, token_id: u32) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, embed.cols)?;
    embedding_into(ctx, embed, token_id, &mut out)?;
    Ok(out)
}

/// Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
pub fn embedding_decode_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    decode_meta: &cudarc::driver::CudaSlice<i32>,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(embed.cols, out.len);

    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (meta_ptr, _gm) = decode_meta.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_decode_cuda(
            embed_ptr as *const ffi::Half,
            meta_ptr as *const i32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Argmax — returns the index of the maximum element
pub fn argmax(ctx: &DeviceContext, x: &DeviceVec) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out_gpu.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                x.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    // Sync before reading result from GPU to CPU
    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;

    Ok(result[0] as u32)
}

/// GPU sampling: temperature → softmax → top-k → top-p → multinomial.
/// Runs entirely on GPU. Only the final token ID (4 bytes) is transferred D2H.
pub fn gpu_sample(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (p_ptr, _gp) = probs_scratch.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = out_gpu.device_ptr_mut(&ctx.stream);

        let inv_temperature = if params.temperature > 0.0 {
            1.0 / params.temperature
        } else {
            1.0
        };

        unsafe {
            ffi::gpu_sample_cuda(
                l_ptr as *const ffi::Half,
                p_ptr as *mut f32,
                o_ptr as *mut i32,
                logits.len as i32,
                inv_temperature,
                params.top_k,
                params.top_p,
                random_val,
                ctx.stream.cu_stream(),
            );
        }
    }

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;

    Ok(result[0] as u32)
}

/// Attention scores: scores[i] = q @ k_cache[i] * scale
pub fn attention_scores(
    ctx: &DeviceContext,
    q: &DeviceVec,
    k_cache: &DeviceVec, // Flattened (seq_len * head_dim)
    seq_len: usize,
    head_dim: usize,
    scale: f32,
) -> Result<DeviceVec> {
    let mut scores = DeviceVec::zeros(ctx, seq_len)?;

    {
        let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
        let (k_ptr, _gk) = k_cache.data.device_ptr(&ctx.stream);
        let (scores_ptr, _gs) = scores.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::attention_scores_cuda(
                q_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                scores_ptr as *mut ffi::Half,
                seq_len as i32,
                head_dim as i32,
                scale,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(scores)
}

/// Attention weighted sum: out = sum(weights[i] * v_cache[i])
pub fn attention_weighted_sum(
    ctx: &DeviceContext,
    weights: &DeviceVec, // (seq_len,)
    v_cache: &DeviceVec, // Flattened (seq_len * head_dim)
    seq_len: usize,
    head_dim: usize,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, head_dim)?;

    {
        let (w_ptr, _gw) = weights.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_cache.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::attention_weighted_sum_cuda(
                w_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                seq_len as i32,
                head_dim as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(out)
}

/// Fused GQA Attention (single token generation)
///
/// 融合所有 attention 操作：
/// - Q/K heads: slice → norm → RoPE
/// - V heads: slice
/// - Attention: scores → softmax → weighted sum
/// - Concat outputs
///
/// 参数:
/// - q_full: Q projection 输出 (num_qheads * head_dim,)
/// - k_full: K projection 输出 (num_kvheads * head_dim,)
/// - v_full: V projection 输出 (num_kvheads * head_dim,)
/// - q_norm_weight, k_norm_weight: RMSNorm weights (head_dim,)
/// - cos_cache, sin_cache: RoPE 缓存 for current position (head_dim/2,)
/// - k_cache, v_cache: KV 历史缓存 (num_kvheads * max_seq * head_dim,)
/// - current_pos: 当前位置
/// - seq_len: 当前序列长度（包含当前 token）
/// - scale: attention scale (1/sqrt(head_dim))
/// - rms_eps: RMSNorm epsilon
///
/// 返回: attention 输出 (num_qheads * head_dim,)
/// Fused GQA Attention into pre-allocated output buffer
/// Fused GQA Attention into pre-allocated output buffer.
/// cos_pos/sin_pos: RoPE values for the current position (head_dim elements).
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_into(
    ctx: &DeviceContext,
    q_full: &DeviceVec,
    k_full: &DeviceVec,
    v_full: &DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_pos: &DeviceVecView<'_>,
    sin_pos: &DeviceVecView<'_>,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut DeviceVec,
    num_qheads: usize,
    num_kvheads: usize,
    head_dim: usize,
    current_pos: usize,
    seq_len: usize,
    scale: f32,
    rms_eps: f32,
) -> Result<()> {
    let (q_ptr, _gq) = q_full.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_full.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_full.data.device_ptr(&ctx.stream);
    let (q_norm_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (k_norm_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gcos) = cos_pos.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gsin) = sin_pos.data.device_ptr(&ctx.stream);
    let (k_cache_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
    let (v_cache_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_gqa_attention_single_token(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            k_cache_ptr as *mut ffi::Half,
            v_cache_ptr as *mut ffi::Half,
            out_ptr as *mut ffi::Half,
            num_qheads as i32,
            num_kvheads as i32,
            (num_qheads / num_kvheads) as i32,
            head_dim as i32,
            current_pos as i32,
            seq_len as i32,
            scale,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Batched prefill attention: replaces per-token attention loop with cuBLAS GEMM.
/// Modifies q_batch and k_batch in-place (QK norm + RoPE).
#[allow(clippy::too_many_arguments)]
/// Scratch buffers for prefill attention (reused across layers to avoid per-layer allocation).
pub struct PrefillAttnScratch {
    pub scores_fp32: CudaSlice<f32>,
    pub softmax_bf16: CudaSlice<half::bf16>,
    pub capacity: usize, // number of elements allocated
}

impl PrefillAttnScratch {
    pub fn new(
        ctx: &DeviceContext,
        num_q_heads: usize,
        total_seq: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let capacity = num_q_heads * total_seq * seq_len;
        let scores_fp32: CudaSlice<f32> = unsafe {
            ctx.stream
                .alloc::<f32>(capacity)
                .map_err(|e| anyhow!("scores alloc failed: {}", e))?
        };
        let softmax_bf16: CudaSlice<half::bf16> = unsafe {
            ctx.stream
                .alloc::<half::bf16>(capacity)
                .map_err(|e| anyhow!("softmax alloc failed: {}", e))?
        };
        Ok(Self {
            scores_fp32,
            softmax_bf16,
            capacity,
        })
    }
}

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
    scratch: &mut PrefillAttnScratch,
    output: &mut HiddenStates,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
    scale: f32,
    rms_eps: f32,
) -> Result<()> {
    let seq_len = q_batch.seq_len;

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
        let (sc_ptr, _gsc) = scratch.scores_fp32.device_ptr_mut(&ctx.stream);
        let (sm_ptr, _gsm) = scratch.softmax_bf16.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::prefill_attention_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                kc_ptr as *mut ffi::Half,
                vc_ptr as *mut ffi::Half,
                o_ptr as *mut ffi::Half,
                sc_ptr as *mut f32,
                sm_ptr as *mut ffi::Half,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                seq_len as i32,
                start_pos as i32,
                scale,
                rms_eps,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(())
}

/// Fused GQA Attention (allocating)
#[allow(clippy::too_many_arguments)]
pub fn fused_attention(
    ctx: &DeviceContext,
    q_full: &DeviceVec,
    k_full: &DeviceVec,
    v_full: &DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_pos: &DeviceVecView<'_>,
    sin_pos: &DeviceVecView<'_>,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    num_qheads: usize,
    num_kvheads: usize,
    head_dim: usize,
    current_pos: usize,
    seq_len: usize,
    scale: f32,
    rms_eps: f32,
) -> Result<DeviceVec> {
    let mut output = DeviceVec::zeros(ctx, num_qheads * head_dim)?;
    fused_attention_into(
        ctx,
        q_full,
        k_full,
        v_full,
        q_norm_weight,
        k_norm_weight,
        cos_pos,
        sin_pos,
        k_cache,
        v_cache,
        &mut output,
        num_qheads,
        num_kvheads,
        head_dim,
        current_pos,
        seq_len,
        scale,
        rms_eps,
    )?;
    Ok(output)
}

/// Fused GQA Attention for decode — reads pos/seq_len from decode_meta (CUDA Graph safe).
/// cos_cache_base/sin_cache_base: full RoPE buffers [max_seq_len * head_dim].
/// decode_meta: [token_id, current_pos, seq_len] on GPU.
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
    decode_meta: &cudarc::driver::CudaSlice<i32>,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut DeviceVec,
    num_qheads: usize,
    num_kvheads: usize,
    head_dim: usize,
    scale: f32,
    rms_eps: f32,
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
            out_ptr as *mut ffi::Half,
            num_qheads as i32,
            num_kvheads as i32,
            (num_qheads / num_kvheads) as i32,
            head_dim as i32,
            scale,
            rms_eps,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

// ============================================================
// Batched ops for prefill (multi-token processing)
// ============================================================

/// GEMM: Y = weight @ X (batched linear projection)
/// weight: [out_dim, in_dim] row-major, X: HiddenStates [in_dim, seq_len], Y: HiddenStates [out_dim, seq_len]
pub fn gemm(ctx: &DeviceContext, weight: &DeviceMatrix, x: &HiddenStates) -> Result<HiddenStates> {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    let out_dim = weight.rows;
    let seq_len = x.seq_len;
    let mut out = HiddenStates::zeros(ctx, out_dim, seq_len)?;

    {
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::gemm_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                out_dim as i32,
                seq_len as i32,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(out)
}

/// Batched RMSNorm: normalize each token's hidden state independently
pub fn rms_norm_batch(
    ctx: &DeviceContext,
    x: &HiddenStates,
    weight: &DeviceVec,
    eps: f32,
) -> Result<HiddenStates> {
    assert_eq!(weight.len, x.hidden_dim);
    let mut out = HiddenStates::zeros(ctx, x.hidden_dim, x.seq_len)?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::rms_norm_batched_cuda(
                x_ptr as *const ffi::Half,
                w_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                x.hidden_dim as i32,
                x.seq_len as i32,
                eps,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(out)
}

/// Batched element-wise add: out = a + b (same shape HiddenStates)
pub fn add_batch(ctx: &DeviceContext, a: &HiddenStates, b: &HiddenStates) -> Result<HiddenStates> {
    assert_eq!(a.hidden_dim, b.hidden_dim);
    assert_eq!(a.seq_len, b.seq_len);
    let n = a.hidden_dim * a.seq_len;
    let mut out = HiddenStates::zeros(ctx, a.hidden_dim, a.seq_len)?;

    {
        let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                n as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(out)
}

/// Batched SiLU+mul: out[i] = silu(gate[i]) * up[i]
pub fn silu_mul_batch(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
) -> Result<HiddenStates> {
    assert_eq!(gate.hidden_dim, up.hidden_dim);
    assert_eq!(gate.seq_len, up.seq_len);
    let n = gate.hidden_dim * gate.seq_len;
    let mut out = HiddenStates::zeros(ctx, gate.hidden_dim, gate.seq_len)?;

    {
        let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
        let (u_ptr, _gu) = up.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        let result = unsafe {
            ffi::silu_mul_triton_aot_cuda(
                g_ptr as *const ffi::Half,
                u_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                n as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
    }

    Ok(out)
}

/// Batched embedding lookup
pub fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<i32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_batched_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Extract a single token's vector from a HiddenStates batch (GPU copy)
pub fn extract_vec(
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
) -> Result<DeviceVec> {
    let offset = token_idx * batch.hidden_dim;
    let len = batch.hidden_dim;
    let mut out = DeviceVec::zeros(ctx, len)?;

    let src_view = batch.data.slice(offset..offset + len);
    ctx.stream
        .memcpy_dtod(&src_view, &mut out.data)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;

    Ok(out)
}

/// Write a single token's vector into a HiddenStates batch (GPU copy)
pub fn write_vec(
    ctx: &DeviceContext,
    batch: &mut HiddenStates,
    token_idx: usize,
    vec: &DeviceVec,
) -> Result<()> {
    let offset = token_idx * batch.hidden_dim;
    let mut dst_view = batch.data.slice_mut(offset..offset + vec.len);

    ctx.stream
        .memcpy_dtod(&vec.data, &mut dst_view)
        .map_err(|e| anyhow!("Device copy failed: {}", e))?;

    Ok(())
}

// ============================================================
// Qwen3.5 ops
// ============================================================

/// (1+weight) RMSNorm into pre-allocated output buffer (Gemma/Qwen3.5 style)
pub fn rms_norm_offset_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(x.len, out.len);

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::rms_norm_offset_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            x.len as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Fused add + (1+weight) RMSNorm: hidden += residual; out = rms_norm_offset(hidden, weight)
pub fn fused_add_rms_norm_offset_into(
    ctx: &DeviceContext,
    hidden: &mut DeviceVec,
    residual: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(hidden.len, residual.len);
    assert_eq!(hidden.len, out.len);

    let (h_ptr, _gh) = hidden.data.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = residual.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_add_rms_norm_offset_cuda(
            h_ptr as *mut ffi::Half,
            r_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            hidden.len as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Per-head RMSNorm with F32 weight + SiLU gate multiplication.
/// x: [num_heads * head_dim], weight: [head_dim] f32, gate: [num_heads * head_dim]
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gated_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &CudaSlice<f32>,
    gate: &DeviceVec,
    out: &mut DeviceVec,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<()> {
    assert_eq!(x.len, num_heads * head_dim);
    assert_eq!(gate.len, x.len);
    assert_eq!(out.len, x.len);

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = gate.data.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::rms_norm_gated_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const f32,
            g_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            num_heads as i32,
            head_dim as i32,
            eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Causal depthwise conv1d decode (single step).
/// Updates conv_state in-place. Applies SiLU activation.
pub fn conv1d_decode_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    conv_weight: &DeviceVec,
    conv_state: &mut DeviceVec,
    out: &mut DeviceVec,
    kernel_size: usize,
) -> Result<()> {
    let num_channels = x.len;
    assert_eq!(out.len, num_channels);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert_eq!(conv_state.len, num_channels * (kernel_size - 1));

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = conv_state.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_decode_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Gated delta rule recurrent decode (single step).
/// qkv: [q_dim + k_dim + v_dim] after conv1d+SiLU
/// b_proj, a_proj: [num_value_heads] bf16 (from gemv)
/// state: [num_value_heads * key_dim * val_dim] f32 (updated in-place)
/// output: [num_value_heads * val_dim] bf16
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_decode_into(
    ctx: &DeviceContext,
    qkv: &DeviceVec,
    b_proj: &DeviceVec,
    a_proj: &DeviceVec,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    output: &mut DeviceVec,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Result<()> {
    let (qkv_ptr, _gq) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gated_delta_rule_decode_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            s_ptr as *mut f32,
            o_ptr as *mut ffi::Half,
            num_key_heads as i32,
            num_value_heads as i32,
            key_dim as i32,
            val_dim as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Fused GQA Attention HD256 — decode variant (reads pos/seq_len from decode_meta).
/// q_full: interleaved [head0_q(256), head0_gate(256), head1_q(256), ...] = [num_qheads * 2 * 256]
/// Output already gated: output *= sigmoid(gate).
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_hd256_decode_into(
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
    num_qheads: usize,
    num_kvheads: usize,
    rotary_dim: usize,
    scale: f32,
    rms_eps: f32,
) -> Result<()> {
    let (q_ptr, _gq) = q_full.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_full.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_full.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_cache_base.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_cache_base.data.device_ptr(&ctx.stream);
    let (meta_ptr, _gm) = decode_meta.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_gqa_attention_hd256_decode(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            meta_ptr as *const i32,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_qheads as i32,
            num_kvheads as i32,
            (num_qheads / num_kvheads) as i32,
            rotary_dim as i32,
            scale,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Fused GQA Attention HD256 — single token variant (scalar pos/seq_len, for prefill).
#[allow(clippy::too_many_arguments)]
pub fn fused_attention_hd256_single_token_into(
    ctx: &DeviceContext,
    q_full: &DeviceVec,
    k_full: &DeviceVec,
    v_full: &DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_pos: &DeviceVecView<'_>,
    sin_pos: &DeviceVecView<'_>,
    k_cache: &mut DeviceVec,
    v_cache: &mut DeviceVec,
    output: &mut DeviceVec,
    num_qheads: usize,
    num_kvheads: usize,
    current_pos: usize,
    seq_len: usize,
    rotary_dim: usize,
    scale: f32,
    rms_eps: f32,
) -> Result<()> {
    let (q_ptr, _gq) = q_full.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k_full.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v_full.data.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
    let (cos_ptr, _gc) = cos_pos.data.device_ptr(&ctx.stream);
    let (sin_ptr, _gs) = sin_pos.data.device_ptr(&ctx.stream);
    let (kc_ptr, _gkc) = k_cache.data.device_ptr_mut(&ctx.stream);
    let (vc_ptr, _gvc) = v_cache.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::fused_gqa_attention_hd256_single_token(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            kc_ptr as *mut ffi::Half,
            vc_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_qheads as i32,
            num_kvheads as i32,
            (num_qheads / num_kvheads) as i32,
            current_pos as i32,
            seq_len as i32,
            rotary_dim as i32,
            scale,
            rms_eps,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// Causal depthwise conv1d prefill (parallel over sequence).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_prefill_into(
    ctx: &DeviceContext,
    x_seq: &DeviceVec,
    conv_weight: &DeviceVec,
    conv_state: &mut DeviceVec,
    out_seq: &mut DeviceVec,
    num_channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Result<()> {
    assert_eq!(x_seq.len, num_channels * seq_len);
    assert_eq!(out_seq.len, num_channels * seq_len);

    let (x_ptr, _gx) = x_seq.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = conv_weight.data.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = conv_state.data.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out_seq.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::conv1d_prefill_cuda(
            x_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            s_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            num_channels as i32,
            seq_len as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn bf16_vec(data: &[f32]) -> Vec<bf16> {
        data.iter().map(|&x| bf16::from_f32(x)).collect()
    }

    fn rms_norm_reference(x: &[bf16], weight: &[bf16], eps: f32, offset: bool) -> Vec<f32> {
        let sum_sq: f32 = x
            .iter()
            .map(|value| {
                let v = value.to_f32();
                v * v
            })
            .sum();
        let inv_rms = 1.0 / ((sum_sq / x.len() as f32) + eps).sqrt();

        x.iter()
            .zip(weight.iter())
            .map(|(value, weight)| {
                let normed = bf16::from_f32(value.to_f32() * inv_rms).to_f32();
                let scale = if offset {
                    1.0 + weight.to_f32()
                } else {
                    weight.to_f32()
                };
                bf16::from_f32(normed * scale).to_f32()
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tol,
                "index {} expected {} got {} (tol {})",
                idx,
                expected,
                actual,
                tol
            );
        }
    }

    #[test]
    fn test_gemv() -> Result<()> {
        let ctx = DeviceContext::new()?;

        // A = [[1, 2, 3], [4, 5, 6]] (2x3) row-major
        // x = [1, 2, 3]
        // y = A @ x = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
        let a_data = bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x_data = bf16_vec(&[1.0, 2.0, 3.0]);

        let a = DeviceMatrix::from_host(&ctx, &a_data, 2, 3)?;
        let x = DeviceVec::from_host(&ctx, &x_data)?;
        let y = linear(&ctx, &x, &a)?;

        let result = y.to_host(&ctx)?;
        assert!(
            (result[0] - 14.0).abs() < 0.1,
            "Expected 14, got {}",
            result[0]
        );
        assert!(
            (result[1] - 32.0).abs() < 0.1,
            "Expected 32, got {}",
            result[1]
        );

        Ok(())
    }

    #[test]
    fn test_argmax() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let x = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 9.0, 3.0, 8.0]))?;
        let token = argmax(&ctx, &x)?;
        assert_eq!(token, 1, "Expected argmax index 1, got {}", token);
        Ok(())
    }

    #[test]
    fn test_argmax_tie_matches_legacy_reduction_order() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let mut host = vec![bf16::from_f32(-1.0); 300];
        host[2] = bf16::from_f32(10.0);
        host[257] = bf16::from_f32(10.0);
        let x = DeviceVec::from_host(&ctx, &host)?;
        let token = argmax(&ctx, &x)?;
        assert_eq!(
            token, 2,
            "Expected legacy reduction-order winner 2, got {}",
            token
        );
        Ok(())
    }

    #[test]
    fn test_rms_norm() -> Result<()> {
        let ctx = DeviceContext::new()?;

        let x_host = bf16_vec(&[1.0, 2.0, 3.0, 4.0]);
        let w_host = bf16_vec(&[1.0, 1.0, 1.0, 1.0]);
        let x = DeviceVec::from_host(&ctx, &x_host)?;
        let w = DeviceVec::from_host(&ctx, &w_host)?;
        let out = rms_norm(&ctx, &x, &w, 1e-6)?;

        let result = out.to_host(&ctx)?;
        let expected = rms_norm_reference(&x_host, &w_host, 1e-6, false);
        assert_close(&result, &expected, 0.01);

        Ok(())
    }

    #[test]
    fn test_rms_norm_batch_multi_tile() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let hidden_dim = 260;
        let seq_len = 2;

        let x_host_f32: Vec<f32> = (0..hidden_dim * seq_len)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.25)
            .collect();
        let w_host_f32: Vec<f32> = (0..hidden_dim)
            .map(|idx| 0.5 + (idx % 11) as f32 * 0.0625)
            .collect();
        let x_host = bf16_vec(&x_host_f32);
        let w_host = bf16_vec(&w_host_f32);

        let x = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&x_host)
                .map_err(|e| anyhow!("H2D copy failed: {}", e))?,
            hidden_dim,
            seq_len,
        };
        let weight = DeviceVec::from_host(&ctx, &w_host)?;
        let out = rms_norm_batch(&ctx, &x, &weight, 1e-6)?;

        let result = ctx
            .stream
            .clone_dtoh(&out.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        let result: Vec<f32> = result.iter().map(|value| value.to_f32()).collect();

        let mut expected = Vec::with_capacity(hidden_dim * seq_len);
        for row in 0..seq_len {
            let start = row * hidden_dim;
            expected.extend(rms_norm_reference(
                &x_host[start..start + hidden_dim],
                &w_host,
                1e-6,
                false,
            ));
        }
        assert_close(&result, &expected, 0.02);

        Ok(())
    }

    #[test]
    fn test_rms_norm_offset() -> Result<()> {
        let ctx = DeviceContext::new()?;

        let x_host = bf16_vec(&[-2.0, -0.5, 0.25, 1.5, 3.0, 0.75, -1.25]);
        let w_host = bf16_vec(&[0.0, 0.5, -0.25, 0.125, 1.0, -0.5, 0.25]);
        let x = DeviceVec::from_host(&ctx, &x_host)?;
        let w = DeviceVec::from_host(&ctx, &w_host)?;
        let mut out = DeviceVec::zeros(&ctx, x_host.len())?;
        rms_norm_offset_into(&ctx, &x, &w, 1e-6, &mut out)?;

        let result = out.to_host(&ctx)?;
        let expected = rms_norm_reference(&x_host, &w_host, 1e-6, true);
        assert_close(&result, &expected, 0.02);

        Ok(())
    }

    #[test]
    fn test_rope() -> Result<()> {
        let ctx = DeviceContext::new()?;

        // Simple test: x = [1, 0, 1, 0], cos = [1, 1, 1, 1], sin = [0, 0, 0, 0]
        // With sin=0, cos=1, output should equal input
        let x = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 0.0, 1.0, 0.0]))?;
        let cos = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 1.0, 1.0, 1.0]))?;
        let sin = DeviceVec::from_host(&ctx, &bf16_vec(&[0.0, 0.0, 0.0, 0.0]))?;
        let out = rope(&ctx, &x, &cos, &sin)?;

        let result = out.to_host(&ctx)?;

        assert!(
            (result[0] - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.0).abs() < 0.01,
            "Expected 0.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.0).abs() < 0.01,
            "Expected 0.0, got {}",
            result[3]
        );

        Ok(())
    }

    #[test]
    fn test_add_and_add_inplace() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let a = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]))?;
        let b = DeviceVec::from_host(&ctx, &bf16_vec(&[0.5, -1.0, 2.5, 3.0]))?;

        let out = add(&ctx, &a, &b)?;
        let result = out.to_host(&ctx)?;
        assert!(
            (result[0] - 1.5).abs() < 0.01,
            "Expected 1.5, got {}",
            result[0]
        );
        assert!(
            (result[1] - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - 5.5).abs() < 0.01,
            "Expected 5.5, got {}",
            result[2]
        );
        assert!(
            (result[3] - 7.0).abs() < 0.01,
            "Expected 7.0, got {}",
            result[3]
        );

        let mut inplace = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]))?;
        add_inplace(&ctx, &mut inplace, &b)?;
        let inplace_result = inplace.to_host(&ctx)?;
        assert!(
            (inplace_result[0] - 1.5).abs() < 0.01,
            "Expected 1.5, got {}",
            inplace_result[0]
        );
        assert!(
            (inplace_result[1] - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            inplace_result[1]
        );
        assert!(
            (inplace_result[2] - 5.5).abs() < 0.01,
            "Expected 5.5, got {}",
            inplace_result[2]
        );
        assert!(
            (inplace_result[3] - 7.0).abs() < 0.01,
            "Expected 7.0, got {}",
            inplace_result[3]
        );

        Ok(())
    }

    #[test]
    fn test_embedding_variants() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let embed = DeviceMatrix::from_host(
            &ctx,
            &bf16_vec(&[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
            3,
            4,
        )?;

        let single = embedding(&ctx, &embed, 2)?;
        let single_host = single.to_host(&ctx)?;
        assert!(
            (single_host[0] - 9.0).abs() < 0.01,
            "Expected 9.0, got {}",
            single_host[0]
        );
        assert!(
            (single_host[3] - 12.0).abs() < 0.01,
            "Expected 12.0, got {}",
            single_host[3]
        );

        let decode_meta = ctx.stream.clone_htod(&[1_i32])?;
        let mut decode_out = DeviceVec::zeros(&ctx, 4)?;
        embedding_decode_into(&ctx, &embed, &decode_meta, &mut decode_out)?;
        let decode_host = decode_out.to_host(&ctx)?;
        assert!(
            (decode_host[0] - 5.0).abs() < 0.01,
            "Expected 5.0, got {}",
            decode_host[0]
        );
        assert!(
            (decode_host[3] - 8.0).abs() < 0.01,
            "Expected 8.0, got {}",
            decode_host[3]
        );

        let token_ids = ctx.stream.clone_htod(&[2_i32, 0_i32])?;
        let mut batch_out = HiddenStates::zeros(&ctx, 4, 2)?;
        embedding_batch(&ctx, &embed, &token_ids, &mut batch_out)?;
        let batch_host = ctx.stream.clone_dtoh(&batch_out.data)?;
        ctx.sync()?;
        assert!(
            (batch_host[0].to_f32() - 9.0).abs() < 0.01,
            "Expected 9.0, got {}",
            batch_host[0]
        );
        assert!(
            (batch_host[3].to_f32() - 12.0).abs() < 0.01,
            "Expected 12.0, got {}",
            batch_host[3]
        );
        assert!(
            (batch_host[4].to_f32() - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            batch_host[4]
        );
        assert!(
            (batch_host[7].to_f32() - 4.0).abs() < 0.01,
            "Expected 4.0, got {}",
            batch_host[7]
        );

        Ok(())
    }

    #[test]
    fn test_extract_write_vec_roundtrip() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let batch = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))?,
            hidden_dim: 3,
            seq_len: 2,
        };

        let extracted = extract_vec(&ctx, &batch, 1)?;
        let extracted_host = extracted.to_host(&ctx)?;
        assert!(
            (extracted_host[0] - 4.0).abs() < 0.01,
            "Expected 4.0, got {}",
            extracted_host[0]
        );
        assert!(
            (extracted_host[2] - 6.0).abs() < 0.01,
            "Expected 6.0, got {}",
            extracted_host[2]
        );

        let replacement = DeviceVec::from_host(&ctx, &bf16_vec(&[7.0, 8.0, 9.0]))?;
        let mut batch = batch;
        write_vec(&ctx, &mut batch, 0, &replacement)?;
        let batch_host = ctx.stream.clone_dtoh(&batch.data)?;
        ctx.sync()?;
        assert!(
            (batch_host[0].to_f32() - 7.0).abs() < 0.01,
            "Expected 7.0, got {}",
            batch_host[0]
        );
        assert!(
            (batch_host[2].to_f32() - 9.0).abs() < 0.01,
            "Expected 9.0, got {}",
            batch_host[2]
        );
        assert!(
            (batch_host[3].to_f32() - 4.0).abs() < 0.01,
            "Expected 4.0, got {}",
            batch_host[3]
        );
        assert!(
            (batch_host[5].to_f32() - 6.0).abs() < 0.01,
            "Expected 6.0, got {}",
            batch_host[5]
        );

        Ok(())
    }

    #[test]
    fn test_attention_scores() -> Result<()> {
        let ctx = DeviceContext::new()?;

        // q = [1, 0], k_cache = [[1, 0], [0, 1]] (2 positions, head_dim=2)
        // scores = [q @ k[0], q @ k[1]] = [1*1+0*0, 1*0+0*1] = [1, 0]
        // with scale=1
        let q = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 0.0]))?;
        let k_cache = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 0.0, 0.0, 1.0]))?; // flattened

        let scores = attention_scores(&ctx, &q, &k_cache, 2, 2, 1.0)?;

        let result = scores.to_host(&ctx)?;

        assert!(
            (result[0] - 1.0).abs() < 0.01,
            "Expected 1.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.0).abs() < 0.01,
            "Expected 0.0, got {}",
            result[1]
        );

        Ok(())
    }

    #[test]
    fn test_attention_weighted_sum() -> Result<()> {
        let ctx = DeviceContext::new()?;

        // weights = [0.5, 0.5], v_cache = [[1, 2], [3, 4]]
        // out = 0.5 * [1, 2] + 0.5 * [3, 4] = [2, 3]
        let weights = DeviceVec::from_host(&ctx, &bf16_vec(&[0.5, 0.5]))?;
        let v_cache = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]))?;

        let out = attention_weighted_sum(&ctx, &weights, &v_cache, 2, 2)?;

        let result = out.to_host(&ctx)?;

        assert!(
            (result[0] - 2.0).abs() < 0.01,
            "Expected 2.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 3.0).abs() < 0.01,
            "Expected 3.0, got {}",
            result[1]
        );

        Ok(())
    }

    #[test]
    fn test_gpu_sample() -> Result<()> {
        use cudarc::driver::CudaSlice;
        let ctx = DeviceContext::new()?;

        // Create logits with a clear winner at index 2 (highest logit)
        // but with temperature sampling, other tokens have a chance
        let logits_data = bf16_vec(&[1.0, 2.0, 10.0, 1.5, 0.5]);
        let logits = DeviceVec::from_host(&ctx, &logits_data)?;
        let mut probs: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(5)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;

        // Test 1: With very low temperature (near-greedy), should pick token 2
        let params = crate::sampler::SamplingParams {
            temperature: 0.01,
            top_k: -1,
            top_p: 1.0,
        };
        let token = gpu_sample(&ctx, &logits, &mut probs, &params, 0.5)?;
        assert_eq!(token, 2, "near-greedy should pick index 2 (highest logit)");

        // Test 2: With high temperature, random_val=0.0 should pick first nonzero token
        let params = crate::sampler::SamplingParams {
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
        };
        let token = gpu_sample(&ctx, &logits, &mut probs, &params, 0.0)?;
        // random_val=0.0 should pick the first token (index 0)
        assert_eq!(token, 0, "random_val=0.0 should pick first token");

        // Test 3: top_k=1 should always pick the highest
        let params = crate::sampler::SamplingParams {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let token = gpu_sample(&ctx, &logits, &mut probs, &params, 0.5)?;
        assert_eq!(token, 2, "top_k=1 should pick highest probability token");

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_triton_decode_attention_matches_cpu_reference() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let num_qheads = 8;
        let num_kvheads = 2;
        let head_dim = 128;
        let max_seq_len = 64;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let eps = 1e-6_f32;

        let q_host: Vec<f32> = (0..num_qheads * head_dim)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.125)
            .collect();
        let k_host: Vec<f32> = (0..num_kvheads * head_dim)
            .map(|i| ((i % 19) as f32 - 9.0) * 0.15)
            .collect();
        let v_host: Vec<f32> = (0..num_kvheads * head_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.2)
            .collect();
        let q_weight_host: Vec<f32> = (0..head_dim)
            .map(|i| 0.9 + (i % 13) as f32 * 0.03)
            .collect();
        let k_weight_host: Vec<f32> = (0..head_dim)
            .map(|i| 0.8 + (i % 11) as f32 * 0.025)
            .collect();

        let half = head_dim / 2;
        let theta = 1_000_000.0_f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / head_dim as f32))
            .collect();
        let mut cos_host = vec![bf16::ZERO; max_seq_len * head_dim];
        let mut sin_host = vec![bf16::ZERO; max_seq_len * head_dim];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = pos as f32 * inv_freq[i];
                let cos = bf16::from_f32(freq.cos());
                let sin = bf16::from_f32(freq.sin());
                cos_host[pos * head_dim + i] = cos;
                cos_host[pos * head_dim + i + half] = cos;
                sin_host[pos * head_dim + i] = sin;
                sin_host[pos * head_dim + i + half] = sin;
            }
        }

        for seq_len in [1_usize, 6_usize] {
            let current_pos = seq_len - 1;
            let cache_len = num_kvheads * 4096 * head_dim;
            let mut k_cache_host = vec![bf16::ZERO; cache_len];
            let mut v_cache_host = vec![bf16::ZERO; cache_len];
            for kv_head in 0..num_kvheads {
                for pos in 0..current_pos {
                    let base = (kv_head * 4096 + pos) * head_dim;
                    for dim in 0..head_dim {
                        k_cache_host[base + dim] = bf16::from_f32(
                            ((kv_head * 31 + pos * 7 + dim) % 41) as f32 * 0.05 - 1.0,
                        );
                        v_cache_host[base + dim] = bf16::from_f32(
                            ((kv_head * 17 + pos * 5 + dim) % 37) as f32 * 0.04 - 0.7,
                        );
                    }
                }
            }

            let q = DeviceVec::from_host(&ctx, &bf16_vec(&q_host))?;
            let k = DeviceVec::from_host(&ctx, &bf16_vec(&k_host))?;
            let v = DeviceVec::from_host(&ctx, &bf16_vec(&v_host))?;
            let q_weight = DeviceVec::from_host(&ctx, &bf16_vec(&q_weight_host))?;
            let k_weight = DeviceVec::from_host(&ctx, &bf16_vec(&k_weight_host))?;
            let cos_cache = DeviceVec::from_host(&ctx, &cos_host)?;
            let sin_cache = DeviceVec::from_host(&ctx, &sin_host)?;
            let decode_meta = ctx
                .stream
                .clone_htod(&[0_i32, current_pos as i32, seq_len as i32])?;
            let mut k_cache = DeviceVec::from_host(&ctx, &k_cache_host)?;
            let mut v_cache = DeviceVec::from_host(&ctx, &v_cache_host)?;
            let mut out = DeviceVec::zeros(&ctx, num_qheads * head_dim)?;

            fused_attention_decode_into(
                &ctx,
                &q,
                &k,
                &v,
                &q_weight,
                &k_weight,
                &cos_cache,
                &sin_cache,
                &decode_meta,
                &mut k_cache,
                &mut v_cache,
                &mut out,
                num_qheads,
                num_kvheads,
                head_dim,
                scale,
                eps,
            )?;

            let out_host = out.to_host(&ctx)?;
            let got_k_cache = k_cache.to_host(&ctx)?;
            let got_v_cache = v_cache.to_host(&ctx)?;

            let q_heads: Vec<Vec<f32>> = q_host.chunks(head_dim).map(|x| x.to_vec()).collect();
            let k_heads: Vec<Vec<f32>> = k_host.chunks(head_dim).map(|x| x.to_vec()).collect();
            let v_heads: Vec<Vec<f32>> = v_host.chunks(head_dim).map(|x| x.to_vec()).collect();
            let gqa_ratio = num_qheads / num_kvheads;

            let mut ref_k_cache: Vec<f32> = k_cache_host.iter().map(|x| x.to_f32()).collect();
            let mut ref_v_cache: Vec<f32> = v_cache_host.iter().map(|x| x.to_f32()).collect();
            let mut ref_out = vec![0.0_f32; num_qheads * head_dim];

            let rms_norm = |head: &[f32], weight: &[f32]| -> Vec<f32> {
                let mean_sq = head.iter().map(|x| x * x).sum::<f32>() / head.len() as f32;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                head.iter()
                    .zip(weight.iter())
                    .map(|(x, w)| x * inv * w)
                    .collect()
            };

            let apply_rope = |head: &[f32]| -> Vec<f32> {
                let mut out = vec![0.0_f32; head_dim];
                for i in 0..half {
                    let cos = cos_host[current_pos * head_dim + i].to_f32();
                    let sin = sin_host[current_pos * head_dim + i].to_f32();
                    let lo = head[i];
                    let hi = head[i + half];
                    out[i] = lo * cos - hi * sin;
                    out[i + half] = lo * sin + hi * cos;
                }
                out
            };

            for kv_head in 0..num_kvheads {
                let k_rot = apply_rope(&rms_norm(&k_heads[kv_head], &k_weight_host));
                let base = (kv_head * 4096 + current_pos) * head_dim;
                ref_k_cache[base..base + head_dim].copy_from_slice(&k_rot);
                ref_v_cache[base..base + head_dim].copy_from_slice(&v_heads[kv_head]);
            }

            for q_head in 0..num_qheads {
                let kv_head = q_head / gqa_ratio;
                let q_rot = apply_rope(&rms_norm(&q_heads[q_head], &q_weight_host));
                let mut scores = vec![0.0_f32; seq_len];
                for (pos, score) in scores.iter_mut().enumerate() {
                    let base = (kv_head * 4096 + pos) * head_dim;
                    *score = (0..head_dim)
                        .map(|dim| ref_k_cache[base + dim] * q_rot[dim])
                        .sum::<f32>()
                        * scale;
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|x| (x - max_score).exp()).collect();
                let sum_exp = exp_scores.iter().sum::<f32>();
                let probs: Vec<f32> = exp_scores.iter().map(|x| x / sum_exp).collect();
                for dim in 0..head_dim {
                    let mut acc = 0.0_f32;
                    for (pos, prob) in probs.iter().enumerate() {
                        let base = (kv_head * 4096 + pos) * head_dim;
                        acc += prob * ref_v_cache[base + dim];
                    }
                    ref_out[q_head * head_dim + dim] = acc;
                }
            }

            let current_base = current_pos * head_dim;
            let max_k_diff = (0..num_kvheads * head_dim)
                .map(|idx| {
                    let kv_head = idx / head_dim;
                    let dim = idx % head_dim;
                    let offset = kv_head * 4096 * head_dim + current_base + dim;
                    (got_k_cache[offset] - ref_k_cache[offset]).abs()
                })
                .fold(0.0_f32, f32::max);
            let max_v_diff = (0..num_kvheads * head_dim)
                .map(|idx| {
                    let kv_head = idx / head_dim;
                    let dim = idx % head_dim;
                    let offset = kv_head * 4096 * head_dim + current_base + dim;
                    (got_v_cache[offset] - ref_v_cache[offset]).abs()
                })
                .fold(0.0_f32, f32::max);
            let max_out_diff = out_host
                .iter()
                .zip(ref_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);

            assert!(max_k_diff < 0.02, "seq_len={seq_len} k_cache diff {max_k_diff}");
            assert!(max_v_diff < 0.02, "seq_len={seq_len} v_cache diff {max_v_diff}");
            assert!(max_out_diff < 0.1, "seq_len={seq_len} output diff {max_out_diff}");
        }

        Ok(())
    }

}
