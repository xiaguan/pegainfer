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

/// Batched QKV projection: (q, k, v) = (Wq@x, Wk@x, Wv@x)
/// Supports GQA where K/V may have different dimensions than Q
pub fn linear_qkv_batched(
    ctx: &DeviceContext,
    x: &DeviceVec,
    wq: &DeviceMatrix,
    wk: &DeviceMatrix,
    wv: &DeviceMatrix,
) -> Result<(DeviceVec, DeviceVec, DeviceVec)> {
    assert_eq!(wq.cols, x.len, "Wq cols {} != x len {}", wq.cols, x.len);
    assert_eq!(wk.cols, x.len, "Wk cols {} != x len {}", wk.cols, x.len);
    assert_eq!(wv.cols, x.len, "Wv cols {} != x len {}", wv.cols, x.len);
    assert_eq!(
        wk.rows, wv.rows,
        "Wk rows {} != Wv rows {}",
        wk.rows, wv.rows
    );

    let mut q = DeviceVec::zeros(ctx, wq.rows)?;
    let mut k = DeviceVec::zeros(ctx, wk.rows)?;
    let mut v = DeviceVec::zeros(ctx, wv.rows)?;

    {
        let (wq_ptr, _gq) = wq.data.device_ptr(&ctx.stream);
        let (wk_ptr, _gk) = wk.data.device_ptr(&ctx.stream);
        let (wv_ptr, _gv) = wv.data.device_ptr(&ctx.stream);
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (q_ptr, _goq) = q.data.device_ptr_mut(&ctx.stream);
        let (k_ptr, _gok) = k.data.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gov) = v.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::gemv_batched_qkv_cuda(
                wq_ptr as *const ffi::Half,
                wk_ptr as *const ffi::Half,
                wv_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                v_ptr as *mut ffi::Half,
                wq.rows as i32, // Mq (Q output dimension)
                wk.rows as i32, // Mk (K/V output dimension)
                wq.cols as i32, // K (input dimension)
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok((q, k, v))
}

/// RMSNorm
pub fn rms_norm(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceVec,
    eps: f32,
) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, x.len)?;

    {
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
    }

    Ok(out)
}

/// Fully fused MLP: gate_proj @ x + up_proj @ x + SiLU + down_proj
pub fn fused_mlp(
    ctx: &DeviceContext,
    x: &DeviceVec,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
) -> Result<DeviceVec> {
    // Dimensions check
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

    let hidden_size = x.len;
    let intermediate_size = gate_proj.rows;
    let mut out = DeviceVec::zeros(ctx, down_proj.rows)?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (gate_ptr, _gg) = gate_proj.data.device_ptr(&ctx.stream);
        let (up_ptr, _gu) = up_proj.data.device_ptr(&ctx.stream);
        let (down_ptr, _gd) = down_proj.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::fused_mlp_cuda(
                x_ptr as *const ffi::Half,
                gate_ptr as *const ffi::Half,
                up_ptr as *const ffi::Half,
                down_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                hidden_size as i32,
                intermediate_size as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

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

/// Element-wise add
pub fn add(ctx: &DeviceContext, a: &DeviceVec, b: &DeviceVec) -> Result<DeviceVec> {
    assert_eq!(a.len, b.len);
    let mut out = DeviceVec::zeros(ctx, a.len)?;

    {
        let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
        let (b_ptr, _gb) = b.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                a.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(out)
}

/// Embedding lookup
pub fn embedding(ctx: &DeviceContext, embed: &DeviceMatrix, token_id: u32) -> Result<DeviceVec> {
    let mut out = DeviceVec::zeros(ctx, embed.cols)?;

    {
        let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::embedding_cuda(
                embed_ptr as *const ffi::Half,
                token_id as i32,
                out_ptr as *mut ffi::Half,
                embed.cols as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    Ok(out)
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
pub fn fused_attention(
    ctx: &DeviceContext,
    q_full: &DeviceVec,
    k_full: &DeviceVec,
    v_full: &DeviceVec,
    q_norm_weight: &DeviceVec,
    k_norm_weight: &DeviceVec,
    cos_cache: &DeviceVec,
    sin_cache: &DeviceVec,
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
    // 输出tensor
    let mut output = DeviceVec::zeros(ctx, num_qheads * head_dim)?;

    {
        let (q_ptr, _gq) = q_full.data.device_ptr(&ctx.stream);
        let (k_ptr, _gk) = k_full.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = v_full.data.device_ptr(&ctx.stream);
        let (q_norm_ptr, _gqn) = q_norm_weight.data.device_ptr(&ctx.stream);
        let (k_norm_ptr, _gkn) = k_norm_weight.data.device_ptr(&ctx.stream);
        let (cos_ptr, _gcos) = cos_cache.data.device_ptr(&ctx.stream);
        let (sin_ptr, _gsin) = sin_cache.data.device_ptr(&ctx.stream);
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
    }

    Ok(output)
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

        unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                n as i32,
                ctx.stream.cu_stream(),
            );
        }
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

        unsafe {
            ffi::silu_mul_cuda(
                g_ptr as *const ffi::Half,
                u_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                n as i32,
                ctx.stream.cu_stream(),
            );
        }
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

    unsafe {
        ffi::embedding_batched_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const i32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            ctx.stream.cu_stream(),
        );
    }

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

    {
        let (src_ptr, _gs) = batch.data.device_ptr(&ctx.stream);
        let (dst_ptr, _gd) = out.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::copy_cuda(
                (src_ptr as *const ffi::Half).add(offset),
                dst_ptr as *mut ffi::Half,
                len as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

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

    {
        let (src_ptr, _gs) = vec.data.device_ptr(&ctx.stream);
        let (dst_ptr, _gd) = batch.data.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::copy_cuda(
                src_ptr as *const ffi::Half,
                (dst_ptr as *mut ffi::Half).add(offset),
                vec.len as i32,
                ctx.stream.cu_stream(),
            );
        }
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

        ctx.sync()?;
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
    fn test_rms_norm() -> Result<()> {
        let ctx = DeviceContext::new()?;

        // x = [1, 2, 3, 4], weight = [1, 1, 1, 1]
        // mean(x^2) = (1+4+9+16)/4 = 7.5
        // rms = sqrt(7.5 + 1e-6) ≈ 2.7386
        // out = x / rms * weight = [0.365, 0.730, 1.095, 1.461]
        let x = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 2.0, 3.0, 4.0]))?;
        let w = DeviceVec::from_host(&ctx, &bf16_vec(&[1.0, 1.0, 1.0, 1.0]))?;
        let out = rms_norm(&ctx, &x, &w, 1e-6)?;

        ctx.sync()?;
        let result = out.to_host(&ctx)?;

        let rms = (7.5f32 + 1e-6).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 0.01);
        assert!((result[1] - 2.0 / rms).abs() < 0.01);

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

        ctx.sync()?;
        let result = out.to_host(&ctx)?;

        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 0.0).abs() < 0.01);
        assert!((result[2] - 1.0).abs() < 0.01);
        assert!((result[3] - 0.0).abs() < 0.01);

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

        ctx.sync()?;
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

        ctx.sync()?;
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
}
