use anyhow::Result;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

/// GEMV on a row sub-range of a matrix: y = A[row_offset..row_offset+M, :] @ x
/// Useful for accessing Q/K/V projections from a fused QKV weight matrix.
pub(crate) fn gemv_rows(
    ctx: &DeviceContext,
    a: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &DeviceVec,
    y: &mut DeviceVec,
) -> Result<()> {
    assert!(
        row_offset + num_rows <= a.rows,
        "gemv_rows: row_offset {} + num_rows {} > a.rows {}",
        row_offset,
        num_rows,
        a.rows
    );
    assert_eq!(a.cols, x.len);
    assert_eq!(num_rows, y.len);

    let (a_ptr, _ga) = a.data.device_ptr(&ctx.stream);
    // Offset by row_offset * cols elements (row-major)
    let a_sub = a_ptr + (row_offset * a.cols * std::mem::size_of::<half::bf16>()) as u64;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = y.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::gemv_cuda(
            a_sub as *const ffi::Half,
            x_ptr as *const ffi::Half,
            y_ptr as *mut ffi::Half,
            num_rows as i32,
            a.cols as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

/// GEMM on a row sub-range of a weight matrix: Y = W[row_offset..row_offset+M, :] @ X
pub(crate) fn gemm_rows_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    row_offset: usize,
    num_rows: usize,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    assert!(row_offset + num_rows <= weight.rows);
    assert_eq!(weight.cols, x.hidden_dim);
    assert_eq!(out.hidden_dim, num_rows);
    assert_eq!(out.seq_len, x.seq_len);

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let w_sub = w_ptr + (row_offset * weight.cols * std::mem::size_of::<half::bf16>()) as u64;
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        if x.seq_len == 1 {
            ffi::gemm_graphsafe_cuda(
                w_sub as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                num_rows as i32,
                1,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            );
        } else {
            ffi::gemm_cuda(
                w_sub as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                num_rows as i32,
                x.seq_len as i32,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            );
        }
    }
}

/// Fused MLP using gate_up_proj [2*intermediate, hidden] concatenated matrix.
/// Phase 1: silu(gate_proj @ x) * (up_proj @ x), Phase 2: down_proj @ act
pub(crate) fn fused_mlp_gate_up_into(
    ctx: &DeviceContext,
    x: &DeviceVec,
    gate_up_proj: &DeviceMatrix,
    intermediate_size: usize,
    down_proj: &DeviceMatrix,
    act: &mut DeviceVec,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(gate_up_proj.rows, 2 * intermediate_size);
    assert_eq!(gate_up_proj.cols, x.len);
    assert_eq!(down_proj.cols, intermediate_size);
    assert_eq!(down_proj.rows, out.len);
    assert_eq!(act.len, intermediate_size);

    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (gu_ptr, _gg) = gate_up_proj.data.device_ptr(&ctx.stream);
    // gate = first `intermediate_size` rows, up = next `intermediate_size` rows
    let gate_ptr = gu_ptr;
    let up_ptr =
        gu_ptr + (intermediate_size * gate_up_proj.cols * std::mem::size_of::<half::bf16>()) as u64;
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
            x.len as i32,
            intermediate_size as i32,
            ctx.stream.cu_stream(),
        );
    }

    Ok(())
}

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
pub(crate) fn linear(
    ctx: &DeviceContext,
    x: &DeviceVec,
    weight: &DeviceMatrix,
) -> Result<DeviceVec> {
    let mut y = DeviceVec::zeros(ctx, weight.rows)?;
    gemv(ctx, weight, x, &mut y)?;
    Ok(y)
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

/// Deinterleave fused QKV GEMM output [qkv_dim, bs] → q [q_dim, bs], k [kv_dim, bs], v [kv_dim, bs].
/// Zero-copy split via a small copy kernel (data is ~73KB at bs=8, takes ~2μs).
pub(crate) fn deinterleave_qkv_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    q: &mut HiddenStates,
    k: &mut HiddenStates,
    v: &mut HiddenStates,
) {
    let q_dim = q.hidden_dim;
    let kv_dim = k.hidden_dim;
    let bs = qkv.seq_len;
    debug_assert_eq!(qkv.hidden_dim, q_dim + 2 * kv_dim);
    debug_assert_eq!(v.hidden_dim, kv_dim);
    debug_assert_eq!(q.seq_len, bs);
    debug_assert_eq!(k.seq_len, bs);
    debug_assert_eq!(v.seq_len, bs);

    let (qkv_ptr, _g0) = qkv.data.device_ptr(&ctx.stream);
    let (q_ptr, _g1) = q.data.device_ptr_mut(&ctx.stream);
    let (k_ptr, _g2) = k.data.device_ptr_mut(&ctx.stream);
    let (v_ptr, _g3) = v.data.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::deinterleave_qkv_cuda(
            qkv_ptr as *const ffi::Half,
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            v_ptr as *mut ffi::Half,
            q_dim as i32,
            kv_dim as i32,
            bs as i32,
            ctx.stream.cu_stream(),
        );
    }
}

/// GEMM: Y = weight @ X (batched linear projection)
/// weight: [out_dim, in_dim] row-major, X: HiddenStates [in_dim, seq_len], Y: HiddenStates [out_dim, seq_len]
pub fn gemm(ctx: &DeviceContext, weight: &DeviceMatrix, x: &HiddenStates) -> Result<HiddenStates> {
    let mut out = HiddenStates::zeros(ctx, weight.rows, x.seq_len)?;
    gemm_into(ctx, weight, x, &mut out);
    Ok(out)
}

/// GEMM into pre-allocated output buffer (zero allocation).
/// For seq_len=1, uses the graph-safe cuBLAS handle (no workspace) for lower
/// latency while preserving numerical parity with the prefill path.
pub(crate) fn gemm_into(
    ctx: &DeviceContext,
    weight: &DeviceMatrix,
    x: &HiddenStates,
    out: &mut HiddenStates,
) {
    assert_eq!(
        weight.cols, x.hidden_dim,
        "weight cols {} != hidden_dim {}",
        weight.cols, x.hidden_dim
    );
    assert_eq!(
        out.hidden_dim, weight.rows,
        "out hidden_dim {} != weight rows {}",
        out.hidden_dim, weight.rows
    );
    assert_eq!(
        out.seq_len, x.seq_len,
        "out seq_len {} != x seq_len {}",
        out.seq_len, x.seq_len
    );

    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
    let (y_ptr, _gy) = out.data.device_ptr_mut(&ctx.stream);

    unsafe {
        if x.seq_len == 1 {
            ffi::gemm_graphsafe_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                weight.rows as i32,
                1,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            );
        } else {
            ffi::gemm_cuda(
                w_ptr as *const ffi::Half,
                x_ptr as *const ffi::Half,
                y_ptr as *mut ffi::Half,
                weight.rows as i32,
                x.seq_len as i32,
                weight.cols as i32,
                ctx.stream.cu_stream(),
            );
        }
    }
}
