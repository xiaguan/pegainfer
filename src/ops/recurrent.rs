use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::model::qwen35::prefill_buffers::GdrChunkwiseScratch35;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Gated delta rule recurrent decode (single step, seq_len=1).
/// Fused CUDA kernel: L2-norm q/k, compute g/beta, decay + rank-1 state update, output.
/// ~15μs/layer on RTX 5070 Ti vs ~33μs for the 7-stage chunk-wise pipeline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gated_delta_rule_decode_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    output: &mut HiddenStates,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Result<()> {
    debug_assert_eq!(qkv.seq_len, 1);
    debug_assert_eq!(b_proj.seq_len, 1);
    debug_assert_eq!(a_proj.seq_len, 1);
    debug_assert_eq!(output.seq_len, 1);

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

/// Causal depthwise conv1d prefill over a HiddenStates batch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv1d_prefill_batch_into(
    ctx: &DeviceContext,
    x_seq: &HiddenStates,
    conv_weight: &DeviceVec,
    conv_state: &mut DeviceVec,
    out_seq: &mut HiddenStates,
    kernel_size: usize,
) {
    let num_channels = x_seq.hidden_dim;
    assert_eq!(out_seq.hidden_dim, num_channels);
    assert_eq!(out_seq.seq_len, x_seq.seq_len);
    assert_eq!(conv_weight.len, num_channels * kernel_size);
    assert_eq!(conv_state.len, num_channels * (kernel_size - 1));

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
            x_seq.seq_len as i32,
            kernel_size as i32,
            ctx.stream.cu_stream(),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_prepare_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    q_out: &mut HiddenStates,
    k_out: &mut HiddenStates,
    v_out: &mut HiddenStates,
    g_out: &mut CudaSlice<f32>,
    beta_out: &mut CudaSlice<f32>,
    num_key_heads: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (qkv_ptr, _gqkv) = qkv.data.device_ptr(&ctx.stream);
    let (b_ptr, _gb) = b_proj.data.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_proj.data.device_ptr(&ctx.stream);
    let (dt_ptr, _gdt) = dt_bias.data.device_ptr(&ctx.stream);
    let (alog_ptr, _gal) = a_log.device_ptr(&ctx.stream);
    let (q_out_ptr, _gqo) = q_out.data.device_ptr_mut(&ctx.stream);
    let (k_out_ptr, _gko) = k_out.data.device_ptr_mut(&ctx.stream);
    let (v_out_ptr, _gvo) = v_out.data.device_ptr_mut(&ctx.stream);
    let (g_out_ptr, _ggo) = g_out.device_ptr_mut(&ctx.stream);
    let (beta_out_ptr, _gbetao) = beta_out.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_prepare_cuda(
            qkv_ptr as *const ffi::Half,
            b_ptr as *const ffi::Half,
            a_ptr as *const ffi::Half,
            dt_ptr as *const ffi::Half,
            alog_ptr as *const f32,
            q_out_ptr as *mut ffi::Half,
            k_out_ptr as *mut ffi::Half,
            v_out_ptr as *mut ffi::Half,
            g_out_ptr as *mut f32,
            beta_out_ptr as *mut f32,
            num_key_heads as i32,
            num_value_heads as i32,
            qkv.hidden_dim as i32,
            qkv.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_cumsum_inplace(
    ctx: &DeviceContext,
    g_cumsum: &mut CudaSlice<f32>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (g_ptr, _gg) = g_cumsum.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_cumsum_cuda(
            g_ptr as *const f32,
            g_ptr as *mut f32,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_a_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    a_tril: &mut CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (a_ptr, _ga) = a_tril.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_a_cuda(
            k_ptr as *const ffi::Half,
            g_ptr as *const f32,
            beta_ptr as *const f32,
            a_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

fn gated_delta_rule_prefill_chunk_solve_into(
    ctx: &DeviceContext,
    a_tril: &CudaSlice<f32>,
    a_inv: &mut CudaSlice<half::bf16>,
    seq_len: usize,
    num_value_heads: usize,
) -> Result<()> {
    let (a_ptr, _ga) = a_tril.device_ptr(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_solve_cuda(
            a_ptr as *const f32,
            ai_ptr as *mut ffi::Half,
            seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_recompute_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    v: &HiddenStates,
    beta: &CudaSlice<f32>,
    w: &mut HiddenStates,
    u: &mut HiddenStates,
    a_inv: &CudaSlice<half::bf16>,
    g_cumsum: &CudaSlice<f32>,
    num_value_heads: usize,
) -> Result<()> {
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = v.data.device_ptr(&ctx.stream);
    let (beta_ptr, _gb) = beta.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr_mut(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr_mut(&ctx.stream);
    let (ai_ptr, _gai) = a_inv.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_recompute_cuda(
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            beta_ptr as *const f32,
            w_ptr as *mut ffi::Half,
            u_ptr as *mut ffi::Half,
            ai_ptr as *const ffi::Half,
            g_ptr as *const f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_state_stage_into(
    ctx: &DeviceContext,
    k: &HiddenStates,
    w: &HiddenStates,
    u: &HiddenStates,
    g_cumsum: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    chunk_state: &mut CudaSlice<f32>,
    v_new: &mut HiddenStates,
    num_value_heads: usize,
) -> Result<()> {
    assert_eq!(k.hidden_dim, w.hidden_dim);
    assert_eq!(u.hidden_dim, v_new.hidden_dim);
    assert_eq!(k.seq_len, w.seq_len);
    assert_eq!(k.seq_len, u.seq_len);
    assert_eq!(k.seq_len, v_new.seq_len);

    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = w.data.device_ptr(&ctx.stream);
    let (u_ptr, _gu) = u.data.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (s_ptr, _gs) = state.device_ptr_mut(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr_mut(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_state_cuda(
            k_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            u_ptr as *const ffi::Half,
            g_ptr as *const f32,
            s_ptr as *const f32,
            cs_ptr as *mut f32,
            vn_ptr as *mut ffi::Half,
            s_ptr as *mut f32,
            k.seq_len as i32,
            num_value_heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule_prefill_chunk_o_stage_into(
    ctx: &DeviceContext,
    q: &HiddenStates,
    k: &HiddenStates,
    v_new: &HiddenStates,
    chunk_state: &CudaSlice<f32>,
    g_cumsum: &CudaSlice<f32>,
    output: &mut HiddenStates,
    num_value_heads: usize,
    scale: f32,
) -> Result<()> {
    assert_eq!(q.hidden_dim, k.hidden_dim);
    assert_eq!(v_new.hidden_dim, output.hidden_dim);
    assert_eq!(q.seq_len, k.seq_len);
    assert_eq!(q.seq_len, v_new.seq_len);
    assert_eq!(q.seq_len, output.seq_len);

    let (q_ptr, _gq) = q.data.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = k.data.device_ptr(&ctx.stream);
    let (vn_ptr, _gvn) = v_new.data.device_ptr(&ctx.stream);
    let (cs_ptr, _gcs) = chunk_state.device_ptr(&ctx.stream);
    let (g_ptr, _gg) = g_cumsum.device_ptr(&ctx.stream);
    let (o_ptr, _go) = output.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::gated_delta_rule_prefill_chunk_o_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            vn_ptr as *const ffi::Half,
            cs_ptr as *const f32,
            g_ptr as *const f32,
            o_ptr as *mut ffi::Half,
            q.seq_len as i32,
            num_value_heads as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Chunk-wise GDR prefill operator contract for Qwen3.5.
///
/// The chunk-wise path is an explicit multi-stage operator with pre-allocated
/// scratch instead of one opaque kernel launch.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_rule_prefill_chunkwise_into(
    ctx: &DeviceContext,
    qkv: &HiddenStates,
    b_proj: &HiddenStates,
    a_proj: &HiddenStates,
    dt_bias: &DeviceVec,
    a_log: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    scratch: &mut GdrChunkwiseScratch35,
    output: &mut HiddenStates,
    num_key_heads: usize,
    num_value_heads: usize,
    key_dim: usize,
    val_dim: usize,
) -> Result<()> {
    assert_eq!(scratch.q_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.k_expanded.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_raw.seq_len, qkv.seq_len);
    assert_eq!(scratch.w.seq_len, qkv.seq_len);
    assert_eq!(scratch.u.seq_len, qkv.seq_len);
    assert_eq!(scratch.v_new.seq_len, qkv.seq_len);
    assert_eq!(scratch.q_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.k_expanded.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.v_raw.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.w.hidden_dim, num_value_heads * key_dim);
    assert_eq!(scratch.u.hidden_dim, num_value_heads * val_dim);
    assert_eq!(scratch.v_new.hidden_dim, num_value_heads * val_dim);

    let expected_gate_len = qkv.seq_len * num_value_heads;
    let expected_chunk_a_len = qkv.seq_len * num_value_heads * GdrChunkwiseScratch35::CHUNK_SIZE;
    let expected_chunk_ai_len = expected_chunk_a_len;
    let expected_chunk_state_len =
        GdrChunkwiseScratch35::num_chunks(qkv.seq_len) * num_value_heads * val_dim * key_dim;
    assert_eq!(scratch.g_cumsum.len(), expected_gate_len);
    assert_eq!(scratch.beta.len(), expected_gate_len);
    assert_eq!(scratch.a_tril.len(), expected_chunk_a_len);
    assert_eq!(scratch.a_inv.len(), expected_chunk_ai_len);
    assert_eq!(scratch.chunk_state.len(), expected_chunk_state_len);

    gated_delta_rule_prefill_chunk_prepare_into(
        ctx,
        qkv,
        b_proj,
        a_proj,
        dt_bias,
        a_log,
        &mut scratch.q_expanded,
        &mut scratch.k_expanded,
        &mut scratch.v_raw,
        &mut scratch.g_cumsum,
        &mut scratch.beta,
        num_key_heads,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_cumsum_inplace(
        ctx,
        &mut scratch.g_cumsum,
        qkv.seq_len,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_a_into(
        ctx,
        &scratch.k_expanded,
        &scratch.g_cumsum,
        &scratch.beta,
        &mut scratch.a_tril,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_solve_into(
        ctx,
        &scratch.a_tril,
        &mut scratch.a_inv,
        qkv.seq_len,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_recompute_into(
        ctx,
        &scratch.k_expanded,
        &scratch.v_raw,
        &scratch.beta,
        &mut scratch.w,
        &mut scratch.u,
        &scratch.a_inv,
        &scratch.g_cumsum,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_state_stage_into(
        ctx,
        &scratch.k_expanded,
        &scratch.w,
        &scratch.u,
        &scratch.g_cumsum,
        state,
        &mut scratch.chunk_state,
        &mut scratch.v_new,
        num_value_heads,
    )?;
    gated_delta_rule_prefill_chunk_o_stage_into(
        ctx,
        &scratch.q_expanded,
        &scratch.k_expanded,
        &scratch.v_new,
        &scratch.chunk_state,
        &scratch.g_cumsum,
        output,
        num_value_heads,
        1.0 / (key_dim as f32).sqrt(),
    )
}
