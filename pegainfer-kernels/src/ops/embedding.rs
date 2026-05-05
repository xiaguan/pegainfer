use anyhow::Result;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

/// Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
pub fn embedding_decode_into(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_id: &CudaSlice<u32>,
    out: &mut DeviceVec,
) -> Result<()> {
    assert_eq!(embed.cols, out.len);

    let (embed_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (token_ptr, _gt) = token_id.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_decode_cuda(
            embed_ptr as *const ffi::Half,
            token_ptr as *const u32,
            out_ptr as *mut ffi::Half,
            embed.cols as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}

/// Batched embedding lookup
pub fn embedding_batch(
    ctx: &DeviceContext,
    embed: &DeviceMatrix,
    token_ids_gpu: &CudaSlice<u32>,
    out: &mut HiddenStates,
) -> Result<()> {
    let (e_ptr, _ge) = embed.data.device_ptr(&ctx.stream);
    let (t_ptr, _gt) = token_ids_gpu.device_ptr(&ctx.stream);
    let (o_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::embedding_batched_cuda(
            e_ptr as *const ffi::Half,
            t_ptr as *const u32,
            o_ptr as *mut ffi::Half,
            embed.cols as i32,
            out.seq_len as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;

    Ok(())
}
