use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

/// Argmax — returns the index of the maximum element.
///
/// Allocates a temporary output buffer. Used by benchmarks; model code uses
/// `gpu_sample_into` for both greedy and non-greedy paths.
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

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;

    Ok(result[0] as u32)
}

/// GPU sampling: temperature → softmax → top-k → top-p → multinomial.
/// Allocates a temporary output buffer — use `gpu_sample_into` for the decode loop.
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

    gpu_sample_core(ctx, logits, probs_scratch, &mut out_gpu, params, random_val)
}

/// GPU sampling into pre-allocated buffers — zero allocation, suitable for decode loop.
///
/// Greedy dispatch: argmax kernel. Non-greedy: full sampling kernel.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_core(ctx, logits, probs_scratch, out, params, random_val)
}

fn gpu_sample_core(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    if params.is_greedy() {
        // Fast path: deterministic argmax (avoids softmax tie-breaking issues with bf16)
        let (x_ptr, _gx) = logits.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                o_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    } else {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (p_ptr, _gp) = probs_scratch.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::gpu_sample_cuda(
                l_ptr as *const ffi::Half,
                p_ptr as *mut f32,
                o_ptr as *mut i32,
                logits.len as i32,
                1.0 / params.temperature,
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
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H sample read failed: {}", e))?;

    Ok(result[0] as u32)
}

/// Batched greedy sampling: one argmax kernel launch for all rows, one sync, one D2H.
///
/// `logits` must be [batch_size, vocab_size] contiguous (HiddenStates with
/// hidden_dim=vocab_size, seq_len=batch_size). `out` must have capacity >= batch_size.
pub fn argmax_batched(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    out: &mut CudaSlice<i32>,
    batch_size: usize,
) -> Result<Vec<u32>> {
    let vocab_size = logits.hidden_dim;

    {
        let (x_ptr, _gx) = logits.data.device_ptr(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_batched_cuda(
                x_ptr as *const ffi::Half,
                o_ptr as *mut i32,
                vocab_size as i32,
                batch_size as i32,
                ctx.stream.cu_stream(),
            );
        }
    }

    ctx.sync()?;

    let result = ctx
        .stream
        .clone_dtoh(out)
        .map_err(|e| anyhow!("D2H batched argmax failed: {}", e))?;

    Ok(result[..batch_size].iter().map(|&x| x as u32).collect())
}
