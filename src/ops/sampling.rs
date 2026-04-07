use anyhow::{Result, anyhow};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use crate::ffi;
use crate::tensor::{DeviceContext, DeviceVec};

const FLASHINFER_TOPK_ROW_STATES_BYTES: usize = 1024 * 1024;

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
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    let mut valid_scratch: CudaSlice<u8> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    gpu_sample_core(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        &mut valid_scratch,
        &mut out_gpu,
        params,
        random_val,
    )
}

/// GPU sampling into pre-allocated buffers — zero allocation, suitable for decode loop.
pub fn gpu_sample_into(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    gpu_sample_core(
        ctx,
        logits,
        probs_scratch,
        top1_value_scratch,
        row_states_scratch,
        valid_scratch,
        out,
        params,
        random_val,
    )
}

fn gpu_sample_core(
    ctx: &DeviceContext,
    logits: &DeviceVec,
    probs_scratch: &mut CudaSlice<f32>,
    top1_value_scratch: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    valid_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
    params: &crate::sampler::SamplingParams,
    random_val: f32,
) -> Result<u32> {
    if params.is_greedy() {
        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (v_ptr, _gv) = top1_value_scratch.device_ptr_mut(&ctx.stream);
        let (r_ptr, _gr) = row_states_scratch.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::flashinfer_top1_cuda(
                l_ptr as *const ffi::Half,
                v_ptr as *mut ffi::Half,
                r_ptr as *mut u8,
                o_ptr as *mut i32,
                logits.len as i32,
                ctx.stream.cu_stream(),
            );
        }
    } else {
        let inv_temperature = 1.0 / params.temperature;
        let top_k = params.top_k;
        let top_p = params.top_p;

        let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (p_ptr, _gp) = probs_scratch.device_ptr_mut(&ctx.stream);
        let (v_ptr, _gv) = valid_scratch.device_ptr_mut(&ctx.stream);
        let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::gpu_sample_flashinfer_cuda(
                l_ptr as *const ffi::Half,
                p_ptr as *mut f32,
                v_ptr as *mut u8,
                o_ptr as *mut i32,
                logits.len as i32,
                inv_temperature,
                top_k,
                top_p,
                u64::from(random_val.to_bits()),
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

pub fn flashinfer_topk_row_states_bytes() -> usize {
    FLASHINFER_TOPK_ROW_STATES_BYTES
}
