#![allow(dead_code)]

use std::hint::black_box;
use std::time::Duration;

use anyhow::{Result, anyhow};
use criterion::{Bencher, BenchmarkGroup, measurement::WallTime};
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer::tensor::{DeviceContext, DeviceMatrix, DeviceVec, HiddenStates};

pub const VECTOR_DIM: usize = 1024;
pub const OUT_DIM: usize = 1024;
pub const INTERMEDIATE_DIM: usize = 4096;
pub const VOCAB_SIZE: usize = 32_768;
pub const BATCH_SEQ_LEN: usize = 64;
pub const ATTN_SEQ_LEN: usize = 64;
pub const MAX_SEQ_LEN: usize = 4096;
pub const Q_HEADS_128: usize = 8;
pub const KV_HEADS_128: usize = 2;
pub const HEAD_DIM_128: usize = 128;
pub const Q_HEADS_256: usize = 4;
pub const KV_HEADS_256: usize = 1;
pub const HEAD_DIM_256: usize = 256;
pub const ROTARY_DIM_256: usize = 64;
pub const LINEAR_KEY_HEADS: usize = 4;
pub const LINEAR_VALUE_HEADS: usize = 8;
pub const LINEAR_KEY_DIM: usize = 64;
pub const LINEAR_VALUE_DIM: usize = 64;
pub const CONV_KERNEL_SIZE: usize = 4;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA_QWEN3: f32 = 1_000_000.0;
pub const ROPE_THETA_QWEN35: f32 = 10_000_000.0;

// Qwen3.5-4B actual model dimensions
pub const QWEN35_4B_HIDDEN: usize = 2560;
pub const QWEN35_4B_Q_HEADS: usize = 16;
pub const QWEN35_4B_KV_HEADS: usize = 4;
pub const QWEN35_4B_HEAD_DIM: usize = 256;
pub const QWEN35_4B_ROTARY_DIM: usize = 64;
pub const QWEN35_4B_LINEAR_K_HEADS: usize = 16;
pub const QWEN35_4B_LINEAR_V_HEADS: usize = 32;
pub const QWEN35_4B_LINEAR_K_DIM: usize = 128;
pub const QWEN35_4B_LINEAR_V_DIM: usize = 128;
pub const QWEN35_4B_ROPE_THETA: f32 = 10_000_000.0;

pub fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    group.sample_size(10);
}

pub fn iter_sync<T, F>(b: &mut Bencher<'_>, ctx: &DeviceContext, mut f: F)
where
    F: FnMut() -> T,
{
    ctx.sync().expect("CUDA pre-benchmark sync failed");
    b.iter(|| {
        ctx.sync().expect("CUDA pre-iteration sync failed");
        let value = f();
        ctx.sync().expect("CUDA post-iteration sync failed");
        black_box(value);
    });
    ctx.sync().expect("CUDA post-benchmark sync failed");
}

pub fn bf16_data(len: usize) -> Vec<bf16> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 97) as f32 - 48.0;
            bf16::from_f32(centered * 0.03125)
        })
        .collect()
}

pub fn bf16_data_scaled(len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 89) as f32 - 44.0;
            bf16::from_f32(centered * scale)
        })
        .collect()
}

pub fn positive_bf16_data(len: usize) -> Vec<bf16> {
    (0..len)
        .map(|idx| bf16::from_f32(0.5 + (idx % 23) as f32 * 0.03125))
        .collect()
}

pub fn f32_data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 79) as f32 - 39.0) * 0.015625)
        .collect()
}

pub fn device_vec(ctx: &DeviceContext, len: usize) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &bf16_data(len))
}

pub fn device_vec_scaled(ctx: &DeviceContext, len: usize, scale: f32) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &bf16_data_scaled(len, scale))
}

pub fn positive_device_vec(ctx: &DeviceContext, len: usize) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &positive_bf16_data(len))
}

pub fn device_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    DeviceMatrix::from_host(ctx, &bf16_data(rows * cols), rows, cols)
}

pub fn embedding_matrix(
    ctx: &DeviceContext,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<DeviceMatrix> {
    DeviceMatrix::from_host(
        ctx,
        &bf16_data_scaled(vocab_size * hidden_size, 0.0078125),
        vocab_size,
        hidden_size,
    )
}

pub fn hidden_states(
    ctx: &DeviceContext,
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    let host = bf16_data(hidden_dim * seq_len);
    let data = ctx
        .stream
        .clone_htod(&host)
        .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
    Ok(HiddenStates {
        data,
        hidden_dim,
        seq_len,
    })
}

pub fn f32_slice(ctx: &DeviceContext, len: usize) -> Result<CudaSlice<f32>> {
    ctx.stream
        .clone_htod(&f32_data(len))
        .map_err(|e| anyhow!("H2D copy failed: {}", e))
}

pub fn zero_f32_slice(ctx: &DeviceContext, len: usize) -> Result<CudaSlice<f32>> {
    ctx.stream
        .alloc_zeros(len)
        .map_err(|e| anyhow!("Alloc failed: {}", e))
}

pub fn token_ids(ctx: &DeviceContext, seq_len: usize, vocab_size: usize) -> Result<CudaSlice<i32>> {
    let host: Vec<i32> = (0..seq_len)
        .map(|idx| (idx % vocab_size.max(1)) as i32)
        .collect();
    ctx.stream
        .clone_htod(&host)
        .map_err(|e| anyhow!("H2D copy failed: {}", e))
}

pub fn decode_meta(
    ctx: &DeviceContext,
    token_id: i32,
    current_pos: usize,
    seq_len: usize,
) -> Result<CudaSlice<i32>> {
    ctx.stream
        .clone_htod(&[token_id, current_pos as i32, seq_len as i32])
        .map_err(|e| anyhow!("H2D copy failed: {}", e))
}

pub fn rope_cache(
    ctx: &DeviceContext,
    max_seq_len: usize,
    dim: usize,
    theta: f32,
) -> Result<(DeviceVec, DeviceVec)> {
    assert_eq!(dim % 2, 0, "RoPE dimension must be even");
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|idx| 1.0 / theta.powf(idx as f32 * 2.0 / dim as f32))
        .collect();
    let mut cos_host = vec![bf16::ZERO; max_seq_len * dim];
    let mut sin_host = vec![bf16::ZERO; max_seq_len * dim];
    for pos in 0..max_seq_len {
        for idx in 0..half {
            let freq = pos as f32 * inv_freq[idx];
            let cos = bf16::from_f32(freq.cos());
            let sin = bf16::from_f32(freq.sin());
            cos_host[pos * dim + idx] = cos;
            cos_host[pos * dim + idx + half] = cos;
            sin_host[pos * dim + idx] = sin;
            sin_host[pos * dim + idx + half] = sin;
        }
    }

    Ok((
        DeviceVec::from_host(ctx, &cos_host)?,
        DeviceVec::from_host(ctx, &sin_host)?,
    ))
}
