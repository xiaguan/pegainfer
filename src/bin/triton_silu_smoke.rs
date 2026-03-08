use anyhow::Result;
use clap::Parser;
use half::bf16;
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, HiddenStates};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 128)]
    seq_len: usize,
    #[arg(long, default_value_t = 11008)]
    hidden_dim: usize,
    #[arg(long, default_value_t = 100)]
    iters: usize,
}

fn hidden_states_from_host(
    ctx: &DeviceContext,
    data: &[bf16],
    hidden_dim: usize,
    seq_len: usize,
) -> Result<HiddenStates> {
    let gpu = ctx.stream.clone_htod(data)?;
    Ok(HiddenStates {
        data: gpu,
        hidden_dim,
        seq_len,
    })
}

fn dtoh(ctx: &DeviceContext, x: &HiddenStates) -> Result<Vec<bf16>> {
    let host = ctx.stream.clone_dtoh(&x.data)?;
    ctx.sync()?;
    Ok(host)
}

fn max_abs_diff(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn bench_cuda(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    iters: usize,
) -> Result<f64> {
    for _ in 0..10 {
        let _ = ops::silu_mul_batch_cuda_ref(ctx, gate, up)?;
    }
    ctx.sync()?;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = ops::silu_mul_batch_cuda_ref(ctx, gate, up)?;
    }
    ctx.sync()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

fn bench_triton(
    ctx: &DeviceContext,
    gate: &HiddenStates,
    up: &HiddenStates,
    iters: usize,
) -> Result<f64> {
    for _ in 0..10 {
        let _ = ops::silu_mul_batch(ctx, gate, up)?;
    }
    ctx.sync()?;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = ops::silu_mul_batch(ctx, gate, up)?;
    }
    ctx.sync()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let ctx = DeviceContext::new()?;
    let n = args.seq_len * args.hidden_dim;

    let gate_host: Vec<bf16> = (0..n)
        .map(|i| {
            let x = ((i % 257) as f32 - 128.0) / 32.0;
            bf16::from_f32(x)
        })
        .collect();
    let up_host: Vec<bf16> = (0..n)
        .map(|i| {
            let x = ((i % 113) as f32 - 56.0) / 24.0;
            bf16::from_f32(x)
        })
        .collect();

    let gate = hidden_states_from_host(&ctx, &gate_host, args.hidden_dim, args.seq_len)?;
    let up = hidden_states_from_host(&ctx, &up_host, args.hidden_dim, args.seq_len)?;

    let cuda_out = ops::silu_mul_batch_cuda_ref(&ctx, &gate, &up)?;
    let triton_out = ops::silu_mul_batch(&ctx, &gate, &up)?;
    let cuda_host = dtoh(&ctx, &cuda_out)?;
    let triton_host = dtoh(&ctx, &triton_out)?;
    let diff = max_abs_diff(&cuda_host, &triton_host);

    let cuda_ms = bench_cuda(&ctx, &gate, &up, args.iters)?;
    let triton_ms = bench_triton(&ctx, &gate, &up, args.iters)?;

    println!(
        "shape seq_len={} hidden_dim={} n={}",
        args.seq_len, args.hidden_dim, n
    );
    println!("max_abs_diff={:.6}", diff);
    println!("cuda_ms={:.6}", cuda_ms);
    println!("triton_ms={:.6}", triton_ms);
    if cuda_ms > 0.0 {
        println!("speedup={:.3}x", cuda_ms / triton_ms);
    }

    Ok(())
}
