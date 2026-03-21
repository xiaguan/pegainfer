use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec, HiddenStates};

use super::common::{
    EPS, HEAD_DIM_256, Q_HEADS_256, configure_group, device_vec, f32_slice, hidden_states, iter_sync,
    positive_device_vec,
};

pub fn bench_qwen35_norm_ops(c: &mut Criterion) {
    let vec_len = Q_HEADS_256 * HEAD_DIM_256;

    let mut group = c.benchmark_group("ops_qwen35_norm");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(vec_len as u64));
    group.bench_function(BenchmarkId::new("rms_norm_offset_into", vec_len), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x = device_vec(&ctx, vec_len).expect("failed to allocate x");
        let weight = positive_device_vec(&ctx, vec_len).expect("failed to allocate offset weight");
        let mut offset_out =
            DeviceVec::zeros(&ctx, vec_len).expect("failed to allocate offset out");
        iter_sync(b, &ctx, || {
            ops::rms_norm_offset_into(&ctx, &x, &weight, EPS, &mut offset_out)
                .expect("rms_norm_offset_into failed");
        });
    });

    group.throughput(Throughput::Elements(vec_len as u64));
    group.bench_function(
        BenchmarkId::new("fused_add_rms_norm_offset_into", vec_len),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let mut hidden = device_vec(&ctx, vec_len).expect("failed to allocate hidden");
            let residual = device_vec(&ctx, vec_len).expect("failed to allocate residual");
            let fused_weight =
                positive_device_vec(&ctx, vec_len).expect("failed to allocate fused weight");
            let mut fused_out =
                DeviceVec::zeros(&ctx, vec_len).expect("failed to allocate fused out");
            iter_sync(b, &ctx, || {
                ops::fused_add_rms_norm_offset_into(
                    &ctx,
                    &mut hidden,
                    &residual,
                    &fused_weight,
                    EPS,
                    &mut fused_out,
                )
                .expect("fused_add_rms_norm_offset_into failed");
            });
        },
    );

    group.throughput(Throughput::Elements(vec_len as u64));
    group.bench_function(BenchmarkId::new("rms_norm_gated_into", vec_len), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x = device_vec(&ctx, vec_len).expect("failed to allocate x");
        let per_head_weight =
            f32_slice(&ctx, HEAD_DIM_256).expect("failed to allocate f32 norm weight");
        let gate = device_vec(&ctx, vec_len).expect("failed to allocate gate");
        let mut gated_out = DeviceVec::zeros(&ctx, vec_len).expect("failed to allocate gated out");
        iter_sync(b, &ctx, || {
            ops::rms_norm_gated_into(
                &ctx,
                &x,
                &per_head_weight,
                &gate,
                &mut gated_out,
                Q_HEADS_256,
                HEAD_DIM_256,
                EPS,
            )
            .expect("rms_norm_gated_into failed");
        });
    });

    group.finish();

    // ========================================================================
    // Batched RMSNorm_offset: per-token loop (current) vs batched kernel (TODO)
    // Qwen3.5 hidden_dim=2560, realistic seq_len values.
    // ========================================================================
    let hidden_dim = 2560;

    for &seq_len in &[128, 2048] {
        let mut group = c.benchmark_group(format!("rms_norm_offset_batched/seq{seq_len}"));
        configure_group(&mut group);
        group.throughput(Throughput::Elements((hidden_dim * seq_len) as u64));

        group.bench_function("batched_kernel", |b| {
            let ctx = DeviceContext::new().expect("ctx");
            let x = hidden_states(&ctx, hidden_dim, seq_len).expect("x");
            let weight = positive_device_vec(&ctx, hidden_dim).expect("weight");
            let mut out = HiddenStates::zeros(&ctx, hidden_dim, seq_len).expect("out");
            iter_sync(b, &ctx, || {
                ops::rms_norm_batch_offset_into(&ctx, &x, &weight, EPS, &mut out)
                    .expect("batched norm");
            });
        });

        group.finish();
    }
}
