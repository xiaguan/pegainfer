use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec};

use super::common::{
    EPS, INTERMEDIATE_DIM, OUT_DIM, QWEN35_4B_HIDDEN, QWEN35_4B_INTERMEDIATE, QWEN35_4B_VOCAB,
    VECTOR_DIM, configure_group, device_matrix, device_vec, iter_sync, positive_device_vec,
};

pub(crate) fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_elementwise");
    configure_group(&mut group);

    let gemv_shapes = [
        ("legacy_1024x1024", OUT_DIM, VECTOR_DIM),
        ("qwen35_q_qkv_8192x2560", 8192, QWEN35_4B_HIDDEN),
        ("qwen35_z_4096x2560", 4096, QWEN35_4B_HIDDEN),
        ("qwen35_kv_1024x2560", 1024, QWEN35_4B_HIDDEN),
        ("qwen35_ba_32x2560", 32, QWEN35_4B_HIDDEN),
        ("qwen35_o_2560x4096", QWEN35_4B_HIDDEN, 4096),
        (
            "qwen35_lm_head_248320x2560",
            QWEN35_4B_VOCAB,
            QWEN35_4B_HIDDEN,
        ),
    ];

    for (label, rows, cols) in gemv_shapes {
        group.throughput(Throughput::Elements((rows * cols) as u64));
        group.bench_function(BenchmarkId::new("gemv", label), |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let matrix = device_matrix(&ctx, rows, cols).expect("failed to allocate matrix");
            let x = device_vec(&ctx, cols).expect("failed to allocate x");
            let mut gemv_out = DeviceVec::zeros(&ctx, rows).expect("failed to allocate gemv out");
            iter_sync(b, &ctx, || {
                ops::gemv(&ctx, &matrix, &x, &mut gemv_out).expect("gemv failed");
            });
        });
    }
    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("rms_norm_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let rms_x = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate rms x");
        let rms_weight =
            positive_device_vec(&ctx, VECTOR_DIM).expect("failed to allocate rms weight");
        let mut rms_out = DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate rms out");
        iter_sync(b, &ctx, || {
            ops::rms_norm_into(&ctx, &rms_x, &rms_weight, EPS, &mut rms_out)
                .expect("rms_norm_into failed");
        });
    });

    group.throughput(Throughput::Elements((INTERMEDIATE_DIM * VECTOR_DIM) as u64));
    group.bench_function(BenchmarkId::new("fused_mlp_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate x");
        let gate_proj = device_matrix(&ctx, INTERMEDIATE_DIM, VECTOR_DIM)
            .expect("failed to allocate gate proj");
        let up_proj =
            device_matrix(&ctx, INTERMEDIATE_DIM, VECTOR_DIM).expect("failed to allocate up proj");
        let down_proj = device_matrix(&ctx, VECTOR_DIM, INTERMEDIATE_DIM)
            .expect("failed to allocate down proj");
        let mut act = DeviceVec::zeros(&ctx, INTERMEDIATE_DIM).expect("failed to allocate act");
        let mut mlp_out = DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate mlp out");
        iter_sync(b, &ctx, || {
            ops::fused_mlp_into(
                &ctx,
                &x,
                &gate_proj,
                &up_proj,
                &down_proj,
                &mut act,
                &mut mlp_out,
            )
            .expect("fused_mlp_into failed");
        });
    });

    group.throughput(Throughput::Elements(
        (QWEN35_4B_HIDDEN * QWEN35_4B_INTERMEDIATE) as u64,
    ));
    group.bench_function(
        BenchmarkId::new("fused_mlp_into", "qwen35_4b_2560x9216"),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let x = device_vec(&ctx, QWEN35_4B_HIDDEN).expect("failed to allocate x");
            let gate_proj = device_matrix(&ctx, QWEN35_4B_INTERMEDIATE, QWEN35_4B_HIDDEN)
                .expect("failed to allocate gate proj");
            let up_proj = device_matrix(&ctx, QWEN35_4B_INTERMEDIATE, QWEN35_4B_HIDDEN)
                .expect("failed to allocate up proj");
            let down_proj = device_matrix(&ctx, QWEN35_4B_HIDDEN, QWEN35_4B_INTERMEDIATE)
                .expect("failed to allocate down proj");
            let mut act =
                DeviceVec::zeros(&ctx, QWEN35_4B_INTERMEDIATE).expect("failed to allocate act");
            let mut mlp_out =
                DeviceVec::zeros(&ctx, QWEN35_4B_HIDDEN).expect("failed to allocate mlp out");
            iter_sync(b, &ctx, || {
                ops::fused_mlp_into(
                    &ctx,
                    &x,
                    &gate_proj,
                    &up_proj,
                    &down_proj,
                    &mut act,
                    &mut mlp_out,
                )
                .expect("fused_mlp_into failed");
            });
        },
    );

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(
        BenchmarkId::new("fused_add_rms_norm_into", VECTOR_DIM),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let mut hidden = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate hidden");
            let residual = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate residual");
            let fused_weight =
                positive_device_vec(&ctx, VECTOR_DIM).expect("failed to allocate fused weight");
            let mut fused_out =
                DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate fused out");
            iter_sync(b, &ctx, || {
                ops::fused_add_rms_norm_into(
                    &ctx,
                    &mut hidden,
                    &residual,
                    &fused_weight,
                    EPS,
                    &mut fused_out,
                )
                .expect("fused_add_rms_norm_into failed");
            });
        },
    );

    group.throughput(Throughput::Elements((VECTOR_DIM * 4) as u64));
    group.bench_function(BenchmarkId::new("argmax", VECTOR_DIM * 4), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let argmax_x = device_vec(&ctx, VECTOR_DIM * 4).expect("failed to allocate argmax input");
        iter_sync(b, &ctx, || {
            let token = ops::argmax(&ctx, &argmax_x).expect("argmax failed");
            black_box(token);
        });
    });

    group.finish();
}
