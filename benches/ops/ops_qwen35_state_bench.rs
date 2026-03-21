use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec};

use super::common::{
    ATTN_SEQ_LEN, CONV_KERNEL_SIZE, EPS, MAX_SEQ_LEN, QWEN35_4B_HEAD_DIM, QWEN35_4B_KV_HEADS,
    QWEN35_4B_LINEAR_K_DIM, QWEN35_4B_LINEAR_K_HEADS, QWEN35_4B_LINEAR_V_DIM,
    QWEN35_4B_LINEAR_V_HEADS, QWEN35_4B_Q_HEADS, QWEN35_4B_ROPE_THETA, QWEN35_4B_ROTARY_DIM,
    configure_group, decode_meta, device_vec, f32_slice, iter_sync, positive_device_vec, rope_cache,
    zero_f32_slice,
};

pub fn bench_qwen35_state_ops(c: &mut Criterion) {
    // Qwen3.5-4B linear attention: q=16×128, k=16×128, v=32×128
    let conv_channels = QWEN35_4B_LINEAR_K_HEADS * QWEN35_4B_LINEAR_K_DIM * 2
        + QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM;
    let scale = 1.0 / (QWEN35_4B_HEAD_DIM as f32).sqrt();

    let mut group = c.benchmark_group("ops_qwen35_state");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(conv_channels as u64));
    group.bench_function("conv1d_decode_into", |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let conv_x = device_vec(&ctx, conv_channels).expect("failed to allocate conv input");
        let conv_weight = device_vec(&ctx, conv_channels * CONV_KERNEL_SIZE)
            .expect("failed to allocate conv weight");
        let mut conv_state = DeviceVec::zeros(&ctx, conv_channels * (CONV_KERNEL_SIZE - 1))
            .expect("failed to allocate conv state");
        let mut conv_out =
            DeviceVec::zeros(&ctx, conv_channels).expect("failed to allocate conv out");
        iter_sync(b, &ctx, || {
            ops::conv1d_decode_into(
                &ctx,
                &conv_x,
                &conv_weight,
                &mut conv_state,
                &mut conv_out,
                CONV_KERNEL_SIZE,
            )
            .expect("conv1d_decode_into failed");
        });
    });

    group.throughput(Throughput::Elements((conv_channels * ATTN_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("conv1d_prefill_into", ATTN_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x_seq =
            device_vec(&ctx, conv_channels * ATTN_SEQ_LEN).expect("failed to allocate conv seq");
        let conv_weight = device_vec(&ctx, conv_channels * CONV_KERNEL_SIZE)
            .expect("failed to allocate conv weight");
        let mut conv_state_prefill = DeviceVec::zeros(&ctx, conv_channels * (CONV_KERNEL_SIZE - 1))
            .expect("failed to allocate prefill conv state");
        let mut conv_prefill_out = DeviceVec::zeros(&ctx, conv_channels * ATTN_SEQ_LEN)
            .expect("failed to allocate prefill conv out");
        iter_sync(b, &ctx, || {
            ops::conv1d_prefill_into(
                &ctx,
                &x_seq,
                &conv_weight,
                &mut conv_state_prefill,
                &mut conv_prefill_out,
                conv_channels,
                ATTN_SEQ_LEN,
                CONV_KERNEL_SIZE,
            )
            .expect("conv1d_prefill_into failed");
        });
    });

    // GDR state: v_heads × k_dim × v_dim = 32 × 128 × 128 = 524288 f32
    group.throughput(Throughput::Elements(
        (QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_K_DIM * QWEN35_4B_LINEAR_V_DIM) as u64,
    ));
    group.bench_function("gated_delta_rule_decode_into", |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let qkv = device_vec(&ctx, conv_channels).expect("failed to allocate qkv");
        let b_proj =
            device_vec(&ctx, QWEN35_4B_LINEAR_V_HEADS).expect("failed to allocate b_proj");
        let a_proj =
            device_vec(&ctx, QWEN35_4B_LINEAR_V_HEADS).expect("failed to allocate a_proj");
        let dt_bias = positive_device_vec(&ctx, QWEN35_4B_LINEAR_V_HEADS)
            .expect("failed to allocate dt_bias");
        let a_log =
            f32_slice(&ctx, QWEN35_4B_LINEAR_V_HEADS).expect("failed to allocate a_log");
        let mut state = zero_f32_slice(
            &ctx,
            QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_K_DIM * QWEN35_4B_LINEAR_V_DIM,
        )
        .expect("failed to allocate recurrent state");
        let mut recurrent_out =
            DeviceVec::zeros(&ctx, QWEN35_4B_LINEAR_V_HEADS * QWEN35_4B_LINEAR_V_DIM)
                .expect("failed to allocate recurrent out");
        iter_sync(b, &ctx, || {
            ops::gated_delta_rule_decode_into(
                &ctx,
                &qkv,
                &b_proj,
                &a_proj,
                &dt_bias,
                &a_log,
                &mut state,
                &mut recurrent_out,
                QWEN35_4B_LINEAR_K_HEADS,
                QWEN35_4B_LINEAR_V_HEADS,
                QWEN35_4B_LINEAR_K_DIM,
                QWEN35_4B_LINEAR_V_DIM,
            )
            .expect("gated_delta_rule_decode_into failed");
        });
    });

    // Full attention decode: q=16×256, kv=4×256, rotary_dim=64
    let q_dim = QWEN35_4B_Q_HEADS * QWEN35_4B_HEAD_DIM;
    let kv_dim = QWEN35_4B_KV_HEADS * QWEN35_4B_HEAD_DIM;
    let cache_len = QWEN35_4B_KV_HEADS * MAX_SEQ_LEN * QWEN35_4B_HEAD_DIM;

    group.throughput(Throughput::Elements(q_dim as u64));
    group.bench_function(
        BenchmarkId::new("fused_attention_hd256_decode_into", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_full = device_vec(&ctx, q_dim * 2).expect("failed to allocate q_full (w/ gate)");
            let k_full = device_vec(&ctx, kv_dim).expect("failed to allocate k_full");
            let v_full = device_vec(&ctx, kv_dim).expect("failed to allocate v_full");
            let q_norm =
                positive_device_vec(&ctx, QWEN35_4B_HEAD_DIM).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, QWEN35_4B_HEAD_DIM).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROPE_THETA)
                    .expect("failed to create rope cache");
            let current_pos = ATTN_SEQ_LEN - 1;
            let decode_meta = decode_meta(&ctx, 7, current_pos, ATTN_SEQ_LEN)
                .expect("failed to allocate decode meta");
            let mut k_cache = DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache = DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut attn_out =
                DeviceVec::zeros(&ctx, q_dim).expect("failed to allocate attention out");
            iter_sync(b, &ctx, || {
                ops::fused_attention_hd256_decode_into(
                    &ctx,
                    &q_full,
                    &k_full,
                    &v_full,
                    &q_norm,
                    &k_norm,
                    &cos_cache,
                    &sin_cache,
                    &decode_meta,
                    &mut k_cache,
                    &mut v_cache,
                    &mut attn_out,
                    QWEN35_4B_Q_HEADS,
                    QWEN35_4B_KV_HEADS,
                    QWEN35_4B_ROTARY_DIM,
                    scale,
                    EPS,
                )
                .expect("fused_attention_hd256_decode_into failed");
            });
        },
    );

    group.throughput(Throughput::Elements(q_dim as u64));
    group.bench_function(
        BenchmarkId::new("fused_attention_hd256_single_token_into", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_full = device_vec(&ctx, q_dim * 2).expect("failed to allocate q_full (w/ gate)");
            let k_full = device_vec(&ctx, kv_dim).expect("failed to allocate k_full");
            let v_full = device_vec(&ctx, kv_dim).expect("failed to allocate v_full");
            let q_norm =
                positive_device_vec(&ctx, QWEN35_4B_HEAD_DIM).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, QWEN35_4B_HEAD_DIM).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROPE_THETA)
                    .expect("failed to create rope cache");
            let current_pos = ATTN_SEQ_LEN - 1;
            let cos_pos = cos_cache.view(current_pos * QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROTARY_DIM);
            let sin_pos = sin_cache.view(current_pos * QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROTARY_DIM);
            let mut k_cache = DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache = DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut attn_out =
                DeviceVec::zeros(&ctx, q_dim).expect("failed to allocate attention out");
            iter_sync(b, &ctx, || {
                ops::fused_attention_hd256_single_token_into(
                    &ctx,
                    &q_full,
                    &k_full,
                    &v_full,
                    &q_norm,
                    &k_norm,
                    &cos_pos,
                    &sin_pos,
                    &mut k_cache,
                    &mut v_cache,
                    &mut attn_out,
                    QWEN35_4B_Q_HEADS,
                    QWEN35_4B_KV_HEADS,
                    current_pos,
                    ATTN_SEQ_LEN,
                    QWEN35_4B_ROTARY_DIM,
                    scale,
                    EPS,
                )
                .expect("fused_attention_hd256_single_token_into failed");
            });
        },
    );

    group.finish();
}

/// Baseline for optimization #2: per-token full-attention prefill loop.
///
/// Measures seq_len calls to fused_attention_hd256_single_token_into with growing context
/// (pos 0..seq_len). This is the dominant bottleneck at 80% GPU time after #1.
/// Target: replace with 1 FA2 call per layer (8 calls total for 8 full-attn layers).
pub fn bench_qwen35_prefill_attn_ops(c: &mut Criterion) {
    let q_dim = QWEN35_4B_Q_HEADS * QWEN35_4B_HEAD_DIM;
    let kv_dim = QWEN35_4B_KV_HEADS * QWEN35_4B_HEAD_DIM;
    let cache_len = QWEN35_4B_KV_HEADS * MAX_SEQ_LEN * QWEN35_4B_HEAD_DIM;
    let scale = 1.0 / (QWEN35_4B_HEAD_DIM as f32).sqrt();

    for &seq_len in &[128usize, 512, 2048] {
        let mut group =
            c.benchmark_group(format!("qwen35_prefill_attn/seq{seq_len}"));
        configure_group(&mut group);
        group.throughput(Throughput::Elements((q_dim * seq_len) as u64));

        group.bench_function("single_token_loop", |b| {
            let ctx = DeviceContext::new().expect("ctx");
            // Pre-allocate fixed per-token buffers (no alloc inside loop)
            let q = device_vec(&ctx, q_dim * 2).expect("q");
            let k = device_vec(&ctx, kv_dim).expect("k");
            let v = device_vec(&ctx, kv_dim).expect("v");
            let q_norm = positive_device_vec(&ctx, QWEN35_4B_HEAD_DIM).expect("q_norm");
            let k_norm = positive_device_vec(&ctx, QWEN35_4B_HEAD_DIM).expect("k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROPE_THETA)
                    .expect("rope");
            let mut k_cache = DeviceVec::zeros(&ctx, cache_len).expect("k_cache");
            let mut v_cache = DeviceVec::zeros(&ctx, cache_len).expect("v_cache");
            let mut attn_out = DeviceVec::zeros(&ctx, q_dim).expect("attn_out");

            iter_sync(b, &ctx, || {
                for t in 0..seq_len {
                    let cos_pos =
                        cos_cache.view(t * QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROTARY_DIM);
                    let sin_pos =
                        sin_cache.view(t * QWEN35_4B_ROTARY_DIM, QWEN35_4B_ROTARY_DIM);
                    ops::fused_attention_hd256_single_token_into(
                        &ctx,
                        &q,
                        &k,
                        &v,
                        &q_norm,
                        &k_norm,
                        &cos_pos,
                        &sin_pos,
                        &mut k_cache,
                        &mut v_cache,
                        &mut attn_out,
                        QWEN35_4B_Q_HEADS,
                        QWEN35_4B_KV_HEADS,
                        t,
                        t + 1,
                        QWEN35_4B_ROTARY_DIM,
                        scale,
                        EPS,
                    )
                    .expect("attn failed");
                }
            });
        });

        group.finish();
    }
}
