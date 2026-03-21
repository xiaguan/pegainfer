use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec, HiddenStates};

use super::common::{
    ATTN_SEQ_LEN, EPS, HEAD_DIM_128, KV_HEADS_128, MAX_SEQ_LEN, Q_HEADS_128, ROPE_THETA_QWEN3,
    configure_group, decode_meta, device_vec, hidden_states, iter_sync, positive_device_vec,
    rope_cache, zero_f32_slice,
};

pub(crate) fn bench_attention_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_attention");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(
        (Q_HEADS_128 * HEAD_DIM_128 * ATTN_SEQ_LEN) as u64,
    ));
    group.bench_function(
        BenchmarkId::new("prefill_attention_batch", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_dim = Q_HEADS_128 * HEAD_DIM_128;
            let kv_dim = KV_HEADS_128 * HEAD_DIM_128;
            let mut q_batch =
                hidden_states(&ctx, q_dim, ATTN_SEQ_LEN).expect("failed to allocate q batch");
            let mut k_batch =
                hidden_states(&ctx, kv_dim, ATTN_SEQ_LEN).expect("failed to allocate k batch");
            let v_batch =
                hidden_states(&ctx, kv_dim, ATTN_SEQ_LEN).expect("failed to allocate v batch");
            let q_norm =
                positive_device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, HEAD_DIM_128, ROPE_THETA_QWEN3)
                    .expect("failed to create rope cache");
            let cache_len = KV_HEADS_128 * MAX_SEQ_LEN * HEAD_DIM_128;
            let mut k_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut output_batch = HiddenStates::zeros(&ctx, q_dim, ATTN_SEQ_LEN)
                .expect("failed to allocate attention batch out");
            iter_sync(b, &ctx, || {
                ops::prefill_attention_batch(
                    &ctx,
                    &mut q_batch,
                    &mut k_batch,
                    &v_batch,
                    &q_norm,
                    &k_norm,
                    &cos_cache,
                    &sin_cache,
                    &mut k_cache,
                    &mut v_cache,
                    &mut output_batch,
                    Q_HEADS_128,
                    KV_HEADS_128,
                    HEAD_DIM_128,
                    0,
                    EPS,
                )
                .expect("prefill_attention_batch failed");
            });
        },
    );

    group.throughput(Throughput::Elements((Q_HEADS_128 * HEAD_DIM_128) as u64));
    group.bench_function(
        BenchmarkId::new("fused_attention_decode_into", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_dim = Q_HEADS_128 * HEAD_DIM_128;
            let kv_dim = KV_HEADS_128 * HEAD_DIM_128;
            let q_full = device_vec(&ctx, q_dim).expect("failed to allocate q_full");
            let k_full = device_vec(&ctx, kv_dim).expect("failed to allocate k_full");
            let v_full = device_vec(&ctx, kv_dim).expect("failed to allocate v_full");
            let q_norm =
                positive_device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, HEAD_DIM_128, ROPE_THETA_QWEN3)
                    .expect("failed to create rope cache");
            let current_pos = ATTN_SEQ_LEN - 1;
            let decode_meta = decode_meta(&ctx, 13, current_pos, ATTN_SEQ_LEN)
                .expect("failed to allocate decode meta");
            let cache_len = KV_HEADS_128 * MAX_SEQ_LEN * HEAD_DIM_128;
            let mut k_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut fused_out =
                DeviceVec::zeros(&ctx, q_dim).expect("failed to allocate fused out");
            let num_kv_splits = 4usize;
            let mut partial_out = zero_f32_slice(&ctx, Q_HEADS_128 * num_kv_splits * HEAD_DIM_128)
                .expect("partial_out");
            let mut partial_m =
                zero_f32_slice(&ctx, Q_HEADS_128 * num_kv_splits).expect("partial_m");
            let mut partial_l =
                zero_f32_slice(&ctx, Q_HEADS_128 * num_kv_splits).expect("partial_l");
            iter_sync(b, &ctx, || {
                ops::fused_attention_decode_into(
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
                    &mut fused_out,
                    &mut partial_out,
                    &mut partial_m,
                    &mut partial_l,
                    Q_HEADS_128,
                    KV_HEADS_128,
                )
                .expect("fused_attention_decode_into failed");
            });
        },
    );

    group.finish();
}
