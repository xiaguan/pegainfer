use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec, HiddenStates};

use super::common::{
    ATTN_SEQ_LEN, BATCH_SEQ_LEN, HEAD_DIM_128, KV_HEADS_128, MAX_SEQ_LEN, Q_HEADS_128,
    ROPE_THETA_QWEN3, VECTOR_DIM, VOCAB_SIZE, configure_group, decode_meta, device_vec,
    embedding_matrix, hidden_states, iter_sync, positive_device_vec, rope_cache, token_ids,
    zero_f32_slice,
};

pub(crate) fn bench_triton_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_triton");
    configure_group(&mut group);

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("add_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let add_batch_a = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate add batch lhs");
        let add_batch_b = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate add batch rhs");
        iter_sync(b, &ctx, || {
            let out = ops::add_batch(&ctx, &add_batch_a, &add_batch_b).expect("add_batch failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("silu_mul_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let gate = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate gate");
        let up = hidden_states(&ctx, VECTOR_DIM, BATCH_SEQ_LEN).expect("failed to allocate up");
        iter_sync(b, &ctx, || {
            let out = ops::silu_mul_batch(&ctx, &gate, &up).expect("silu_mul_batch failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("embedding_decode_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_id = 17_u32;
        let mut embed_out =
            DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate embedding out");
        let decode_meta_embed =
            decode_meta(&ctx, token_id as i32, 0, 1).expect("failed to allocate decode meta");
        iter_sync(b, &ctx, || {
            ops::embedding_decode_into(&ctx, &embed, &decode_meta_embed, &mut embed_out)
                .expect("embedding_decode_into failed");
        });
    });

    group.throughput(Throughput::Elements((VECTOR_DIM * BATCH_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("embedding_batch", BATCH_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_ids_gpu =
            token_ids(&ctx, BATCH_SEQ_LEN, VOCAB_SIZE).expect("failed to allocate token ids");
        let mut embed_batch_out = HiddenStates::zeros(&ctx, VECTOR_DIM, BATCH_SEQ_LEN)
            .expect("failed to allocate batched embedding out");
        iter_sync(b, &ctx, || {
            ops::embedding_batch(&ctx, &embed, &token_ids_gpu, &mut embed_batch_out)
                .expect("embedding_batch failed");
        });
    });

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
            let decode_meta_attn = decode_meta(&ctx, 13, current_pos, ATTN_SEQ_LEN)
                .expect("failed to allocate attention decode meta");
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
                    &decode_meta_attn,
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
