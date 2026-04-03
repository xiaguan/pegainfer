use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use cudarc::driver::CudaSlice;
use pegainfer::ops;
use pegainfer::sampler::SamplingParams;
use pegainfer::tensor::{DeviceContext, DeviceVec, HiddenStates};

use super::common::{
    BATCH_SEQ_LEN, VECTOR_DIM, VOCAB_SIZE, configure_group, decode_token_id, embedding_matrix,
    iter_sync, token_ids,
};

pub(crate) fn bench_embedding_sampling_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_embedding_sampling");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("embedding_decode_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let embed = embedding_matrix(&ctx, VOCAB_SIZE, VECTOR_DIM)
            .expect("failed to allocate embedding matrix");
        let token_id = 17_u32;
        let mut embed_out =
            DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate embedding out");
        let decode_token = decode_token_id(&ctx, token_id).expect("failed to allocate token id");
        iter_sync(b, &ctx, || {
            ops::embedding_decode_into(&ctx, &embed, &decode_token, &mut embed_out)
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

    group.throughput(Throughput::Elements(VOCAB_SIZE as u64));
    group.bench_function(BenchmarkId::new("gpu_sample_into", VOCAB_SIZE), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let logits = DeviceVec::zeros(&ctx, VOCAB_SIZE).expect("failed to allocate logits");
        let mut probs: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(VOCAB_SIZE)
            .expect("failed to allocate probs scratch");
        let mut out_gpu: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(1)
            .expect("failed to allocate output token");
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            ..Default::default()
        };
        iter_sync(b, &ctx, || {
            let token =
                ops::gpu_sample_into(&ctx, &logits, &mut probs, &mut out_gpu, &params, 0.37)
                    .expect("gpu_sample_into failed");
            black_box(token);
        });
    });

    group.finish();
}
