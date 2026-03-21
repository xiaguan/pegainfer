mod ops;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches,
    ops::ops_elementwise_bench::bench_elementwise_ops,
    ops::ops_embedding_sampling_bench::bench_embedding_sampling_ops,
    ops::ops_attention_bench::bench_attention_ops,
    ops::ops_batched_bench::bench_batched_ops,
    ops::ops_qwen35_norm_bench::bench_qwen35_norm_ops,
    ops::ops_qwen35_state_bench::bench_qwen35_state_ops,
    ops::ops_qwen35_state_bench::bench_qwen35_prefill_attn_ops,
    ops::ops_triton_bench::bench_triton_ops
);
criterion_main!(benches);
