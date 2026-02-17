// HuggingFace tokenizers 0.22 (BPE) encode/decode throughput benchmark.
// Model: Qwen3-4B (vocab_size ~152K)
//
// CPU: AMD EPYC 7402 24-Core @ 2.8GHz (48 threads, single-threaded bench)
//
// Results (2025-02-12):
//
// | tokens |  encode time | encode thrpt  |  decode time | decode thrpt  |
// |--------|--------------|---------------|--------------|---------------|
// |    100 |      157 us  |  638 Ktok/s   |       17 us  |  5.9 Mtok/s   |
// |     1K |     1.57 ms  |  638 Ktok/s   |      173 us  |  5.8 Mtok/s   |
// |    10K |     23.0 ms  |  434 Ktok/s   |     1.80 ms  |  5.5 Mtok/s   |
// |    50K |      121 ms  |  415 Ktok/s   |      9.3 ms  |  5.4 Mtok/s   |
// |   100K |      204 ms  |  491 Ktok/s   |     19.6 ms  |  5.1 Mtok/s   |
// |   200K |      450 ms  |  444 Ktok/s   |     43.7 ms  |  4.6 Mtok/s   |

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenizers::Tokenizer;

const MODEL_PATH: &str = "models/Qwen3-4B/tokenizer.json";

// Target token counts to benchmark
const TOKEN_COUNTS: &[usize] = &[100, 1_000, 10_000, 50_000, 100_000, 200_000];

fn load_tokenizer() -> Tokenizer {
    Tokenizer::from_file(MODEL_PATH).expect("failed to load tokenizer")
}

/// Build a text that encodes to approximately `target_tokens` tokens
/// by repeating a seed paragraph and trimming.
fn make_text(tokenizer: &Tokenizer, target_tokens: usize) -> String {
    let seed = "The quick brown fox jumps over the lazy dog. \
                Rust is a systems programming language focused on safety and performance. \
                Large language models process text by splitting it into tokens. ";
    let seed_enc = tokenizer.encode(seed, false).unwrap();
    let seed_len = seed_enc.get_ids().len();

    // Over-allocate by 2x to ensure we have enough tokens after encoding the joined text
    let repeats = (target_tokens / seed_len) * 2 + 2;
    let long_text: String = seed.repeat(repeats);

    let enc = tokenizer.encode(long_text.as_str(), false).unwrap();
    assert!(
        enc.get_ids().len() >= target_tokens,
        "seed too short: got {} tokens, need {}",
        enc.get_ids().len(),
        target_tokens
    );
    let ids: Vec<u32> = enc.get_ids()[..target_tokens].to_vec();
    tokenizer.decode(&ids, true).unwrap()
}

/// Build a token id vector of exactly `n` tokens by cycling encoded seed text.
fn make_ids(tokenizer: &Tokenizer, n: usize) -> Vec<u32> {
    let seed = "The quick brown fox jumps over the lazy dog. \
                Rust is a systems programming language focused on safety and performance. ";
    let seed_enc = tokenizer.encode(seed, false).unwrap();
    let seed_ids = seed_enc.get_ids();
    seed_ids.iter().copied().cycle().take(n).collect()
}

fn bench_encode(c: &mut Criterion) {
    let tokenizer = load_tokenizer();

    let mut group = c.benchmark_group("encode");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    for &n in TOKEN_COUNTS {
        let text = make_text(&tokenizer, n);
        // Throughput in "tokens" (elements)
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &text, |b, text| {
            b.iter(|| {
                let enc = tokenizer.encode(black_box(text.as_str()), false).unwrap();
                black_box(enc);
            });
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let tokenizer = load_tokenizer();

    let mut group = c.benchmark_group("decode");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    for &n in TOKEN_COUNTS {
        let ids = make_ids(&tokenizer, n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &ids, |b, ids| {
            b.iter(|| {
                let text = tokenizer.decode(black_box(ids), true).unwrap();
                black_box(text);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
