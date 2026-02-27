//! Minimal decode benchmark for nsys profiling.
//!
//! Usage:
//!   nsys profile -o decode_trace --force-overwrite \
//!     cargo test -r --test bench_decode -- --nocapture
//!
//!   nsys stats decode_trace.nsys-rep

use pegainfer::model::{ModelRuntimeConfig, Qwen3Model};
use pegainfer::sampler::SamplingParams;
use pegainfer::tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const CUDA_GRAPH_PROMPT: &str = "Explain why LLM prefill and decode phases have different performance characteristics, and give one optimization strategy for each phase.";
const CUDA_GRAPH_WARMUP_ITERS: usize = 5;
const CUDA_GRAPH_BENCH_ITERS: usize = 20;
const CUDA_GRAPH_MAX_NEW_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug)]
struct RunMetrics {
    ttft: Duration,
    tbt_sum: Duration,
    tbt_count: usize,
    emitted_tokens: usize,
    elapsed: Duration,
}

#[derive(Debug, Default)]
struct BenchMetrics {
    ttft_sum: Duration,
    run_count: usize,
    tbt_sum: Duration,
    tbt_count: usize,
    emitted_tokens: usize,
    elapsed_sum: Duration,
}

impl BenchMetrics {
    fn add_run(&mut self, run: RunMetrics) {
        self.ttft_sum += run.ttft;
        self.run_count += 1;
        self.tbt_sum += run.tbt_sum;
        self.tbt_count += run.tbt_count;
        self.emitted_tokens += run.emitted_tokens;
        self.elapsed_sum += run.elapsed;
    }

    fn avg_ttft_ms(&self) -> f64 {
        if self.run_count == 0 {
            return 0.0;
        }
        (self.ttft_sum.as_secs_f64() * 1000.0) / self.run_count as f64
    }

    fn avg_tbt_ms(&self) -> f64 {
        if self.tbt_count == 0 {
            return 0.0;
        }
        (self.tbt_sum.as_secs_f64() * 1000.0) / self.tbt_count as f64
    }

    fn tps(&self) -> f64 {
        if self.emitted_tokens == 0 || self.elapsed_sum.is_zero() {
            return 0.0;
        }
        self.emitted_tokens as f64 / self.elapsed_sum.as_secs_f64()
    }
}

fn run_single_generation(
    model: &mut Qwen3Model,
    prompt_tokens: &[u32],
    sampling: &SamplingParams,
    rng: &mut StdRng,
) -> RunMetrics {
    let start = Instant::now();
    let mut first_token_at: Option<Instant> = None;
    let mut prev_token_at: Option<Instant> = None;
    let mut tbt_sum = Duration::ZERO;
    let mut tbt_count = 0usize;
    let mut emitted_tokens = 0usize;

    model
        .generate_streaming_with_callback(
            prompt_tokens,
            CUDA_GRAPH_MAX_NEW_TOKENS,
            sampling,
            rng,
            |_token_id| {
                let now = Instant::now();
                if first_token_at.is_none() {
                    first_token_at = Some(now);
                } else if let Some(prev) = prev_token_at {
                    tbt_sum += now - prev;
                    tbt_count += 1;
                }
                prev_token_at = Some(now);
                emitted_tokens += 1;
                true
            },
        )
        .expect("generation failed");

    let elapsed = start.elapsed();
    let ttft = first_token_at.map_or(elapsed, |t| t - start);

    RunMetrics {
        ttft,
        tbt_sum,
        tbt_count,
        emitted_tokens,
        elapsed,
    }
}

fn run_cuda_graph_benchmark(enable_cuda_graph: bool, prompt_tokens: &[u32]) -> BenchMetrics {
    let mut model = Qwen3Model::from_safetensors_with_runtime(
        MODEL_PATH,
        ModelRuntimeConfig { enable_cuda_graph },
    )
    .expect("Failed to load model");
    let sampling = SamplingParams::default();
    let mut rng = StdRng::seed_from_u64(42);

    eprintln!(
        "[warmup][cuda_graph={}] {} runs",
        enable_cuda_graph, CUDA_GRAPH_WARMUP_ITERS
    );
    for _ in 0..CUDA_GRAPH_WARMUP_ITERS {
        let _ = run_single_generation(&mut model, prompt_tokens, &sampling, &mut rng);
    }

    eprintln!(
        "[bench][cuda_graph={}] {} runs",
        enable_cuda_graph, CUDA_GRAPH_BENCH_ITERS
    );
    let mut metrics = BenchMetrics::default();
    for _ in 0..CUDA_GRAPH_BENCH_ITERS {
        metrics.add_run(run_single_generation(
            &mut model,
            prompt_tokens,
            &sampling,
            &mut rng,
        ));
    }
    metrics
}

#[test]
fn bench_decode_steps() {
    pegainfer::logging::init_stderr("warn");

    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    let mut model = Qwen3Model::from_safetensors(MODEL_PATH).expect("Failed to load model");

    let prompt = "Tell me a story";
    let prompt_tokens = tokenizer.encode(prompt).expect("encode failed");

    let mut rng = StdRng::seed_from_u64(42);
    let greedy = SamplingParams::default();

    // Warmup: prefill + a few decode steps (cuBLAS warmup, JIT, etc.)
    eprintln!("[warmup] prefill + 5 decode steps...");
    let warmup_tokens = model
        .generate(&prompt_tokens, 6, &greedy, &mut rng)
        .expect("warmup failed");
    eprintln!("[warmup] done, {} tokens total", warmup_tokens.len());

    // Now do a fresh generate with 20 decode steps for profiling
    eprintln!("[bench] prefill + 20 decode steps...");
    let start = Instant::now();
    let tokens = model
        .generate(&prompt_tokens, 21, &greedy, &mut rng)
        .expect("bench failed");
    let elapsed = start.elapsed();

    let decode_tokens = tokens.len() - prompt_tokens.len();
    let tpot = elapsed.as_secs_f64() * 1000.0 / decode_tokens as f64;
    eprintln!(
        "[bench] {} decode tokens in {:.2?}, TPOT={:.2}ms ({:.1} tok/s)",
        decode_tokens,
        elapsed,
        tpot,
        decode_tokens as f64 / elapsed.as_secs_f64()
    );
}

#[test]
#[ignore]
fn bench_cuda_graph_ttft_tbt_tps() {
    pegainfer::logging::init_stderr("warn");

    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    let prompt_tokens = tokenizer
        .encode(CUDA_GRAPH_PROMPT)
        .expect("Failed to encode prompt");

    eprintln!(
        "[bench] prompt_tokens={}, max_new_tokens={}, warmup={}, benchmark={}",
        prompt_tokens.len(),
        CUDA_GRAPH_MAX_NEW_TOKENS,
        CUDA_GRAPH_WARMUP_ITERS,
        CUDA_GRAPH_BENCH_ITERS
    );

    let metrics_with_graph = run_cuda_graph_benchmark(true, &prompt_tokens);
    let metrics_without_graph = run_cuda_graph_benchmark(false, &prompt_tokens);

    assert!(
        metrics_with_graph.emitted_tokens > 0 && metrics_without_graph.emitted_tokens > 0,
        "benchmark emitted zero tokens"
    );

    eprintln!(
        "[result][cuda_graph=true ] TTFT={:.2}ms, TBT={:.2}ms, TPS={:.1} tok/s",
        metrics_with_graph.avg_ttft_ms(),
        metrics_with_graph.avg_tbt_ms(),
        metrics_with_graph.tps()
    );
    eprintln!(
        "[result][cuda_graph=false] TTFT={:.2}ms, TBT={:.2}ms, TPS={:.1} tok/s",
        metrics_without_graph.avg_ttft_ms(),
        metrics_without_graph.avg_tbt_ms(),
        metrics_without_graph.tps()
    );
    if metrics_without_graph.tps() > 0.0 {
        eprintln!(
            "[result] TPS speedup (cuda_graph on/off): {:.2}x",
            metrics_with_graph.tps() / metrics_without_graph.tps()
        );
    }
}
