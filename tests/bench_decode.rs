//! Minimal decode benchmark for nsys profiling.
//!
//! Usage:
//!   nsys profile -o decode_trace --force-overwrite \
//!     cargo test -r --test bench_decode -- --nocapture
//!
//!   nsys stats decode_trace.nsys-rep

use rust_llm::model::Qwen3Model;
use rust_llm::tokenizer::Tokenizer;
use std::time::Instant;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[test]
fn bench_decode_steps() {
    rust_llm::logging::init_stderr("warn");

    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    let model = Qwen3Model::from_safetensors(MODEL_PATH).expect("Failed to load model");

    let prompt = "Tell me a story";
    let prompt_tokens = tokenizer.encode(prompt).expect("encode failed");

    // Warmup: prefill + a few decode steps (cuBLAS warmup, JIT, etc.)
    eprintln!("[warmup] prefill + 5 decode steps...");
    let warmup_tokens = model.generate(&prompt_tokens, 6).expect("warmup failed");
    eprintln!("[warmup] done, {} tokens total", warmup_tokens.len());

    // Now do a fresh generate with 20 decode steps for profiling
    eprintln!("[bench] prefill + 20 decode steps...");
    let start = Instant::now();
    let tokens = model.generate(&prompt_tokens, 21).expect("bench failed");
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
