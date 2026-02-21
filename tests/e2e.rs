use std::path::PathBuf;
use std::time::Instant;

use fastrace::prelude::*;
use log::info;

use pegainfer::model::Qwen3Model;
use pegainfer::sampler::SamplingParams;
use pegainfer::tokenizer::Tokenizer;
use pegainfer::trace_reporter::FileReporter;
use rand::SeedableRng;
use rand::rngs::StdRng;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

fn init_logging() {
    pegainfer::logging::init_stderr("info");
}

fn init_tracing() -> PathBuf {
    let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("traces");
    std::fs::create_dir_all(&trace_dir).expect("Failed to create traces dir");
    fastrace::set_reporter(
        FileReporter::new(trace_dir.clone()),
        fastrace::collector::Config::default(),
    );
    info!("Tracing enabled: {}", trace_dir.display());
    trace_dir
}

#[test]
fn test_e2e_generation() {
    init_logging();
    let trace_dir = init_tracing();

    info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");

    info!("Loading GPU model...");
    let start = Instant::now();
    let model = Qwen3Model::from_safetensors(MODEL_PATH).expect("Failed to load GPU model");
    info!("Model loaded in {:.2?}", start.elapsed());

    let cases = [
        // English narrative
        ("Tell me a story", 100),
        ("My name is", 100),
        // Math
        ("What is 2 + 2?", 30),
        // Chinese
        ("今天天气真好", 50),
        ("请介绍一下中国的首都", 50),
        // Code
        ("Write a Python function to reverse a string", 80),
    ];

    let mut rng = StdRng::seed_from_u64(42);
    let greedy = SamplingParams::default();

    for (prompt, max_tokens) in &cases {
        info!("=== Test: \"{}\" ===", prompt);

        let root = Span::root("generate", SpanContext::random());
        let _guard = root.set_local_parent();

        let prompt_tokens = tokenizer.encode(prompt).expect("Failed to encode");
        info!("Prompt tokens: {:?}", prompt_tokens);

        let start = Instant::now();
        let output_tokens = model
            .generate(&prompt_tokens, *max_tokens, &greedy, &mut rng)
            .expect("Generation failed");
        let elapsed = start.elapsed();

        let new_tokens = &output_tokens[prompt_tokens.len()..];
        let generated_text = tokenizer.decode(new_tokens).expect("Failed to decode");

        let tokens_per_sec = new_tokens.len() as f64 / elapsed.as_secs_f64();
        info!(
            "Generated {} tokens in {:.2?} ({:.1} tok/s)",
            new_tokens.len(),
            elapsed,
            tokens_per_sec
        );
        info!("Output: \"{}\"", generated_text);

        assert!(
            !new_tokens.is_empty(),
            "No tokens generated for prompt: {}",
            prompt
        );
    }

    fastrace::flush();
    info!("Traces written to {}", trace_dir.display());
}
