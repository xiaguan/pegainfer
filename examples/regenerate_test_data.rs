//! Regenerate test_data/Qwen3-4B.json using pegainfer's greedy output.
//!
//! Uses the scheduler path (with CUDA Graph) to match the e2e test execution.
//! Run: cargo run -r --example regenerate_test_data

use pegainfer::model::{ModelRuntimeConfig, Qwen3Model};
use pegainfer::sampler::SamplingParams;
use pegainfer::scheduler::{self, SchedulerRequest, TokenEvent};
use pegainfer::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const OUTPUT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/Qwen3-4B.json");

#[derive(Deserialize)]
struct InputData {
    model_name: String,
    cases: Vec<InputCase>,
}

#[derive(Deserialize)]
struct InputCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    #[allow(dead_code)]
    output: String,
}

#[derive(Serialize)]
struct OutputData {
    model_name: String,
    engine: String,
    cases: Vec<OutputCase>,
}

#[derive(Serialize)]
struct OutputCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    output: String,
}

fn main() {
    pegainfer::logging::init_stderr("info");

    let content = std::fs::read_to_string(OUTPUT_PATH).expect("Failed to read test data");
    let input: InputData = serde_json::from_str(&content).expect("Failed to parse JSON");

    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    let model = Qwen3Model::from_safetensors_with_runtime(
        MODEL_PATH,
        ModelRuntimeConfig {
            enable_cuda_graph: true,
        },
    )
    .expect("Failed to load model");

    let handle = scheduler::start(model, 42).expect("Failed to start scheduler");

    let mut cases = Vec::new();
    for case in &input.cases {
        let prompt_tokens = tokenizer.encode(&case.prompt).expect("encode failed");
        let (token_tx, mut token_rx) = mpsc::unbounded_channel();

        handle
            .submit(SchedulerRequest {
                prompt_tokens,
                params: SamplingParams::default(),
                max_tokens: case.max_new_tokens,
                token_tx,
                logprobs: 0,
                echo: false,
            })
            .expect("submit failed");

        let mut tokens = Vec::new();
        loop {
            match token_rx.blocking_recv() {
                Some(TokenEvent::Token { id, .. }) => tokens.push(id),
                Some(TokenEvent::PromptTokens { .. }) => {}
                Some(TokenEvent::Finished { .. }) => break,
                None => panic!("scheduler channel closed without Finished"),
            }
        }

        let output_text = tokenizer.decode(&tokens).expect("decode failed");

        eprintln!("[{}] prompt={:?}", case.name, case.prompt);
        eprintln!("  output={:?}", output_text);

        cases.push(OutputCase {
            name: case.name.clone(),
            prompt: case.prompt.clone(),
            max_new_tokens: case.max_new_tokens,
            output: output_text,
        });
    }

    // Drop handle to stop scheduler before writing
    drop(handle);

    let output = OutputData {
        model_name: input.model_name,
        engine: "pegainfer".to_string(),
        cases,
    };

    let json = serde_json::to_string_pretty(&output).expect("serialize failed");
    std::fs::write(OUTPUT_PATH, json).expect("Failed to write test data");
    eprintln!("\nWrote {OUTPUT_PATH}");
}
