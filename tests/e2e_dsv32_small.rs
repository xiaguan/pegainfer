/// DSV3.2 greedy regression test against checked-in JSON golden data.
///
/// This is the small/frequent E2E path: run the real DSV3.2 scheduler over
/// the full 44-prompt fixture set and compare the final decoded string only.
/// The long teacher-forced top-K logprob harness lives in `tests/e2e_dsv32.rs`
/// and remains the deeper alignment check.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use log::info;
use pegainfer::model::DsV32Executor;
use pegainfer::sampler::SamplingParams;
use pegainfer::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent};
use pegainfer::server_engine::FinishReason;
use pegainfer::tokenizer::Tokenizer;
use serde::Deserialize;
use tokio::sync::mpsc;

const DEFAULT_MODEL_PATH: &str = "/data/models/DeepSeek-V3.2";

fn get_model_path() -> String {
    let path = std::env::var("PEGAINFER_DSV32_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    info!("Using model path: {}", path);
    path
}

fn get_test_data_path(model_path: &str) -> PathBuf {
    let model_name = Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path);
    let test_data_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(format!("{model_name}.json"));
    info!("Using test data path: {}", test_data_path.display());
    test_data_path
}

#[derive(Debug, Deserialize)]
struct TestData {
    cases: Vec<TestCase>,
}

#[derive(Debug, Deserialize)]
struct TestCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    output: String,
}

fn load_test_cases(path: &Path) -> Vec<TestCase> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let data: TestData = serde_json::from_str(&content).expect("Failed to parse JSON");
    data.cases
}

fn parse_device_ordinals() -> Vec<usize> {
    let raw = std::env::var("PEGAINFER_DSV32_DEVICE_ORDINALS")
        .unwrap_or_else(|_| "0,1,2,3,4,5,6,7".to_string());
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid device ordinal `{s}`"))
        })
        .collect()
}

fn parse_tp_size() -> usize {
    std::env::var("PEGAINFER_DSV32_TP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1)
}

fn parse_case_filter() -> Option<String> {
    std::env::var("PEGAINFER_DSV32_CASE_FILTER")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_max_cases() -> Option<usize> {
    std::env::var("PEGAINFER_DSV32_MAX_CASES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0)
}

fn init_logging() {
    pegainfer::logging::init_stderr("info");
}

fn generate_tokens(
    handle: &SchedulerHandle,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let prompt_tokens = tokenizer.encode(prompt).expect("encode failed");
    let (token_tx, mut token_rx) = mpsc::unbounded_channel();

    handle
        .submit(SchedulerRequest {
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
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
            Some(TokenEvent::Finished { finish_reason, .. }) => {
                return (tokens, finish_reason);
            }
            None => panic!("scheduler channel closed without Finished"),
        }
    }
}

#[test]
#[ignore = "requires 8 GPUs and DeepSeek-V3.2 weights; run manually on H20"]
fn test_e2e_dsv32_small_generation() {
    init_logging();

    let model_path = get_model_path();
    let test_data_path = get_test_data_path(&model_path);
    let mut cases = load_test_cases(&test_data_path);

    let case_filter = parse_case_filter();
    if let Some(filter) = &case_filter {
        cases.retain(|case| case.name.contains(filter));
    }
    if let Some(max_cases) = parse_max_cases() {
        cases.truncate(max_cases);
    }
    assert!(
        !cases.is_empty(),
        "no cases matched filter {:?} in {}",
        case_filter,
        test_data_path.display()
    );

    let device_ordinals = parse_device_ordinals();
    assert_eq!(
        device_ordinals.len(),
        8,
        "test_e2e_dsv32_small_generation expects 8 device ordinals, got {device_ordinals:?}"
    );
    let tp_size = parse_tp_size();
    assert_eq!(
        device_ordinals.len() % tp_size,
        0,
        "invalid parallel config: world_size={} tp_size={tp_size}",
        device_ordinals.len()
    );

    info!("Loading DeepSeek-V3.2 executor...");
    let start = Instant::now();
    let tokenizer = Arc::new(Tokenizer::from_file(&model_path).expect("Failed to load tokenizer"));
    let executor = DsV32Executor::load(&model_path, &device_ordinals, tp_size)
        .unwrap_or_else(|e| panic!("DsV32Executor::load failed (tp_size={tp_size}): {e}"));
    let handle =
        pegainfer::scheduler_dsv32::start(executor, 42).expect("Failed to start DSV3.2 scheduler");
    info!("Engine loaded in {:.2?}", start.elapsed());

    info!("=== DSV3.2 greedy correctness ===");
    let mut failures = Vec::new();
    for case in &cases {
        info!("--- [{}] {} ---", case.name, case.prompt);
        let start = Instant::now();
        let (tokens, finish_reason) =
            generate_tokens(&handle, &tokenizer, &case.prompt, case.max_new_tokens);
        let elapsed = start.elapsed();
        let text = tokenizer.decode(&tokens).expect("decode failed");
        let tok_s = tokens.len() as f64 / elapsed.as_secs_f64();

        info!(
            "  {} tokens in {:.2?} ({:.1} tok/s) finish={:?}",
            tokens.len(),
            elapsed,
            tok_s,
            finish_reason
        );

        if text.is_empty() {
            failures.push(format!(
                "case {} produced empty output (prompt={:?})",
                case.name, case.prompt
            ));
            continue;
        }
        if tokens.len() >= case.max_new_tokens && finish_reason != FinishReason::Length {
            failures.push(format!(
                "case {} should finish by length when emitting max_new_tokens: got {:?}",
                case.name, finish_reason
            ));
        }
        if text != case.output {
            failures.push(format!(
                "greedy output mismatch for case {}\n  prompt:    {:?}\n  got:      {:?}\n  expected: {:?}",
                case.name, case.prompt, text, case.output
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "DSV3.2 small E2E mismatches: {} case(s)\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}
