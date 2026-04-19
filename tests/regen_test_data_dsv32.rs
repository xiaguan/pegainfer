/// Regenerate DSV3.2 greedy golden data in `test_data/DeepSeek-V3.2.json`
/// from the 44-prompt SGLang corpus using pegainfer's current scheduler path.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use pegainfer::model::DsV32Executor;
use pegainfer::sampler::SamplingParams;
use pegainfer::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent};
use pegainfer::tokenizer::Tokenizer;
use serde::Deserialize;
use tokio::sync::mpsc;

const DEFAULT_MODEL_PATH: &str = "/data/models/DeepSeek-V3.2";
const DEFAULT_MANIFEST_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/dsv32_sglang_ref/manifest.json"
);
const EXPECTED_SCHEMA: &str = "dsv32_sglang_ref.v1";

#[derive(Debug, Deserialize)]
struct SglangManifest {
    schema_version: String,
    cases: Vec<SglangCase>,
}

#[derive(Debug, Deserialize)]
struct SglangCase {
    name: String,
    prompt: String,
    generated_token_ids: Vec<u32>,
}

fn model_name(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

fn test_data_path(model_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(format!("{model_name}.json"))
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

fn generate_text(
    handle: &SchedulerHandle,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> String {
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
            Some(TokenEvent::Finished { .. }) => break,
            None => panic!("scheduler closed"),
        }
    }

    tokenizer.decode(&tokens).expect("decode failed")
}

fn write_golden_json(output_path: &Path, model_name: &str, cases_json: &[serde_json::Value]) {
    let data = serde_json::json!({
        "model_name": model_name,
        "engine": "pegainfer",
        "cases": cases_json,
    });
    let json = serde_json::to_string_pretty(&data).unwrap();
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create test data directory");
    }
    std::fs::write(output_path, &json).expect("Failed to write test data");
    eprintln!("Wrote {}", output_path.display());
}

#[test]
#[ignore = "regenerates checked-in DSV3.2 JSON golden data from pegainfer greedy output"]
fn regen_test_data_dsv32() {
    pegainfer::logging::init_stderr("info");

    let model_path = std::env::var("PEGAINFER_DSV32_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    let manifest_path = std::env::var("PEGAINFER_DSV32_SGLANG_REF_MANIFEST")
        .unwrap_or_else(|_| DEFAULT_MANIFEST_PATH.to_string());
    let model_name = model_name(&model_path);
    let output_path = test_data_path(&model_name);

    let manifest_content = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", manifest_path));
    let manifest: SglangManifest =
        serde_json::from_str(&manifest_content).expect("Failed to parse manifest JSON");
    assert_eq!(
        manifest.schema_version, EXPECTED_SCHEMA,
        "unexpected manifest schema version"
    );
    assert!(
        !manifest.cases.is_empty(),
        "manifest contains no cases: {}",
        manifest_path
    );

    let device_ordinals = parse_device_ordinals();
    assert_eq!(
        device_ordinals.len(),
        8,
        "regen_test_data_dsv32 expects 8 device ordinals, got {device_ordinals:?}"
    );
    let tp_size = parse_tp_size();
    assert_eq!(
        device_ordinals.len() % tp_size,
        0,
        "invalid parallel config: world_size={} tp_size={tp_size}",
        device_ordinals.len()
    );

    let tokenizer = Arc::new(Tokenizer::from_file(&model_path).expect("Failed to load tokenizer"));
    let executor = DsV32Executor::load(&model_path, &device_ordinals, tp_size)
        .unwrap_or_else(|e| panic!("DsV32Executor::load failed (tp_size={tp_size}): {e}"));
    let handle =
        pegainfer::scheduler_dsv32::start(executor, 42).expect("Failed to start DSV3.2 scheduler");

    let mut cases_json = Vec::with_capacity(manifest.cases.len());
    for case in manifest.cases {
        let max_new_tokens = case.generated_token_ids.len();
        let output = generate_text(&handle, &tokenizer, &case.prompt, max_new_tokens);
        eprintln!(
            "[{}] max_new_tokens={} output={:?}",
            case.name, max_new_tokens, output
        );
        cases_json.push(serde_json::json!({
            "name": case.name,
            "prompt": case.prompt,
            "max_new_tokens": max_new_tokens,
            "output": output,
        }));
    }

    write_golden_json(&output_path, &model_name, &cases_json);
}
