/// E2E greedy regression test for Qwen3.5-4B.
///
/// Loads the model, runs greedy decoding on the standard test cases, and
/// compares against golden data in `test_data/Qwen3.5-4B.json`.
use std::path::{Path, PathBuf};

use log::info;
use pegainfer::model::{GenerationState, ModelForward, Qwen35Model};
use pegainfer::sampler::SamplingParams;
use pegainfer::tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");

fn get_model_path() -> String {
    let path = std::env::var("PEGAINFER_E2E_MODEL_PATH")
        .unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
    info!("Using model path: {}", path);
    path
}

fn get_test_data_path(model_path: &str) -> PathBuf {
    let name = Path::new(model_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(model_path);
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join(format!("{name}.json"));
    info!("Using test data path: {}", p.display());
    p
}

#[derive(Deserialize)]
struct TestData {
    cases: Vec<TestCase>,
}

#[derive(Deserialize)]
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

fn generate_text<M: ModelForward>(
    model: &M,
    state: &mut M::State,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    rng: &mut StdRng,
) -> String {
    let tokens = tokenizer.encode(prompt).expect("encode failed");
    state.reset().expect("reset failed");
    model.forward(&tokens, state).expect("prefill failed");
    let sampling = SamplingParams::default();

    let mut out = Vec::new();
    let mut last = model.select_token(state, &sampling, rng).expect("select_token");
    out.push(last);

    for _ in 1..max_tokens {
        if model.is_stop_token(last) {
            break;
        }
        model.forward(&[last], state).expect("decode failed");
        last = model.select_token(state, &sampling, rng).expect("select_token");
        out.push(last);
    }

    tokenizer.decode(&out).expect("decode failed")
}

#[test]
fn test_e2e_qwen35_generation() {
    pegainfer::logging::init_stderr("info");

    let model_path = get_model_path();
    let test_data_path = get_test_data_path(&model_path);
    let cases = load_test_cases(&test_data_path);

    info!("Loading Qwen3.5 model...");
    let model =
        Qwen35Model::from_safetensors_with_options(&model_path, /*enable_cuda_graph=*/ true)
            .expect("Failed to load model");
    let mut state = model.create_state().expect("Failed to create state");
    let tokenizer = Tokenizer::from_file(&model_path).expect("Failed to load tokenizer");
    let mut rng = StdRng::seed_from_u64(42);
    info!("Model loaded");

    info!("=== Qwen3.5-4B greedy correctness ===");
    let mut pass = 0;
    let mut fail = 0;
    for case in &cases {
        let output = generate_text(&model, &mut state, &tokenizer, &case.prompt, case.max_new_tokens, &mut rng);
        if output == case.output {
            info!("  PASS: {:?}", case.name);
            pass += 1;
        } else {
            eprintln!("  FAIL: {:?}", case.name);
            eprintln!("    expected: {:?}", case.output);
            eprintln!("    got:      {:?}", output);
            fail += 1;
        }
    }

    assert_eq!(fail, 0, "{fail} / {} cases failed (see output above)", pass + fail);
    info!("All {} cases passed", pass);
}
