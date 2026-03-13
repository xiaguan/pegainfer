/// Regenerate model-specific golden data in test_data/<model_name>.json using greedy decoding.
use std::path::{Path, PathBuf};

use pegainfer::sampler::SamplingParams;
use pegainfer::server_engine::{CompleteRequest, RealServerEngine, ServerEngine};

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum PromptStyle {
    Base,
    Instruct,
}

struct Case {
    name: &'static str,
    prompt: &'static str,
    max_new_tokens: usize,
}

const CASES: &[Case] = &[
    Case {
        name: "tell_story",
        prompt: "Tell me a story",
        max_new_tokens: 50,
    },
    Case {
        name: "my_name",
        prompt: "My name is",
        max_new_tokens: 50,
    },
    Case {
        name: "math",
        prompt: "What is 2 + 2?",
        max_new_tokens: 30,
    },
    Case {
        name: "chinese_weather",
        prompt: "今天天气真好",
        max_new_tokens: 50,
    },
    Case {
        name: "chinese_capital",
        prompt: "请介绍一下中国的首都",
        max_new_tokens: 50,
    },
    Case {
        name: "python_code",
        prompt: "Write a Python function to reverse a string",
        max_new_tokens: 50,
    },
];

fn model_path() -> String {
    std::env::var("PEGAINFER_E2E_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
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

fn prompt_style(model_name: &str) -> PromptStyle {
    if model_name.to_ascii_lowercase().contains("instruct") {
        PromptStyle::Instruct
    } else {
        PromptStyle::Base
    }
}

fn wrap_prompt(prompt: &str, style: PromptStyle) -> String {
    match style {
        PromptStyle::Base => prompt.to_string(),
        PromptStyle::Instruct => format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
    }
}

#[test]
fn regen_test_data() {
    pegainfer::logging::init_stderr("info");

    let model_path = model_path();
    let model_name = model_name(&model_path);
    let prompt_style = prompt_style(&model_name);
    let test_data_path = test_data_path(&model_name);

    eprintln!(
        "Regenerating golden data: model={}, path={}, prompt_style={:?}, output={}",
        model_name,
        model_path,
        prompt_style,
        test_data_path.display()
    );

    let mut engine = RealServerEngine::load(&model_path, 42).expect("Failed to load model");

    let mut cases_json = Vec::new();
    for case in CASES {
        let prompt = wrap_prompt(case.prompt, prompt_style);
        let req = CompleteRequest {
            prompt: prompt.clone(),
            max_tokens: case.max_new_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                ..Default::default()
            },
            stop: None,
        };
        let resp = engine.complete(req).expect("complete failed");
        let output = resp.text;
        eprintln!(
            "[{}] raw_prompt={:?} prompt={:?} output={:?}",
            case.name, case.prompt, prompt, output
        );
        cases_json.push(serde_json::json!({
            "name": case.name,
            "prompt": prompt,
            "max_new_tokens": case.max_new_tokens,
            "output": output,
        }));
    }

    let data = serde_json::json!({
        "model_name": model_name,
        "engine": "pegainfer",
        "cases": cases_json,
    });
    let json = serde_json::to_string_pretty(&data).unwrap();
    if let Some(parent) = test_data_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create test data directory");
    }
    std::fs::write(&test_data_path, json).expect("Failed to write test data");
    eprintln!("Wrote {}", test_data_path.display());
}
