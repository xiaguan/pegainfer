/// Generate greedy reference outputs from our own Qwen3.5 engine.
/// Run with: cargo test -r --test gen_test_data_35 -- --nocapture
use pegainfer::sampler::SamplingParams;
use pegainfer::server_engine::{CompleteRequest, EngineOptions, Qwen35ServerEngine, ServerEngine};
use serde::Serialize;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3.5-4B");
const OUTPUT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/Qwen3.5-4B.json");

#[derive(Serialize)]
struct TestData {
    cases: Vec<TestCase>,
}

#[derive(Serialize)]
struct TestCase {
    name: String,
    prompt: String,
    max_new_tokens: usize,
    output: String,
}

#[test]
fn generate_qwen35_test_data() {
    pegainfer::logging::init_stderr("info");

    let mut engine = Qwen35ServerEngine::load_with_options(
        MODEL_PATH,
        42,
        EngineOptions {
            enable_cuda_graph: false,
        },
    )
    .expect("Failed to load engine");

    let prompts: Vec<(&str, &str, usize)> = vec![
        // English
        ("hello", "Hello", 50),
        ("capital_france", "The capital of France is", 30),
        ("quick_fox", "The quick brown fox", 50),
        ("tell_story", "Tell me a story", 50),
        (
            "python_prime",
            "Write a Python function to check if a number is prime.",
            80,
        ),
        (
            "quantum_simple",
            "Explain quantum computing in simple terms.",
            80,
        ),
        // Math
        ("math_add", "What is 2+2?", 30),
        ("math_multiply", "What is 12 times 15?", 30),
        ("math_sqrt", "What is the square root of 144?", 30),
        // Chinese
        ("chinese_capital", "中国的首都是哪里？", 50),
        ("chinese_weather", "今天天气真好", 50),
        ("chinese_math", "请计算：7乘以8等于多少？", 30),
        ("chinese_translate", "请把Hello World翻译成中文", 30),
    ];

    let mut cases = Vec::new();
    for (name, prompt, max_tokens) in &prompts {
        eprintln!("\n--- {name}: \"{prompt}\" (max_tokens={max_tokens}) ---");
        let out = engine
            .complete(CompleteRequest {
                prompt: (*prompt).to_string(),
                max_tokens: *max_tokens,
                sampling: SamplingParams::default(),
                stop: None,
            })
            .expect("complete() failed");

        eprintln!(
            "  tokens: {}, finish: {:?}",
            out.usage.completion_tokens, out.finish_reason
        );
        let preview: String = out.text.chars().take(80).collect();
        eprintln!("  output: {:?}", preview);

        cases.push(TestCase {
            name: (*name).to_string(),
            prompt: (*prompt).to_string(),
            max_new_tokens: *max_tokens,
            output: out.text,
        });
    }

    let data = TestData { cases };
    let json = serde_json::to_string_pretty(&data).unwrap();
    std::fs::write(OUTPUT_PATH, &json).expect("Failed to write test data");
    eprintln!("\nWrote {OUTPUT_PATH} ({} cases)", prompts.len());
}
