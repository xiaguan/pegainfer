//! E2E test for paged attention decode path.
//!
//! Uses single-token prompts (BOS token) to bypass the prefill path,
//! exercising only the paged KV cache + FlashInfer decode kernel.

use pegainfer::model::{GenerationState, ModelForward, ModelRuntimeConfig, Qwen3Model};
use pegainfer::sampler::SamplingParams;
use rand::SeedableRng;
use rand::rngs::StdRng;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

fn get_model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| MODEL_PATH.to_string())
}

#[test]
fn paged_attention_single_token_decode() {
    let model_path = get_model_path();
    let model = Qwen3Model::from_safetensors_with_runtime(
        &model_path,
        ModelRuntimeConfig {
            enable_cuda_graph: true,
        },
    )
    .expect("Failed to load model");

    let mut state = model.create_state().expect("Failed to create state");
    let mut rng = StdRng::seed_from_u64(42);
    let params = SamplingParams {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        ignore_eos: true,
    };

    // Use BOS token as single-token "prompt" — goes through decode path, not prefill
    let bos_token = 151643u32; // Qwen3 BOS token

    // First forward: single token → decode path
    model
        .forward(&[bos_token], &mut state)
        .expect("First forward failed");
    let token1 = model
        .select_token(&mut state, &params, &mut rng)
        .expect("select_token failed");

    eprintln!("Token 1: {token1}");

    // Generate a few more tokens to exercise multi-step paged decode
    let mut tokens = vec![token1];
    for i in 0..9 {
        model
            .forward(&[*tokens.last().unwrap()], &mut state)
            .expect(&format!("Forward step {i} failed"));
        let next = model
            .select_token(&mut state, &params, &mut rng)
            .expect("select_token failed");
        eprintln!("Token {}: {next}", i + 2);
        tokens.push(next);
    }

    eprintln!("Generated 10 tokens from BOS: {:?}", tokens);

    // Basic sanity: should produce non-zero, valid tokens
    for &t in &tokens {
        assert!(t < 151936, "Token {t} out of vocab range");
    }

    // Verify determinism: reset and regenerate with same seed
    state.reset().expect("reset failed");
    let mut rng2 = StdRng::seed_from_u64(42);

    model
        .forward(&[bos_token], &mut state)
        .expect("Second forward failed");
    let token1_again = model
        .select_token(&mut state, &params, &mut rng2)
        .expect("select_token failed");
    assert_eq!(
        token1, token1_again,
        "Paged attention decode is not deterministic"
    );

    eprintln!("Determinism check passed");
}
