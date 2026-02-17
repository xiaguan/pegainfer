//! Accuracy tests comparing rust-llm outputs against HuggingFace Transformers golden data

use anyhow::Result;
use rust_llm::model::Qwen3Model;
use serde::Deserialize;
use std::fs;

/// Golden reference data from HuggingFace Transformers
#[derive(Debug, Deserialize)]
struct GoldenData {
    name: String,
    prompt: String,
    reason: String,
    prompt_tokens: Vec<u32>,
    #[allow(dead_code)]
    model_path: String,
    #[allow(dead_code)]
    model_type: String,
    #[allow(dead_code)]
    torch_version: String,
    vocab_size: usize,
    logits: LogitsData,
}

#[derive(Debug, Deserialize)]
struct LogitsData {
    #[allow(dead_code)]
    shape: Vec<usize>,
    #[allow(dead_code)]
    dtype: String,
    data: String, // base64-encoded bf16 bytes
}

impl GoldenData {
    /// Load golden data from JSON file
    fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let data: GoldenData = serde_json::from_str(&content)?;
        Ok(data)
    }

    /// Decode logits from base64 BF16 to Vec<f32>
    fn decode_logits(&self) -> Result<Vec<f32>> {
        use base64::prelude::*;

        // Decode base64 to bytes
        let bytes = BASE64_STANDARD.decode(&self.logits.data)?;

        // BF16 is stored as u16 (2 bytes per value)
        assert_eq!(bytes.len(), self.vocab_size * 2, "Logits size mismatch");

        // Convert BF16 bytes to f32
        let mut logits = Vec::with_capacity(self.vocab_size);
        for chunk in bytes.chunks_exact(2) {
            let bf16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            // BF16 to F32: BF16 is F32 with lower 16 bits zeroed
            let f32_bits = (bf16_bits as u32) << 16;
            logits.push(f32::from_bits(f32_bits));
        }

        Ok(logits)
    }
}

/// Compute max absolute difference between two vectors
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(f32::NEG_INFINITY, f32::max)
}

/// Compute mean absolute difference
fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f32
}

/// Compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    dot / (norm_a * norm_b)
}

/// Assert logits are close with detailed error messages
fn assert_logits_close(
    rust_logits: &[f32],
    golden_logits: &[f32],
    test_name: &str,
    max_diff_threshold: f32,
    mean_diff_threshold: f32,
    cosine_threshold: f32,
) {
    let max_diff = max_abs_diff(rust_logits, golden_logits);
    let mean_diff = mean_abs_diff(rust_logits, golden_logits);
    let cosine = cosine_similarity(rust_logits, golden_logits);

    println!("\n{} Accuracy Metrics:", test_name);
    println!("  Max abs diff:       {:.6}", max_diff);
    println!("  Mean abs diff:      {:.6}", mean_diff);
    println!("  Cosine similarity:  {:.8}", cosine);

    // Find indices of top-5 largest diffs for debugging
    let mut diffs: Vec<(usize, f32)> = rust_logits
        .iter()
        .zip(golden_logits)
        .enumerate()
        .map(|(i, (r, g))| (i, (r - g).abs()))
        .collect();
    diffs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top-5 largest diffs:");
    for (i, diff) in diffs.iter().take(5) {
        println!(
            "    Token {:6}: rust={:8.4}, golden={:8.4}, diff={:.6}",
            i, rust_logits[*i], golden_logits[*i], diff
        );
    }

    // Assertions
    assert!(
        max_diff < max_diff_threshold,
        "{}: Max abs diff {:.6} exceeds threshold {:.6}",
        test_name,
        max_diff,
        max_diff_threshold
    );
    assert!(
        mean_diff < mean_diff_threshold,
        "{}: Mean abs diff {:.6} exceeds threshold {:.6}",
        test_name,
        mean_diff,
        mean_diff_threshold
    );
    assert!(
        cosine > cosine_threshold,
        "{}: Cosine similarity {:.8} below threshold {:.8}",
        test_name,
        cosine,
        cosine_threshold
    );

    println!("  ✓ All thresholds passed\n");
}

/// Run accuracy test for a single golden data file
fn test_accuracy_case(golden_path: &str, model_path: &str) -> Result<()> {
    // Load golden data
    let golden = GoldenData::load(golden_path)?;
    println!("\nTesting: {} ({})", golden.name, golden.reason);
    println!("  Prompt: {}", golden.prompt);
    println!("  Tokens: {:?}", golden.prompt_tokens);

    let golden_logits = golden.decode_logits()?;
    println!("  Loaded golden logits: {} values", golden_logits.len());

    // Load model and run inference
    let model = Qwen3Model::from_safetensors(model_path)?;
    println!("  Running rust-llm inference...");
    let rust_logits = model.forward_logits(&golden.prompt_tokens)?;
    println!("  Got rust-llm logits: {} values", rust_logits.len());

    assert_eq!(
        rust_logits.len(),
        golden_logits.len(),
        "Logits length mismatch"
    );

    // Compare with thresholds
    // Note: Thresholds account for:
    // - BF16 precision (~7-8 significant bits)
    // - GPU (cuBLAS) vs CPU (PyTorch) GEMM implementation differences
    // - Accumulation of numerical errors across 36 transformer layers
    assert_logits_close(
        &rust_logits,
        &golden_logits,
        &golden.name,
        0.8,      // max_abs_diff threshold (worst: 0.75 on minimal)
        0.12,     // mean_abs_diff threshold (worst: 0.111 on minimal)
        0.9997,   // cosine_similarity threshold (worst: 0.99976 on medium_en)
    );

    Ok(())
}

#[test]
#[ignore] // Requires GPU
fn test_accuracy_minimal() -> Result<()> {
    test_accuracy_case(
        "../golden_data/minimal.json",
        "../rust-llm/models/Qwen3-4B",
    )
}

#[test]
#[ignore] // Requires GPU
fn test_accuracy_short_en() -> Result<()> {
    test_accuracy_case(
        "../golden_data/short_en.json",
        "../rust-llm/models/Qwen3-4B",
    )
}

#[test]
#[ignore] // Requires GPU
fn test_accuracy_medium_en() -> Result<()> {
    test_accuracy_case(
        "../golden_data/medium_en.json",
        "../rust-llm/models/Qwen3-4B",
    )
}

#[test]
#[ignore] // Requires GPU
fn test_accuracy_long_en() -> Result<()> {
    test_accuracy_case(
        "../golden_data/long_en.json",
        "../rust-llm/models/Qwen3-4B",
    )
}

#[test]
#[ignore] // Requires GPU
fn test_accuracy_multilingual() -> Result<()> {
    test_accuracy_case(
        "../golden_data/multilingual.json",
        "../rust-llm/models/Qwen3-4B",
    )
}

/// Run all accuracy tests in sequence (avoids repeated model loading)
#[test]
#[ignore] // Requires GPU
fn test_accuracy_all() -> Result<()> {
    let model_path = "../rust-llm/models/Qwen3-4B";

    let test_cases = vec![
        ("../golden_data/minimal.json", "minimal"),
        ("../golden_data/short_en.json", "short_en"),
        ("../golden_data/medium_en.json", "medium_en"),
        ("../golden_data/long_en.json", "long_en"),
        ("../golden_data/multilingual.json", "multilingual"),
    ];

    println!("\n========================================");
    println!("Running all accuracy tests");
    println!("========================================");

    for (golden_path, name) in test_cases {
        println!("\n>>> Test case: {}", name);
        test_accuracy_case(golden_path, model_path)?;
    }

    println!("\n========================================");
    println!("✓ All accuracy tests passed!");
    println!("========================================\n");

    Ok(())
}
