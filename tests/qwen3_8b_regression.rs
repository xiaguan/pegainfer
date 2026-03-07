use std::collections::BTreeSet;

use pegainfer::qwen3_config::Config;
use pegainfer::tokenizer::Tokenizer;
use pegainfer::weight_loader::load_shard_info;

const MODEL_4B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const MODEL_8B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-8B");

#[test]
fn test_qwen3_8b_uses_untied_lm_head_metadata() {
    let config_4b = Config::from_file(MODEL_4B_PATH).expect("load 4B config");
    let config_8b = Config::from_file(MODEL_8B_PATH).expect("load 8B config");
    let (_, weights_4b) = load_shard_info(MODEL_4B_PATH).expect("load 4B shard info");
    let (_, weights_8b) = load_shard_info(MODEL_8B_PATH).expect("load 8B shard info");

    assert!(config_4b.tie_word_embeddings);
    assert!(!config_8b.tie_word_embeddings);
    assert_eq!(config_4b.lm_head_tensor_name(), "model.embed_tokens.weight");
    assert_eq!(config_8b.lm_head_tensor_name(), "lm_head.weight");

    assert!(weights_4b.contains_key("model.embed_tokens.weight"));
    assert!(!weights_4b.contains_key("lm_head.weight"));
    assert!(weights_8b.contains_key("model.embed_tokens.weight"));
    assert!(weights_8b.contains_key("lm_head.weight"));
}

#[test]
fn test_qwen3_8b_only_adds_lm_head_weight_vs_4b() {
    let (_, weights_4b) = load_shard_info(MODEL_4B_PATH).expect("load 4B shard info");
    let (_, weights_8b) = load_shard_info(MODEL_8B_PATH).expect("load 8B shard info");

    let weights_4b: BTreeSet<_> = weights_4b.keys().cloned().collect();
    let weights_8b: BTreeSet<_> = weights_8b.keys().cloned().collect();
    let extra_in_8b: Vec<_> = weights_8b.difference(&weights_4b).cloned().collect();

    assert_eq!(extra_in_8b, vec!["lm_head.weight".to_string()]);
}

#[test]
fn test_qwen3_4b_and_8b_tokenizers_stay_compatible() {
    let tokenizer_4b = Tokenizer::from_file(MODEL_4B_PATH).expect("load 4B tokenizer");
    let tokenizer_8b = Tokenizer::from_file(MODEL_8B_PATH).expect("load 8B tokenizer");
    let sample = "Rust 和 CUDA are fun";

    assert_eq!(tokenizer_4b.vocab_size(), tokenizer_8b.vocab_size());
    assert_eq!(
        tokenizer_4b
            .encode(sample)
            .expect("encode with 4B tokenizer"),
        tokenizer_8b
            .encode(sample)
            .expect("encode with 8B tokenizer")
    );
}
