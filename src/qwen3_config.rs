use anyhow::Result;
use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

    #[test]
    fn test_load_config() {
        let config = Config::from_file(MODEL_PATH).unwrap();

        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1000000.0);
        assert_eq!(config.bos_token_id, 151643);
        assert_eq!(config.eos_token_id, 151645);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_config_gqa_ratio() {
        let config = Config::from_file(MODEL_PATH).unwrap();

        // GQA: Q heads / KV heads = 4, meaning 4 Q heads share 1 KV head
        let gqa_ratio = config.num_attention_heads / config.num_key_value_heads;
        assert_eq!(gqa_ratio, 4);

        assert_eq!(config.head_dim, 128);
    }
}
