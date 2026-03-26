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
    #[serde(skip)]
    pub stop_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct GenerationConfig {
    eos_token_id: EosTokenIds,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(token_id) => vec![token_id],
            Self::Multiple(token_ids) => token_ids,
        }
    }
}

impl Config {
    pub fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.stop_token_ids = Self::load_stop_token_ids(model_path, config.eos_token_id)?;
        Ok(config)
    }

    pub fn lm_head_tensor_name(&self) -> &'static str {
        if self.tie_word_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        }
    }

    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    fn load_stop_token_ids(model_path: &str, fallback_eos_token_id: u32) -> Result<Vec<u32>> {
        let generation_config_path = format!("{}/generation_config.json", model_path);
        match fs::read_to_string(&generation_config_path) {
            Ok(content) => {
                let generation_config: GenerationConfig = serde_json::from_str(&content)?;
                let mut stop_token_ids = generation_config.eos_token_id.into_vec();
                stop_token_ids.dedup();
                Ok(stop_token_ids)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok(vec![fallback_eos_token_id])
            }
            Err(err) => Err(err.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
    const MODEL_8B_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-8B");

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
        assert_eq!(config.stop_token_ids, vec![151645, 151643]);
        assert!(config.is_stop_token(151645));
        assert!(config.is_stop_token(151643));
        assert_eq!(config.lm_head_tensor_name(), "model.embed_tokens.weight");
    }

    #[test]
    fn test_config_gqa_ratio() {
        let config = Config::from_file(MODEL_PATH).unwrap();

        // GQA: Q heads / KV heads = 4, meaning 4 Q heads share 1 KV head
        let gqa_ratio = config.num_attention_heads / config.num_key_value_heads;
        assert_eq!(gqa_ratio, 4);

        assert_eq!(config.head_dim, 128);
    }

    #[test]
    #[ignore = "requires Qwen3-8B model"]
    fn test_load_8b_config() {
        let config = Config::from_file(MODEL_8B_PATH).unwrap();

        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.intermediate_size, 12288);
        assert_eq!(config.num_hidden_layers, 36);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1000000.0);
        assert_eq!(config.bos_token_id, 151643);
        assert_eq!(config.eos_token_id, 151645);
        assert!(!config.tie_word_embeddings);
        assert_eq!(config.stop_token_ids, vec![151645, 151643]);
        assert!(config.is_stop_token(151645));
        assert!(config.is_stop_token(151643));
        assert_eq!(config.lm_head_tensor_name(), "lm_head.weight");
    }
}
