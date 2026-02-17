use anyhow::Result;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer_path = format!("{}/tokenizer.json", path);
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

    #[test]
    fn test_load_tokenizer() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();

        let text = "Hello, world!";
        let ids = tokenizer.encode(text).unwrap();
        // ids: [9707, 11, 1879, 0]
        assert_eq!(ids, vec![9707, 11, 1879, 0]);

        let decoded = tokenizer.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_chinese() {
        let tokenizer = Tokenizer::from_file(MODEL_PATH).unwrap();

        let text = "你好，世界！";
        let ids = tokenizer.encode(text).unwrap();

        let decoded = tokenizer.decode(&ids).unwrap();

        assert_eq!(decoded, text);
    }
}
