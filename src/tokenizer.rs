use anyhow::{Result, anyhow};
use tokenizers::Tokenizer as HfTokenizer;
use tokenizers::tokenizer::{
    DecodeStream as HfDecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper,
};

type InnerDecodeStream<'a> = HfDecodeStream<
    'a,
    ModelWrapper,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

pub struct Tokenizer {
    inner: HfTokenizer,
}

pub(crate) struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    inner: InnerDecodeStream<'a>,
    token_ids: Vec<u32>,
    emitted_text: String,
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

    pub(crate) fn incremental_decoder(&self) -> IncrementalDecoder<'_> {
        IncrementalDecoder {
            tokenizer: self,
            inner: self.inner.decode_stream(true),
            token_ids: Vec::new(),
            emitted_text: String::new(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

impl IncrementalDecoder<'_> {
    pub(crate) fn step(&mut self, token_id: u32) -> Result<Option<String>> {
        self.token_ids.push(token_id);
        let chunk = self
            .inner
            .step(token_id)
            .map_err(|e| anyhow!("Streaming decode error for token {}: {}", token_id, e))?;

        if let Some(ref text) = chunk {
            self.emitted_text.push_str(text);
        }

        Ok(chunk)
    }

    /// Text emitted so far by `step()` (and `finish()`). Used for stop-sequence checks.
    pub(crate) fn emitted_text(&self) -> &str {
        &self.emitted_text
    }

    pub(crate) fn finish(&mut self) -> Result<Option<String>> {
        let decoded = self.tokenizer.decode(&self.token_ids)?;
        let suffix = decoded.strip_prefix(&self.emitted_text).ok_or_else(|| {
            anyhow!(
                "Streaming decoder state mismatch: emitted text is not a prefix of final decode"
            )
        })?;

        if suffix.is_empty() {
            Ok(None)
        } else {
            self.emitted_text.push_str(suffix);
            Ok(Some(suffix.to_string()))
        }
    }
}
