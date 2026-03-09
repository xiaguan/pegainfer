use std::fmt;
use std::path::Path;

use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

use crate::model::{ModelRuntimeConfig, Qwen3Model, StreamingStats};
use crate::qwen35_model::Qwen35Model;
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;

/// If `text` ends with any of `stops`, return the text with that suffix removed.
/// Prefers the longest matching stop when several match at the end.
fn strip_stop_suffix(text: &str, stops: &[String]) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for s in stops {
        let s = s.as_str();
        if s.is_empty() {
            continue;
        }
        if text.ends_with(s) {
            let len: usize = s.len();
            if best.map_or(true, |(l, _)| len > l) {
                best = Some((len, s));
            }
        }
    }
    best.map(|(len, _)| text[..text.len() - len].to_string())
}

/// If `new_full` (accumulated text) ends with any of `stops`, return the delta to send
/// (from `sent_len` up to but not including the stop) and the matching stop.
/// Prefers the longest matching stop when several match at the end.
fn truncate_at_stop<'a>(
    new_full: &str,
    sent_len: usize,
    stops: &[&'a str],
) -> Option<(String, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if new_full.ends_with(s) {
            let len = s.len();
            if best.map_or(true, |(l, _)| len > l) {
                best = Some((len, *s));
            }
        }
    }
    best.map(|(stop_len, stop)| {
        let end = new_full.len().saturating_sub(stop_len);
        let to_send = if end >= sent_len {
            new_full[sent_len..end].to_string()
        } else {
            String::new()
        };
        (to_send, stop)
    })
}

pub struct CompleteRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    pub stop: Option<Vec<String>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Length,
    Stop,
}

impl FinishReason {
    pub fn as_openai_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Stop => "stop",
        }
    }
}

pub struct CompleteOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct StreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

pub trait ServerEngine: Send {
    fn model_id(&self) -> &str;

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput>;

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()>;
}

// ============================================================================
// Model type detection
// ============================================================================

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Qwen3,
    Qwen35,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qwen3 => write!(f, "Qwen3"),
            Self::Qwen35 => write!(f, "Qwen3.5"),
        }
    }
}

/// Detect model type from config.json.
pub fn detect_model_type(model_path: &str) -> Result<ModelType> {
    let config_path = format!("{}/config.json", model_path);
    let content = std::fs::read_to_string(&config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    // Qwen3.5 has nested text_config
    if json.get("text_config").is_some() {
        return Ok(ModelType::Qwen35);
    }

    Ok(ModelType::Qwen3)
}

// ============================================================================
// Engine options
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    pub enable_cuda_graph: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
        }
    }
}

// ============================================================================
// GenerativeModel trait — shared by Qwen3 and Qwen3.5
// ============================================================================

pub trait GenerativeModel: Send {
    fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>>;

    fn generate_streaming_with_callback(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        callback: impl FnMut(u32) -> bool,
    ) -> Result<StreamingStats>;
}

impl GenerativeModel for Qwen3Model {
    fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        self.generate(prompt_tokens, max_new_tokens, params, rng)
    }

    fn generate_streaming_with_callback(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        callback: impl FnMut(u32) -> bool,
    ) -> Result<StreamingStats> {
        self.generate_streaming_with_callback(prompt_tokens, max_new_tokens, params, rng, callback)
    }
}

impl GenerativeModel for Qwen35Model {
    fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        self.generate(prompt_tokens, max_new_tokens, params, rng)
    }

    fn generate_streaming_with_callback(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        rng: &mut StdRng,
        callback: impl FnMut(u32) -> bool,
    ) -> Result<StreamingStats> {
        self.generate_streaming_with_callback(prompt_tokens, max_new_tokens, params, rng, callback)
    }
}

// ============================================================================
// Generic server engine — shared complete/complete_stream logic
// ============================================================================

pub struct GenericServerEngine<M: GenerativeModel> {
    model_id: String,
    model: M,
    tokenizer: Tokenizer,
    rng: StdRng,
}

impl<M: GenerativeModel> ServerEngine for GenericServerEngine<M> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let output_tokens =
            self.model
                .generate(&prompt_tokens, req.max_tokens, &req.sampling, &mut self.rng)?;
        let completion_tokens = output_tokens.len().saturating_sub(prompt_tokens.len());
        let mut text = self
            .tokenizer
            .decode(&output_tokens[prompt_tokens.len()..])?;
        let mut finish_reason = if completion_tokens >= req.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        if let Some(ref stops) = req.stop {
            if let Some(stripped) = strip_stop_suffix(&text, stops) {
                text = stripped;
                finish_reason = FinishReason::Stop;
            }
        }
        let usage = Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: output_tokens.len(),
        };
        Ok(CompleteOutput {
            text,
            finish_reason,
            usage,
        })
    }

    fn complete_stream(
        &mut self,
        req: CompleteRequest,
        tx: UnboundedSender<StreamDelta>,
    ) -> Result<()> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let mut decoder = self.tokenizer.incremental_decoder();
        let mut decode_error = None;
        let stops: Option<Vec<&str>> = req
            .stop
            .as_ref()
            .map(|v| v.iter().map(String::as_str).filter(|s| !s.is_empty()).collect());
        let mut sent_len: usize = 0;
        let stopped_by_stop_sequence = std::cell::Cell::new(false);

        let stats = self.model.generate_streaming_with_callback(
            &prompt_tokens,
            req.max_tokens,
            &req.sampling,
            &mut self.rng,
            |token_id| match decoder.step(token_id) {
                Ok(Some(text_delta)) => {
                    if let Some(ref stop_list) = stops {
                        let new_full = {
                            let emitted = decoder.emitted_text();
                            emitted.to_string()
                        };
                        if let Some((to_send, stopped)) =
                            truncate_at_stop(&new_full, sent_len, stop_list)
                        {
                            if !to_send.is_empty() {
                                if tx.send(StreamDelta {
                                    text_delta: to_send,
                                    finish_reason: None,
                                    usage: None,
                                }).is_err() {
                                    return false;
                                }
                            }
                            sent_len = new_full.len() - stopped.len();
                            stopped_by_stop_sequence.set(true);
                            return false;
                        }
                        let to_send = &new_full[sent_len..];
                        sent_len = new_full.len();
                        tx.send(StreamDelta {
                            text_delta: to_send.to_string(),
                            finish_reason: None,
                            usage: None,
                        })
                        .is_ok()
                    } else {
                        tx.send(StreamDelta {
                            text_delta: text_delta,
                            finish_reason: None,
                            usage: None,
                        })
                        .is_ok()
                    }
                }
                Ok(None) => true,
                Err(err) => {
                    decode_error = Some(err);
                    false
                }
            },
        )?;

        if let Some(err) = decode_error {
            return Err(err);
        }

        if stats.consumer_dropped {
            return Ok(());
        }

        if let Some(text_delta) = decoder.finish()? {
            if let Some(ref stop_list) = stops {
                let new_full = decoder.emitted_text().to_string();
                if let Some((to_send, _)) = truncate_at_stop(&new_full, sent_len, stop_list) {
                    if !to_send.is_empty() {
                        let _ = tx.send(StreamDelta {
                            text_delta: to_send,
                            finish_reason: None,
                            usage: None,
                        });
                    }
                } else {
                    let to_send = &new_full[sent_len..];
                    if !to_send.is_empty() {
                        let _ = tx.send(StreamDelta {
                            text_delta: to_send.to_string(),
                            finish_reason: None,
                            usage: None,
                        });
                    }
                }
            } else {
                let _ = tx.send(StreamDelta {
                    text_delta: text_delta,
                    finish_reason: None,
                    usage: None,
                });
            }
        }

        let finish_reason = if stopped_by_stop_sequence.get() {
            FinishReason::Stop
        } else if stats.hit_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let _ = tx.send(StreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(Usage {
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: stats.emitted_tokens,
                total_tokens: prompt_tokens.len() + stats.emitted_tokens,
            }),
        });

        Ok(())
    }
}

// ============================================================================
// Public engine constructors
// ============================================================================

fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

pub type RealServerEngine = GenericServerEngine<Qwen3Model>;
pub type Qwen35ServerEngine = GenericServerEngine<Qwen35Model>;

impl RealServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        Self::load_with_options(model_path, seed, EngineOptions::default())
    }

    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_path)?;
        let model = Qwen3Model::from_safetensors_with_runtime(
            model_path,
            ModelRuntimeConfig {
                enable_cuda_graph: options.enable_cuda_graph,
            },
        )?;
        let rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            model_id: model_id_from_path(model_path),
            model,
            tokenizer,
            rng,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

impl Qwen35ServerEngine {
    pub fn load_with_options(model_path: &str, seed: u64, options: EngineOptions) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_path)?;
        let model =
            Qwen35Model::from_safetensors_with_options(model_path, options.enable_cuda_graph)?;
        let rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            model_id: model_id_from_path(model_path),
            model,
            tokenizer,
            rng,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

#[cfg(test)]
mod tests {
    use super::{strip_stop_suffix, truncate_at_stop};

    #[test]
    fn test_strip_stop_suffix() {
        let stops: Vec<String> = vec!["\n".into(), "END".into()];
        assert_eq!(strip_stop_suffix("hello\n", &stops), Some("hello".to_string()));
        assert_eq!(strip_stop_suffix("hello\n\n", &stops), Some("hello\n".to_string()));
        assert_eq!(strip_stop_suffix("helloEND", &stops), Some("hello".to_string()));
        assert_eq!(strip_stop_suffix("hello", &stops), None);
        assert_eq!(strip_stop_suffix("", &stops), None);
    }

    #[test]
    fn test_truncate_at_stop() {
        let stops = ["\n"];
        assert_eq!(
            truncate_at_stop("hello\n", 0, &stops),
            Some(("hello".to_string(), "\n"))
        );
        assert_eq!(truncate_at_stop("hello", 0, &stops), None);
        assert_eq!(
            truncate_at_stop("ab\n", 2, &stops),
            Some(("".to_string(), "\n"))
        );
    }
}
