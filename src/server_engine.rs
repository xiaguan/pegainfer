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

pub struct CompleteRequest {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
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
        let text = self
            .tokenizer
            .decode(&output_tokens[prompt_tokens.len()..])?;
        let finish_reason = if completion_tokens >= req.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
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

        let stats = self.model.generate_streaming_with_callback(
            &prompt_tokens,
            req.max_tokens,
            &req.sampling,
            &mut self.rng,
            |token_id| match decoder.step(token_id) {
                Ok(Some(text_delta)) => tx
                    .send(StreamDelta {
                        text_delta,
                        finish_reason: None,
                        usage: None,
                    })
                    .is_ok(),
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
            let _ = tx.send(StreamDelta {
                text_delta,
                finish_reason: None,
                usage: None,
            });
        }

        let finish_reason = if stats.hit_eos {
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
