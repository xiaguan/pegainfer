use anyhow::Result;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc::Sender;

use crate::model::Qwen3Model;
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

#[derive(Clone, Copy, Debug)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub struct CompleteOutput {
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

pub struct StreamDelta {
    pub text_delta: String,
    pub finish_reason: Option<FinishReason>,
}

pub trait ServerEngine: Send {
    fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput>;

    fn complete_stream(&mut self, req: CompleteRequest, tx: Sender<StreamDelta>) -> Result<()>;
}

pub struct RealServerEngine {
    model: Qwen3Model,
    tokenizer: Tokenizer,
    rng: StdRng,
}

impl RealServerEngine {
    pub fn load(model_path: &str, seed: u64) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_path)?;
        let model = Qwen3Model::from_safetensors(model_path)?;
        let rng = StdRng::seed_from_u64(seed);
        Ok(Self {
            model,
            tokenizer,
            rng,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

impl ServerEngine for RealServerEngine {
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

    fn complete_stream(&mut self, req: CompleteRequest, tx: Sender<StreamDelta>) -> Result<()> {
        let prompt_tokens = self.tokenizer.encode(&req.prompt)?;

        // TODO: Buffer incomplete subword/UTF-8 sequences before sending deltas.
        let stats = self.model.generate_streaming_with_callback(
            &prompt_tokens,
            req.max_tokens,
            &req.sampling,
            &mut self.rng,
            |token_id| {
                let text_delta = self.tokenizer.decode(&[token_id]).unwrap_or_else(|e| {
                    log::warn!("Failed to decode token {}: {}", token_id, e);
                    "\u{FFFD}".to_string()
                });
                tx.blocking_send(StreamDelta {
                    text_delta,
                    finish_reason: None,
                })
                .is_ok()
            },
        )?;

        if stats.consumer_dropped {
            return Ok(());
        }

        let finish_reason = if stats.hit_eos {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        let _ = tx.blocking_send(StreamDelta {
            text_delta: String::new(),
            finish_reason: Some(finish_reason),
        });

        Ok(())
    }
}
