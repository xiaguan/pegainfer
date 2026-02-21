use serde::{Deserialize, Serialize};

use crate::server_engine::{CompleteOutput, StreamDelta};

// OpenAI-compatible /v1/completions request
#[derive(Debug, Deserialize)]
pub(super) struct CompletionRequest {
    pub(super) model: Option<String>,
    pub(super) prompt: String,
    pub(super) max_tokens: Option<usize>,
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<i32>,
    #[allow(dead_code)]
    pub(super) n: Option<usize>,
    pub(super) stream: Option<bool>,
    #[allow(dead_code)]
    pub(super) stop: Option<Vec<String>>,
}

impl CompletionRequest {
    pub(super) fn max_tokens_or_default(&self) -> usize {
        self.max_tokens.unwrap_or(16)
    }

    pub(super) fn stream_or_default(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub(super) fn model_or_default(&self) -> String {
        self.model
            .clone()
            .unwrap_or_else(|| "qwen3-4b-gpu".to_string())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl CompletionResponse {
    pub(super) fn from_output(model: String, created: u64, output: CompleteOutput) -> Self {
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created,
            model,
            choices: vec![Choice {
                text: output.text,
                index: 0,
                logprobs: None,
                finish_reason: output.finish_reason.as_openai_str().to_string(),
            }],
            usage: Usage {
                prompt_tokens: output.usage.prompt_tokens,
                completion_tokens: output.usage.completion_tokens,
                total_tokens: output.usage.total_tokens,
            },
        }
    }
}

// OpenAI-compatible SSE streaming chunk
#[derive(Debug, Serialize)]
pub(super) struct StreamChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: Option<String>,
}

impl StreamChunk {
    pub(super) fn from_delta(
        request_id: &str,
        created: u64,
        model: &str,
        delta: StreamDelta,
    ) -> Self {
        Self {
            id: request_id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                text: delta.text_delta,
                index: 0,
                logprobs: None,
                finish_reason: delta
                    .finish_reason
                    .map(|reason| reason.as_openai_str().to_string()),
            }],
        }
    }
}
