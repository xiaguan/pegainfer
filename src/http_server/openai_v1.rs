use serde::{Deserialize, Serialize};

use crate::server_engine::StreamDelta;

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
    pub(super) stream_options: Option<StreamOptions>,
    pub(super) stop: Option<Vec<String>>,
    pub(super) ignore_eos: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub(super) struct StreamOptions {
    pub(super) include_usage: Option<bool>,
}

impl CompletionRequest {
    pub(super) fn max_tokens_or_default(&self) -> usize {
        self.max_tokens.unwrap_or(16)
    }

    pub(super) fn stream_or_default(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    pub(super) fn include_usage_or_default(&self) -> bool {
        self.stream_options
            .as_ref()
            .and_then(|options| options.include_usage)
            .unwrap_or(false)
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
#[allow(clippy::struct_field_names)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

impl CompletionResponse {
    pub(super) fn from_parts(
        model: String,
        created: u64,
        text: String,
        finish_reason: crate::server_engine::FinishReason,
        usage: crate::server_engine::Usage,
    ) -> Self {
        Self {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created,
            model,
            choices: vec![Choice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: finish_reason.as_openai_str().to_string(),
            }],
            usage: Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
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
        delta: &StreamDelta,
    ) -> Self {
        Self {
            id: request_id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            choices: vec![StreamChoice {
                text: delta.text_delta.clone(),
                index: 0,
                logprobs: None,
                finish_reason: delta
                    .finish_reason
                    .map(|reason| reason.as_openai_str().to_string()),
            }],
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct StreamUsageChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    usage: Usage,
}

impl StreamUsageChunk {
    pub(super) fn from_usage(
        request_id: &str,
        created: u64,
        model: &str,
        usage: crate::server_engine::Usage,
    ) -> Self {
        Self {
            id: request_id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            usage: Usage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            },
        }
    }
}
