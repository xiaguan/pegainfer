mod openai_v1;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use futures_util::{StreamExt, stream};
use log::{error, info, warn};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::sampler::SamplingParams;
use crate::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent};
use crate::server_engine::{
    FinishReason, StreamDelta, Usage, truncate_at_first_stop, truncate_at_stop,
};
use crate::tokenizer::Tokenizer;
use openai_v1::{
    CompletionRequest, CompletionResponse, LogprobsResponse, StreamChunk, StreamUsageChunk,
};

struct AppState {
    handle: SchedulerHandle,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, StatusCode> {
    let max_tokens = req.max_tokens_or_default();
    let stream = req.stream_or_default();
    let include_usage = req.include_usage_or_default();
    let requested_model = req.model.clone();
    let loaded_model = state.model_id.clone();
    let prompt_len = req.prompt.len();
    let logprobs_n = req.logprobs.unwrap_or(0);
    let echo = req.echo.unwrap_or(false);

    if req.prompt.trim().is_empty() {
        warn!("Rejecting empty prompt request");
        return Err(StatusCode::BAD_REQUEST);
    }

    if let Some(ref requested_model) = requested_model
        && requested_model != &loaded_model
    {
        warn!(
            "Request model '{}' does not match loaded model '{}'; using loaded model",
            requested_model, loaded_model
        );
    }

    info!(
        "Received request: prompt_len={}, max_tokens={}, stream={}",
        prompt_len, max_tokens, stream,
    );

    // Tokenize prompt (CPU, ~μs)
    let prompt_tokens = state.tokenizer.encode(&req.prompt).map_err(|e| {
        error!("Tokenization error: {e}");
        StatusCode::BAD_REQUEST
    })?;
    let prompt_token_count = prompt_tokens.len();

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: req.top_k.unwrap_or(-1),
        top_p: req.top_p.unwrap_or(1.0),
        ignore_eos: req.ignore_eos.unwrap_or(false),
    };

    // Per-request channel: scheduler → this handler
    let (token_tx, token_rx) = mpsc::unbounded_channel();

    // Submit to scheduler
    state
        .handle
        .submit(SchedulerRequest {
            prompt_tokens,
            params: sampling,
            max_tokens,
            token_tx,
            logprobs: logprobs_n,
            echo,
        })
        .map_err(|_| {
            error!("Scheduler thread has exited");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    if stream {
        Ok(handle_streaming(
            state,
            token_rx,
            req.stop,
            prompt_token_count,
            loaded_model,
            include_usage,
        )
        .into_response())
    } else {
        handle_non_streaming(
            state,
            token_rx,
            req.stop,
            prompt_token_count,
            loaded_model,
            logprobs_n,
        )
        .await
    }
}

// ── Non-streaming ───────────────────────────────────────────────────────

async fn handle_non_streaming(
    state: Arc<AppState>,
    mut token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    stop: Option<Vec<String>>,
    prompt_token_count: usize,
    loaded_model: String,
    logprobs_requested: usize,
) -> Result<Response, StatusCode> {
    let request_start = Instant::now();

    // Collect all tokens and logprobs
    let mut all_token_ids: Vec<u32> = Vec::new();
    let mut all_logprobs: Vec<Option<crate::server_engine::TokenLogprob>> = Vec::new();
    let mut prompt_token_ids: Vec<u32> = Vec::new();
    let mut prompt_logprobs: Vec<Option<crate::server_engine::TokenLogprob>> = Vec::new();
    let mut finish_reason = FinishReason::Stop;
    let mut completion_tokens = 0;

    while let Some(event) = token_rx.recv().await {
        match event {
            TokenEvent::Token { id, logprob } => {
                all_token_ids.push(id);
                all_logprobs.push(logprob);
            }
            TokenEvent::PromptTokens { ids, logprobs } => {
                prompt_token_ids = ids;
                prompt_logprobs = logprobs;
            }
            TokenEvent::Finished {
                finish_reason: fr,
                completion_tokens: ct,
                ..
            } => {
                finish_reason = fr;
                completion_tokens = ct;
                break;
            }
        }
    }

    // Detokenize completion tokens
    let mut text = state.tokenizer.decode(&all_token_ids).map_err(|e| {
        error!("Detokenization error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // If echo, prepend prompt text
    let echo_prefix = if !prompt_token_ids.is_empty() {
        let prompt_text = state.tokenizer.decode(&prompt_token_ids).map_err(|e| {
            error!("Prompt detokenization error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;
        let prefix_len = prompt_text.len();
        text = format!("{prompt_text}{text}");
        Some(prefix_len)
    } else {
        None
    };

    // Stop sequence truncation
    if let Some(ref stops) = stop
        && let Some(truncated) = truncate_at_first_stop(&text, stops)
    {
        text = truncated;
        finish_reason = FinishReason::Stop;
    }

    // Build logprobs response if requested
    let logprobs_response = if logprobs_requested > 0 {
        let tokenizer = &state.tokenizer;

        // Combine prompt + completion tokens for the response
        let mut tokens = Vec::new();
        let mut token_logprobs = Vec::new();
        let mut top_logprobs = Vec::new();
        let mut text_offset = Vec::new();
        let mut offset = 0usize;

        // Echo prompt tokens
        for (i, &tid) in prompt_token_ids.iter().enumerate() {
            let tok_str = tokenizer.decode(&[tid]).unwrap_or_default();
            text_offset.push(offset);
            offset += tok_str.len();
            tokens.push(tok_str);
            // First prompt token has no logprob
            let lp = prompt_logprobs.get(i).and_then(|l| l.as_ref());
            token_logprobs.push(lp.map(|l| l.logprob));
            top_logprobs.push(lp.map(|l| {
                l.top_logprobs
                    .iter()
                    .map(|&(tid, lp)| {
                        (tokenizer.decode(&[tid]).unwrap_or_default(), lp)
                    })
                    .collect()
            }));
        }

        // Completion tokens
        if let Some(prefix_len) = echo_prefix {
            offset = prefix_len;
        } else {
            offset = 0;
        }
        for (i, &tid) in all_token_ids.iter().enumerate() {
            let tok_str = tokenizer.decode(&[tid]).unwrap_or_default();
            text_offset.push(offset);
            offset += tok_str.len();
            tokens.push(tok_str);
            let lp = all_logprobs.get(i).and_then(|l| l.as_ref());
            token_logprobs.push(lp.map(|l| l.logprob));
            top_logprobs.push(lp.map(|l| {
                l.top_logprobs
                    .iter()
                    .map(|&(tid, lp)| {
                        (tokenizer.decode(&[tid]).unwrap_or_default(), lp)
                    })
                    .collect()
            }));
        }

        Some(LogprobsResponse {
            tokens,
            token_logprobs,
            top_logprobs,
            text_offset,
        })
    } else {
        None
    };

    let total_time = request_start.elapsed();
    info!(
        "Request completed: total_time={:.2}ms, prompt_tokens={}, completion_tokens={}",
        total_time.as_secs_f64() * 1000.0,
        prompt_token_count,
        completion_tokens
    );

    let usage = Usage {
        prompt_tokens: prompt_token_count,
        completion_tokens,
        total_tokens: prompt_token_count + completion_tokens,
    };
    let response = CompletionResponse::from_parts(
        loaded_model,
        now_secs(),
        text,
        finish_reason,
        usage,
        logprobs_response,
    );
    Ok(Json(response).into_response())
}

// ── Streaming ───────────────────────────────────────────────────────────

fn handle_streaming(
    state: Arc<AppState>,
    token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    stop: Option<Vec<String>>,
    prompt_token_count: usize,
    loaded_model: String,
    include_usage: bool,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = now_secs();

    // Bridge channel: spawn_blocking decodes tokens → delta_rx feeds SSE
    let (delta_tx, delta_rx) = mpsc::unbounded_channel::<StreamDelta>();

    let tokenizer = state.tokenizer.clone();
    tokio::task::spawn_blocking(move || {
        streaming_bridge(tokenizer, token_rx, delta_tx, stop, prompt_token_count);
    });

    let model = loaded_model;
    let rid = request_id;
    let stream = UnboundedReceiverStream::new(delta_rx).flat_map(move |delta| {
        let usage = delta.usage;
        let is_terminal = delta.finish_reason.is_some();

        let chunk = StreamChunk::from_delta(&rid, created, &model, &delta);
        let mut events = vec![Ok::<_, Infallible>(
            Event::default().data(serde_json::to_string(&chunk).unwrap()),
        )];

        if include_usage
            && is_terminal
            && let Some(usage) = usage
        {
            let usage_chunk = StreamUsageChunk::from_usage(&rid, created, &model, usage);
            events.push(Ok(
                Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
            ));
        }

        stream::iter(events)
    });

    let done_stream = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

    Sse::new(stream.chain(done_stream))
}

/// Runs in spawn_blocking: receives TokenEvents, incrementally decodes,
/// handles stop sequences, sends StreamDeltas.
fn streaming_bridge(
    tokenizer: Arc<Tokenizer>,
    mut token_rx: mpsc::UnboundedReceiver<TokenEvent>,
    delta_tx: mpsc::UnboundedSender<StreamDelta>,
    stop: Option<Vec<String>>,
    prompt_token_count: usize,
) {
    let mut decoder = tokenizer.incremental_decoder();
    let mut sent_len: usize = 0;
    let mut token_count: usize = 0;
    let stops: Vec<&str> = stop.as_ref().map_or(Vec::new(), |v| {
        v.iter()
            .map(String::as_str)
            .filter(|s| !s.is_empty())
            .collect()
    });

    loop {
        match token_rx.blocking_recv() {
            Some(TokenEvent::PromptTokens { .. }) => {
                // Echo tokens handled in non-streaming; skip in streaming for now
            }
            Some(TokenEvent::Token { id, .. }) => {
                token_count += 1;
                match decoder.step(id) {
                    Ok(Some(text_delta)) => {
                        if !stops.is_empty() {
                            let full = decoder.emitted_text().to_string();
                            if let Some((to_send, _)) = truncate_at_stop(&full, sent_len, &stops) {
                                if !to_send.is_empty()
                                    && delta_tx
                                        .send(StreamDelta {
                                            text_delta: to_send,
                                            finish_reason: None,
                                            usage: None,
                                        })
                                        .is_err()
                                {
                                    return;
                                }
                                // Stop sequence hit — send terminal delta, drop token_rx
                                let _ = delta_tx.send(StreamDelta {
                                    text_delta: String::new(),
                                    finish_reason: Some(FinishReason::Stop),
                                    usage: Some(Usage {
                                        prompt_tokens: prompt_token_count,
                                        completion_tokens: token_count,
                                        total_tokens: prompt_token_count + token_count,
                                    }),
                                });
                                return;
                            }
                            let to_send = &full[sent_len..];
                            sent_len = full.len();
                            if !to_send.is_empty()
                                && delta_tx
                                    .send(StreamDelta {
                                        text_delta: to_send.to_string(),
                                        finish_reason: None,
                                        usage: None,
                                    })
                                    .is_err()
                            {
                                return;
                            }
                        } else if delta_tx
                            .send(StreamDelta {
                                text_delta,
                                finish_reason: None,
                                usage: None,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(None) => {} // partial byte buffered
                    Err(e) => {
                        warn!("Incremental decode error: {e}");
                        return;
                    }
                }
            }
            Some(TokenEvent::Finished {
                finish_reason,
                completion_tokens,
                ..
            }) => {
                // Flush decoder
                if let Ok(Some(tail)) = decoder.finish() {
                    if !stops.is_empty() {
                        let full = decoder.emitted_text().to_string();
                        if let Some((to_send, _)) = truncate_at_stop(&full, sent_len, &stops) {
                            if !to_send.is_empty() {
                                let _ = delta_tx.send(StreamDelta {
                                    text_delta: to_send,
                                    finish_reason: None,
                                    usage: None,
                                });
                            }
                            let _ = delta_tx.send(StreamDelta {
                                text_delta: String::new(),
                                finish_reason: Some(FinishReason::Stop),
                                usage: Some(Usage {
                                    prompt_tokens: prompt_token_count,
                                    completion_tokens,
                                    total_tokens: prompt_token_count + completion_tokens,
                                }),
                            });
                            return;
                        }
                        let to_send = &full[sent_len..];
                        if !to_send.is_empty() {
                            let _ = delta_tx.send(StreamDelta {
                                text_delta: to_send.to_string(),
                                finish_reason: None,
                                usage: None,
                            });
                        }
                    } else {
                        let _ = delta_tx.send(StreamDelta {
                            text_delta: tail,
                            finish_reason: None,
                            usage: None,
                        });
                    }
                }

                // Terminal delta
                let _ = delta_tx.send(StreamDelta {
                    text_delta: String::new(),
                    finish_reason: Some(finish_reason),
                    usage: Some(Usage {
                        prompt_tokens: prompt_token_count,
                        completion_tokens,
                        total_tokens: prompt_token_count + completion_tokens,
                    }),
                });
                return;
            }
            None => {
                // Scheduler dropped without Finished — treat as error
                return;
            }
        }
    }
}

// ── App builder ─────────────────────────────────────────────────────────

pub fn build_app(handle: SchedulerHandle, tokenizer: Arc<Tokenizer>, model_id: String) -> Router {
    let state = Arc::new(AppState {
        handle,
        tokenizer,
        model_id,
    });

    Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::util::ServiceExt;

    /// Create a mock scheduler that responds to each request with fixed tokens.
    fn mock_scheduler(tokens: Vec<u32>, finish_reason: FinishReason) -> SchedulerHandle {
        let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<SchedulerRequest>();

        tokio::spawn(async move {
            while let Some(req) = submit_rx.recv().await {
                for &t in &tokens {
                    if req
                        .token_tx
                        .send(TokenEvent::Token {
                            id: t,
                            logprob: None,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: tokens.len(),
                });
            }
        });

        SchedulerHandle { submit_tx }
    }

    fn test_tokenizer() -> Arc<Tokenizer> {
        let model_path = std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B").to_string()
        });
        Arc::new(Tokenizer::from_file(&model_path).expect("tokenizer"))
    }

    #[tokio::test]
    async fn completion_response_uses_loaded_model_id() {
        let tokenizer = test_tokenizer();
        // Token 9707 = "Hello" in Qwen3 tokenizer
        let handle = mock_scheduler(vec![9707], FinishReason::Stop);
        let app = build_app(handle, tokenizer, "Qwen3-4B".to_string());

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-8b","prompt":"hello","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(payload["model"], "Qwen3-4B");
    }

    #[tokio::test]
    async fn streaming_response_uses_loaded_model_id() {
        let tokenizer = test_tokenizer();
        let handle = mock_scheduler(vec![9707], FinishReason::Stop);
        let app = build_app(handle, tokenizer, "Qwen3-8B".to_string());

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(
            payload.contains(r#""model":"Qwen3-8B""#),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"));
    }

    #[tokio::test]
    async fn streaming_response_includes_usage_when_requested() {
        let tokenizer = test_tokenizer();
        let handle = mock_scheduler(vec![9707], FinishReason::Stop);
        let app = build_app(handle, tokenizer, "Qwen3-4B".to_string());

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"hello","max_tokens":1,"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let payload = String::from_utf8(body.to_vec()).unwrap();

        assert!(payload.contains(r#""usage":{"#), "payload={payload}");
    }

    #[tokio::test]
    async fn completion_rejects_empty_prompt() {
        let tokenizer = test_tokenizer();
        let handle = mock_scheduler(vec![], FinishReason::Stop);
        let app = build_app(handle, tokenizer, "Qwen3-4B".to_string());

        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"qwen3-4b","prompt":"   ","max_tokens":1}"#,
            ))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
