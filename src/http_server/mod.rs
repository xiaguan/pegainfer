mod openai_v1;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use fastrace::local::LocalSpan;
use fastrace::prelude::*;
use futures_util::{StreamExt, stream};
use log::{error, info, warn};
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::sampler::SamplingParams;
use crate::server_engine::{CompleteRequest, ServerEngine};
use openai_v1::{CompletionRequest, CompletionResponse, StreamChunk, StreamUsageChunk};

struct AppState {
    engine: Mutex<Box<dyn ServerEngine>>,
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
    let loaded_model = {
        let guard = state.engine.lock().await;
        guard.model_id().to_string()
    };
    let prompt_len = req.prompt.len();

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

    let root = Span::root("request", SpanContext::random());
    let local_guard = root.set_local_parent();
    LocalSpan::add_properties(|| {
        [
            ("prompt_len", prompt_len.to_string()),
            ("max_tokens", max_tokens.to_string()),
        ]
    });

    let engine_req = CompleteRequest {
        prompt: req.prompt,
        max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(0.0),
            top_k: req.top_k.unwrap_or(-1),
            top_p: req.top_p.unwrap_or(1.0),
        },
    };

    // Drop non-Send guard before awaiting task joins.
    drop(local_guard);

    if stream {
        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let created = now_secs();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        let state_clone = state.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = state_clone.engine.blocking_lock();
            if let Err(e) = guard.complete_stream(engine_req, tx) {
                error!("Streaming generation error: {}", e);
            }
        });

        let stream = UnboundedReceiverStream::new(rx).flat_map(move |delta| {
            let usage = delta.usage;
            let is_terminal = delta.finish_reason.is_some();

            let chunk = StreamChunk::from_delta(&request_id, created, &loaded_model, delta);
            let mut events = vec![Ok::<_, Infallible>(
                Event::default().data(serde_json::to_string(&chunk).unwrap()),
            )];

            if include_usage
                && is_terminal
                && let Some(usage) = usage
            {
                let usage_chunk =
                    StreamUsageChunk::from_usage(&request_id, created, &loaded_model, usage);
                events.push(Ok(
                    Event::default().data(serde_json::to_string(&usage_chunk).unwrap())
                ));
            }

            stream::iter(events)
        });

        let done_stream =
            stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });

        Ok(Sse::new(stream.chain(done_stream)).into_response())
    } else {
        let request_start = Instant::now();
        let state_clone = state.clone();
        let output = tokio::task::spawn_blocking(move || {
            let mut guard = state_clone.engine.blocking_lock();
            guard.complete(engine_req)
        })
        .await
        .map_err(|e| {
            error!("Task join error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .map_err(|e| {
            error!("Generation error: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

        let total_time = request_start.elapsed();
        info!(
            "Request completed: total_time={:.2}ms, prompt_tokens={}, completion_tokens={}",
            total_time.as_secs_f64() * 1000.0,
            output.usage.prompt_tokens,
            output.usage.completion_tokens
        );

        let response = CompletionResponse::from_output(loaded_model, now_secs(), output);
        Ok(Json(response).into_response())
    }
}

pub fn build_app(engine: Box<dyn ServerEngine>) -> Router {
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
    });

    Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    use anyhow::Result;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::util::ServiceExt;

    use crate::server_engine::{CompleteOutput, FinishReason, StreamDelta, Usage};

    struct MockEngine {
        model_id: String,
    }

    impl MockEngine {
        fn new(model_id: &str) -> Self {
            Self {
                model_id: model_id.to_string(),
            }
        }
    }

    impl ServerEngine for MockEngine {
        fn model_id(&self) -> &str {
            &self.model_id
        }

        fn complete(&mut self, req: CompleteRequest) -> Result<CompleteOutput> {
            Ok(CompleteOutput {
                text: format!("ok:{}", req.prompt),
                finish_reason: FinishReason::Length,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
            })
        }

        fn complete_stream(
            &mut self,
            _req: CompleteRequest,
            tx: tokio::sync::mpsc::UnboundedSender<StreamDelta>,
        ) -> Result<()> {
            let _ = tx.send(StreamDelta {
                text_delta: "ok".to_string(),
                finish_reason: None,
                usage: None,
            });
            let _ = tx.send(StreamDelta {
                text_delta: String::new(),
                finish_reason: Some(FinishReason::Stop),
                usage: Some(Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                }),
            });
            Ok(())
        }
    }

    #[tokio::test]
    async fn completion_response_uses_loaded_model_id() {
        let app = build_app(Box::new(MockEngine::new("Qwen3-4B")));
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
        assert_eq!(payload["choices"][0]["text"], "ok:hello");
    }

    #[tokio::test]
    async fn streaming_response_uses_loaded_model_id() {
        let app = build_app(Box::new(MockEngine::new("Qwen3-8B")));
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
        assert!(
            !payload.contains(r#""model":"qwen3-4b""#),
            "payload={payload}"
        );
        assert!(payload.contains("[DONE]"));
    }

    #[tokio::test]
    async fn streaming_response_includes_usage_when_requested() {
        let app = build_app(Box::new(MockEngine::new("Qwen3-4B")));
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

        assert!(
            payload
                .contains(r#""usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}"#),
            "payload={payload}"
        );
    }
}
