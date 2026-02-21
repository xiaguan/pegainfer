mod openai_v1;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use fastrace::local::LocalSpan;
use fastrace::prelude::*;
use futures_util::stream;
use log::{error, info};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::sampler::SamplingParams;
use crate::server_engine::{CompleteRequest, ServerEngine};
use openai_v1::{CompletionRequest, CompletionResponse, StreamChunk};

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
    let model_name = req.model_or_default();
    let prompt_len = req.prompt.len();

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
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        let state_clone = state.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = state_clone.engine.blocking_lock();
            if let Err(e) = guard.complete_stream(engine_req, tx) {
                error!("Streaming generation error: {}", e);
            }
        });

        let stream = ReceiverStream::new(rx).map(move |delta| {
            let chunk = StreamChunk::from_delta(&request_id, created, &model_name, delta);
            let json = serde_json::to_string(&chunk).unwrap();
            Ok::<_, Infallible>(Event::default().data(json))
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

        let response = CompletionResponse::from_output(model_name, now_secs(), output);
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
