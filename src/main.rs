use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use clap::Parser;
use fastrace::local::LocalSpan;
use fastrace::prelude::*;
use log::{error, info};
use pegainfer::logging;
use pegainfer::model::Qwen3Model;
use pegainfer::sampler::SamplingParams;
use pegainfer::tokenizer::Tokenizer;
use pegainfer::trace_reporter::FileReporter;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "pegainfer", about = "Qwen3 GPU inference server")]
struct Args {
    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Enable request tracing and write trace JSON files to this directory
    #[arg(long)]
    trace_output_path: Option<PathBuf>,
}

struct ModelState {
    model: Qwen3Model,
    rng: StdRng,
}

struct AppState {
    model: Mutex<ModelState>,
    tokenizer: Tokenizer,
}

// OpenAI-compatible /v1/completions request
#[derive(Debug, Deserialize)]
struct CompletionRequest {
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i32>,
    #[allow(dead_code)]
    n: Option<usize>,
    stream: Option<bool>,
    #[allow(dead_code)]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
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

// SSE streaming chunk (OpenAI-compatible)
#[derive(Debug, Serialize)]
struct StreamChunk {
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
    let max_tokens = req.max_tokens.unwrap_or(16);
    let stream = req.stream.unwrap_or(false);
    let model_name = req.model.unwrap_or_else(|| "qwen3-4b-gpu".to_string());

    info!(
        "Received request: prompt_len={}, max_tokens={}, stream={}",
        req.prompt.len(),
        max_tokens,
        stream,
    );

    let root = Span::root("request", SpanContext::random());
    let _guard = root.set_local_parent();
    LocalSpan::add_properties(|| {
        [
            ("prompt_len", req.prompt.len().to_string()),
            ("max_tokens", max_tokens.to_string()),
        ]
    });

    let prompt_tokens = {
        let _span = LocalSpan::enter_with_local_parent("tokenize_encode");
        state
            .tokenizer
            .encode(&req.prompt)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    let sampling_params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: req.top_k.unwrap_or(-1),
        top_p: req.top_p.unwrap_or(1.0),
    };

    // Drop non-Send span guard before any .await boundary
    drop(_guard);

    if stream {
        let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
        let created = now_secs();
        let (tx, rx) = tokio::sync::mpsc::channel::<u32>(32);

        let state_clone = state.clone();
        let prompt_tokens_clone = prompt_tokens.clone();

        tokio::task::spawn_blocking(move || {
            let mut guard = state_clone.model.blocking_lock();
            let s = &mut *guard;
            let result = s.model.generate_streaming(
                &prompt_tokens_clone,
                max_tokens,
                &sampling_params,
                &mut s.rng,
                tx,
            );
            if let Err(e) = result {
                error!("Streaming generation error: {}", e);
            }
        });

        let state_for_decode = state.clone();
        // TODO: Buffer incomplete subword/UTF-8 sequences before sending SSE events.
        // BPE tokens can be word fragments (e.g. "un" + "belie" + "vable"); byte-level
        // tokens may decode to invalid UTF-8 (�). Hold partial text until it forms
        // complete characters, similar to mini-sglang's find_printable_text approach.
        let stream = ReceiverStream::new(rx).map(move |token_id| {
            let text = state_for_decode.tokenizer.decode(&[token_id]).unwrap_or_else(|e| {
                log::warn!("Failed to decode token {}: {}", token_id, e);
                "\u{FFFD}".to_string()
            });
            let chunk = StreamChunk {
                id: request_id.clone(),
                object: "text_completion",
                created,
                model: model_name.clone(),
                choices: vec![StreamChoice {
                    text,
                    index: 0,
                    logprobs: None,
                    finish_reason: None,
                }],
            };
            let json = serde_json::to_string(&chunk).unwrap();
            Ok::<_, Infallible>(Event::default().data(json))
        });

        // Append [DONE] sentinel after the token stream ends
        let done_stream = futures_util::stream::once(async {
            Ok::<_, Infallible>(Event::default().data("[DONE]"))
        });

        let full_stream = stream.chain(done_stream);

        Ok(Sse::new(full_stream).into_response())
    } else {
        // Non-streaming path: run blocking GPU work on spawn_blocking
        let request_start = Instant::now();
        let prompt_len = prompt_tokens.len();
        let state_clone = state.clone();

        let output_tokens = tokio::task::spawn_blocking(move || {
            let mut guard = state_clone.model.blocking_lock();
            let s = &mut *guard;
            s.model
                .generate(&prompt_tokens, max_tokens, &sampling_params, &mut s.rng)
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

        let new_tokens = &output_tokens[prompt_len..];
        let generated_text = state
            .tokenizer
            .decode(new_tokens)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

        let total_time = request_start.elapsed();
        info!(
            "Request completed: total_time={:.2}ms, prompt_tokens={}, completion_tokens={}",
            total_time.as_secs_f64() * 1000.0,
            prompt_len,
            new_tokens.len()
        );

        let response = CompletionResponse {
            id: format!("cmpl-{}", uuid::Uuid::new_v4()),
            object: "text_completion",
            created: now_secs(),
            model: model_name,
            choices: vec![Choice {
                text: generated_text,
                index: 0,
                logprobs: None,
                finish_reason: "length".to_string(),
            }],
            usage: Usage {
                prompt_tokens: prompt_len,
                completion_tokens: new_tokens.len(),
                total_tokens: output_tokens.len(),
            },
        };

        Ok(Json(response).into_response())
    }
}

#[tokio::main]
async fn main() {
    logging::init_default();

    let args = Args::parse();

    if let Some(ref trace_path) = args.trace_output_path {
        std::fs::create_dir_all(trace_path).expect("Failed to create trace output directory");
        fastrace::set_reporter(
            FileReporter::new(trace_path.clone()),
            fastrace::collector::Config::default(),
        );
        info!("Tracing enabled: output_dir={}", trace_path.display());
    }

    info!("=== Rust LLM Server - Qwen3 (GPU) ===");

    info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    info!("Tokenizer loaded: vocab_size={}", tokenizer.vocab_size());

    info!("Loading model to GPU...");
    let start = Instant::now();
    let model = Qwen3Model::from_safetensors(MODEL_PATH).expect("Failed to load model");
    info!("Model loaded: elapsed_ms={}", start.elapsed().as_millis());

    let rng = StdRng::seed_from_u64(42);
    let state = Arc::new(AppState {
        model: Mutex::new(ModelState { model, rng }),
        tokenizer,
    });

    let app = Router::new()
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    if args.trace_output_path.is_some() {
        info!("Flushing pending traces...");
        fastrace::flush();
    }
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C handler");
    info!("Shutdown signal received");
}
