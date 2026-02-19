use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use clap::Parser;
use fastrace::local::LocalSpan;
use fastrace::prelude::*;
use log::{error, info};
use pegainfer::model::Qwen3Model;
use pegainfer::logging;
use pegainfer::sampler::SamplingParams;
use pegainfer::tokenizer::Tokenizer;
use pegainfer::trace_reporter::FileReporter;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

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

struct AppState {
    model: Qwen3Model,
    tokenizer: Tokenizer,
    rng: StdRng,
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
    #[allow(dead_code)]
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

async fn completions(
    State(state): State<Arc<Mutex<AppState>>>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let request_start = Instant::now();
    let max_tokens = req.max_tokens.unwrap_or(16);
    let prompt_len = req.prompt.len();

    info!(
        "Received request: prompt_len={}, max_tokens={}",
        prompt_len, max_tokens
    );

    let mut state = state.lock().await;

    // Set up trace root span (no-op if no reporter is configured)
    let root = Span::root("request", SpanContext::random());
    let _guard = root.set_local_parent();
    LocalSpan::add_properties(|| {
        [
            ("prompt_len", prompt_len.to_string()),
            ("max_tokens", max_tokens.to_string()),
        ]
    });

    // Encode prompt
    let prompt_tokens = {
        let _span = LocalSpan::enter_with_local_parent("tokenize_encode");
        state
            .tokenizer
            .encode(&req.prompt)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    // Generate (GPU)
    let sampling_params = SamplingParams {
        temperature: req.temperature.unwrap_or(0.0),
        top_k: req.top_k.unwrap_or(-1),
        top_p: req.top_p.unwrap_or(1.0),
    };
    let output_tokens = {
        let s = &mut *state;
        s.model
            .generate(&prompt_tokens, max_tokens, &sampling_params, &mut s.rng)
            .map_err(|e| {
                error!("Generation error: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?
    };

    // Decode only the new tokens
    let new_tokens = &output_tokens[prompt_tokens.len()..];
    let generated_text = {
        let _span = LocalSpan::enter_with_local_parent("tokenize_decode");
        state
            .tokenizer
            .decode(new_tokens)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
    };

    let total_time = request_start.elapsed();
    LocalSpan::add_property(|| {
        (
            "total_time_ms",
            format!("{:.2}", total_time.as_secs_f64() * 1000.0),
        )
    });

    info!(
        "Request completed: total_time={:.2}ms, prompt_tokens={}, completion_tokens={}",
        total_time.as_secs_f64() * 1000.0,
        prompt_tokens.len(),
        new_tokens.len()
    );

    let response = CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        object: "text_completion",
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: req.model.unwrap_or_else(|| "qwen3-4b-gpu".to_string()),
        choices: vec![Choice {
            text: generated_text,
            index: 0,
            logprobs: None,
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: new_tokens.len(),
            total_tokens: output_tokens.len(),
        },
    };

    Ok(Json(response))
}

#[tokio::main]
async fn main() {
    // Initialize logger
    logging::init_default();

    let args = Args::parse();

    // Set up tracing if output path specified
    if let Some(ref trace_path) = args.trace_output_path {
        std::fs::create_dir_all(trace_path).expect("Failed to create trace output directory");
        fastrace::set_reporter(
            FileReporter::new(trace_path.clone()),
            fastrace::collector::Config::default(),
        );
        info!("Tracing enabled: output_dir={}", trace_path.display());
    }

    info!("=== Rust LLM Server - Qwen3 (GPU) ===");

    // Load tokenizer
    info!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(MODEL_PATH).expect("Failed to load tokenizer");
    info!("Tokenizer loaded: vocab_size={}", tokenizer.vocab_size());

    // Load GPU model
    info!("Loading model to GPU...");
    let start = Instant::now();
    let model = Qwen3Model::from_safetensors(MODEL_PATH).expect("Failed to load model");
    info!("Model loaded: elapsed_ms={}", start.elapsed().as_millis());

    let rng = StdRng::seed_from_u64(42);
    let state = Arc::new(Mutex::new(AppState { model, tokenizer, rng }));

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

    // Flush pending traces before exit
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
