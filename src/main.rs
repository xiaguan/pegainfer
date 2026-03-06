use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use log::info;
use pegainfer::http_server::build_app;
use pegainfer::logging;
use pegainfer::server_engine::{EngineOptions, RealServerEngine};
use pegainfer::trace_reporter::FileReporter;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "pegainfer", about = "Qwen3 GPU inference server")]
struct Args {
    /// Model directory containing config, tokenizer, and safetensor shards
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: PathBuf,

    /// Port to listen on
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Enable CUDA Graph capture/replay on decode path (`--cuda-graph=false` to disable)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// Enable request tracing and write trace JSON files to this directory
    #[arg(long)]
    trace_output_path: Option<PathBuf>,
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
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: model_path={}, cuda_graph={}",
        args.model_path.display(),
        args.cuda_graph
    );
    let engine = RealServerEngine::load_with_options(
        args.model_path
            .to_str()
            .expect("Model path must be valid UTF-8"),
        42,
        EngineOptions {
            enable_cuda_graph: args.cuda_graph,
        },
    )
    .expect("Failed to load engine");
    info!(
        "Engine loaded: elapsed_ms={}, vocab_size={}",
        start.elapsed().as_millis(),
        engine.vocab_size()
    );

    let app = build_app(Box::new(engine));
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
