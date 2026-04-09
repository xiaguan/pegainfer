use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use log::info;
use pegainfer::http_server::build_app;
use pegainfer::logging;
use pegainfer::model::{ModelRuntimeConfig, Qwen3Model, Qwen35Model};
use pegainfer::server_engine::{ModelType, detect_model_type};
use pegainfer::tokenizer::Tokenizer;
use pegainfer::trace_reporter::FileReporter;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

#[derive(Parser)]
#[command(name = "pegainfer", about = "Qwen3/3.5 GPU inference server")]
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

    /// CUDA device ordinal for single-GPU Qwen3 loads
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,

    /// Tensor-parallel world size for Qwen3
    #[arg(long, default_value_t = 1)]
    tp_size: usize,

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

    let model_path = args
        .model_path
        .to_str()
        .expect("Model path must be valid UTF-8");
    let model_type = detect_model_type(model_path).expect("Failed to detect model type");

    info!("=== Rust LLM Server - {} (GPU) ===", model_type);
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime options: model_path={}, cuda_graph={}, device_ordinal={}, tp_size={}",
        args.model_path.display(),
        args.cuda_graph,
        args.device_ordinal,
        args.tp_size
    );

    let app = match model_type {
        ModelType::Qwen3 => {
            let tokenizer =
                Arc::new(Tokenizer::from_file(model_path).expect("Failed to load tokenizer"));
            let model_id = pegainfer::server_engine::model_id_from_path(model_path);
            let handle = if args.tp_size == 1 {
                let model = Qwen3Model::from_safetensors_with_runtime(
                    model_path,
                    ModelRuntimeConfig {
                        enable_cuda_graph: args.cuda_graph,
                        tensor_parallel: None,
                        device_ordinal: args.device_ordinal,
                    },
                )
                .expect("Failed to load Qwen3 model");
                pegainfer::scheduler::start(model, 42).expect("Failed to start scheduler")
            } else {
                let device_ordinals: Vec<usize> = (0..args.tp_size).collect();
                pegainfer::scheduler::start_tensor_parallel(
                    model_path,
                    args.cuda_graph,
                    &device_ordinals,
                    42,
                )
                .expect("Failed to start tensor-parallel scheduler")
            };

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis(),);

            build_app(handle, tokenizer, model_id)
        }
        ModelType::Qwen35 => {
            let model = Qwen35Model::from_safetensors_with_options(model_path, args.cuda_graph)
                .expect("Failed to load Qwen3.5 model");

            let tokenizer =
                Arc::new(Tokenizer::from_file(model_path).expect("Failed to load tokenizer"));
            let model_id = pegainfer::server_engine::model_id_from_path(model_path);

            let handle = pegainfer::scheduler_qwen35::start(model, 42)
                .expect("Failed to start Qwen3.5 scheduler");

            info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());

            build_app(handle, tokenizer, model_id)
        }
    };

    let addr = format!("0.0.0.0:{}", args.port);
    info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(
        axum::serve::ListenerExt::tap_io(listener, |tcp_stream| {
            let _ = tcp_stream.set_nodelay(true);
        }),
        app,
    )
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
