//! In-process inference benchmark CLI.
//!
//! Usage:
//!   cargo run -r --bin bench_serving -- [GLOBAL_OPTIONS] <SUBCOMMAND> [OPTIONS]
//!
//! Examples:
//!   cargo run -r --bin bench_serving -- request --prompt "Tell me a story" --output-len 128
//!   cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
//!   cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512 --output-lens 32,128
//!   cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32

use std::fmt::Write as _;
use std::fs;
use std::io::{IsTerminal, stdout};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::{ASCII_FULL_CONDENSED, UTF8_FULL_CONDENSED};
use comfy_table::{Cell, CellAlignment, Table};
use log::{debug, info};
use pegainfer::logging;
use pegainfer::model::{
    GenerationState, ModelForward, ModelRuntimeConfig, Qwen3Model, Qwen35Model,
};
use pegainfer::sampler::SamplingParams;
use pegainfer::server_engine::{ModelType, detect_model_type};
use pegainfer::tokenizer::Tokenizer;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");
const DEFAULT_REQUEST_PROMPT: &str = "Tell me a story";
const DEFAULT_CURVE_PROMPT_LEN: usize = 512;
const SYNTHETIC_PATTERN: &str = "token_id = 100 + (idx % 1000)";
const TOP_LEVEL_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- request
  cargo run -r --bin bench_serving -- request --prompt \"Tell me a story about Rust\" --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64
  cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
  cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
  cargo run -r --bin bench_serving -- --format json --out bench.json request --prompt-len 512 --output-len 64";
const REQUEST_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- request
  cargo run -r --bin bench_serving -- request --prompt \"Tell me a story about Rust\" --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-file prompts/story.txt --output-len 128
  cargo run -r --bin bench_serving -- request --prompt-len 512 --output-len 64 --warmup 3 --iters 10";
const MATRIX_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- matrix
  cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
  cargo run -r --bin bench_serving -- --format json --out matrix.json matrix --prompt-lens 128,512 --output-lens 64,256";
const CURVE_EXAMPLES: &str = "\
Examples:
  cargo run -r --bin bench_serving -- curve
  cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
  cargo run -r --bin bench_serving -- curve --prompt \"Summarize KV cache behavior\" --output-len 128 --window 16";

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Measure one request shape end-to-end.
    #[command(after_help = REQUEST_EXAMPLES)]
    Request(RequestArgs),
    /// Sweep prompt_len x output_len and summarize each cell.
    #[command(after_help = MATRIX_EXAMPLES)]
    Matrix(MatrixArgs),
    /// Measure TPOT as context grows during decode.
    #[command(after_help = CURVE_EXAMPLES)]
    Curve(CurveArgs),
}

#[derive(Parser, Debug)]
#[command(
    name = "bench_serving",
    about = "pegainfer in-process inference benchmark",
    after_help = TOP_LEVEL_EXAMPLES
)]
struct Cli {
    /// Model directory (contains config.json, tokenizer, safetensors)
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    model_path: String,

    /// Enable CUDA graph on decode path
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graph: bool,

    /// Render result to terminal as text or structured JSON
    #[arg(long, default_value = "text")]
    format: OutputFormat,

    /// Optional label to tag this benchmark run
    #[arg(long)]
    label: Option<String>,

    /// Optional output path for the rendered report
    #[arg(long)]
    out: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Clone, ClapArgs)]
struct PromptInputArgs {
    /// Inline prompt text
    #[arg(long, conflicts_with_all = ["prompt_file", "prompt_len"])]
    prompt: Option<String>,

    /// Read prompt text from file
    #[arg(long, conflicts_with_all = ["prompt", "prompt_len"])]
    prompt_file: Option<String>,

    /// Use a synthetic prompt with exactly this many token ids
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    prompt_len: Option<usize>,
}

#[derive(Debug, Clone, ClapArgs)]
struct RunArgs {
    /// Warmup iterations
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// Measured iterations
    #[arg(long, default_value_t = 20)]
    iters: usize,

    /// RNG seed (matters once sampling becomes non-greedy)
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Debug, ClapArgs)]
struct RequestArgs {
    #[command(flatten)]
    prompt_input: PromptInputArgs,

    /// Max generated tokens
    #[arg(long, default_value_t = 64)]
    output_len: usize,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, ClapArgs)]
struct MatrixArgs {
    /// Synthetic prompt lengths to sweep
    #[arg(long, value_delimiter = ',', default_value = "32,128,512,2048")]
    prompt_lens: Vec<usize>,

    /// Output lengths to sweep
    #[arg(long, value_delimiter = ',', default_value = "32,128,256")]
    output_lens: Vec<usize>,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, ClapArgs)]
struct CurveArgs {
    #[command(flatten)]
    prompt_input: PromptInputArgs,

    /// Max generated tokens
    #[arg(long, default_value_t = 256)]
    output_len: usize,

    /// Group decode positions into windows of this size
    #[arg(long, default_value_t = 32)]
    window: usize,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Debug, Clone, Serialize)]
struct RunInfo {
    command: &'static str,
    model_path: String,
    model_type: String,
    cuda_graph: bool,
    load_ms: f64,
    label: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PromptDescriptor {
    source: String,
    prompt_tokens: usize,
    prompt_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct DurationStats {
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    samples: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CountStats {
    min: usize,
    max: usize,
    avg: f64,
    samples: usize,
}

#[derive(Debug, Clone, Serialize)]
struct RequestWorkload {
    prompt: PromptDescriptor,
    output_len: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct RequestMetrics {
    ttft_ms: DurationStats,
    first_decode_step_ms: Option<DurationStats>,
    steady_tpot_ms: Option<DurationStats>,
    e2e_ms: DurationStats,
    generated_tokens: CountStats,
    request_tok_s: Option<f64>,
    decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct RequestReport {
    run: RunInfo,
    workload: RequestWorkload,
    metrics: RequestMetrics,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixWorkload {
    prompt_lens: Vec<usize>,
    output_lens: Vec<usize>,
    warmup: usize,
    iters: usize,
    seed: u64,
    synthetic_pattern: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixCell {
    prompt_len: usize,
    output_len: usize,
    ttft_ms: DurationStats,
    e2e_ms: DurationStats,
    first_decode_step_ms: Option<DurationStats>,
    steady_tpot_ms: Option<DurationStats>,
    generated_tokens: CountStats,
    request_tok_s: Option<f64>,
    decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct MatrixReport {
    run: RunInfo,
    workload: MatrixWorkload,
    cells: Vec<MatrixCell>,
}

#[derive(Debug, Clone, Serialize)]
struct CurveWorkload {
    prompt: PromptDescriptor,
    output_len: usize,
    window: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
}

#[derive(Debug, Clone, Serialize)]
struct CurveWindow {
    ctx_start: usize,
    ctx_end: usize,
    tpot_ms: DurationStats,
    decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct CurveReport {
    run: RunInfo,
    workload: CurveWorkload,
    windows: Vec<CurveWindow>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BenchReport {
    Request(RequestReport),
    Matrix(MatrixReport),
    Curve(CurveReport),
}

fn dur_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn percentiles(sorted: &[Duration]) -> (Duration, Duration, Duration, Duration, Duration) {
    assert!(!sorted.is_empty());
    let n = sorted.len();
    let sum: Duration = sorted.iter().sum();
    let avg = sum / n as u32;
    let p = |pct: f64| sorted[((pct / 100.0) * (n - 1) as f64).round() as usize];
    (avg, p(50.0), p(95.0), p(99.0), sorted[n - 1])
}

fn summarize_durations(samples: &[Duration]) -> DurationStats {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let (avg, p50, p95, p99, max) = percentiles(&sorted);
    DurationStats {
        avg_ms: dur_ms(avg),
        p50_ms: dur_ms(p50),
        p95_ms: dur_ms(p95),
        p99_ms: dur_ms(p99),
        max_ms: dur_ms(max),
        samples: sorted.len(),
    }
}

fn summarize_counts(samples: &[usize]) -> CountStats {
    assert!(!samples.is_empty());
    let min = *samples.iter().min().unwrap();
    let max = *samples.iter().max().unwrap();
    let sum: usize = samples.iter().sum();
    CountStats {
        min,
        max,
        avg: sum as f64 / samples.len() as f64,
        samples: samples.len(),
    }
}

fn aggregate_tok_s(tokens: usize, total: Duration) -> Option<f64> {
    if tokens == 0 || total.is_zero() {
        None
    } else {
        Some(tokens as f64 / total.as_secs_f64())
    }
}

fn new_table() -> Table {
    let mut table = Table::new();
    if stdout().is_terminal() {
        table.load_preset(UTF8_FULL_CONDENSED);
        table.apply_modifier(UTF8_ROUND_CORNERS);
    } else {
        table.load_preset(ASCII_FULL_CONDENSED);
    }
    table
}

fn key_cell(label: impl Into<String>) -> Cell {
    Cell::new(label.into())
}

fn value_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into())
}

fn numeric_cell(value: impl Into<String>) -> Cell {
    Cell::new(value.into()).set_alignment(CellAlignment::Right)
}

fn format_rate(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.2}"))
        .unwrap_or_else(|| "-".to_string())
}

fn format_duration_ms(value: f64) -> String {
    format!("{value:.2}")
}

fn format_count_avg(value: f64) -> String {
    format!("{value:.2}")
}

fn push_table(out: &mut String, table: &Table) {
    out.push_str(&table.to_string());
    out.push('\n');
}

fn render_run_summary(report: &RunInfo) -> Table {
    let mut table = new_table();
    table.add_row(vec![
        key_cell("model"),
        value_cell(format!("{} ({})", report.model_path, report.model_type)),
    ]);
    table.add_row(vec![
        key_cell("cuda_graph"),
        value_cell(report.cuda_graph.to_string()),
    ]);
    table.add_row(vec![
        key_cell("load_ms"),
        numeric_cell(format_duration_ms(report.load_ms)),
    ]);
    if let Some(label) = &report.label {
        table.add_row(vec![key_cell("label"), value_cell(label.clone())]);
    }
    table
}

fn render_request_meta(report: &RequestReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_source"),
        value_cell(report.workload.prompt.source.clone()),
    ]);
    table.add_row(vec![
        key_cell("prompt_tokens"),
        numeric_cell(report.workload.prompt.prompt_tokens.to_string()),
    ]);
    if let Some(preview) = &report.workload.prompt.prompt_preview {
        table.add_row(vec![
            key_cell("prompt"),
            value_cell(format!("\"{preview}\"")),
        ]);
    }
    table.add_row(vec![
        key_cell("output_len"),
        numeric_cell(report.workload.output_len.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

fn render_duration_table(rows: Vec<(String, DurationStats)>) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("avg_ms").set_alignment(CellAlignment::Right),
        Cell::new("p50_ms").set_alignment(CellAlignment::Right),
        Cell::new("p95_ms").set_alignment(CellAlignment::Right),
        Cell::new("p99_ms").set_alignment(CellAlignment::Right),
        Cell::new("max_ms").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for (label, stats) in rows {
        table.add_row(vec![
            key_cell(label),
            numeric_cell(format_duration_ms(stats.avg_ms)),
            numeric_cell(format_duration_ms(stats.p50_ms)),
            numeric_cell(format_duration_ms(stats.p95_ms)),
            numeric_cell(format_duration_ms(stats.p99_ms)),
            numeric_cell(format_duration_ms(stats.max_ms)),
            numeric_cell(stats.samples.to_string()),
        ]);
    }
    table
}

fn render_request_summary(report: &RequestReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("metric"),
        Cell::new("value").set_alignment(CellAlignment::Right),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_avg"),
        numeric_cell(format_count_avg(report.metrics.generated_tokens.avg)),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_min"),
        numeric_cell(report.metrics.generated_tokens.min.to_string()),
    ]);
    table.add_row(vec![
        key_cell("generated_tokens_max"),
        numeric_cell(report.metrics.generated_tokens.max.to_string()),
    ]);
    table.add_row(vec![
        key_cell("generated_token_runs"),
        numeric_cell(report.metrics.generated_tokens.samples.to_string()),
    ]);
    table.add_row(vec![
        key_cell("request_tok_s"),
        numeric_cell(format_rate(report.metrics.request_tok_s)),
    ]);
    table.add_row(vec![
        key_cell("decode_tok_s"),
        numeric_cell(format_rate(report.metrics.decode_tok_s)),
    ]);
    table
}

fn render_matrix_meta(report: &MatrixReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_lens"),
        value_cell(
            report
                .workload
                .prompt_lens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    table.add_row(vec![
        key_cell("output_lens"),
        value_cell(
            report
                .workload
                .output_lens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ),
    ]);
    table.add_row(vec![
        key_cell("synthetic_pattern"),
        value_cell(report.workload.synthetic_pattern),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

fn render_matrix_table(report: &MatrixReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("prompt_tok").set_alignment(CellAlignment::Right),
        Cell::new("output_tok").set_alignment(CellAlignment::Right),
        Cell::new("ttft_avg").set_alignment(CellAlignment::Right),
        Cell::new("ttft_p95").set_alignment(CellAlignment::Right),
        Cell::new("e2e_avg").set_alignment(CellAlignment::Right),
        Cell::new("req_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("decode_tok/s").set_alignment(CellAlignment::Right),
        Cell::new("gen_avg").set_alignment(CellAlignment::Right),
    ]);
    for cell in &report.cells {
        table.add_row(vec![
            numeric_cell(cell.prompt_len.to_string()),
            numeric_cell(cell.output_len.to_string()),
            numeric_cell(format_duration_ms(cell.ttft_ms.avg_ms)),
            numeric_cell(format_duration_ms(cell.ttft_ms.p95_ms)),
            numeric_cell(format_duration_ms(cell.e2e_ms.avg_ms)),
            numeric_cell(format_rate(cell.request_tok_s)),
            numeric_cell(format_rate(cell.decode_tok_s)),
            numeric_cell(format_count_avg(cell.generated_tokens.avg)),
        ]);
    }
    table
}

fn render_curve_meta(report: &CurveReport) -> Table {
    let mut table = render_run_summary(&report.run);
    table.add_row(vec![
        key_cell("prompt_source"),
        value_cell(report.workload.prompt.source.clone()),
    ]);
    table.add_row(vec![
        key_cell("prompt_tokens"),
        numeric_cell(report.workload.prompt.prompt_tokens.to_string()),
    ]);
    if let Some(preview) = &report.workload.prompt.prompt_preview {
        table.add_row(vec![
            key_cell("prompt"),
            value_cell(format!("\"{preview}\"")),
        ]);
    }
    table.add_row(vec![
        key_cell("output_len"),
        numeric_cell(report.workload.output_len.to_string()),
    ]);
    table.add_row(vec![
        key_cell("window"),
        numeric_cell(report.workload.window.to_string()),
    ]);
    table.add_row(vec![
        key_cell("warmup / iters"),
        value_cell(format!(
            "{} / {}",
            report.workload.warmup, report.workload.iters
        )),
    ]);
    table.add_row(vec![
        key_cell("seed"),
        numeric_cell(report.workload.seed.to_string()),
    ]);
    table
}

fn render_curve_table(report: &CurveReport) -> Table {
    let mut table = new_table();
    table.set_header(vec![
        Cell::new("ctx_range"),
        Cell::new("avg_ms").set_alignment(CellAlignment::Right),
        Cell::new("p50_ms").set_alignment(CellAlignment::Right),
        Cell::new("p95_ms").set_alignment(CellAlignment::Right),
        Cell::new("p99_ms").set_alignment(CellAlignment::Right),
        Cell::new("tok/s").set_alignment(CellAlignment::Right),
        Cell::new("samples").set_alignment(CellAlignment::Right),
    ]);
    for window in &report.windows {
        table.add_row(vec![
            value_cell(format!("{}-{}", window.ctx_start, window.ctx_end)),
            numeric_cell(format_duration_ms(window.tpot_ms.avg_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p50_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p95_ms)),
            numeric_cell(format_duration_ms(window.tpot_ms.p99_ms)),
            numeric_cell(format_rate(window.decode_tok_s)),
            numeric_cell(window.tpot_ms.samples.to_string()),
        ]);
    }
    table
}

fn truncate_preview(text: &str, limit: usize) -> String {
    let one_line = text.replace('\n', "\\n");
    if one_line.chars().count() <= limit {
        return one_line;
    }
    let mut truncated = String::new();
    for ch in one_line.chars().take(limit) {
        truncated.push(ch);
    }
    truncated.push_str("...");
    truncated
}

fn synthetic_prompt_tokens(len: usize) -> Vec<u32> {
    (0..len).map(|i| ((i % 1000) + 100) as u32).collect()
}

#[derive(Debug, Clone)]
struct PromptSpec {
    descriptor: PromptDescriptor,
    tokens: Vec<u32>,
}

fn resolve_prompt_input(
    args: &PromptInputArgs,
    tokenizer: &Tokenizer,
    default_text: Option<&str>,
    default_prompt_len: Option<usize>,
) -> Result<PromptSpec> {
    match (&args.prompt, &args.prompt_file, args.prompt_len) {
        (Some(prompt), None, None) => Ok(PromptSpec {
            descriptor: PromptDescriptor {
                source: "text".to_string(),
                prompt_tokens: tokenizer.encode(prompt)?.len(),
                prompt_preview: Some(truncate_preview(prompt, 96)),
            },
            tokens: tokenizer.encode(prompt)?,
        }),
        (None, Some(path), None) => {
            let prompt = fs::read_to_string(path)
                .with_context(|| format!("failed to read prompt file: {path}"))?;
            let tokens = tokenizer.encode(&prompt)?;
            Ok(PromptSpec {
                descriptor: PromptDescriptor {
                    source: format!("file:{path}"),
                    prompt_tokens: tokens.len(),
                    prompt_preview: Some(truncate_preview(&prompt, 96)),
                },
                tokens,
            })
        }
        (None, None, Some(prompt_len)) => {
            ensure!(prompt_len > 0, "--prompt-len must be > 0");
            Ok(PromptSpec {
                descriptor: PromptDescriptor {
                    source: format!("synthetic:{SYNTHETIC_PATTERN}"),
                    prompt_tokens: prompt_len,
                    prompt_preview: None,
                },
                tokens: synthetic_prompt_tokens(prompt_len),
            })
        }
        (None, None, None) => {
            if let Some(prompt) = default_text {
                let tokens = tokenizer.encode(prompt)?;
                Ok(PromptSpec {
                    descriptor: PromptDescriptor {
                        source: "text".to_string(),
                        prompt_tokens: tokens.len(),
                        prompt_preview: Some(truncate_preview(prompt, 96)),
                    },
                    tokens,
                })
            } else if let Some(prompt_len) = default_prompt_len {
                Ok(PromptSpec {
                    descriptor: PromptDescriptor {
                        source: format!("synthetic:{SYNTHETIC_PATTERN}"),
                        prompt_tokens: prompt_len,
                        prompt_preview: None,
                    },
                    tokens: synthetic_prompt_tokens(prompt_len),
                })
            } else {
                unreachable!("default prompt source must be provided");
            }
        }
        _ => unreachable!("clap enforces prompt input conflicts"),
    }
}

struct GenTimings {
    ttft: Duration,
    tbt: Vec<Duration>,
    total: Duration,
    emitted_tokens: usize,
}

trait BenchModel {
    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        rng: &mut StdRng,
    ) -> GenTimings;
}

fn run_timed<F>(prompt_tokens: &[u32], max_new_tokens: usize, mut generate: F) -> GenTimings
where
    F: FnMut(&[u32], usize, &mut dyn FnMut(u32) -> bool) -> Result<()>,
{
    let start = Instant::now();
    let mut first_at: Option<Instant> = None;
    let mut prev_at: Option<Instant> = None;
    let mut emitted_tokens = 0usize;
    let mut tbt = Vec::with_capacity(max_new_tokens.saturating_sub(1));

    generate(prompt_tokens, max_new_tokens, &mut |_tok| {
        let now = Instant::now();
        emitted_tokens += 1;
        if first_at.is_none() {
            first_at = Some(now);
        } else if let Some(prev) = prev_at {
            tbt.push(now - prev);
        }
        prev_at = Some(now);
        true
    })
    .expect("generation failed");

    let total = start.elapsed();
    let ttft = first_at.map_or(total, |t| t - start);
    GenTimings {
        ttft,
        tbt,
        total,
        emitted_tokens,
    }
}

struct ModelWithState<M: ModelForward> {
    model: M,
    state: M::State,
}

impl<M: ModelForward> BenchModel for ModelWithState<M> {
    fn timed_generation(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        sampling: &SamplingParams,
        rng: &mut StdRng,
    ) -> GenTimings {
        run_timed(prompt_tokens, max_new_tokens, |toks, n, cb| {
            self.state.reset()?;
            self.model.forward(toks, &mut self.state)?;
            let mut last = self.model.select_token(&mut self.state, sampling, rng)?;
            if !cb(last) {
                return Ok(());
            }
            for _ in 1..n {
                self.model.forward(&[last], &mut self.state)?;
                last = self.model.select_token(&mut self.state, sampling, rng)?;
                if !cb(last) {
                    break;
                }
            }
            Ok(())
        })
    }
}

fn normalize_sizes(values: &[usize], flag: &str) -> Result<Vec<usize>> {
    ensure!(!values.is_empty(), "{flag} must not be empty");
    ensure!(values.iter().all(|v| *v > 0), "{flag} values must be > 0");
    let mut normalized = values.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    Ok(normalized)
}

fn validate_run_args(args: &RunArgs) -> Result<()> {
    ensure!(args.iters > 0, "--iters must be > 0");
    Ok(())
}

fn measure_timings(
    model: &mut dyn BenchModel,
    prompt_tokens: &[u32],
    output_len: usize,
    run: &RunArgs,
) -> Result<Vec<GenTimings>> {
    ensure!(output_len > 0, "--output-len must be > 0");
    validate_run_args(run)?;

    let sampling = SamplingParams::default();
    let mut rng = StdRng::seed_from_u64(run.seed);

    for _ in 0..run.warmup {
        model.timed_generation(prompt_tokens, output_len, &sampling, &mut rng);
    }

    let mut timings = Vec::with_capacity(run.iters);
    for _ in 0..run.iters {
        timings.push(model.timed_generation(prompt_tokens, output_len, &sampling, &mut rng));
    }
    Ok(timings)
}

fn build_request_metrics(timings: &[GenTimings]) -> RequestMetrics {
    let ttfts: Vec<Duration> = timings.iter().map(|t| t.ttft).collect();
    let e2e: Vec<Duration> = timings.iter().map(|t| t.total).collect();
    let first_steps: Vec<Duration> = timings
        .iter()
        .filter_map(|t| t.tbt.first().copied())
        .collect();
    let steady: Vec<Duration> = timings
        .iter()
        .flat_map(|t| t.tbt.iter().skip(1).copied())
        .collect();
    let generated: Vec<usize> = timings.iter().map(|t| t.emitted_tokens).collect();

    let total_emitted: usize = timings.iter().map(|t| t.emitted_tokens).sum();
    let total_request_time: Duration = timings.iter().map(|t| t.total).sum();
    let total_decode_steps: usize = timings
        .iter()
        .map(|t| t.emitted_tokens.saturating_sub(1))
        .sum();
    let total_decode_time: Duration = timings.iter().flat_map(|t| t.tbt.iter()).copied().sum();

    RequestMetrics {
        ttft_ms: summarize_durations(&ttfts),
        first_decode_step_ms: (!first_steps.is_empty()).then(|| summarize_durations(&first_steps)),
        steady_tpot_ms: (!steady.is_empty()).then(|| summarize_durations(&steady)),
        e2e_ms: summarize_durations(&e2e),
        generated_tokens: summarize_counts(&generated),
        request_tok_s: aggregate_tok_s(total_emitted, total_request_time),
        decode_tok_s: aggregate_tok_s(total_decode_steps, total_decode_time),
    }
}

fn run_info(cli: &Cli, command: &'static str, model_type: ModelType, load_ms: f64) -> RunInfo {
    RunInfo {
        command,
        model_path: cli.model_path.clone(),
        model_type: format!("{model_type:?}"),
        cuda_graph: cli.cuda_graph,
        load_ms,
        label: cli.label.clone(),
    }
}

fn bench_request(
    model: &mut dyn BenchModel,
    tokenizer: &Tokenizer,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    args: &RequestArgs,
) -> Result<BenchReport> {
    let prompt = resolve_prompt_input(
        &args.prompt_input,
        tokenizer,
        Some(DEFAULT_REQUEST_PROMPT),
        None,
    )?;
    info!(
        "Starting request benchmark: prompt_tokens={} output_len={} warmup={} iters={} seed={}",
        prompt.descriptor.prompt_tokens,
        args.output_len,
        args.run.warmup,
        args.run.iters,
        args.run.seed
    );
    let timings = measure_timings(model, &prompt.tokens, args.output_len, &args.run)?;
    Ok(BenchReport::Request(RequestReport {
        run: run_info(cli, "request", model_type, load_ms),
        workload: RequestWorkload {
            prompt: prompt.descriptor,
            output_len: args.output_len,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
        },
        metrics: build_request_metrics(&timings),
    }))
}

fn bench_matrix(
    model: &mut dyn BenchModel,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    args: &MatrixArgs,
) -> Result<BenchReport> {
    validate_run_args(&args.run)?;
    let prompt_lens = normalize_sizes(&args.prompt_lens, "--prompt-lens")?;
    let output_lens = normalize_sizes(&args.output_lens, "--output-lens")?;
    info!(
        "Starting matrix benchmark: prompt_lens={:?} output_lens={:?} warmup={} iters={} seed={}",
        prompt_lens, output_lens, args.run.warmup, args.run.iters, args.run.seed
    );

    let mut cells = Vec::with_capacity(prompt_lens.len() * output_lens.len());
    for &prompt_len in &prompt_lens {
        let prompt_tokens = synthetic_prompt_tokens(prompt_len);
        for &output_len in &output_lens {
            debug!(
                "Running matrix cell: prompt_len={} output_len={}",
                prompt_len, output_len
            );
            let timings = measure_timings(model, &prompt_tokens, output_len, &args.run)?;
            let metrics = build_request_metrics(&timings);
            cells.push(MatrixCell {
                prompt_len,
                output_len,
                ttft_ms: metrics.ttft_ms,
                e2e_ms: metrics.e2e_ms,
                first_decode_step_ms: metrics.first_decode_step_ms,
                steady_tpot_ms: metrics.steady_tpot_ms,
                generated_tokens: metrics.generated_tokens,
                request_tok_s: metrics.request_tok_s,
                decode_tok_s: metrics.decode_tok_s,
            });
        }
    }

    Ok(BenchReport::Matrix(MatrixReport {
        run: run_info(cli, "matrix", model_type, load_ms),
        workload: MatrixWorkload {
            prompt_lens,
            output_lens,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
            synthetic_pattern: SYNTHETIC_PATTERN,
        },
        cells,
    }))
}

fn bench_curve(
    model: &mut dyn BenchModel,
    tokenizer: &Tokenizer,
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    args: &CurveArgs,
) -> Result<BenchReport> {
    ensure!(args.window > 0, "--window must be > 0");
    ensure!(args.output_len >= 2, "--output-len must be >= 2 for curve");

    let prompt = resolve_prompt_input(
        &args.prompt_input,
        tokenizer,
        None,
        Some(DEFAULT_CURVE_PROMPT_LEN),
    )?;
    info!(
        "Starting curve benchmark: prompt_tokens={} output_len={} window={} warmup={} iters={} seed={}",
        prompt.descriptor.prompt_tokens,
        args.output_len,
        args.window,
        args.run.warmup,
        args.run.iters,
        args.run.seed
    );
    let timings = measure_timings(model, &prompt.tokens, args.output_len, &args.run)?;

    let mut tbt_by_pos: Vec<Vec<Duration>> = Vec::new();
    for timing in &timings {
        for (idx, &duration) in timing.tbt.iter().enumerate() {
            if idx >= tbt_by_pos.len() {
                tbt_by_pos.push(Vec::with_capacity(args.run.iters));
            }
            tbt_by_pos[idx].push(duration);
        }
    }

    let mut windows = Vec::new();
    let mut pos = 0usize;
    while pos < tbt_by_pos.len() {
        let end = (pos + args.window).min(tbt_by_pos.len());
        let mut samples = Vec::new();
        for bucket in &tbt_by_pos[pos..end] {
            samples.extend_from_slice(bucket);
        }
        if !samples.is_empty() {
            let stats = summarize_durations(&samples);
            windows.push(CurveWindow {
                ctx_start: prompt.descriptor.prompt_tokens + pos + 1,
                ctx_end: prompt.descriptor.prompt_tokens + end,
                decode_tok_s: (stats.avg_ms > 0.0).then(|| 1000.0 / stats.avg_ms),
                tpot_ms: stats,
            });
        }
        pos = end;
    }

    Ok(BenchReport::Curve(CurveReport {
        run: run_info(cli, "curve", model_type, load_ms),
        workload: CurveWorkload {
            prompt: prompt.descriptor,
            output_len: args.output_len,
            window: args.window,
            warmup: args.run.warmup,
            iters: args.run.iters,
            seed: args.run.seed,
        },
        windows,
    }))
}

fn render_text(report: &BenchReport) -> String {
    let mut out = String::new();
    match report {
        BenchReport::Request(report) => {
            let _ = writeln!(out, "bench_serving request\n");
            push_table(&mut out, &render_request_meta(report));
            out.push('\n');
            push_table(
                &mut out,
                &render_duration_table(
                    std::iter::once(("ttft_ms".to_string(), report.metrics.ttft_ms.clone()))
                        .chain(
                            report
                                .metrics
                                .first_decode_step_ms
                                .clone()
                                .into_iter()
                                .map(|stats| ("first_decode_step_ms".to_string(), stats)),
                        )
                        .chain(
                            report
                                .metrics
                                .steady_tpot_ms
                                .clone()
                                .into_iter()
                                .map(|stats| ("steady_tpot_ms".to_string(), stats)),
                        )
                        .chain(std::iter::once((
                            "e2e_ms".to_string(),
                            report.metrics.e2e_ms.clone(),
                        )))
                        .collect(),
                ),
            );
            out.push('\n');
            push_table(&mut out, &render_request_summary(report));
        }
        BenchReport::Matrix(report) => {
            let _ = writeln!(out, "bench_serving matrix\n");
            push_table(&mut out, &render_matrix_meta(report));
            out.push('\n');
            push_table(&mut out, &render_matrix_table(report));
        }
        BenchReport::Curve(report) => {
            let _ = writeln!(out, "bench_serving curve\n");
            push_table(&mut out, &render_curve_meta(report));
            out.push('\n');
            push_table(&mut out, &render_curve_table(report));
        }
    }
    out
}

fn emit_report(cli: &Cli, report: &BenchReport) -> Result<()> {
    let rendered = match cli.format {
        OutputFormat::Text => render_text(report),
        OutputFormat::Json => serde_json::to_string_pretty(report)?,
    };

    if let Some(path) = &cli.out {
        fs::write(path, &rendered).with_context(|| format!("failed to write report to {path}"))?;
        info!("Wrote benchmark report to {}", path);
    }

    println!("{rendered}");
    Ok(())
}

fn run_command(
    cli: &Cli,
    model_type: ModelType,
    load_ms: f64,
    model: &mut dyn BenchModel,
    tokenizer: &Tokenizer,
) -> Result<BenchReport> {
    match &cli.command {
        Command::Request(args) => bench_request(model, tokenizer, cli, model_type, load_ms, args),
        Command::Matrix(args) => bench_matrix(model, cli, model_type, load_ms, args),
        Command::Curve(args) => bench_curve(model, tokenizer, cli, model_type, load_ms, args),
    }
}

fn main() -> Result<()> {
    logging::init_default();

    let cli = Cli::parse();
    debug!(
        "bench_serving starting: command={} model_path={} cuda_graph={} format={:?}",
        match &cli.command {
            Command::Request(_) => "request",
            Command::Matrix(_) => "matrix",
            Command::Curve(_) => "curve",
        },
        cli.model_path,
        cli.cuda_graph,
        cli.format
    );
    let model_type = detect_model_type(&cli.model_path)?;
    debug!("Detected model type: {:?}", model_type);
    let runtime = ModelRuntimeConfig {
        enable_cuda_graph: cli.cuda_graph,
    };
    let load_start = Instant::now();

    let report = match model_type {
        ModelType::Qwen3 => {
            let model = Qwen3Model::from_safetensors_with_runtime(&cli.model_path, runtime)?;
            let state = model.create_state()?;
            let tokenizer = Tokenizer::from_file(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = ModelWithState { model, state };
            run_command(&cli, model_type, load_ms, &mut bench, &tokenizer)?
        }
        ModelType::Qwen35 => {
            let model = Qwen35Model::from_safetensors_with_options(
                &cli.model_path,
                runtime.enable_cuda_graph,
            )?;
            let state = model.create_state()?;
            let tokenizer = Tokenizer::from_file(&cli.model_path)?;
            let load_ms = dur_ms(load_start.elapsed());
            let mut bench = ModelWithState { model, state };
            run_command(&cli, model_type, load_ms, &mut bench, &tokenizer)?
        }
    };

    emit_report(&cli, &report)
}
