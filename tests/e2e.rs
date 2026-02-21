use std::path::PathBuf;
use std::time::{Duration, Instant};

use fastrace::prelude::*;
use log::info;
use tokio::sync::mpsc;

use pegainfer::sampler::SamplingParams;
use pegainfer::server_engine::{
    CompleteRequest, FinishReason, RealServerEngine, ServerEngine, StreamDelta,
};
use pegainfer::trace_reporter::FileReporter;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models/Qwen3-4B");

fn init_logging() {
    pegainfer::logging::init_stderr("info");
}

fn init_tracing() -> PathBuf {
    let trace_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("traces");
    std::fs::create_dir_all(&trace_dir).expect("Failed to create traces dir");
    fastrace::set_reporter(
        FileReporter::new(trace_dir.clone()),
        fastrace::collector::Config::default(),
    );
    info!("Tracing enabled: {}", trace_dir.display());
    trace_dir
}

fn make_request(prompt: &str, max_tokens: usize) -> CompleteRequest {
    CompleteRequest {
        prompt: prompt.to_string(),
        max_tokens,
        sampling: SamplingParams::default(),
    }
}

fn drain_deltas(rx: &mut mpsc::UnboundedReceiver<StreamDelta>) -> Vec<StreamDelta> {
    let mut deltas = Vec::new();
    while let Ok(delta) = rx.try_recv() {
        deltas.push(delta);
    }
    deltas
}

#[test]
fn test_e2e_generation() {
    init_logging();
    let trace_dir = init_tracing();

    info!("Loading engine...");
    let start = Instant::now();
    let mut engine = RealServerEngine::load(MODEL_PATH, 42).expect("Failed to load engine");
    info!("Engine loaded in {:.2?}", start.elapsed());

    let cases: &[(&str, usize)] = &[
        ("Tell me a story", 100),
        ("My name is", 100),
        ("What is 2 + 2?", 30),
        ("今天天气真好", 50),
        ("请介绍一下中国的首都", 50),
        ("Write a Python function to reverse a string", 80),
    ];

    // ── 1. Non-streaming correctness ──────────────────────────────────────

    info!("=== Phase 1: Non-streaming ===");
    for &(prompt, max_tokens) in cases {
        info!("--- complete: \"{}\" ---", prompt);
        let root = Span::root("complete", SpanContext::random());
        let _guard = root.set_local_parent();

        let start = Instant::now();
        let out = engine
            .complete(make_request(prompt, max_tokens))
            .expect("complete() failed");
        let elapsed = start.elapsed();

        let tok_s = out.usage.completion_tokens as f64 / elapsed.as_secs_f64();
        info!(
            "  {} tokens in {:.2?} ({:.1} tok/s) finish={:?}",
            out.usage.completion_tokens, elapsed, tok_s, out.finish_reason
        );
        info!("  Output: \"{}\"", out.text);

        assert!(!out.text.is_empty(), "empty output for: {}", prompt);
        assert_eq!(
            out.usage.prompt_tokens + out.usage.completion_tokens,
            out.usage.total_tokens,
            "usage mismatch for: {}",
            prompt
        );
        if out.usage.completion_tokens >= max_tokens {
            assert_eq!(out.finish_reason, FinishReason::Length);
        }
    }

    // ── 2. Streaming correctness + TTFT/TPOT ────────────────────────────

    info!("=== Phase 2: Streaming ===");
    for &(prompt, max_tokens) in cases {
        info!("--- stream: \"{}\" ---", prompt);
        let root = Span::root("stream", SpanContext::random());
        let _guard = root.set_local_parent();

        let (tx, mut rx) = mpsc::unbounded_channel();
        let req = make_request(prompt, max_tokens);
        let start = Instant::now();

        // Run generation in a background thread, measure delta arrival on main thread.
        std::thread::scope(|s| {
            s.spawn(|| {
                engine.complete_stream(req, tx).expect("complete_stream() failed");
            });

            let mut deltas: Vec<StreamDelta> = Vec::new();
            let mut ttft = Duration::ZERO;
            let mut tpot_intervals: Vec<Duration> = Vec::new();
            let mut prev_time = start;

            loop {
                match rx.try_recv() {
                    Ok(delta) => {
                        let now = Instant::now();
                        if deltas.is_empty() {
                            ttft = now - start;
                        } else if delta.finish_reason.is_none() {
                            tpot_intervals.push(now - prev_time);
                        }
                        prev_time = now;
                        let done = delta.finish_reason.is_some();
                        deltas.push(delta);
                        if done {
                            break;
                        }
                    }
                    Err(mpsc::error::TryRecvError::Empty) => {
                        std::hint::spin_loop();
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }

            assert!(!deltas.is_empty(), "no deltas for: {}", prompt);

            // All but last: finish_reason must be None
            for (i, d) in deltas[..deltas.len() - 1].iter().enumerate() {
                assert!(
                    d.finish_reason.is_none(),
                    "delta {} has unexpected finish_reason for: {}",
                    i,
                    prompt
                );
            }

            // Last delta: finish_reason must be Some
            let last = deltas.last().unwrap();
            assert!(
                last.finish_reason.is_some(),
                "last delta missing finish_reason for: {}",
                prompt
            );

            let streamed: String = deltas.iter().map(|d| d.text_delta.as_str()).collect();
            let token_count = deltas.len() - 1; // exclude final sentinel delta
            let avg_tpot = if tpot_intervals.is_empty() {
                Duration::ZERO
            } else {
                tpot_intervals.iter().sum::<Duration>() / tpot_intervals.len() as u32
            };

            info!(
                "  {} tokens, ttft={:.2?}, avg_tpot={:.2?} ({:.1} tok/s), finish={:?}",
                token_count,
                ttft,
                avg_tpot,
                1.0 / avg_tpot.as_secs_f64(),
                last.finish_reason.unwrap()
            );
            info!("  Streamed: \"{}\"", streamed);
        });
    }

    // ── 3. Streaming / non-streaming consistency (greedy → deterministic) ─

    info!("=== Phase 3: Consistency ===");
    for &(prompt, max_tokens) in cases {
        info!("--- consistency: \"{}\" ---", prompt);

        let non_stream = engine
            .complete(make_request(prompt, max_tokens))
            .expect("complete() failed");

        let (tx, mut rx) = mpsc::unbounded_channel();
        engine
            .complete_stream(make_request(prompt, max_tokens), tx)
            .expect("complete_stream() failed");
        let deltas = drain_deltas(&mut rx);
        let streamed: String = deltas.iter().map(|d| d.text_delta.as_str()).collect();

        assert_eq!(
            non_stream.text, streamed,
            "stream/non-stream text mismatch for: {}",
            prompt
        );
        info!("  PASS ({} chars)", non_stream.text.len());
    }

    // ── 4. Consumer drop safety ───────────────────────────────────────────

    info!("=== Phase 4: Consumer drop ===");
    {
        let (tx, rx) = mpsc::unbounded_channel();
        drop(rx);
        let result = engine.complete_stream(make_request("Hello", 10), tx);
        assert!(
            result.is_ok(),
            "complete_stream should not panic on dropped consumer"
        );
        info!("  PASS: consumer drop handled");
    }

    fastrace::flush();
    info!("Traces written to {}", trace_dir.display());
}
