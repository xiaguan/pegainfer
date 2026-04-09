//! Scheduler: dedicated GPU thread that batches concurrent requests.
//!
//! HTTP handlers tokenize prompts and submit `SchedulerRequest` via channel.
//! The scheduler batch-prefills all pending requests in one forward pass, then
//! batch-decodes all active requests. Per-request tokens flow back through
//! individual channels.

mod effects;
mod plan;
mod resolve;

use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::model::Qwen3Model;
use crate::model_executor::SingleGpuQwen3Executor;
use crate::sampler::SamplingParams;
use crate::server_engine::{FinishReason, TokenLogprob};

use self::effects::apply_effects;
use self::plan::{build_next_plan, execute_plan};
use self::resolve::{SampleScratch, resolve_step};

// ── Public types ────────────────────────────────────────────────────────

/// Request submitted to the scheduler by an HTTP handler.
pub struct SchedulerRequest {
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub token_tx: mpsc::UnboundedSender<TokenEvent>,
    /// Number of top log-probabilities to return per token (0 = disabled).
    pub logprobs: usize,
    /// If true, echo prompt tokens (with logprobs) before generation.
    pub echo: bool,
}

/// Events sent from the scheduler back to a per-request HTTP handler.
pub enum TokenEvent {
    /// A new token was generated.
    Token {
        id: u32,
        logprob: Option<TokenLogprob>,
    },
    /// Echo: prompt tokens with their logprobs (sent before generation tokens).
    PromptTokens {
        ids: Vec<u32>,
        logprobs: Vec<Option<TokenLogprob>>,
    },
    /// Generation finished (EOS, max_tokens, or error).
    Finished {
        finish_reason: FinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
}

/// Handle for submitting requests to the scheduler. Cheaply cloneable.
#[derive(Clone)]
pub struct SchedulerHandle {
    pub(crate) submit_tx: mpsc::UnboundedSender<SchedulerRequest>,
}

impl SchedulerHandle {
    /// Submit a request. Returns Err if the scheduler thread has exited.
    pub fn submit(
        &self,
        req: SchedulerRequest,
    ) -> Result<(), mpsc::error::SendError<SchedulerRequest>> {
        self.submit_tx.send(req)
    }
}

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded.
pub(super) struct ActiveRequestState {
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) kv: crate::kv_pool::KvState,
    pub(super) last_token: u32,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    pub(super) logprobs: usize,
}

// ── Entry point ─────────────────────────────────────────────────────────

/// Start the scheduler thread. Returns a handle for submitting requests.
///
/// The scheduler exclusively owns the request lifecycle and KV allocation state.
/// No Mutex — only this thread touches the GPU.
pub fn start(model: Qwen3Model, seed: u64) -> Result<SchedulerHandle> {
    let executor = SingleGpuQwen3Executor::new(model)?;

    // Sampling scratch — reused across prefill and unified-step sampling
    let sample_scratch = SampleScratch::new(&executor)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(executor, submit_rx, sample_scratch, seed);
        })
        .expect("failed to spawn scheduler thread");

    Ok(SchedulerHandle { submit_tx })
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop(
    mut executor: SingleGpuQwen3Executor,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    mut sample_scratch: SampleScratch,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequestState> = Vec::new();
    // Requests that could not be admitted due to KV budget pressure.
    // Held here so they aren't lost; re-evaluated every loop iteration.
    let mut deferred: Vec<SchedulerRequest> = Vec::new();

    info!("Scheduler ready");

    loop {
        // 1. Drain all incoming requests into deferred.
        while let Ok(req) = submit_rx.try_recv() {
            deferred.push(req);
        }

        // 2. Nothing active and nothing deferred → block until a request arrives.
        if active.is_empty() && deferred.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => deferred.push(req),
                None => {
                    info!("Scheduler: all handles dropped, exiting");
                    return;
                }
            }
            while let Ok(req) = submit_rx.try_recv() {
                deferred.push(req);
            }
        }

        // 3. Admission control: admit deferred requests only if the KV pool
        //    has enough pages for their prefill, after reserving one page per
        //    active request for its next decode step.
        let page_size = executor.kv_pool().layout().page_size;
        let decode_reserve = active.len(); // one page per active request
        let mut budget = executor
            .kv_pool()
            .available_pages()
            .saturating_sub(decode_reserve);
        let mut pending: Vec<SchedulerRequest> = Vec::new();
        let mut still_deferred: Vec<SchedulerRequest> = Vec::new();
        for req in deferred.drain(..) {
            let needed = req.prompt_tokens.len().div_ceil(page_size);
            if needed <= budget {
                budget -= needed;
                pending.push(req);
            } else {
                still_deferred.push(req);
            }
        }
        deferred = still_deferred;

        let Some(plan) = build_next_plan(!active.is_empty(), pending) else {
            continue;
        };
        let artifacts = match execute_plan(&mut executor, &mut active, plan, &mut rng) {
            Ok(v) => v,
            Err(e) => {
                warn!("Execution step failed: {e}");
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Stop,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                continue;
            }
        };
        let effects = resolve_step(&executor, &active, artifacts, &mut sample_scratch, &mut rng);
        apply_effects(&mut active, effects);
    }
}
