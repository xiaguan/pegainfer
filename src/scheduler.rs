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
use crate::model_executor::{ModelExecutor, Qwen3Executor, RequestId};
use crate::sampler::SamplingParams;
use crate::server_engine::{FinishReason, TokenLogprob};

use self::effects::apply_effects;
use self::plan::{build_next_plan, execute_plan};
use self::resolve::resolve_step;

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
    pub(super) request_id: RequestId,
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) last_token: u32,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    pub(super) logprobs: usize,
}

pub(super) struct PendingRequest {
    pub(super) request_id: RequestId,
    pub(super) prompt_tokens: Vec<u32>,
    pub(super) params: SamplingParams,
    pub(super) max_tokens: usize,
    pub(super) token_tx: mpsc::UnboundedSender<TokenEvent>,
    pub(super) logprobs: usize,
    pub(super) echo: bool,
}

impl PendingRequest {
    fn from_scheduler_request(request_id: RequestId, req: SchedulerRequest) -> Self {
        Self {
            request_id,
            prompt_tokens: req.prompt_tokens,
            params: req.params,
            max_tokens: req.max_tokens,
            token_tx: req.token_tx,
            logprobs: req.logprobs,
            echo: req.echo,
        }
    }
}

// ── Entry point ─────────────────────────────────────────────────────────

/// Start the scheduler thread. Returns a handle for submitting requests.
///
/// The scheduler exclusively owns request lifecycle and batching decisions.
pub fn start(model: Qwen3Model, seed: u64) -> Result<SchedulerHandle> {
    let executor = Qwen3Executor::single(model)?;
    Ok(start_with_executor(executor, seed))
}

pub fn start_qwen3(
    model_path: &str,
    enable_cuda_graph: bool,
    device_ordinals: &[usize],
    seed: u64,
) -> Result<SchedulerHandle> {
    let executor = Qwen3Executor::from_runtime(model_path, enable_cuda_graph, device_ordinals)?;
    Ok(start_with_executor(executor, seed))
}

pub(crate) fn start_with_executor(executor: Qwen3Executor, seed: u64) -> SchedulerHandle {
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(executor, submit_rx, seed);
        })
        .expect("failed to spawn scheduler thread");

    SchedulerHandle { submit_tx }
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop(
    mut executor: Qwen3Executor,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequestState> = Vec::new();
    let mut next_request_id = 0u64;
    // Requests that could not be admitted due to KV budget pressure.
    // Held here so they aren't lost; re-evaluated every loop iteration.
    let mut deferred: Vec<PendingRequest> = Vec::new();

    info!("Scheduler ready");

    loop {
        // 1. Drain all incoming requests into deferred.
        while let Ok(req) = submit_rx.try_recv() {
            deferred.push(PendingRequest::from_scheduler_request(
                RequestId(next_request_id),
                req,
            ));
            next_request_id += 1;
        }

        // 2. Nothing active and nothing deferred → block until a request arrives.
        if active.is_empty() && deferred.is_empty() {
            if let Some(req) = submit_rx.blocking_recv() {
                deferred.push(PendingRequest::from_scheduler_request(
                    RequestId(next_request_id),
                    req,
                ));
                next_request_id += 1;
            } else {
                info!("Scheduler: all handles dropped, exiting");
                return;
            }
            while let Ok(req) = submit_rx.try_recv() {
                deferred.push(PendingRequest::from_scheduler_request(
                    RequestId(next_request_id),
                    req,
                ));
                next_request_id += 1;
            }
        }

        // 3. Admission control: admit deferred requests only if the KV pool
        //    has enough pages for their prefill, after reserving one page per
        //    active request for its next decode step.
        let page_size = executor.page_size();
        let decode_reserve = active.len(); // one page per active request
        let mut budget = executor.available_pages().saturating_sub(decode_reserve);
        let mut pending: Vec<PendingRequest> = Vec::new();
        let mut still_deferred: Vec<PendingRequest> = Vec::new();
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
        let effects = resolve_step(&executor, &active, artifacts);
        apply_effects(&mut executor, &mut active, effects);
    }
}
