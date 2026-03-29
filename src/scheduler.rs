//! Scheduler: dedicated GPU thread that batches concurrent requests.
//!
//! HTTP handlers tokenize prompts and submit `SchedulerRequest` via channel.
//! The scheduler prefills serially (one request at a time) then batch-decodes
//! all active requests in a single forward pass. Per-request tokens flow back
//! through individual channels.

use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::kv_pool::KvState;
use crate::model::qwen3::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::model::{ModelForward, Qwen3Model, Qwen3State};
use crate::sampler::SamplingParams;
use crate::server_engine::FinishReason;

// ── Public types ────────────────────────────────────────────────────────

/// Request submitted to the scheduler by an HTTP handler.
pub struct SchedulerRequest {
    pub prompt_tokens: Vec<u32>,
    pub params: SamplingParams,
    pub max_tokens: usize,
    pub token_tx: mpsc::UnboundedSender<TokenEvent>,
}

/// Events sent from the scheduler back to a per-request HTTP handler.
pub enum TokenEvent {
    /// A new token was generated.
    Token(u32),
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
struct ActiveRequest {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    kv: KvState,
    last_token: u32,
    generated_count: usize,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
}

// ── Entry point ─────────────────────────────────────────────────────────

/// Start the scheduler thread. Returns a handle for submitting requests.
///
/// The scheduler exclusively owns the model, KV pool, and batch decode buffers.
/// No Mutex — only this thread touches the GPU.
pub fn start(model: Qwen3Model, seed: u64) -> Result<SchedulerHandle> {
    let max_bucket = *BATCH_BUCKETS.last().unwrap();
    let bufs = model.create_batch_decode_bufs(max_bucket)?;
    let prefill_state = model.create_state()?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(model, submit_rx, bufs, prefill_state, seed);
        })
        .expect("failed to spawn scheduler thread");

    Ok(SchedulerHandle { submit_tx })
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop(
    model: Qwen3Model,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    mut bufs: BatchDecodeBuffers,
    mut prefill_state: Qwen3State,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest> = Vec::new();

    info!("Scheduler ready (max_batch={})", bufs.max_batch_size);

    loop {
        // 1. Prefill-priority: drain all pending requests before next decode step
        while let Ok(req) = submit_rx.try_recv() {
            prefill_one(&model, &mut prefill_state, &mut active, req, &mut rng);
        }

        // 2. Nothing active → block until a request arrives (thread sleeps, no spin)
        if active.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => {
                    prefill_one(&model, &mut prefill_state, &mut active, req, &mut rng);
                }
                None => {
                    // All senders dropped — server shutting down
                    info!("Scheduler: all handles dropped, exiting");
                    return;
                }
            }
        }

        // 3. After prefill, active may still be empty (e.g. receiver already dropped)
        if active.is_empty() {
            continue;
        }

        // 4. One batch decode step
        let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

        if let Err(e) = model.batch_decode(&token_ids, &mut kv_refs, &mut bufs) {
            // Fatal for this batch — retire all active requests
            warn!("batch_decode error: {e}");
            for req in active.drain(..) {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
            }
            continue;
        }

        // 5. Sample per-request + dispatch + retire finished
        let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
        let tokens = match model.select_tokens_batch_varied(&mut bufs, &params_refs, &mut rng) {
            Ok(t) => t,
            Err(e) => {
                warn!("sampling error: {e}");
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

        let mut i = 0;
        while i < active.len() {
            let token = tokens[i];
            let req = &mut active[i];
            req.generated_count += 1;

            let is_eos = !req.params.ignore_eos && model.is_stop_token(token);
            let at_limit = req.generated_count >= req.max_tokens;

            if is_eos || at_limit {
                let finish_reason = if is_eos {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                };
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason,
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.generated_count,
                });
                active.swap_remove(i);
            } else if req.token_tx.send(TokenEvent::Token(token)).is_err() {
                // Receiver dropped — client disconnected or stop sequence detected
                active.swap_remove(i);
            } else {
                req.last_token = token;
                i += 1;
            }
        }
    }
}

// ── Prefill helper ──────────────────────────────────────────────────────

fn prefill_one(
    model: &Qwen3Model,
    state: &mut Qwen3State,
    active: &mut Vec<ActiveRequest>,
    req: SchedulerRequest,
    rng: &mut StdRng,
) {
    let prompt_len = req.prompt_tokens.len();

    // Prefill (writes into state.kv_state which was swapped to empty after previous call)
    if let Err(e) = model.forward(&req.prompt_tokens, state) {
        warn!("Prefill failed for {prompt_len}-token prompt: {e}");
        return;
    }

    // Sample first token
    let first_token = match model.select_token(state, &req.params, rng) {
        Ok(t) => t,
        Err(e) => {
            warn!("First token sampling failed: {e}");
            return;
        }
    };

    // Immediate EOS — no decode needed
    if !req.params.ignore_eos && model.is_stop_token(first_token) {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Stop,
            prompt_tokens: prompt_len,
            completion_tokens: 0,
        });
        // state.kv_state still holds prefilled pages — swap with fresh
        let _ = state.take_kv_state(model.alloc_kv());
        return;
    }

    // Send first token
    if req.token_tx.send(TokenEvent::Token(first_token)).is_err() {
        let _ = state.take_kv_state(model.alloc_kv());
        return;
    }

    // max_tokens == 1 → done after first token
    if req.max_tokens <= 1 {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: prompt_len,
            completion_tokens: 1,
        });
        let _ = state.take_kv_state(model.alloc_kv());
        return;
    }

    // Move prefilled KV into active set, leave state with a fresh allocation
    let kv = state.take_kv_state(model.alloc_kv());

    active.push(ActiveRequest {
        token_tx: req.token_tx,
        kv,
        last_token: first_token,
        generated_count: 1,
        max_tokens: req.max_tokens,
        prompt_len,
        params: req.params,
    });
}
