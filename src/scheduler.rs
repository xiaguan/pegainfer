//! Scheduler: dedicated GPU thread that batches concurrent requests.
//!
//! HTTP handlers tokenize prompts and submit `SchedulerRequest` via channel.
//! The scheduler batch-prefills all pending requests in one forward pass, then
//! batch-decodes all active requests. Per-request tokens flow back through
//! individual channels.

use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::kv_pool::KvState;
use crate::model::qwen3::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::model::{ModelForward, Qwen3Model};
use crate::sampler::SamplingParams;
use crate::server_engine::FinishReason;
use crate::tensor::DeviceVec;

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

    // Sampling scratch — reused across prefill and decode sampling
    let sample_scratch = SampleScratch::new(&model)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler".into())
        .spawn(move || {
            scheduler_loop(model, submit_rx, bufs, sample_scratch, seed);
        })
        .expect("failed to spawn scheduler thread");

    Ok(SchedulerHandle { submit_tx })
}

/// Scratch buffers for GPU sampling (reused across all prefill sampling).
struct SampleScratch {
    probs: cudarc::driver::CudaSlice<f32>,
    out: cudarc::driver::CudaSlice<i32>,
}

impl SampleScratch {
    fn new(model: &Qwen3Model) -> Result<Self> {
        let vocab_size = model.config().vocab_size;
        let ctx = model.device_ctx();
        Ok(Self {
            probs: ctx.stream.alloc_zeros(vocab_size)?,
            out: ctx.stream.alloc_zeros(1)?,
        })
    }
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop(
    model: Qwen3Model,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    mut bufs: BatchDecodeBuffers,
    mut sample_scratch: SampleScratch,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest> = Vec::new();

    info!("Scheduler ready (max_batch={})", bufs.max_batch_size);

    loop {
        // 1. Drain all pending requests
        let mut pending: Vec<SchedulerRequest> = Vec::new();
        while let Ok(req) = submit_rx.try_recv() {
            pending.push(req);
        }

        // 2. Nothing active and nothing pending → block until a request arrives
        if active.is_empty() && pending.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => pending.push(req),
                None => {
                    info!("Scheduler: all handles dropped, exiting");
                    return;
                }
            }
            // Drain any others that arrived while we were blocked
            while let Ok(req) = submit_rx.try_recv() {
                pending.push(req);
            }
        }

        // 3. Batch prefill all pending requests in one forward pass
        if !pending.is_empty() {
            prefill_batch(&model, &mut active, pending, &mut sample_scratch, &mut rng);
        }

        // 4. After prefill, active may still be empty
        if active.is_empty() {
            continue;
        }

        // 5. One batch decode step
        let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

        if let Err(e) = model.batch_decode(&token_ids, &mut kv_refs, &mut bufs) {
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

        // 6. Sample per-request + dispatch + retire finished
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
                active.swap_remove(i);
            } else {
                req.last_token = token;
                i += 1;
            }
        }
    }
}

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    model: &Qwen3Model,
    active: &mut Vec<ActiveRequest>,
    pending: Vec<SchedulerRequest>,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let mut kv_states: Vec<KvState> = (0..pending.len()).map(|_| model.alloc_kv()).collect();

    let logits_vec = match model.batch_prefill(&prompts, &mut kv_states) {
        Ok(v) => v,
        Err(e) => {
            warn!("Batch prefill failed: {e}");
            return;
        }
    };

    // Process each request: sample first token, handle EOS/limits, add to active
    for (i, req) in pending.into_iter().enumerate() {
        let prompt_len = req.prompt_tokens.len();

        let first_token = match sample_from_logits(model, &logits_vec[i], scratch, &req.params, rng)
        {
            Ok(t) => t,
            Err(e) => {
                warn!("First token sampling failed for request {i}: {e}");
                continue;
            }
        };

        if !req.params.ignore_eos && model.is_stop_token(first_token) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }

        if req.token_tx.send(TokenEvent::Token(first_token)).is_err() {
            continue;
        }

        if req.max_tokens <= 1 {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            continue;
        }

        // Take ownership of this request's KV state
        let kv = std::mem::replace(&mut kv_states[i], model.alloc_kv());
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
}

fn sample_from_logits(
    model: &Qwen3Model,
    logits: &DeviceVec,
    scratch: &mut SampleScratch,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> Result<u32> {
    let random_val: f32 = rand::RngExt::random(rng);
    crate::ops::gpu_sample_into(
        model.device_ctx(),
        logits,
        &mut scratch.probs,
        &mut scratch.out,
        params,
        random_val,
    )
}
