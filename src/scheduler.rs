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
use crate::server_engine::{FinishReason, TokenLogprob};
use crate::tensor::DeviceVec;

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
struct ActiveRequest {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    kv: KvState,
    last_token: u32,
    generated_count: usize,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    logprobs: usize,
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
    top1_value: cudarc::driver::CudaSlice<half::bf16>,
    row_states: cudarc::driver::CudaSlice<u8>,
    valid: cudarc::driver::CudaSlice<u8>,
    out: cudarc::driver::CudaSlice<i32>,
}

impl SampleScratch {
    fn new(model: &Qwen3Model) -> Result<Self> {
        let vocab_size = model.config().vocab_size;
        let ctx = model.device_ctx();
        Ok(Self {
            probs: ctx.stream.alloc_zeros(vocab_size)?,
            top1_value: ctx.stream.alloc_zeros(1)?,
            row_states: ctx
                .stream
                .alloc_zeros(crate::ops::flashinfer_topk_row_states_bytes())?,
            valid: ctx.stream.alloc_zeros(1)?,
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

        let have_pending = !pending.is_empty();

        if have_pending && !active.is_empty() {
            // ── Unified step: prefill + decode in one forward pass ──
            unified_step_sched(&model, &mut active, pending, &mut sample_scratch, &mut rng);
        } else if have_pending {
            // ── Pure prefill (no active decode requests yet) ────────
            prefill_batch(&model, &mut active, pending, &mut sample_scratch, &mut rng);
        }

        if active.is_empty() {
            continue;
        }

        // ── Pure decode step (CUDA Graph enabled) ──────────────────
        // Only when no pending arrived: unified_step already did one decode.
        if !have_pending {
            decode_step(&model, &mut active, &mut bufs, &mut rng);
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

    let any_echo = pending.iter().any(|r| r.echo);
    let (logits_vec, all_position_logits) =
        match model.batch_prefill(&prompts, &mut kv_states, any_echo) {
            Ok(v) => v,
            Err(e) => {
                warn!("Batch prefill failed: {e}");
                return;
            }
        };

    // Process each request: sample first token, handle EOS/limits, add to active
    let seq_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    let mut token_offset = 0usize;
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

        // Extract logprobs if requested
        let logprob = if req.logprobs > 0 {
            extract_logprobs(model, &logits_vec[i], first_token, req.logprobs).ok()
        } else {
            None
        };

        // Echo: send prompt tokens with logprobs computed from all-position logits.
        // Token at position j has logprob from logits at position j-1.
        // First token (j=0) has no conditioning context → logprob = None.
        if req.echo {
            let prompt_len_local = req.prompt_tokens.len();
            let mut echo_logprobs: Vec<Option<TokenLogprob>> = Vec::with_capacity(prompt_len_local);
            echo_logprobs.push(None); // first token has no logprob
            if let Some(ref all_logits) = all_position_logits {
                for j in 1..prompt_len_local {
                    let prev_pos = token_offset + j - 1; // position in concatenated sequence
                    let target_token = req.prompt_tokens[j];
                    let lp = crate::ops::extract_vec(model.device_ctx(), all_logits, prev_pos)
                        .ok()
                        .and_then(|logits_vec| {
                            let logits_f32 = logits_vec.to_host(model.device_ctx()).ok()?;
                            compute_logprobs_from_cpu(&logits_f32, target_token, req.logprobs)
                        });
                    echo_logprobs.push(lp);
                }
            } else {
                for _ in 1..prompt_len_local {
                    echo_logprobs.push(None);
                }
            }
            let _ = req.token_tx.send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }
        token_offset += seq_lens[i];

        if !req.params.ignore_eos && model.is_stop_token(first_token) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }

        if req
            .token_tx
            .send(TokenEvent::Token {
                id: first_token,
                logprob,
            })
            .is_err()
        {
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
            logprobs: req.logprobs,
        });
    }
}

// ── Unified step (prefill + decode in one forward pass) ────────────────

fn unified_step_sched(
    model: &Qwen3Model,
    active: &mut Vec<ActiveRequest>,
    pending: Vec<SchedulerRequest>,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    // Build prefill inputs
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let mut prefill_kv_states: Vec<KvState> =
        (0..pending.len()).map(|_| model.alloc_kv()).collect();

    // Build decode inputs
    let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let mut decode_kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    // Run unified forward pass
    let (prefill_logits, decode_logits) = match model.unified_step(
        &prompts,
        &mut prefill_kv_states,
        &decode_tokens,
        &mut decode_kv_refs,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("Unified step failed: {e}");
            return;
        }
    };

    // Process decode results FIRST (before adding prefill results to active,
    // since decode_logits only has entries for the original active requests).
    process_decode_logits(model, active, &decode_logits, scratch, rng);

    // Process prefill results: sample first token, add to active
    for (i, req) in pending.into_iter().enumerate() {
        let prompt_len = req.prompt_tokens.len();

        let first_token =
            match sample_from_logits(model, &prefill_logits[i], scratch, &req.params, rng) {
                Ok(t) => t,
                Err(e) => {
                    warn!("First token sampling failed for request {i}: {e}");
                    continue;
                }
            };

        let logprob = if req.logprobs > 0 {
            extract_logprobs(model, &prefill_logits[i], first_token, req.logprobs).ok()
        } else {
            None
        };

        if req.echo {
            let echo_logprobs = vec![None; req.prompt_tokens.len()];
            let _ = req.token_tx.send(TokenEvent::PromptTokens {
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }

        if !req.params.ignore_eos && model.is_stop_token(first_token) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }

        if req
            .token_tx
            .send(TokenEvent::Token {
                id: first_token,
                logprob,
            })
            .is_err()
        {
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

        let kv = std::mem::replace(&mut prefill_kv_states[i], model.alloc_kv());
        active.push(ActiveRequest {
            token_tx: req.token_tx,
            kv,
            last_token: first_token,
            generated_count: 1,
            max_tokens: req.max_tokens,
            prompt_len,
            params: req.params,
            logprobs: req.logprobs,
        });
    }
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    model: &Qwen3Model,
    active: &mut Vec<ActiveRequest>,
    bufs: &mut BatchDecodeBuffers,
    rng: &mut StdRng,
) {
    let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    if let Err(e) = model.batch_decode(&token_ids, &mut kv_refs, bufs) {
        warn!("batch_decode error: {e}");
        for req in active.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
        }
        return;
    }

    // Snapshot logits to CPU BEFORE sampling (sampling may modify bufs.logits in-place)
    let any_logprobs = active.iter().any(|r| r.logprobs > 0);
    let cpu_logits: Vec<Option<Vec<f32>>> = if any_logprobs {
        (0..active.len())
            .map(|i| {
                if active[i].logprobs > 0 {
                    crate::ops::extract_vec(model.device_ctx(), &bufs.logits, i)
                        .ok()
                        .and_then(|v| v.to_host(model.device_ctx()).ok())
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![None; active.len()]
    };

    let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
    let tokens = match model.select_tokens_batch_varied(bufs, &params_refs, rng) {
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
            return;
        }
    };

    // Compute logprobs from cached CPU logits
    let logprobs_vec: Vec<Option<TokenLogprob>> = cpu_logits
        .into_iter()
        .enumerate()
        .map(|(i, logits_opt)| {
            logits_opt.and_then(|logits_f32| {
                compute_logprobs_from_cpu(&logits_f32, tokens[i], active[i].logprobs)
            })
        })
        .collect();

    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec);
}

/// Process decode logits from unified step: sample, extract logprobs, dispatch.
fn process_decode_logits(
    model: &Qwen3Model,
    active: &mut Vec<ActiveRequest>,
    decode_logits: &[DeviceVec],
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    // Sample one token per active request + optional logprobs
    let mut tokens = Vec::with_capacity(active.len());
    let mut logprobs_vec: Vec<Option<TokenLogprob>> = Vec::with_capacity(active.len());
    for (i, logits) in decode_logits.iter().enumerate() {
        match sample_from_logits(model, logits, scratch, &active[i].params, rng) {
            Ok(t) => {
                let lp = if active[i].logprobs > 0 {
                    extract_logprobs(model, logits, t, active[i].logprobs).ok()
                } else {
                    None
                };
                tokens.push(t);
                logprobs_vec.push(lp);
            }
            Err(e) => {
                warn!("decode sampling error: {e}");
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Finished {
                        finish_reason: FinishReason::Stop,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.generated_count,
                    });
                }
                return;
            }
        }
    }

    dispatch_decode_tokens(model, active, &tokens, &logprobs_vec);
}

/// Dispatch sampled decode tokens: send events, check EOS/limits, retire finished.
///
/// `tokens` and `logprobs` are indexed by original position in `active`.
/// Retirements collected first, then applied in reverse to avoid index invalidation.
fn dispatch_decode_tokens(
    model: &Qwen3Model,
    active: &mut Vec<ActiveRequest>,
    tokens: &[u32],
    logprobs: &[Option<TokenLogprob>],
) {
    let n = active.len();
    let mut to_retire = Vec::new();

    for i in 0..n {
        let token = tokens[i];
        let logprob = logprobs[i].clone();
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
            to_retire.push(i);
        } else if req
            .token_tx
            .send(TokenEvent::Token { id: token, logprob })
            .is_err()
        {
            to_retire.push(i);
        } else {
            req.last_token = token;
        }
    }

    // Remove in reverse order so swap_remove doesn't invalidate earlier indices
    for &i in to_retire.iter().rev() {
        active.swap_remove(i);
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
        &mut scratch.top1_value,
        &mut scratch.row_states,
        &mut scratch.valid,
        &mut scratch.out,
        params,
        random_val,
    )
}

/// Extract log-probabilities from GPU logits for a given sampled token.
fn extract_logprobs(
    model: &Qwen3Model,
    logits: &DeviceVec,
    sampled_token: u32,
    top_k: usize,
) -> Result<TokenLogprob> {
    let logits_f32 = logits.to_host(model.device_ctx())?;
    compute_logprobs_from_cpu(&logits_f32, sampled_token, top_k)
        .ok_or_else(|| anyhow::anyhow!("logprobs computation failed"))
}

/// Compute log-probabilities from CPU f32 logits.
fn compute_logprobs_from_cpu(
    logits_f32: &[f32],
    sampled_token: u32,
    top_k: usize,
) -> Option<TokenLogprob> {
    if logits_f32.is_empty() {
        return None;
    }

    let max_val = logits_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_f32.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();

    let sampled_logprob = logits_f32[sampled_token as usize] - log_sum_exp;

    // Top-k: O(V*k) scan keeping k largest logits
    let k = top_k.min(logits_f32.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    if k > 0 {
        let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (idx, &val) in logits_f32.iter().enumerate() {
            if best.len() < k || val > best.last().unwrap().1 {
                let pos = best.partition_point(|&(_, v)| v > val);
                best.insert(pos, (idx as u32, val));
                if best.len() > k {
                    best.pop();
                }
            }
        }
        for (idx, val) in best {
            top.push((idx, val - log_sum_exp));
        }
    }

    Some(TokenLogprob {
        logprob: sampled_logprob,
        top_logprobs: top,
    })
}
