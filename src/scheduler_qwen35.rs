//! Scheduler for Qwen3.5: dedicated GPU thread that batches concurrent requests.
//!
//! Mirrors the Qwen3 scheduler but manages:
//! - `RecurrentState` alongside `KvState` (linear attention layers)
//! - `BatchDecodeGraphState` for CUDA Graph batch decode (stable-address slots)
//!
//! Prefill allocates temporary `RecurrentState`s, then D2D-copies them into
//! graph slots. On request retirement, swap-remove compaction keeps slots dense.

use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::kv_pool::KvState;
use crate::model::qwen35::batch_decode_graph::BatchDecodeGraphState;
use crate::model::qwen35::recurrent_state::RecurrentState;
use crate::model::{ModelForward, Qwen35Model};
use crate::sampler::SamplingParams;
use crate::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent};
use crate::server_engine::FinishReason;
use crate::tensor::DeviceVec;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
struct ActiveRequest35 {
    token_tx: mpsc::UnboundedSender<TokenEvent>,
    kv: KvState,
    /// Index into `BatchDecodeGraphState.slot_states`.
    graph_slot_idx: usize,
    last_token: u32,
    generated_count: usize,
    max_tokens: usize,
    prompt_len: usize,
    params: SamplingParams,
}

/// Scratch buffers for GPU sampling (reused across prefill sampling).
struct SampleScratch {
    probs: cudarc::driver::CudaSlice<f32>,
    out: cudarc::driver::CudaSlice<i32>,
}

impl SampleScratch {
    fn new(model: &Qwen35Model) -> Result<Self> {
        let vocab_size = model.config().vocab_size;
        let ctx = model.device_ctx();
        Ok(Self {
            probs: ctx.stream.alloc_zeros(vocab_size)?,
            out: ctx.stream.alloc_zeros(1)?,
        })
    }
}

// ── Entry point ─────────────────────────────────────────────────────────

/// Start the Qwen3.5 scheduler thread with default max batch size (64).
pub fn start(model: Qwen35Model, seed: u64) -> Result<SchedulerHandle> {
    start_with_capacity(model, seed, crate::model::qwen35::batch_decode_graph::MAX_BATCH)
}

/// Start the Qwen3.5 scheduler thread with a custom max batch size.
///
/// Lower `max_batch` reduces GPU memory usage (each slot holds a full
/// RecurrentState for all linear attention layers).
pub fn start_with_capacity(model: Qwen35Model, seed: u64, max_batch: usize) -> Result<SchedulerHandle> {
    let graph_state = model.create_batch_decode_graph_state_with_capacity(max_batch)?;
    let sample_scratch = SampleScratch::new(&model)?;

    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler-qwen35".into())
        .spawn(move || {
            scheduler_loop(model, submit_rx, graph_state, sample_scratch, seed, max_batch);
        })
        .expect("failed to spawn Qwen3.5 scheduler thread");

    Ok(SchedulerHandle { submit_tx })
}

// ── Main loop ───────────────────────────────────────────────────────────

fn scheduler_loop(
    model: Qwen35Model,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    mut graph_state: BatchDecodeGraphState,
    mut sample_scratch: SampleScratch,
    seed: u64,
    max_batch: usize,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut active: Vec<ActiveRequest35> = Vec::new();
    let mut deferred: Vec<SchedulerRequest> = Vec::new();

    info!("Qwen3.5 scheduler ready (max_batch={})", max_batch);

    loop {
        // 1. Drain all pending requests (deferred from last iteration + channel)
        let mut pending = std::mem::take(&mut deferred);
        while let Ok(req) = submit_rx.try_recv() {
            pending.push(req);
        }

        // 2. Nothing active and nothing pending → block until a request arrives
        if active.is_empty() && pending.is_empty() {
            match submit_rx.blocking_recv() {
                Some(req) => pending.push(req),
                None => {
                    info!("Qwen3.5 scheduler: all handles dropped, exiting");
                    return;
                }
            }
            while let Ok(req) = submit_rx.try_recv() {
                pending.push(req);
            }
        }

        // Cap pending to available slot capacity; defer overflow to next iteration.
        let available = max_batch.saturating_sub(active.len());
        if pending.len() > available {
            deferred = pending.split_off(available);
        }

        let have_pending = !pending.is_empty();

        if have_pending && !active.is_empty() {
            unified_step_sched(
                &model,
                &mut active,
                pending,
                &mut graph_state,
                &mut sample_scratch,
                &mut rng,
            );
        } else if have_pending {
            prefill_batch(
                &model,
                &mut active,
                pending,
                &mut graph_state,
                &mut sample_scratch,
                &mut rng,
            );
        }

        if active.is_empty() {
            continue;
        }

        // Pure decode step — only when no pending arrived (unified_step already did one decode).
        if !have_pending {
            decode_step(&model, &mut active, &mut graph_state, &mut rng);
        }
    }
}

// ── Batch prefill ───────────────────────────────────────────────────────

fn prefill_batch(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    pending: Vec<SchedulerRequest>,
    graph_state: &mut BatchDecodeGraphState,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let mut kv_states: Vec<KvState> = (0..pending.len()).map(|_| model.alloc_kv()).collect();

    // Allocate temporary recurrent states for prefill
    let mut rec_states: Vec<RecurrentState> = (0..pending.len())
        .map(|_| RecurrentState::new(model.device_ctx(), model.config()).unwrap())
        .collect();
    let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();

    let logits_vec = match model.batch_prefill(&prompts, &mut kv_states, &mut rec_refs) {
        Ok(v) => v,
        Err(e) => {
            warn!("Qwen3.5 batch prefill failed: {e}");
            for req in pending {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            return;
        }
    };

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

        // Assign a graph slot and copy recurrent state into it
        let slot_idx = active.len();
        graph_state
            .copy_state_to_slot(model.device_ctx(), &rec_states[i], slot_idx)
            .expect("copy recurrent state to slot failed");

        let kv = std::mem::replace(&mut kv_states[i], model.alloc_kv());
        active.push(ActiveRequest35 {
            token_tx: req.token_tx,
            kv,
            graph_slot_idx: slot_idx,
            last_token: first_token,
            generated_count: 1,
            max_tokens: req.max_tokens,
            prompt_len,
            params: req.params,
        });
    }
}

// ── Unified step (prefill + decode in one forward pass) ────────────────

fn unified_step_sched(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    pending: Vec<SchedulerRequest>,
    graph_state: &mut BatchDecodeGraphState,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    // Build prefill inputs
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let mut prefill_kv_states: Vec<KvState> =
        (0..pending.len()).map(|_| model.alloc_kv()).collect();

    // Allocate temporary recurrent states for prefill
    let mut rec_states: Vec<RecurrentState> = (0..pending.len())
        .map(|_| RecurrentState::new(model.device_ctx(), model.config()).unwrap())
        .collect();
    let mut rec_refs: Vec<&mut RecurrentState> = rec_states.iter_mut().collect();

    // Build decode inputs
    let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let mut decode_kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    // Run unified forward pass
    let (prefill_logits, decode_logits) = match model.unified_step(
        &prompts,
        &mut prefill_kv_states,
        &mut rec_refs,
        &decode_tokens,
        &mut decode_kv_refs,
        graph_state,
    ) {
        Ok(v) => v,
        Err(e) => {
            warn!("Qwen3.5 unified step failed: {e}");
            // Notify all pending requests
            for req in pending {
                let _ = req.token_tx.send(TokenEvent::Finished {
                    finish_reason: FinishReason::Stop,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            // Notify all active decode requests
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

    // Process decode results FIRST (before adding prefill results to active)
    process_decode_logits(model, active, &decode_logits, graph_state, scratch, rng);

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

        // Assign graph slot and copy recurrent state
        let slot_idx = active.len();
        graph_state
            .copy_state_to_slot(model.device_ctx(), &rec_states[i], slot_idx)
            .expect("copy recurrent state to slot failed");

        let kv = std::mem::replace(&mut prefill_kv_states[i], model.alloc_kv());
        active.push(ActiveRequest35 {
            token_tx: req.token_tx,
            kv,
            graph_slot_idx: slot_idx,
            last_token: first_token,
            generated_count: 1,
            max_tokens: req.max_tokens,
            prompt_len,
            params: req.params,
        });
    }
}

// ── Decode step (pure decode, CUDA Graph enabled) ──────────────────────

fn decode_step(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    rng: &mut StdRng,
) {
    let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
    let mut kv_refs: Vec<&mut KvState> = active.iter_mut().map(|r| &mut r.kv).collect();

    if let Err(e) = model.batch_decode_graph(&token_ids, &mut kv_refs, graph_state) {
        warn!("Qwen3.5 batch_decode_graph error: {e}");
        for req in active.drain(..) {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
        }
        return;
    }

    let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
    let tokens = match model.select_tokens_batch_varied(
        &mut graph_state.buffers,
        &params_refs,
        rng,
    ) {
        Ok(t) => t,
        Err(e) => {
            warn!("Qwen3.5 sampling error: {e}");
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

    dispatch_decode_tokens(model, active, &tokens, graph_state);
}

/// Process decode logits from unified step: sample and dispatch.
fn process_decode_logits(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    decode_logits: &[DeviceVec],
    graph_state: &mut BatchDecodeGraphState,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) {
    let mut tokens = Vec::with_capacity(active.len());
    for (i, logits) in decode_logits.iter().enumerate() {
        match sample_from_logits(model, logits, scratch, &active[i].params, rng) {
            Ok(t) => tokens.push(t),
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

    dispatch_decode_tokens(model, active, &tokens, graph_state);
}

/// Dispatch sampled decode tokens: send events, check EOS/limits, retire finished.
///
/// When a request is retired (swap_remove), D2D-copy the last slot's recurrent
/// state into the vacated slot to keep slots 0..active.len() dense.
fn dispatch_decode_tokens(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    tokens: &[u32],
    graph_state: &mut BatchDecodeGraphState,
) {
    let mut i = 0;
    while i < active.len() {
        let token = tokens[i];
        let req = &mut active[i];
        req.generated_count += 1;

        let is_eos = !req.params.ignore_eos && model.is_stop_token(token);
        let at_limit = req.generated_count >= req.max_tokens;

        if is_eos {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            compact_slot(model, active, graph_state, i);
        } else if at_limit {
            // Send the final token, then Finished.
            let _ = req.token_tx.send(TokenEvent::Token(token));
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.generated_count,
            });
            compact_slot(model, active, graph_state, i);
        } else if req.token_tx.send(TokenEvent::Token(token)).is_err() {
            compact_slot(model, active, graph_state, i);
        } else {
            req.last_token = token;
            i += 1;
        }
    }
}

/// Remove request at `idx` via swap_remove and compact graph slots.
///
/// After swap_remove, the element that was at `active.len()-1` (before remove)
/// now sits at `idx`. Its graph slot must be copied into the vacated slot so
/// that slots 0..active.len() remain dense.
fn compact_slot(
    model: &Qwen35Model,
    active: &mut Vec<ActiveRequest35>,
    graph_state: &mut BatchDecodeGraphState,
    idx: usize,
) {
    let last = active.len() - 1;
    active.swap_remove(idx);

    if idx < active.len() {
        // The element that was at `last` is now at `idx`.
        // Copy its recurrent state from slot `last` to slot `idx`.
        let src_slot = active[idx].graph_slot_idx;
        debug_assert_eq!(src_slot, last);

        // D2D copy: graph_state.slot_states[last] → graph_state.slot_states[idx]
        // We can't borrow two slots mutably at once, so use raw index copy.
        let ctx = model.device_ctx();
        let src = &graph_state.slot_states[last];
        // Copy layer by layer using the public fields
        for layer_idx in 0..src.layers.len() {
            let (src_part, dst_part) = if idx < last {
                let (left, right) = graph_state.slot_states.split_at_mut(last);
                (&right[0].layers[layer_idx], &mut left[idx].layers[layer_idx])
            } else {
                unreachable!("idx < active.len() <= last");
            };

            ctx.stream
                .memcpy_dtod(&src_part.state, &mut dst_part.state)
                .expect("compact slot state copy failed");
            ctx.stream
                .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                .expect("compact slot conv_state copy failed");
        }
        graph_state.slot_states[idx].seq_len = graph_state.slot_states[last].seq_len;

        active[idx].graph_slot_idx = idx;
    }
}

fn sample_from_logits(
    model: &Qwen35Model,
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
