//! Minimal DSV3.2 scheduler: single-threaded request execution over the
//! multi-GPU executor.
//!
//! This keeps the serving path honest to the current sparse-prefill main path:
//! the prompt runs through sparse prefill once, then each generation step
//! reuses MLA KV via `DsV32Executor::decode`.
//! It is correctness-first and intentionally avoids reintroducing the retired
//! dense-prefill prompt path.

use std::thread;

use anyhow::Result;
use log::{info, warn};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use crate::model::DsV32Executor;
use crate::sampler::SamplingParams;
use crate::scheduler::{SchedulerHandle, SchedulerRequest, TokenEvent};
use crate::server_engine::{FinishReason, TokenLogprob};

pub fn start(executor: DsV32Executor, seed: u64) -> Result<SchedulerHandle> {
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    thread::Builder::new()
        .name("scheduler-dsv32".into())
        .spawn(move || scheduler_loop(executor, submit_rx, seed))
        .expect("failed to spawn DSV3.2 scheduler thread");

    Ok(SchedulerHandle { submit_tx })
}

fn scheduler_loop(
    executor: DsV32Executor,
    mut submit_rx: mpsc::UnboundedReceiver<SchedulerRequest>,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    info!("DSV3.2 scheduler ready (serial prefill + decode-reuse mode)");

    loop {
        let Some(req) = submit_rx.blocking_recv() else {
            info!("DSV3.2 scheduler: all handles dropped, exiting");
            return;
        };

        if let Err(err) = handle_request(&executor, req, &mut rng) {
            warn!("DSV3.2 request failed: {err}");
        }
    }
}

fn handle_request(executor: &DsV32Executor, req: SchedulerRequest, rng: &mut StdRng) -> Result<()> {
    if req.echo {
        let _ = req.token_tx.send(TokenEvent::PromptTokens {
            ids: req.prompt_tokens.clone(),
            logprobs: vec![None; req.prompt_tokens.len()],
        });
    }

    if req.max_tokens == 0 {
        let _ = req.token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            prompt_tokens: req.prompt_tokens.len(),
            completion_tokens: 0,
        });
        return Ok(());
    }

    let mut context_tokens = req.prompt_tokens.clone();
    let prompt_len = context_tokens.len();
    let mut emitted = 0usize;

    executor.reset_generation_state()?;
    let positions: Vec<i32> = (0..context_tokens.len()).map(|i| i as i32).collect();
    let mut logits_bf16 = executor.prefill(&context_tokens, &positions)?;

    while emitted < req.max_tokens {
        let logits_f32: Vec<f32> = logits_bf16.iter().map(|&x| x.to_f32()).collect();
        let token = sample_from_logits(&logits_f32, &req.params, rng)?;
        let logprob = if req.logprobs > 0 {
            compute_logprobs_from_cpu(&logits_f32, token, req.logprobs)
        } else {
            None
        };

        emitted += 1;
        if req
            .token_tx
            .send(TokenEvent::Token { id: token, logprob })
            .is_err()
        {
            return Ok(());
        }

        let is_eos = !req.params.ignore_eos && executor.is_stop_token(token);
        let at_limit = emitted >= req.max_tokens;
        if is_eos || at_limit {
            let _ = req.token_tx.send(TokenEvent::Finished {
                finish_reason: if is_eos {
                    FinishReason::Stop
                } else {
                    FinishReason::Length
                },
                prompt_tokens: prompt_len,
                completion_tokens: emitted,
            });
            return Ok(());
        }

        context_tokens.push(token);
        logits_bf16 = executor.decode(token)?;
    }

    Ok(())
}

fn sample_from_logits(logits: &[f32], params: &SamplingParams, rng: &mut StdRng) -> Result<u32> {
    anyhow::ensure!(!logits.is_empty(), "empty logits");

    if params.is_greedy() {
        let (idx, _) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("non-empty logits");
        return Ok(idx as u32);
    }

    let temperature = if params.temperature > 0.0 {
        params.temperature
    } else {
        1.0
    };

    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, logit)| (idx, logit / temperature))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if params.top_k > 0 {
        let keep = params.top_k as usize;
        if keep < indexed.len() {
            indexed.truncate(keep);
        }
    }

    let max_logit = indexed
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = indexed
        .into_iter()
        .map(|(idx, logit)| (idx, (logit - max_logit).exp()))
        .collect();

    if params.top_p < 1.0 {
        let full_total: f32 = probs.iter().map(|(_, p)| *p).sum();
        let mut cumulative = 0.0f32;
        let mut kept = Vec::with_capacity(probs.len());
        for (idx, prob) in probs {
            kept.push((idx, prob));
            cumulative += prob;
            if full_total > 0.0 && cumulative / full_total >= params.top_p {
                break;
            }
        }
        probs = kept;
    }

    let total_prob: f32 = probs.iter().map(|(_, p)| *p).sum();
    anyhow::ensure!(
        total_prob.is_finite() && total_prob > 0.0,
        "invalid sampling distribution"
    );

    let mut draw = rng.random::<f32>() * total_prob;
    for (idx, prob) in probs {
        draw -= prob;
        if draw <= 0.0 {
            return Ok(idx as u32);
        }
    }

    Ok(logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty logits")
        .0 as u32)
}

fn compute_logprobs_from_cpu(
    logits_f32: &[f32],
    sampled_token: u32,
    top_k: usize,
) -> Option<TokenLogprob> {
    if logits_f32.is_empty() || sampled_token as usize >= logits_f32.len() {
        return None;
    }

    let max_val = logits_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits_f32.iter().map(|&x| (x - max_val).exp()).sum();
    let log_sum_exp = max_val + sum_exp.ln();
    let sampled_logprob = logits_f32[sampled_token as usize] - log_sum_exp;

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
