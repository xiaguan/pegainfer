use anyhow::Result;
use log::warn;
use rand::rngs::StdRng;

use crate::model_executor::{ModelExecutor, Qwen3Executor, RequestKvState};
use crate::sampler::SamplingParams;
use crate::server_engine::{FinishReason, TokenLogprob};
use crate::tensor::{DeviceVec, HiddenStates};

use super::effects::{DecodeEffect, PendingEffect, PromptEchoEffect, StepEffects};
use super::plan::ExecutionArtifacts;
use super::{ActiveRequestState, SchedulerRequest};

pub(super) struct SampleScratch {
    probs: cudarc::driver::CudaSlice<f32>,
    top1_value: cudarc::driver::CudaSlice<half::bf16>,
    row_states: cudarc::driver::CudaSlice<u8>,
    valid: cudarc::driver::CudaSlice<u8>,
    out: cudarc::driver::CudaSlice<i32>,
}

impl SampleScratch {
    pub(super) fn new(executor: &Qwen3Executor) -> Result<Self> {
        let vocab_size = executor.vocab_size();
        let ctx = executor.device_ctx();
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

pub(super) fn resolve_step(
    executor: &Qwen3Executor,
    active: &[ActiveRequestState],
    artifacts: ExecutionArtifacts,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) -> StepEffects {
    match artifacts {
        ExecutionArtifacts::Prefill {
            pending,
            mut kv_states,
            result,
        } => resolve_prefill_outputs(
            executor,
            pending,
            &mut kv_states,
            result.logits,
            result.all_position_logits,
            scratch,
            rng,
            true,
        ),
        ExecutionArtifacts::Decode { cpu_logits, tokens } => StepEffects {
            prompt_echoes: Vec::new(),
            pending: Vec::new(),
            decode: resolve_decode_outputs(executor, active, &cpu_logits, &tokens),
        },
        ExecutionArtifacts::Unified {
            pending,
            mut prefill_kv_states,
            result,
        } => {
            let mut effects = resolve_prefill_outputs(
                executor,
                pending,
                &mut prefill_kv_states,
                result.prefill_logits,
                None,
                scratch,
                rng,
                false,
            );
            effects.decode =
                resolve_decode_logits(executor, active, &result.decode_logits, scratch, rng);
            effects
        }
    }
}

fn resolve_prefill_outputs(
    executor: &Qwen3Executor,
    pending: Vec<SchedulerRequest>,
    kv_states: &mut [RequestKvState],
    logits_vec: Vec<DeviceVec>,
    all_position_logits: Option<HiddenStates>,
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
    compute_prompt_logprobs: bool,
) -> StepEffects {
    let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
    let seq_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    let mut effects = StepEffects::empty();
    let mut token_offset = 0usize;
    for (i, req) in pending.into_iter().enumerate() {
        let prompt_len = req.prompt_tokens.len();

        let first_token =
            match sample_from_logits(executor, &logits_vec[i], scratch, &req.params, rng) {
                Ok(t) => t,
                Err(e) => {
                    warn!("First token sampling failed for request {i}: {e}");
                    effects.pending.push(PendingEffect::Drop);
                    continue;
                }
            };

        let logprob = if req.logprobs > 0 {
            extract_logprobs(executor, &logits_vec[i], first_token, req.logprobs).ok()
        } else {
            None
        };

        if req.echo {
            let echo_logprobs =
                if compute_prompt_logprobs {
                    let prompt_len_local = req.prompt_tokens.len();
                    let mut echo_logprobs: Vec<Option<TokenLogprob>> =
                        Vec::with_capacity(prompt_len_local);
                    echo_logprobs.push(None);
                    if let Some(ref all_logits) = all_position_logits {
                        for j in 1..prompt_len_local {
                            let prev_pos = token_offset + j - 1;
                            let target_token = req.prompt_tokens[j];
                            let lp = crate::ops::extract_vec(
                                executor.device_ctx(),
                                all_logits,
                                prev_pos,
                            )
                            .ok()
                            .and_then(|logits_vec| {
                                let logits_f32 = logits_vec.to_host(executor.device_ctx()).ok()?;
                                compute_logprobs_from_cpu(&logits_f32, target_token, req.logprobs)
                            });
                            echo_logprobs.push(lp);
                        }
                    } else {
                        for _ in 1..prompt_len_local {
                            echo_logprobs.push(None);
                        }
                    }
                    echo_logprobs
                } else {
                    vec![None; req.prompt_tokens.len()]
                };
            effects.prompt_echoes.push(PromptEchoEffect {
                token_tx: req.token_tx.clone(),
                ids: req.prompt_tokens.clone(),
                logprobs: echo_logprobs,
            });
        }
        token_offset += seq_lens[i];

        if !req.params.ignore_eos && executor.is_stop_token(first_token) {
            effects.pending.push(PendingEffect::Finish {
                token_tx: req.token_tx,
                finish_reason: FinishReason::Stop,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            continue;
        }

        if req.max_tokens <= 1 {
            effects.pending.push(PendingEffect::EmitAndFinish {
                token_tx: req.token_tx,
                token: first_token,
                logprob,
                finish_reason: FinishReason::Length,
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
            continue;
        }

        let kv = std::mem::replace(&mut kv_states[i], executor.alloc_kv());
        effects.pending.push(PendingEffect::Promote {
            state: ActiveRequestState {
                token_tx: req.token_tx,
                kv,
                last_token: first_token,
                generated_count: 1,
                max_tokens: req.max_tokens,
                prompt_len,
                params: req.params,
                logprobs: req.logprobs,
            },
            first_token,
            logprob,
        });
    }

    effects
}

fn resolve_decode_outputs(
    executor: &Qwen3Executor,
    active: &[ActiveRequestState],
    cpu_logits: &[Option<Vec<f32>>],
    tokens: &[u32],
) -> Vec<DecodeEffect> {
    cpu_logits
        .iter()
        .enumerate()
        .map(|(i, logits_opt)| {
            let req = &active[i];
            let token = tokens[i];
            let completion_tokens = req.generated_count + 1;
            let logprob = logits_opt
                .as_ref()
                .and_then(|logits_f32| compute_logprobs_from_cpu(logits_f32, token, req.logprobs));
            let is_eos = !req.params.ignore_eos && executor.is_stop_token(token);
            let at_limit = completion_tokens >= req.max_tokens;
            if is_eos || at_limit {
                DecodeEffect::Finish {
                    index: i,
                    finish_reason: if is_eos {
                        FinishReason::Stop
                    } else {
                        FinishReason::Length
                    },
                    completion_tokens,
                }
            } else {
                DecodeEffect::EmitAndContinue {
                    index: i,
                    token,
                    logprob,
                    completion_tokens,
                }
            }
        })
        .collect()
}

fn resolve_decode_logits(
    executor: &Qwen3Executor,
    active: &[ActiveRequestState],
    decode_logits: &[DeviceVec],
    scratch: &mut SampleScratch,
    rng: &mut StdRng,
) -> Vec<DecodeEffect> {
    let mut tokens = Vec::with_capacity(active.len());
    let mut logprobs_vec: Vec<Option<TokenLogprob>> = Vec::with_capacity(active.len());
    for (i, logits) in decode_logits.iter().enumerate() {
        match sample_from_logits(executor, logits, scratch, &active[i].params, rng) {
            Ok(t) => {
                let lp = if active[i].logprobs > 0 {
                    extract_logprobs(executor, logits, t, active[i].logprobs).ok()
                } else {
                    None
                };
                tokens.push(t);
                logprobs_vec.push(lp);
            }
            Err(e) => {
                warn!("decode sampling error: {e}");
                return active
                    .iter()
                    .enumerate()
                    .map(|(idx, _)| DecodeEffect::Drop { index: idx })
                    .collect();
            }
        }
    }

    tokens
        .into_iter()
        .enumerate()
        .map(|(i, token)| {
            let req = &active[i];
            let completion_tokens = req.generated_count + 1;
            let is_eos = !req.params.ignore_eos && executor.is_stop_token(token);
            let at_limit = completion_tokens >= req.max_tokens;
            if is_eos || at_limit {
                DecodeEffect::Finish {
                    index: i,
                    finish_reason: if is_eos {
                        FinishReason::Stop
                    } else {
                        FinishReason::Length
                    },
                    completion_tokens,
                }
            } else {
                DecodeEffect::EmitAndContinue {
                    index: i,
                    token,
                    logprob: logprobs_vec[i].clone(),
                    completion_tokens,
                }
            }
        })
        .collect()
}

fn sample_from_logits(
    executor: &Qwen3Executor,
    logits: &DeviceVec,
    scratch: &mut SampleScratch,
    params: &SamplingParams,
    rng: &mut StdRng,
) -> Result<u32> {
    let random_val: f32 = rand::RngExt::random(rng);
    crate::ops::gpu_sample_into(
        executor.device_ctx(),
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

fn extract_logprobs(
    executor: &Qwen3Executor,
    logits: &DeviceVec,
    sampled_token: u32,
    top_k: usize,
) -> Result<TokenLogprob> {
    let logits_f32 = logits.to_host(executor.device_ctx())?;
    compute_logprobs_from_cpu(&logits_f32, sampled_token, top_k)
        .ok_or_else(|| anyhow::anyhow!("logprobs computation failed"))
}

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
