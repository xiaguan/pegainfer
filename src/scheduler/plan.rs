use anyhow::Result;
use rand::rngs::StdRng;

use crate::model_executor::{
    DecodePlan, ModelExecutor, PrefillPlan, PrefillResult, Qwen3Executor, RequestKvState,
    UnifiedPlan, UnifiedResult,
};
use crate::sampler::SamplingParams;

use super::{ActiveRequestState, SchedulerRequest};

pub(super) enum ExecutionPlan {
    Prefill { pending: Vec<SchedulerRequest> },
    Decode,
    Unified { pending: Vec<SchedulerRequest> },
}

pub(super) enum ExecutionArtifacts {
    Prefill {
        pending: Vec<SchedulerRequest>,
        kv_states: Vec<RequestKvState>,
        result: PrefillResult,
    },
    Decode {
        cpu_logits: Vec<Option<Vec<f32>>>,
        tokens: Vec<u32>,
    },
    Unified {
        pending: Vec<SchedulerRequest>,
        prefill_kv_states: Vec<RequestKvState>,
        result: UnifiedResult,
    },
}

pub(super) fn build_next_plan(
    have_active: bool,
    pending: Vec<SchedulerRequest>,
) -> Option<ExecutionPlan> {
    if !pending.is_empty() && have_active {
        Some(ExecutionPlan::Unified { pending })
    } else if !pending.is_empty() {
        Some(ExecutionPlan::Prefill { pending })
    } else if have_active {
        Some(ExecutionPlan::Decode)
    } else {
        None
    }
}

pub(super) fn execute_plan(
    executor: &mut Qwen3Executor,
    active: &mut [ActiveRequestState],
    plan: ExecutionPlan,
    rng: &mut StdRng,
) -> Result<ExecutionArtifacts> {
    match plan {
        ExecutionPlan::Prefill { pending } => {
            let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
            let mut kv_states: Vec<RequestKvState> =
                (0..pending.len()).map(|_| executor.alloc_kv()).collect();
            let any_echo = pending.iter().any(|r| r.echo);
            let result = executor.execute_prefill(PrefillPlan {
                prompts: &prompts,
                kv_states: &mut kv_states,
                echo: any_echo,
            })?;
            Ok(ExecutionArtifacts::Prefill {
                pending,
                kv_states,
                result,
            })
        }
        ExecutionPlan::Decode => {
            let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
            let active_len = active.len();
            let requested_topk: Vec<usize> = active.iter().map(|r| r.logprobs).collect();
            let any_logprobs = requested_topk.iter().any(|&k| k > 0);
            let params_owned: Vec<SamplingParams> = active.iter().map(|r| r.params).collect();
            let params_refs: Vec<&SamplingParams> = params_owned.iter().collect();

            let (cpu_logits, tokens) = {
                let mut kv_refs: Vec<&mut RequestKvState> =
                    active.iter_mut().map(|r| &mut r.kv).collect();
                let decode_step = executor.begin_decode(DecodePlan {
                    token_ids: &token_ids,
                    kv_states: &mut kv_refs,
                })?;
                let cpu_logits = if any_logprobs {
                    decode_step.snapshot_cpu_logits(&requested_topk)
                } else {
                    vec![None; active_len]
                };
                let tokens = decode_step.sample_tokens(&params_refs, rng)?;
                (cpu_logits, tokens)
            };

            Ok(ExecutionArtifacts::Decode { cpu_logits, tokens })
        }
        ExecutionPlan::Unified { pending } => {
            let prompts: Vec<&[u32]> = pending.iter().map(|r| r.prompt_tokens.as_slice()).collect();
            let mut prefill_kv_states: Vec<RequestKvState> =
                (0..pending.len()).map(|_| executor.alloc_kv()).collect();
            let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
            let mut decode_kv_refs: Vec<&mut RequestKvState> =
                active.iter_mut().map(|r| &mut r.kv).collect();
            let result = executor.execute_unified(UnifiedPlan {
                prefill_prompts: &prompts,
                prefill_kv_states: &mut prefill_kv_states,
                decode_tokens: &decode_tokens,
                decode_kv_states: &mut decode_kv_refs,
            })?;
            Ok(ExecutionArtifacts::Unified {
                pending,
                prefill_kv_states,
                result,
            })
        }
    }
}
