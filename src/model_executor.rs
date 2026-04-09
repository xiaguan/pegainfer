use anyhow::Result;
use rand::rngs::StdRng;

use crate::kv_pool::{KvPool, KvState};
use crate::model::qwen3::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::model::{ModelForward, Qwen3Model};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

pub(crate) struct PrefillPlan<'a> {
    pub prompts: &'a [&'a [u32]],
    pub kv_states: &'a mut [KvState],
    pub echo: bool,
}

pub(crate) struct DecodePlan<'a> {
    pub token_ids: &'a [u32],
    pub kv_states: &'a mut [&'a mut KvState],
}

pub(crate) struct UnifiedPlan<'a> {
    pub prefill_prompts: &'a [&'a [u32]],
    pub prefill_kv_states: &'a mut [KvState],
    pub decode_tokens: &'a [u32],
    pub decode_kv_states: &'a mut [&'a mut KvState],
}

pub(crate) struct PrefillResult {
    pub logits: Vec<DeviceVec>,
    pub all_position_logits: Option<HiddenStates>,
}

pub(crate) struct UnifiedResult {
    pub prefill_logits: Vec<DeviceVec>,
    pub decode_logits: Vec<DeviceVec>,
}

pub(crate) struct DecodeStep<'a> {
    executor: &'a mut SingleGpuQwen3Executor,
    batch_size: usize,
}

impl<'a> DecodeStep<'a> {
    pub(crate) fn snapshot_cpu_logits(&self, requested_topk: &[usize]) -> Vec<Option<Vec<f32>>> {
        (0..self.batch_size)
            .map(|i| {
                if requested_topk[i] > 0 {
                    ops::extract_vec(self.executor.device_ctx(), &self.executor.bufs.logits, i)
                        .ok()
                        .and_then(|v| v.to_host(self.executor.device_ctx()).ok())
                } else {
                    None
                }
            })
            .collect()
    }

    pub(crate) fn sample_tokens(
        self,
        params: &[&SamplingParams],
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        self.executor
            .model
            .select_tokens_batch_varied(&mut self.executor.bufs, params, rng)
    }
}

pub(crate) trait ModelExecutor: Send {
    fn execute_prefill<'a>(&mut self, plan: PrefillPlan<'a>) -> Result<PrefillResult>;
    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>>;
    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult>;
}

pub(crate) struct SingleGpuQwen3Executor {
    model: Qwen3Model,
    bufs: BatchDecodeBuffers,
}

impl SingleGpuQwen3Executor {
    pub(crate) fn new(model: Qwen3Model) -> Result<Self> {
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = model.create_batch_decode_bufs(max_bucket)?;
        Ok(Self { model, bufs })
    }

    pub(crate) fn alloc_kv(&self) -> KvState {
        self.model.alloc_kv()
    }

    pub(crate) fn kv_pool(&self) -> &KvPool {
        self.model.kv_pool()
    }

    pub(crate) fn device_ctx(&self) -> &DeviceContext {
        self.model.device_ctx()
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        self.model.is_stop_token(token_id)
    }
}

impl ModelExecutor for SingleGpuQwen3Executor {
    fn execute_prefill<'a>(&mut self, plan: PrefillPlan<'a>) -> Result<PrefillResult> {
        let (logits, all_position_logits) =
            self.model
                .batch_prefill(plan.prompts, plan.kv_states, plan.echo)?;
        Ok(PrefillResult {
            logits,
            all_position_logits,
        })
    }

    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>> {
        self.model
            .batch_decode(plan.token_ids, plan.kv_states, &mut self.bufs)?;
        Ok(DecodeStep {
            executor: self,
            batch_size: plan.token_ids.len(),
        })
    }

    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult> {
        let (prefill_logits, decode_logits) = self.model.unified_step(
            plan.prefill_prompts,
            plan.prefill_kv_states,
            plan.decode_tokens,
            plan.decode_kv_states,
        )?;
        Ok(UnifiedResult {
            prefill_logits,
            decode_logits,
        })
    }
}
