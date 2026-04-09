use std::thread;

use anyhow::Result;
use rand::rngs::StdRng;

use crate::kv_pool::KvState;
use crate::model::qwen3::batch_decode_buffers::{BATCH_BUCKETS, BatchDecodeBuffers};
use crate::model::{ModelForward, ModelRuntimeConfig, Qwen3Model, TensorParallelConfig};
use crate::ops;
use crate::sampler::SamplingParams;
use crate::tensor::{DeviceContext, DeviceVec, HiddenStates};

pub(crate) struct RequestKvState {
    shards: Vec<KvState>,
}

#[derive(Clone, Copy)]
struct KvStatePtr(*mut KvState);

// SAFETY: Each pointer targets a single rank-local KV shard. Callers only
// construct one pointer per (request, rank) pair, so no two worker threads
// mutate the same shard concurrently.
unsafe impl Send for KvStatePtr {}

impl KvStatePtr {
    unsafe fn as_mut<'a>(self) -> &'a mut KvState {
        unsafe { &mut *self.0 }
    }
}

#[derive(Clone, Copy)]
struct BatchDecodeBuffersPtr(*mut BatchDecodeBuffers);

// SAFETY: Each pointer targets one rank-local decode buffer set and is handed
// to exactly one worker thread for the duration of a single decode step.
unsafe impl Send for BatchDecodeBuffersPtr {}

impl BatchDecodeBuffersPtr {
    unsafe fn as_mut<'a>(self) -> &'a mut BatchDecodeBuffers {
        unsafe { &mut *self.0 }
    }
}

impl RequestKvState {
    pub(crate) fn new(shards: Vec<KvState>) -> Self {
        Self { shards }
    }

    fn shard_mut(&mut self, rank: usize) -> &mut KvState {
        &mut self.shards[rank]
    }

    fn shard_ptr(&mut self, rank: usize) -> KvStatePtr {
        KvStatePtr(&mut self.shards[rank])
    }
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

fn bind_model_thread(model: &Qwen3Model) -> Result<()> {
    unsafe {
        let err = crate::ffi::cuda_set_device(model.device_ctx().device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on worker thread: cudaError={}",
                model.device_ctx().device_ordinal,
                err
            ));
        }
    }
    model
        .device_ctx()
        .ctx
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("Failed to bind CUDA context to thread: {e}"))?;
    unsafe {
        crate::ffi::cublas_init();
    }
    Ok(())
}

pub(crate) struct PrefillPlan<'a> {
    pub prompts: &'a [&'a [u32]],
    pub kv_states: &'a mut [RequestKvState],
    pub echo: bool,
}

pub(crate) struct DecodePlan<'a> {
    pub token_ids: &'a [u32],
    pub kv_states: &'a mut [&'a mut RequestKvState],
}

pub(crate) struct UnifiedPlan<'a> {
    pub prefill_prompts: &'a [&'a [u32]],
    pub prefill_kv_states: &'a mut [RequestKvState],
    pub decode_tokens: &'a [u32],
    pub decode_kv_states: &'a mut [&'a mut RequestKvState],
}

pub(crate) struct PrefillResult {
    pub logits: Vec<DeviceVec>,
    pub all_position_logits: Option<HiddenStates>,
}

pub(crate) struct UnifiedResult {
    pub prefill_logits: Vec<DeviceVec>,
    pub decode_logits: Vec<DeviceVec>,
}

pub(crate) enum DecodeStep<'a> {
    Single(SingleDecodeStep<'a>),
    TensorParallel(TensorParallelDecodeStep<'a>),
}

impl<'a> DecodeStep<'a> {
    pub(crate) fn snapshot_cpu_logits(&self, requested_topk: &[usize]) -> Vec<Option<Vec<f32>>> {
        match self {
            Self::Single(step) => step.snapshot_cpu_logits(requested_topk),
            Self::TensorParallel(step) => step.snapshot_cpu_logits(requested_topk),
        }
    }

    pub(crate) fn sample_tokens(
        self,
        params: &[&SamplingParams],
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Single(step) => step.sample_tokens(params, rng),
            Self::TensorParallel(step) => step.sample_tokens(params, rng),
        }
    }
}

pub(crate) struct SingleDecodeStep<'a> {
    executor: &'a mut SingleGpuQwen3Executor,
    batch_size: usize,
}

impl<'a> SingleDecodeStep<'a> {
    fn snapshot_cpu_logits(&self, requested_topk: &[usize]) -> Vec<Option<Vec<f32>>> {
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

    fn sample_tokens(self, params: &[&SamplingParams], rng: &mut StdRng) -> Result<Vec<u32>> {
        self.executor
            .model
            .select_tokens_batch_varied(&mut self.executor.bufs, params, rng)
    }
}

pub(crate) struct TensorParallelDecodeStep<'a> {
    executor: &'a mut TensorParallelQwen3Executor,
    batch_size: usize,
}

impl<'a> TensorParallelDecodeStep<'a> {
    fn snapshot_cpu_logits(&self, requested_topk: &[usize]) -> Vec<Option<Vec<f32>>> {
        (0..self.batch_size)
            .map(|i| {
                if requested_topk[i] > 0 {
                    ops::extract_vec(
                        self.executor.primary_device_ctx(),
                        &self.executor.primary_bufs().logits,
                        i,
                    )
                    .ok()
                    .and_then(|v| v.to_host(self.executor.primary_device_ctx()).ok())
                } else {
                    None
                }
            })
            .collect()
    }

    fn sample_tokens(self, params: &[&SamplingParams], rng: &mut StdRng) -> Result<Vec<u32>> {
        let executor = self.executor;
        let model = &executor.models[0];
        let bufs = &mut executor.bufs[0];
        model.select_tokens_batch_varied(bufs, params, rng)
    }
}

pub(crate) trait ModelExecutor: Send {
    fn alloc_kv(&self) -> RequestKvState;
    fn page_size(&self) -> usize;
    fn available_pages(&self) -> usize;
    fn device_ctx(&self) -> &DeviceContext;
    fn vocab_size(&self) -> usize;
    fn is_stop_token(&self, token_id: u32) -> bool;

    fn execute_prefill<'a>(&mut self, plan: PrefillPlan<'a>) -> Result<PrefillResult>;
    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>>;
    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult>;
}

pub(crate) enum Qwen3Executor {
    Single(SingleGpuQwen3Executor),
    TensorParallel(TensorParallelQwen3Executor),
}

impl Qwen3Executor {
    pub(crate) fn single(model: Qwen3Model) -> Result<Self> {
        Ok(Self::Single(SingleGpuQwen3Executor::new(model)?))
    }

    pub(crate) fn tensor_parallel(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        Ok(Self::TensorParallel(TensorParallelQwen3Executor::new(
            model_path,
            enable_cuda_graph,
            device_ordinals,
        )?))
    }
}

impl ModelExecutor for Qwen3Executor {
    fn alloc_kv(&self) -> RequestKvState {
        match self {
            Self::Single(executor) => executor.alloc_kv(),
            Self::TensorParallel(executor) => executor.alloc_kv(),
        }
    }

    fn page_size(&self) -> usize {
        match self {
            Self::Single(executor) => executor.page_size(),
            Self::TensorParallel(executor) => executor.page_size(),
        }
    }

    fn available_pages(&self) -> usize {
        match self {
            Self::Single(executor) => executor.available_pages(),
            Self::TensorParallel(executor) => executor.available_pages(),
        }
    }

    fn device_ctx(&self) -> &DeviceContext {
        match self {
            Self::Single(executor) => executor.device_ctx(),
            Self::TensorParallel(executor) => executor.device_ctx(),
        }
    }

    fn vocab_size(&self) -> usize {
        match self {
            Self::Single(executor) => executor.vocab_size(),
            Self::TensorParallel(executor) => executor.vocab_size(),
        }
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        match self {
            Self::Single(executor) => executor.is_stop_token(token_id),
            Self::TensorParallel(executor) => executor.is_stop_token(token_id),
        }
    }

    fn execute_prefill<'a>(&mut self, plan: PrefillPlan<'a>) -> Result<PrefillResult> {
        match self {
            Self::Single(executor) => executor.execute_prefill(plan),
            Self::TensorParallel(executor) => executor.execute_prefill(plan),
        }
    }

    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>> {
        match self {
            Self::Single(executor) => executor.begin_decode(plan),
            Self::TensorParallel(executor) => executor.begin_decode(plan),
        }
    }

    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult> {
        match self {
            Self::Single(executor) => executor.execute_unified(plan),
            Self::TensorParallel(executor) => executor.execute_unified(plan),
        }
    }
}

pub(crate) struct SingleGpuQwen3Executor {
    model: Qwen3Model,
    bufs: BatchDecodeBuffers,
}

impl SingleGpuQwen3Executor {
    fn new(model: Qwen3Model) -> Result<Self> {
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = model.create_batch_decode_bufs(max_bucket)?;
        Ok(Self { model, bufs })
    }

    fn page_size_inner(&self) -> usize {
        self.model.kv_pool().layout().page_size
    }

    fn available_pages_inner(&self) -> usize {
        self.model.kv_pool().available_pages()
    }

    fn device_ctx_inner(&self) -> &DeviceContext {
        self.model.device_ctx()
    }

    fn vocab_size_inner(&self) -> usize {
        self.model.config().vocab_size
    }

    fn is_stop_token_inner(&self, token_id: u32) -> bool {
        self.model.is_stop_token(token_id)
    }
}

impl ModelExecutor for SingleGpuQwen3Executor {
    fn alloc_kv(&self) -> RequestKvState {
        RequestKvState::new(vec![self.model.alloc_kv()])
    }

    fn page_size(&self) -> usize {
        self.page_size_inner()
    }

    fn available_pages(&self) -> usize {
        self.available_pages_inner()
    }

    fn device_ctx(&self) -> &DeviceContext {
        self.device_ctx_inner()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size_inner()
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.is_stop_token_inner(token_id)
    }

    fn execute_prefill<'a>(&mut self, plan: PrefillPlan<'a>) -> Result<PrefillResult> {
        bind_model_thread(&self.model)?;
        let _cublas_guard = CublasThreadGuard;
        let mut local_kv_states: Vec<&mut KvState> = plan
            .kv_states
            .iter_mut()
            .map(|state| state.shard_mut(0))
            .collect();
        let (logits, all_position_logits) =
            self.model
                .batch_prefill(plan.prompts, &mut local_kv_states, plan.echo)?;
        Ok(PrefillResult {
            logits,
            all_position_logits,
        })
    }

    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>> {
        bind_model_thread(&self.model)?;
        let _cublas_guard = CublasThreadGuard;
        let mut local_kv_states: Vec<&mut KvState> = plan
            .kv_states
            .iter_mut()
            .map(|state| state.shard_mut(0))
            .collect();
        self.model
            .batch_decode(plan.token_ids, &mut local_kv_states, &mut self.bufs)?;
        Ok(DecodeStep::Single(SingleDecodeStep {
            executor: self,
            batch_size: plan.token_ids.len(),
        }))
    }

    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult> {
        bind_model_thread(&self.model)?;
        let _cublas_guard = CublasThreadGuard;
        let mut prefill_kv_states: Vec<&mut KvState> = plan
            .prefill_kv_states
            .iter_mut()
            .map(|state| state.shard_mut(0))
            .collect();
        let mut decode_kv_states: Vec<&mut KvState> = plan
            .decode_kv_states
            .iter_mut()
            .map(|state| state.shard_mut(0))
            .collect();
        let (prefill_logits, decode_logits) = self.model.unified_step(
            plan.prefill_prompts,
            &mut prefill_kv_states,
            plan.decode_tokens,
            &mut decode_kv_states,
        )?;
        Ok(UnifiedResult {
            prefill_logits,
            decode_logits,
        })
    }
}

pub(crate) struct TensorParallelQwen3Executor {
    models: Vec<Qwen3Model>,
    bufs: Vec<BatchDecodeBuffers>,
}

impl TensorParallelQwen3Executor {
    fn new(model_path: &str, enable_cuda_graph: bool, device_ordinals: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            !device_ordinals.is_empty(),
            "tensor-parallel executor requires at least one device"
        );
        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                },
            )?);
        }

        let streams = models
            .iter()
            .map(|m| m.device_ctx().stream.clone())
            .collect();
        let comms = cudarc::nccl::safe::Comm::from_devices(streams)
            .map_err(|e| anyhow::anyhow!("failed to initialize NCCL comms: {e:?}"))?;
        for (model, comm) in models.iter_mut().zip(comms) {
            model.attach_tp_comm(comm);
        }

        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let mut bufs = Vec::with_capacity(world_size);
        for model in &models {
            bufs.push(model.create_batch_decode_bufs(max_bucket)?);
        }

        Ok(Self { models, bufs })
    }

    fn primary_model(&self) -> &Qwen3Model {
        &self.models[0]
    }

    fn primary_bufs(&self) -> &BatchDecodeBuffers {
        &self.bufs[0]
    }

    fn primary_device_ctx(&self) -> &DeviceContext {
        self.primary_model().device_ctx()
    }

    fn page_size_inner(&self) -> usize {
        self.primary_model().kv_pool().layout().page_size
    }

    fn available_pages_inner(&self) -> usize {
        self.models
            .iter()
            .map(|model| model.kv_pool().available_pages())
            .min()
            .unwrap_or(0)
    }

    fn device_ctx_inner(&self) -> &DeviceContext {
        self.primary_device_ctx()
    }

    fn vocab_size_inner(&self) -> usize {
        self.primary_model().config().vocab_size
    }

    fn is_stop_token_inner(&self, token_id: u32) -> bool {
        self.primary_model().is_stop_token(token_id)
    }
}

impl ModelExecutor for TensorParallelQwen3Executor {
    fn alloc_kv(&self) -> RequestKvState {
        RequestKvState::new(self.models.iter().map(|model| model.alloc_kv()).collect())
    }

    fn page_size(&self) -> usize {
        self.page_size_inner()
    }

    fn available_pages(&self) -> usize {
        self.available_pages_inner()
    }

    fn device_ctx(&self) -> &DeviceContext {
        self.device_ctx_inner()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size_inner()
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.is_stop_token_inner(token_id)
    }

    fn execute_prefill<'a>(&mut self, plan: PrefillPlan<'a>) -> Result<PrefillResult> {
        let tp_size = self.models.len();
        let mut kv_by_rank: Vec<Vec<KvStatePtr>> = (0..tp_size)
            .map(|_| Vec::with_capacity(plan.kv_states.len()))
            .collect();
        for kv_state in plan.kv_states.iter_mut() {
            for rank in 0..tp_size {
                kv_by_rank[rank].push(kv_state.shard_ptr(rank));
            }
        }

        thread::scope(|scope| -> Result<PrefillResult> {
            let mut handles = Vec::new();
            for rank in 1..tp_size {
                let model = &self.models[rank];
                let prompts = plan.prompts;
                let echo = plan.echo;
                let local_kv_ptrs = std::mem::take(&mut kv_by_rank[rank]);
                handles.push(scope.spawn(move || {
                    bind_model_thread(model)?;
                    let _cublas_guard = CublasThreadGuard;
                    let mut local_kv_states: Vec<&mut KvState> = local_kv_ptrs
                        .into_iter()
                        .map(|ptr| unsafe { ptr.as_mut() })
                        .collect();
                    model.batch_prefill(prompts, &mut local_kv_states, echo)
                }));
            }

            bind_model_thread(&self.models[0])?;
            let _cublas_guard = CublasThreadGuard;
            let result = {
                let model = &self.models[0];
                let local_kv_ptrs = std::mem::take(&mut kv_by_rank[0]);
                let mut local_kv_states: Vec<&mut KvState> = local_kv_ptrs
                    .into_iter()
                    .map(|ptr| unsafe { ptr.as_mut() })
                    .collect();
                model.batch_prefill(plan.prompts, &mut local_kv_states, plan.echo)?
            };

            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("tensor-parallel prefill thread panicked"))??;
            }

            Ok(PrefillResult {
                logits: result.0,
                all_position_logits: result.1,
            })
        })
    }

    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>> {
        let tp_size = self.models.len();
        let mut kv_by_rank: Vec<Vec<KvStatePtr>> = (0..tp_size)
            .map(|_| Vec::with_capacity(plan.kv_states.len()))
            .collect();
        for kv_state in plan.kv_states.iter_mut() {
            for rank in 0..tp_size {
                kv_by_rank[rank].push(kv_state.shard_ptr(rank));
            }
        }
        let buf_ptrs: Vec<BatchDecodeBuffersPtr> = self
            .bufs
            .iter_mut()
            .map(|bufs| BatchDecodeBuffersPtr(bufs as *mut BatchDecodeBuffers))
            .collect();

        thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            for rank in 1..tp_size {
                let model = &self.models[rank];
                let token_ids = plan.token_ids;
                let local_kv_ptrs = std::mem::take(&mut kv_by_rank[rank]);
                let bufs_ptr = buf_ptrs[rank];
                handles.push(scope.spawn(move || {
                    bind_model_thread(model)?;
                    let _cublas_guard = CublasThreadGuard;
                    let mut local_kv_states: Vec<&mut KvState> = local_kv_ptrs
                        .into_iter()
                        .map(|ptr| unsafe { ptr.as_mut() })
                        .collect();
                    let bufs = unsafe { bufs_ptr.as_mut() };
                    model.batch_decode(token_ids, &mut local_kv_states, bufs)
                }));
            }

            bind_model_thread(&self.models[0])?;
            let _cublas_guard = CublasThreadGuard;
            {
                let model = &self.models[0];
                let local_kv_ptrs = std::mem::take(&mut kv_by_rank[0]);
                let mut local_kv_states: Vec<&mut KvState> = local_kv_ptrs
                    .into_iter()
                    .map(|ptr| unsafe { ptr.as_mut() })
                    .collect();
                let bufs = unsafe { buf_ptrs[0].as_mut() };
                model.batch_decode(plan.token_ids, &mut local_kv_states, bufs)?;
            }

            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("tensor-parallel decode thread panicked"))??;
            }

            Ok(())
        })?;

        Ok(DecodeStep::TensorParallel(TensorParallelDecodeStep {
            executor: self,
            batch_size: plan.token_ids.len(),
        }))
    }

    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult> {
        let tp_size = self.models.len();
        let mut prefill_kv_by_rank: Vec<Vec<KvStatePtr>> = (0..tp_size)
            .map(|_| Vec::with_capacity(plan.prefill_kv_states.len()))
            .collect();
        for kv_state in plan.prefill_kv_states.iter_mut() {
            for rank in 0..tp_size {
                prefill_kv_by_rank[rank].push(kv_state.shard_ptr(rank));
            }
        }
        let mut decode_kv_by_rank: Vec<Vec<KvStatePtr>> = (0..tp_size)
            .map(|_| Vec::with_capacity(plan.decode_kv_states.len()))
            .collect();
        for kv_state in plan.decode_kv_states.iter_mut() {
            for rank in 0..tp_size {
                decode_kv_by_rank[rank].push(kv_state.shard_ptr(rank));
            }
        }

        thread::scope(|scope| -> Result<UnifiedResult> {
            let mut handles = Vec::new();
            for rank in 1..tp_size {
                let model = &self.models[rank];
                let prompts = plan.prefill_prompts;
                let decode_tokens = plan.decode_tokens;
                let prefill_kv_ptrs = std::mem::take(&mut prefill_kv_by_rank[rank]);
                let decode_kv_ptrs = std::mem::take(&mut decode_kv_by_rank[rank]);
                handles.push(scope.spawn(move || {
                    bind_model_thread(model)?;
                    let _cublas_guard = CublasThreadGuard;
                    let mut prefill_kv_states: Vec<&mut KvState> = prefill_kv_ptrs
                        .into_iter()
                        .map(|ptr| unsafe { ptr.as_mut() })
                        .collect();
                    let mut decode_kv_states: Vec<&mut KvState> = decode_kv_ptrs
                        .into_iter()
                        .map(|ptr| unsafe { ptr.as_mut() })
                        .collect();
                    model.unified_step(
                        prompts,
                        &mut prefill_kv_states,
                        decode_tokens,
                        &mut decode_kv_states,
                    )
                }));
            }

            bind_model_thread(&self.models[0])?;
            let _cublas_guard = CublasThreadGuard;
            let result = {
                let model = &self.models[0];
                let prefill_kv_ptrs = std::mem::take(&mut prefill_kv_by_rank[0]);
                let decode_kv_ptrs = std::mem::take(&mut decode_kv_by_rank[0]);
                let mut prefill_kv_states: Vec<&mut KvState> = prefill_kv_ptrs
                    .into_iter()
                    .map(|ptr| unsafe { ptr.as_mut() })
                    .collect();
                let mut decode_kv_states: Vec<&mut KvState> = decode_kv_ptrs
                    .into_iter()
                    .map(|ptr| unsafe { ptr.as_mut() })
                    .collect();
                model.unified_step(
                    plan.prefill_prompts,
                    &mut prefill_kv_states,
                    plan.decode_tokens,
                    &mut decode_kv_states,
                )?
            };

            for handle in handles {
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("tensor-parallel unified thread panicked"))??;
            }

            Ok(UnifiedResult {
                prefill_logits: result.0,
                decode_logits: result.1,
            })
        })
    }
}
