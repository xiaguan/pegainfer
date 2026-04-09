use std::sync::mpsc;
use std::thread;

use anyhow::Result;
use rand::rngs::StdRng;

use crate::kv_pool::{KvPool, KvState};
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
struct TokenSlicePtr {
    ptr: *const u32,
    len: usize,
}

// SAFETY: These pointers borrow scheduler-owned request token buffers for the
// duration of one synchronous executor step. The sender waits for the worker
// response before those borrows can end.
unsafe impl Send for TokenSlicePtr {}

impl TokenSlicePtr {
    fn from_slice(tokens: &[u32]) -> Self {
        Self {
            ptr: tokens.as_ptr(),
            len: tokens.len(),
        }
    }

    unsafe fn as_slice<'a>(self) -> &'a [u32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
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
                    ops::extract_vec(
                        self.executor.device_ctx(),
                        &self.executor.lane.decode_bufs().logits,
                        i,
                    )
                    .ok()
                    .and_then(|v| v.to_host(self.executor.device_ctx()).ok())
                } else {
                    None
                }
            })
            .collect()
    }

    fn sample_tokens(self, params: &[&SamplingParams], rng: &mut StdRng) -> Result<Vec<u32>> {
        self.executor.lane.select_tokens(params, rng)
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
                        &self.executor.primary_lane.decode_bufs().logits,
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
        self.executor.primary_lane.select_tokens(params, rng)
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

    pub(crate) fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        anyhow::ensure!(
            !device_ordinals.is_empty(),
            "Qwen3 executor requires at least one device"
        );
        if device_ordinals.len() == 1 {
            let model = Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: None,
                    device_ordinal: device_ordinals[0],
                },
            )?;
            Self::single(model)
        } else {
            Ok(Self::TensorParallel(TensorParallelQwen3Executor::new(
                model_path,
                enable_cuda_graph,
                device_ordinals,
            )?))
        }
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

struct LocalQwen3Lane {
    model: Qwen3Model,
    bufs: BatchDecodeBuffers,
}

impl LocalQwen3Lane {
    fn new(model: Qwen3Model) -> Result<Self> {
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = model.create_batch_decode_bufs(max_bucket)?;
        Ok(Self { model, bufs })
    }

    fn bind(&self) -> Result<CublasThreadGuard> {
        bind_model_thread(&self.model)?;
        Ok(CublasThreadGuard)
    }

    fn alloc_kv(&self) -> KvState {
        self.model.alloc_kv()
    }

    fn kv_pool(&self) -> &KvPool {
        self.model.kv_pool()
    }

    fn device_ctx(&self) -> &DeviceContext {
        self.model.device_ctx()
    }

    fn vocab_size(&self) -> usize {
        self.model.config().vocab_size
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.model.is_stop_token(token_id)
    }

    fn decode_bufs(&self) -> &BatchDecodeBuffers {
        &self.bufs
    }

    fn select_tokens(&mut self, params: &[&SamplingParams], rng: &mut StdRng) -> Result<Vec<u32>> {
        self.model
            .select_tokens_batch_varied(&mut self.bufs, params, rng)
    }

    fn execute_prefill(
        &mut self,
        prompts: &[&[u32]],
        kv_states: &mut [&mut KvState],
        echo: bool,
    ) -> Result<(Vec<DeviceVec>, Option<HiddenStates>)> {
        self.model.batch_prefill(prompts, kv_states, echo)
    }

    fn execute_decode(&mut self, token_ids: &[u32], kv_states: &mut [&mut KvState]) -> Result<()> {
        self.model
            .batch_decode(token_ids, kv_states, &mut self.bufs)
    }

    fn execute_unified(
        &mut self,
        prefill_prompts: &[&[u32]],
        prefill_kv_states: &mut [&mut KvState],
        decode_tokens: &[u32],
        decode_kv_states: &mut [&mut KvState],
    ) -> Result<(Vec<DeviceVec>, Vec<DeviceVec>)> {
        self.model.unified_step(
            prefill_prompts,
            prefill_kv_states,
            decode_tokens,
            decode_kv_states,
        )
    }
}

pub(crate) struct SingleGpuQwen3Executor {
    lane: LocalQwen3Lane,
}

impl SingleGpuQwen3Executor {
    fn new(model: Qwen3Model) -> Result<Self> {
        Ok(Self {
            lane: LocalQwen3Lane::new(model)?,
        })
    }

    fn page_size_inner(&self) -> usize {
        self.lane.kv_pool().layout().page_size
    }

    fn available_pages_inner(&self) -> usize {
        self.lane.kv_pool().available_pages()
    }

    fn device_ctx_inner(&self) -> &DeviceContext {
        self.lane.device_ctx()
    }

    fn vocab_size_inner(&self) -> usize {
        self.lane.vocab_size()
    }

    fn is_stop_token_inner(&self, token_id: u32) -> bool {
        self.lane.is_stop_token(token_id)
    }
}

impl ModelExecutor for SingleGpuQwen3Executor {
    fn alloc_kv(&self) -> RequestKvState {
        RequestKvState::new(vec![self.lane.alloc_kv()])
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
        let _cublas_guard = self.lane.bind()?;
        let mut local_kv_states: Vec<&mut KvState> = plan
            .kv_states
            .iter_mut()
            .map(|state| state.shard_mut(0))
            .collect();
        let (logits, all_position_logits) =
            self.lane
                .execute_prefill(plan.prompts, &mut local_kv_states, plan.echo)?;
        Ok(PrefillResult {
            logits,
            all_position_logits,
        })
    }

    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>> {
        let _cublas_guard = self.lane.bind()?;
        let mut local_kv_states: Vec<&mut KvState> = plan
            .kv_states
            .iter_mut()
            .map(|state| state.shard_mut(0))
            .collect();
        self.lane
            .execute_decode(plan.token_ids, &mut local_kv_states)?;
        Ok(DecodeStep::Single(SingleDecodeStep {
            executor: self,
            batch_size: plan.token_ids.len(),
        }))
    }

    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult> {
        let _cublas_guard = self.lane.bind()?;
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
        let (prefill_logits, decode_logits) = self.lane.execute_unified(
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
    primary_lane: LocalQwen3Lane,
    kv_pools: Vec<KvPool>,
    workers: Vec<RankWorker>,
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

        let kv_pools = models.iter().map(|model| model.kv_pool().clone()).collect();
        let mut lanes = models
            .into_iter()
            .map(LocalQwen3Lane::new)
            .collect::<Result<Vec<_>>>()?;
        let primary_lane = lanes.remove(0);
        let workers = lanes
            .into_iter()
            .enumerate()
            .map(|(index, lane)| RankWorker::spawn(index + 1, lane))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            primary_lane,
            kv_pools,
            workers,
        })
    }

    fn tp_size(&self) -> usize {
        self.kv_pools.len()
    }

    fn primary_model(&self) -> &Qwen3Model {
        &self.primary_lane.model
    }

    fn primary_device_ctx(&self) -> &DeviceContext {
        self.primary_model().device_ctx()
    }

    fn page_size_inner(&self) -> usize {
        self.primary_model().kv_pool().layout().page_size
    }

    fn available_pages_inner(&self) -> usize {
        self.kv_pools
            .iter()
            .map(KvPool::available_pages)
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

    fn shard_kv_ptrs(&self, kv_states: &mut [RequestKvState]) -> Vec<Vec<KvStatePtr>> {
        let tp_size = self.tp_size();
        let mut kv_by_rank: Vec<Vec<KvStatePtr>> = (0..tp_size)
            .map(|_| Vec::with_capacity(kv_states.len()))
            .collect();
        for kv_state in kv_states.iter_mut() {
            for (rank, rank_kvs) in kv_by_rank.iter_mut().enumerate() {
                rank_kvs.push(kv_state.shard_ptr(rank));
            }
        }
        kv_by_rank
    }

    fn shard_kv_ptr_refs(&self, kv_states: &mut [&mut RequestKvState]) -> Vec<Vec<KvStatePtr>> {
        let tp_size = self.tp_size();
        let mut kv_by_rank: Vec<Vec<KvStatePtr>> = (0..tp_size)
            .map(|_| Vec::with_capacity(kv_states.len()))
            .collect();
        for kv_state in kv_states.iter_mut() {
            for (rank, rank_kvs) in kv_by_rank.iter_mut().enumerate() {
                rank_kvs.push(kv_state.shard_ptr(rank));
            }
        }
        kv_by_rank
    }

    fn wait_for_workers(
        pending: Vec<mpsc::Receiver<Result<()>>>,
        op_name: &'static str,
    ) -> Result<()> {
        for recv in pending {
            recv.recv()
                .map_err(|_| anyhow::anyhow!("tensor-parallel {op_name} worker dropped"))??;
        }
        Ok(())
    }

    fn bind_primary(&self) -> Result<CublasThreadGuard> {
        self.primary_lane.bind()
    }

    fn dispatch_prefill_workers(
        &self,
        prompts: Vec<TokenSlicePtr>,
        mut kv_by_rank: Vec<Vec<KvStatePtr>>,
        echo: bool,
    ) -> Result<Vec<mpsc::Receiver<Result<()>>>> {
        let mut pending = Vec::with_capacity(self.workers.len());
        for (rank, worker) in self.workers.iter().enumerate() {
            pending.push(worker.execute_prefill(
                prompts.clone(),
                std::mem::take(&mut kv_by_rank[rank + 1]),
                echo,
            )?);
        }
        Ok(pending)
    }

    fn dispatch_decode_workers(
        &self,
        token_ids: Vec<u32>,
        mut kv_by_rank: Vec<Vec<KvStatePtr>>,
    ) -> Result<Vec<mpsc::Receiver<Result<()>>>> {
        let mut pending = Vec::with_capacity(self.workers.len());
        for (rank, worker) in self.workers.iter().enumerate() {
            pending.push(
                worker
                    .execute_decode(token_ids.clone(), std::mem::take(&mut kv_by_rank[rank + 1]))?,
            );
        }
        Ok(pending)
    }

    fn dispatch_unified_workers(
        &self,
        prefill_prompts: Vec<TokenSlicePtr>,
        mut prefill_kv_by_rank: Vec<Vec<KvStatePtr>>,
        decode_tokens: Vec<u32>,
        mut decode_kv_by_rank: Vec<Vec<KvStatePtr>>,
    ) -> Result<Vec<mpsc::Receiver<Result<()>>>> {
        let mut pending = Vec::with_capacity(self.workers.len());
        for (rank, worker) in self.workers.iter().enumerate() {
            pending.push(worker.execute_unified(
                prefill_prompts.clone(),
                std::mem::take(&mut prefill_kv_by_rank[rank + 1]),
                decode_tokens.clone(),
                std::mem::take(&mut decode_kv_by_rank[rank + 1]),
            )?);
        }
        Ok(pending)
    }
}

impl ModelExecutor for TensorParallelQwen3Executor {
    fn alloc_kv(&self) -> RequestKvState {
        RequestKvState::new(self.kv_pools.iter().map(KvPool::alloc).collect())
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
        let mut kv_by_rank = self.shard_kv_ptrs(plan.kv_states);
        let prompts: Vec<TokenSlicePtr> = plan
            .prompts
            .iter()
            .map(|tokens| TokenSlicePtr::from_slice(tokens))
            .collect();
        let pending = self.dispatch_prefill_workers(prompts, kv_by_rank.clone(), plan.echo)?;

        let _cublas_guard = self.bind_primary()?;
        let result = {
            let local_kv_ptrs = std::mem::take(&mut kv_by_rank[0]);
            let mut local_kv_states: Vec<&mut KvState> = local_kv_ptrs
                .into_iter()
                .map(|ptr| unsafe { ptr.as_mut() })
                .collect();
            self.primary_lane
                .execute_prefill(plan.prompts, &mut local_kv_states, plan.echo)?
        };

        Self::wait_for_workers(pending, "prefill")?;

        Ok(PrefillResult {
            logits: result.0,
            all_position_logits: result.1,
        })
    }

    fn begin_decode<'a>(&'a mut self, plan: DecodePlan<'a>) -> Result<DecodeStep<'a>> {
        let mut kv_by_rank = self.shard_kv_ptr_refs(plan.kv_states);
        let token_ids = plan.token_ids.to_vec();
        let pending = self.dispatch_decode_workers(token_ids, kv_by_rank.clone())?;

        let _cublas_guard = self.bind_primary()?;
        {
            let local_kv_ptrs = std::mem::take(&mut kv_by_rank[0]);
            let mut local_kv_states: Vec<&mut KvState> = local_kv_ptrs
                .into_iter()
                .map(|ptr| unsafe { ptr.as_mut() })
                .collect();
            self.primary_lane
                .execute_decode(plan.token_ids, &mut local_kv_states)?;
        }

        Self::wait_for_workers(pending, "decode")?;

        Ok(DecodeStep::TensorParallel(TensorParallelDecodeStep {
            executor: self,
            batch_size: plan.token_ids.len(),
        }))
    }

    fn execute_unified<'a>(&mut self, plan: UnifiedPlan<'a>) -> Result<UnifiedResult> {
        let mut prefill_kv_by_rank = self.shard_kv_ptrs(plan.prefill_kv_states);
        let mut decode_kv_by_rank = self.shard_kv_ptr_refs(plan.decode_kv_states);

        let prefill_prompts: Vec<TokenSlicePtr> = plan
            .prefill_prompts
            .iter()
            .map(|tokens| TokenSlicePtr::from_slice(tokens))
            .collect();
        let decode_tokens = plan.decode_tokens.to_vec();
        let pending = self.dispatch_unified_workers(
            prefill_prompts,
            prefill_kv_by_rank.clone(),
            decode_tokens,
            decode_kv_by_rank.clone(),
        )?;

        let _cublas_guard = self.bind_primary()?;
        let result = {
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
            self.primary_lane.execute_unified(
                plan.prefill_prompts,
                &mut prefill_kv_states,
                plan.decode_tokens,
                &mut decode_kv_states,
            )?
        };

        Self::wait_for_workers(pending, "unified")?;

        Ok(UnifiedResult {
            prefill_logits: result.0,
            decode_logits: result.1,
        })
    }
}

impl Drop for TensorParallelQwen3Executor {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

enum WorkerCommand {
    Prefill {
        prompts: Vec<TokenSlicePtr>,
        kv_ptrs: Vec<KvStatePtr>,
        echo: bool,
        resp: mpsc::Sender<Result<()>>,
    },
    Decode {
        token_ids: Vec<u32>,
        kv_ptrs: Vec<KvStatePtr>,
        resp: mpsc::Sender<Result<()>>,
    },
    Unified {
        prefill_prompts: Vec<TokenSlicePtr>,
        prefill_kv_ptrs: Vec<KvStatePtr>,
        decode_tokens: Vec<u32>,
        decode_kv_ptrs: Vec<KvStatePtr>,
        resp: mpsc::Sender<Result<()>>,
    },
    Shutdown,
}

struct RankWorker {
    tx: mpsc::Sender<WorkerCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RankWorker {
    fn spawn(rank: usize, mut lane: LocalQwen3Lane) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let (startup_tx, startup_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name(format!("qwen3-tp-rank-{rank}"))
            .spawn(move || {
                let startup = lane.bind();
                match startup {
                    Ok(_guard) => {
                        let _ = startup_tx.send(Ok(()));
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                WorkerCommand::Prefill {
                                    prompts,
                                    kv_ptrs,
                                    echo,
                                    resp,
                                } => {
                                    let prompts: Vec<&[u32]> = prompts
                                        .into_iter()
                                        .map(|ptr| unsafe { ptr.as_slice() })
                                        .collect();
                                    let mut kv_states: Vec<&mut KvState> = kv_ptrs
                                        .into_iter()
                                        .map(|ptr| unsafe { ptr.as_mut() })
                                        .collect();
                                    let result = lane
                                        .execute_prefill(&prompts, &mut kv_states, echo)
                                        .map(|_| ());
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::Decode {
                                    token_ids,
                                    kv_ptrs,
                                    resp,
                                } => {
                                    let mut kv_states: Vec<&mut KvState> = kv_ptrs
                                        .into_iter()
                                        .map(|ptr| unsafe { ptr.as_mut() })
                                        .collect();
                                    let result =
                                        lane.execute_decode(&token_ids, &mut kv_states).map(|_| ());
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::Unified {
                                    prefill_prompts,
                                    prefill_kv_ptrs,
                                    decode_tokens,
                                    decode_kv_ptrs,
                                    resp,
                                } => {
                                    let prefill_prompts: Vec<&[u32]> = prefill_prompts
                                        .into_iter()
                                        .map(|ptr| unsafe { ptr.as_slice() })
                                        .collect();
                                    let mut prefill_kv_states: Vec<&mut KvState> = prefill_kv_ptrs
                                        .into_iter()
                                        .map(|ptr| unsafe { ptr.as_mut() })
                                        .collect();
                                    let mut decode_kv_states: Vec<&mut KvState> = decode_kv_ptrs
                                        .into_iter()
                                        .map(|ptr| unsafe { ptr.as_mut() })
                                        .collect();
                                    let result = lane
                                        .execute_unified(
                                            &prefill_prompts,
                                            &mut prefill_kv_states,
                                            &decode_tokens,
                                            &mut decode_kv_states,
                                        )
                                        .map(|_| ());
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::Shutdown => break,
                            }
                        }
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn tensor-parallel worker {rank}: {e}"))?;
        startup_rx.recv().map_err(|_| {
            anyhow::anyhow!("tensor-parallel worker {rank} exited during startup")
        })??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    fn execute_prefill(
        &self,
        prompts: Vec<TokenSlicePtr>,
        kv_ptrs: Vec<KvStatePtr>,
        echo: bool,
    ) -> Result<mpsc::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx
            .send(WorkerCommand::Prefill {
                prompts,
                kv_ptrs,
                echo,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel prefill worker channel closed"))?;
        Ok(resp_rx)
    }

    fn execute_decode(
        &self,
        token_ids: Vec<u32>,
        kv_ptrs: Vec<KvStatePtr>,
    ) -> Result<mpsc::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx
            .send(WorkerCommand::Decode {
                token_ids,
                kv_ptrs,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel decode worker channel closed"))?;
        Ok(resp_rx)
    }

    fn execute_unified(
        &self,
        prefill_prompts: Vec<TokenSlicePtr>,
        prefill_kv_ptrs: Vec<KvStatePtr>,
        decode_tokens: Vec<u32>,
        decode_kv_ptrs: Vec<KvStatePtr>,
    ) -> Result<mpsc::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx
            .send(WorkerCommand::Unified {
                prefill_prompts,
                prefill_kv_ptrs,
                decode_tokens,
                decode_kv_ptrs,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel unified worker channel closed"))?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
