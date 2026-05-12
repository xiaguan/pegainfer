# DeepSeek V4 Sub-80 Decode

**Created**: 2026-05-12
**Status**: complete

## TL;DR

After the grouped MoE pointer-cache commit, DeepSeek V4 fixed-token long bench was `83.37-89.65ms/token`. Linux rank-worker CPU affinity moved the fixed-token long bench to `72.88-73.60ms/token` while preserving the fixed hash and exact E2E, and profile evidence showed attention hidden `AllReduce_Sum_f32` wall/skew falling from `1.023/1.006ms` to `0.861/0.843ms`. Replacing hot-path temporary zero-fill allocations with uninitialized allocations then moved the fixed-token long bench to `63.35-64.51ms/token` with the same fixed hash and exact E2E. The sub-70 objective is complete; the remaining larger opportunity is still launch/skew reduction, because post-uninit f32 all-reduce skew is essentially unchanged.

## Preparation

- **Read**:
  - `docs/index.md` - confirmed the just-completed 90ms project and the active MoE AG/RS history.
  - `docs/projects/deepseek-v4-90ms-decode.md` - confirmed pointer caching moved the long bench below 90ms and reduced the nsys reduce-scatter synchronization window.
  - `docs/projects/deepseek-moe-tilelang-review.md` - confirmed the earlier rank-stage proof method: group NCCL kernels into logical collectives and separate arrival skew from post-arrival tail.
- **Relevant history**:
  - The latest effective commit is `ee01f0b feat(deepseek-v4): cache moe grouped expert pointers`.
  - Post-change nsys still shows large f32 all-reduce windows: `AllReduce_Sum_f32 = 5803.26ms`, while `ReduceScatter_Sum_f32` dropped to `2040.33ms`.
- **Plan**:
  1. Inspect `/tmp/dsv4_90ms_ptr_cache_profile.sqlite` on 5090 and reproduce the logical-collective grouping method for post-change all-reduce/reduce-scatter/all-gather kernels.
  2. Identify the largest remaining collective windows by sequence position and infer whether they align with attention all-reduce, MoE reduce-scatter, embedding, indexer score all-reduce, or final logits all-gather.
  3. Choose the next implementation slice from evidence, not from kernel-name intuition.
  4. Keep fixed-token long bench and exact E2E as the gate for any code change.
- **Risks / open questions**:
  - The current trace may lack NVTX labels at collective call sites, so the first pass may need sequence-position inference before adding scoped labels.
  - Nsight Systems kernel duration is still synchronization-window evidence, not pure transfer time.

## Execution Log

### Step 1: group post-cache NCCL kernels
- Parsed `/tmp/dsv4_90ms_ptr_cache_profile.sqlite` on 5090 and grouped consecutive NCCL kernels by 8 ranks to estimate logical collective windows.
- Post-cache profile command source: fixed-token bench with `--prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`; nsys profiler overhead makes absolute TPOT higher, so the trace is used for relative window decomposition.
- Main result:

| NCCL kernel | Kernels | Rank groups | Kernel duration sum | Logical wall sum | Start-skew sum | Post-arrival tail sum | Avg wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AllReduce_Sum_f32` | 9048 | 1131 | `5803.26ms` | `1940.21ms` | `1370.35ms` | `569.86ms` | `1715.5us` |
| `ReduceScatter_Sum_f32` | 6075 | 759 | `2040.33ms` | `378.97ms` | `367.46ms` | `11.50ms` | `499.3us` |
| `AllGather` | 6656 | 832 | `1013.17ms` | `149.57ms` | `138.16ms` | `11.41ms` | `179.8us` |
| `AllReduce_Sum_bf16` | 144 | 18 | `10.01ms` | `2.06ms` | `1.69ms` | `0.37ms` | `114.4us` |

- Interpretation:
  - The remaining dominant collective window is f32 all-reduce, not grouped MoE FP4 compute.
  - `AllReduce_Sum_f32` still contains a large rank-arrival component: `1370.35ms / 1940.21ms` logical wall over the profiled run.
  - `ReduceScatter_Sum_f32` improved after grouped MoE pointer caching, but its wall is still almost entirely arrival skew, so NCCL duration must not be read as pure transfer time.
  - The next code slice should label or isolate f32 all-reduce call sites before changing algorithms, because the kernel name alone mixes attention hidden all-reduce, indexer score reductions, and any other f32 all-reduce with the same element count.

### Step 2: correct f32 all-reduce grouping and pick first code slice
- The first pass grouped consecutive f32 NCCL kernels by 8. That is unsafe when one logical collective has large start skew: the group can contain duplicate devices and miss another device.
- Corrected the grouping by joining `CUPTI_ACTIVITY_KIND_KERNEL.correlationId` to `CUPTI_ACTIVITY_KIND_RUNTIME.globalTid`, aligning each rank worker's f32 all-reduce sequence, and dropping the trace-boundary region.
- Corrected result over the stable aligned region:

| f32 all-reduce class | Groups | Logical wall sum | Start-skew sum | Tail sum | Avg wall | p50 wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention hidden `1x4096` | 747 | `764.50ms` | `751.34ms` | `13.16ms` | `1.023ms` | `1.045ms` |
| ratio-4 indexer score | 364 | `148.83ms` | `141.28ms` | `7.56ms` | `0.409ms` | `0.353ms` |

- Interpretation:
  - The dominant f32 all-reduce window is still attention hidden all-reduce, but the reliable per-window scale is about `1ms`, not the larger values produced by naive grouping.
  - The time is almost entirely rank-arrival skew, not post-arrival NCCL tail.
  - The first implementation slice targets repeated deterministic decode top-k index generation before attention all-reduce. `window_topk_indices_decode` is identical for every layer at the same `start_pos`; non-overlap compressed indices are identical for all layers sharing the same ratio.
- Code attempt:
  - Added `DecodeTopkIndices`, built once per rank per decode token.
  - Reused cached window indices in ratio-0, ratio-4, and non-overlap compressed attention.
  - Reused cached deterministic compressed indices for non-overlap compressed attention.
  - No new `bs=1` or `seq_len=1` specialization was added.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 validation:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
  - Fixed-token bench round 1: `steady_tpot_ms.avg = 87.691639`, hash `6346f03343d75a65`.
  - Fixed-token bench round 2: `steady_tpot_ms.avg = 86.373297`, hash `6346f03343d75a65`.
- Decision:
  - Do not keep the top-k cache code. It is correct, but it does not move the fixed-token long bench below the existing `83.37-89.65ms` post-pointer-cache range.
  - The result suggests deterministic top-k index generation is not a dominant contributor to the remaining hidden all-reduce arrival skew.

### Step 3: separate host launch skew from GPU arrival skew
- Joined f32 NCCL kernels to CUDA runtime launch records through `correlationId`.
- Corrected aligned hidden all-reduce groups show:

| Class | Kernel wall avg | Kernel start-skew avg | Runtime launch start-skew avg | Enqueue-to-kernel max avg |
| --- | ---: | ---: | ---: | ---: |
| attention hidden `1x4096` | `1.023ms` | `1.006ms` | `4.324ms` | `3.327ms` |
| ratio-4 indexer score | `0.409ms` | `0.388ms` | `4.437ms` | `4.056ms` |

- Interpretation:
  - Rank worker host launch timing is more scattered than the final GPU NCCL start timing.
  - Stream backlog hides part of the host scatter, but the remaining hidden all-reduce arrival skew is still about `1ms` per hidden collective.
  - This makes rank worker CPU scheduling a plausible contributor, so the next reversible attempt pins rank worker threads to distinct CPUs from the current Linux cpuset.
- Code attempt:
  - Added Linux-only rank worker CPU affinity at thread startup.
  - Affinity failure logs a warning and does not fail model startup.
  - This is not a `bs=1` or `seq_len=1` specialization.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 validation:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
  - Rank workers pinned successfully to CPUs `0..7`.
  - Fixed-token bench round 1: `steady_tpot_ms.avg = 72.879603`, p50 `72.159356`, p95 `76.381629`, p99 `77.569638`, hash `6346f03343d75a65`.
  - Fixed-token bench round 2: `steady_tpot_ms.avg = 73.601627`, p50 `72.909497`, p95 `76.869231`, p99 `78.121858`, hash `6346f03343d75a65`.
- Post-change profile:

| Class | Kernel wall avg before | Kernel wall avg after | Start-skew avg before | Start-skew avg after |
| --- | ---: | ---: | ---: | ---: |
| attention hidden `1x4096` | `1.023ms` | `0.861ms` | `1.006ms` | `0.843ms` |
| ratio-4 indexer score | `0.409ms` | `0.366ms` | `0.388ms` | `0.346ms` |

- Decision:
  - Keep rank-worker CPU affinity. It directly reduces corrected f32 all-reduce arrival skew and moves the fixed-token long bench below 80ms with stable hashes.
  - This does not reach the stricter sub-70 target yet; continue with the next largest remaining rank-arrival source.
- Rejected follow-up:
  - Tried leaving CPU0 to scheduler/main by pinning rank workers to CPUs `1..8`.
  - Fixed-token bench regressed to `steady_tpot_ms.avg = 76.729985`, p50 `75.934987`, p95 `80.571165`, hash `6346f03343d75a65`.
  - Keep the direct rank-to-first-cpuset-CPU mapping (`rank -> cpus[rank]`) on this machine.
  - Tried pinning the scheduler to CPU8 through the shared affinity helper.
  - This failed badly because Linux thread affinity is inherited: the scheduler narrowed its own mask before spawning rank workers, so every rank worker saw CPU8 as its only allowed CPU and pinned there. Fixed-token bench regressed to `steady_tpot_ms.avg = 196.885794`, hash `6346f03343d75a65`.
  - Do not pin the scheduler before rank workers are spawned unless the original process cpuset is preserved separately.

### Step 4: NUMA-aware worker placement trace
- 5090 topology says GPU0-3 are local to NUMA0 CPUs `0-31,64-95`; GPU4-7 are local to NUMA1 CPUs `32-63,96-127`.
- Tried NUMA-aware worker placement:
  - rank0-3 -> CPU0-3
  - rank4-7 -> CPU32-35
- Fixed-token bench regressed despite correct output hash:
  - Round 1: `steady_tpot_ms.avg = 99.245409`, hash `6346f03343d75a65`.
  - Round 2: `steady_tpot_ms.avg = 96.795315`, hash `6346f03343d75a65`.
  - Node1-first dispatch order did not help: `steady_tpot_ms.avg = 100.331638`.
  - Adding a per-token scheduler/worker barrier did not help: `steady_tpot_ms.avg = 100.416945`.
- Added temporary hard-coded `info!` nanosecond trace for the first decode steps. The trace split scheduler dispatch, worker receive, barrier release, early embedding, and per-layer host enqueue progress.
- Evidence from `/tmp/dsv4_decode_trace_layers_numa_barrier.log`:
  - Barrier release spread was only a few microseconds, so scheduler dispatch order was not the dominant cause.
  - With NUMA-aware worker placement, final layer host progress spread stayed high:

| start_pos | max after-layer spread | layer | final layer spread | final node1-node0 gap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `11427.7us` | 38 | `7917.8us` | `3724.3us` |
| 2 | `8923.7us` | 33 | `5199.1us` | `3021.7us` |
| 3 | `11221.2us` | 37 | `6878.4us` | `3891.7us` |
| 4 | `7920.5us` | 39 | `3700.0us` | `2253.6us` |

- Control run from `/tmp/dsv4_decode_trace_layers_compact_barrier.log` kept the same barrier and trace but used compact rank worker placement (`rank -> first cpuset CPUs`):

| start_pos | max after-layer spread | layer | final layer spread | final node1-node0 gap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `5444.3us` | 39 | `3042.6us` | `1473.4us` |
| 2 | `5876.4us` | 29 | `2242.4us` | `745.7us` |
| 3 | `4125.6us` | 27 | `2490.0us` | `287.8us` |
| 4 | `4492.6us` | 33 | `1904.6us` | `-226.9us` |

- Decision:
  - Reject NUMA-aware worker placement for the current direct runtime. The topology choice is reasonable in isolation, but this direct decode path is dominated by per-rank host enqueue progression. Splitting workers across sockets makes that progression less uniform and increases rank-arrival skew before collectives.
  - Keep compact same-socket worker placement for now. A future NUMA-aware design needs a different launch/scheduling shape, not just a different CPU list.
  - The next performance slice should reduce per-layer host launch progression or capture/replay more of the decode path, because the host trace shows skew accumulating across layers before final completion.

### Step 5: explain the NUMA regression
- Corrected GPU ordinal mapping from `nvidia-smi --query-gpu=index,pci.bus_id,uuid`:

| GPU ordinal | PCI bus | NUMA | local CPUs |
| ---: | --- | ---: | --- |
| 0 | `0000:16:00.0` | 0 | `0-31,64-95` |
| 1 | `0000:27:00.0` | 0 | `0-31,64-95` |
| 2 | `0000:38:00.0` | 0 | `0-31,64-95` |
| 3 | `0000:5a:00.0` | 0 | `0-31,64-95` |
| 4 | `0000:98:00.0` | 1 | `32-63,96-127` |
| 5 | `0000:a8:00.0` | 1 | `32-63,96-127` |
| 6 | `0000:c8:00.0` | 1 | `32-63,96-127` |
| 7 | `0000:d8:00.0` | 1 | `32-63,96-127` |

- The ordinal-to-NUMA split was not wrong. The sysfs scan also showed an extra NVIDIA PCI function at `0000:03:00.0`, but it is not one of the 8 RTX 5090 ordinals used by DeepSeek V4.
- Nsight CUDA API + OSRT traces:
  - Compact same-socket trace: `/tmp/dsv4_compact_api.sqlite`, short-profile steady TPOT `117.094128ms`.
  - Split NUMA trace: `/tmp/dsv4_numa_api.sqlite`, short-profile steady TPOT `151.669444ms`.
- CUDA runtime API comparison showed small but systematic host API slowdowns under split NUMA placement:

| Placement | Rank group | `cudaLaunchKernel` p50 | `cudaEventRecord` p50 | `cudaStreamWaitEvent` p50 |
| --- | --- | ---: | ---: | ---: |
| compact CPU0-7 | rank0-3 | `~3.33-3.36us` | `~2.47-2.53us` | `~0.70-0.72us` |
| compact CPU0-7 | rank4-7 | `~3.60-3.63us` | `~2.70-2.73us` | `~0.71-0.72us` |
| split NUMA | rank0-3 | `~3.46-3.50us` | `~2.82-2.93us` | `~0.83-0.86us` |
| split NUMA | rank4-7 | `~3.97-4.04us` | `~3.49-3.54us` | `~0.86-0.88us` |

- OSRT showed the stronger signal: `pthread_rwlock_wrlock` inside the profiled CUDA/runtime path became slower for every rank when workers were split across sockets.

| Placement | `pthread_rwlock_wrlock` p50 | p95 | per-rank total |
| --- | ---: | ---: | ---: |
| compact CPU0-7 | `~47.9-48.6us` | `~129.8-133.2us` | `~1141-1156ms` |
| split NUMA | `~60.9-62.7us` | `~164.6-171.2us` | `~1539-1590ms` |

- Tested the lock/cacheline-bounce hypothesis by placing all 8 rank workers on node1 CPUs `32-39`, while restoring normal context/weight initialization:
  - Fixed-token bench: `steady_tpot_ms.avg = 76.479541`, p50 `75.788856`, p95 `80.033321`, hash `6346f03343d75a65`.
  - This matches the compact node0 class and is much faster than split NUMA `96-100ms`.
- Tried moving context creation and weight load into the pinned NUMA rank thread:
  - Fixed-token bench regressed further to `steady_tpot_ms.avg = 105.223935`, hash `6346f03343d75a65`.
  - This rejects the first-touch/context-init hypothesis.
- Interpretation:
  - The regression is not caused by wrong GPU ordinal mapping or PCIe locality.
  - The current decode path launches many tiny kernels and CUDA event operations from 8 host threads. CUDA runtime/driver host-side synchronization (`pthread_rwlock_wrlock`, futex, and related shared state) is hot enough that splitting rank workers across sockets hurts more than making GPU4-7 launches originate from their PCIe-local CPUs helps.
  - Compact same-socket worker placement is intentionally kept because it minimizes cross-socket lock/cacheline bouncing in the current host-launch-heavy runtime.

### Step 6: remove hot temporary zero-fill
- Motivation:
  - The NUMA experiment showed the current decode path is host-launch-heavy: many small kernels, CUDA events, and collectives are issued by 8 rank worker threads per token.
  - Several hot intermediate buffers were allocated with `alloc_zeros` even though they are immediately and fully written by a CUDA kernel or NCCL collective before being read.
  - On this path, zero-fill is not free: it can introduce extra CUDA memset work and extra host API traffic before useful compute.
- Code change:
  - Added `HcHiddenStates::uninit`.
  - Converted fully overwritten hot outputs from zeroed allocation to uninitialized allocation in:
    - HC projection/head/pre/post helpers.
    - embedding, RMSNorm, linears, SwiGLU, residual add, logits.
    - BF16 all-gather, u32 all-gather, f32 reduce-scatter, f32 all-reduce scratch, HC post all-reduce output.
    - MoE route outputs, expanded input, reduced output, and grouped FP4 local expert output.
    - deterministic attention top-k index output buffers.
  - Kept zeroed buffers where zero initialization is semantically required, especially MoE mapping/cursor/count buffers used as counters or scatter state.
  - This is not a `bs=1` or `seq_len=1` specialization; it removes unnecessary initialization for general fully-overwritten temporaries.
- Local validation:
  - `cargo fmt --check` passed.
  - `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- 5090 validation:
  - Exact E2E passed: `All 20 DeepSeek V4 exact cases passed`.
  - Fixed-token bench command:
    - `cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`
  - Round 1: `steady_tpot_ms.avg = 63.346991`, p50 `62.503928`, p95 `67.055793`, p99 `68.302441`, hash `6346f03343d75a65`.
  - Round 2: `steady_tpot_ms.avg = 64.507894`, p50 `63.693685`, p95 `68.098382`, p99 `69.279796`, hash `6346f03343d75a65`.
- Post-change profile:
  - First attempt with `--delay=35 --duration=12` missed the decode window because this faster run completed before capture started; the resulting sqlite only had metadata. This is a profiling setup error, not a runtime result.
  - Reran with `--delay=30 --duration=8 --trace=cuda,nvtx,osrt,cublas --export=sqlite`.
  - Profile artifact: `/tmp/dsv4_uninit_profile.sqlite`.
  - Under nsys, short `1x32` TPOT was distorted upward to `107.467198ms/token`; use the trace for composition, not absolute throughput.
  - Corrected f32 all-reduce grouping over the captured stable region:

| f32 all-reduce group | Groups | Logical wall sum | Start-skew sum | Tail sum | Avg wall | p50 wall | p95 wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all aligned `AllReduce_Sum_f32` | 1397 | `1201.89ms` | `1175.86ms` | `26.03ms` | `0.860ms` | `0.919ms` | `1.631ms` |

  - This is essentially unchanged from the post-affinity profile (`attention hidden 1x4096 avg wall 0.861ms`, avg skew 0.843ms).
  - Therefore this optimization did not make NCCL transfer faster and did not materially reduce f32 collective arrival skew.
  - The likely win is removal of unnecessary zero-fill/API work before useful kernels. The post-change trace still has semantic zeroing for stateful buffers, but the remaining `cuMemsetD8Async` count is far lower relative to the old hot path shape. Treat this as a host/API cleanup win, not a communication win.
- 5090 topology note:
  - `nvidia-smi topo -m` reports GPU0-3 on NUMA0 and GPU4-7 on NUMA1; GPU0-3 to GPU4-7 links are `SYS`, while intra-group links are `NODE`.
  - That topology is real, but current results say the direct runtime is more sensitive to cross-socket CUDA runtime/driver host synchronization than to GPU-local CPU placement.
- Decision:
  - Keep the uninitialized hot-output allocation change.
  - The next larger slice should still target rank-arrival skew or launch reduction; uninit got us below 70ms by removing overhead around the hot path, not by fixing the remaining f32 all-reduce skew.

## Debrief

- **Outcome**:
  - DeepSeek V4 decode reached stable sub-70 on the fixed long bench without adding `bs=1` or `seq_len=1` specialization.
  - Fixed bench evidence on 5090:
    - Round 1: `--prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`, `steady_tpot_ms.avg = 63.346991`, hash `6346f03343d75a65`.
    - Round 2: `--prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42`, `steady_tpot_ms.avg = 64.507894`, hash `6346f03343d75a65`.
  - Exact E2E evidence on 5090: `All 20 DeepSeek V4 exact cases passed`.
  - Local compile gate: `cargo fmt --check` and `cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4` passed.
- **Completion audit**:
  - `stable sub-70` - covered by two fixed long bench rounds at `63.35ms` and `64.51ms`.
  - `no bs=1 / seq_len=1 specialization` - covered by the retained code changes: worker affinity and uninitialized fully-overwritten temporaries are general runtime behavior, not shape-specific branches.
  - `split attention local to 1x4096 f32 all-reduce arrival skew` - covered by corrected per-rank CUPTI grouping: attention hidden `1x4096` was separated from ratio-4 indexer score all-reduce.
  - `reduce a cause of skew` - covered by rank-worker CPU affinity: attention hidden `AllReduce_Sum_f32` wall/skew fell from `1.023/1.006ms` to `0.861/0.843ms`.
  - `avoid misreading NCCL wait as transfer` - covered by reporting logical wall, start-skew, and post-arrival tail separately, and by noting post-uninit throughput improved without f32 all-reduce skew improving.
  - `document every attempt` - covered by Steps 1-6, including the rejected top-k cache, rejected NUMA-aware placement, rejected scheduler pinning, and the kept affinity/uninit changes.
- **Pitfalls encountered**:
  - Naive grouping of consecutive NCCL kernels by 8 ranks is wrong when start skew is large; grouping must align per rank worker through `correlationId`/`globalTid`.
  - 5090's topology makes GPU0-3 local to NUMA0 and GPU4-7 local to NUMA1, but splitting rank workers across sockets regressed TPOT to `96-100ms` because CUDA runtime/driver host synchronization became more expensive.
  - Pinning the scheduler before spawning workers narrowed the inherited affinity mask and accidentally placed all workers on the scheduler CPU.
  - An nsys attempt with `--delay=35` missed the faster decode window after uninit; profiling delay must be revisited after major speedups.
- **Lessons learned**:
  - In this host-launch-heavy direct runtime, compact same-socket rank workers are better than PCIe-local split-NUMA workers.
  - Zeroed temporary allocations are a major cost when every intermediate is immediately overwritten; use uninitialized allocation only for buffers with a clear full-writer before any read.
  - A TPOT drop should not be attributed to communication unless the corrected collective grouping shows wall/skew/tail movement.
- **Follow-ups**:
  - Reprofile the post-uninit path with a longer/decode-centered capture and choose the next launch-reduction or CUDA-graph slice from the remaining f32 all-reduce skew.
  - Revisit NUMA-aware placement only after reducing host launch/API synchronization enough that PCIe locality can matter again.
