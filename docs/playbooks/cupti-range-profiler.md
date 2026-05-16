# CUPTI Range Profiler Notes

**Status**: Active
**TL;DR**: CUPTI Range Profiler works for pegainfer kernel reports, but the NVPerf stack on RTX 5090/CUDA 12.9 is sensitive to user range names. Keep range names short and treat them as profiler IDs only; store full model/op/shape metadata in the JSON report. Run one unprofiled pre-measure launch before `cuptiRangeProfilerStart` so the first measured range does not include CUDA lazy initialization. Qwen3 attention reports store raw CUPTI metric names and values, including tensor-pipe/BF16-HMMA peak percentages for attention-core questions.

## Current Use

- Rust wrapper: `pegainfer-cupti`
- C++ Range Profiler bridge: `pegainfer-cupti/csrc/range_profiler.cpp`
- Qwen3 paged decode report tool: `pegainfer-qwen3-4b/src/bin/qwen3_kernel_report.rs`

The report runner should enable CUPTI by default. Use `--no-cupti` only for latency-only local validation or when the host profiler stack is unavailable.

## RTX 5090 Finding

Environment:

- RTX 5090
- driver `575.57.08`
- CUDA toolkit `12.9`
- `PEGAINFER_CUDA_SM=120`

The standalone snapshot runner initially crashed inside NVPerf when decoding counters:

```text
NVPW_CUDA_Profiler_DecodeCounters
libnvperf_host.so
libcupti.so.12
pegainfer_cupti_profile_range
```

The crash was not caused by the attention case, the Rust callback trampoline, context overlap, or correctness pre-runs. The same wrapper and same kernel passed after shortening the user range name.

Observed bad range name:

```text
qwen3_kernel_report/paged_decode_attention/non_partition/bs1/kv128
```

Observed stable range name pattern:

```text
qk/non_partition/b1/k128
qpf/qk/b1/s128
qpf/kv/b1/s128
qpf/attn/b1/s128
```

Operational rule until we test more CUDA/NVPerf versions: keep CUPTI range names under roughly 48 ASCII bytes, avoid verbose path-like names, and put full semantics in structured output fields instead of the range name.

## Measurement Rules

- Run one unprofiled `launch_once(path)` plus stream sync before the profiled range. Without this, the first profiled case can include lazy CUDA/module initialization. On RTX 5090, `bs=1,ctx=1024,non_partition` reported about `3528us` before this pre-measure launch and `87us` after it.
- Clear L2 in the prepare callback before `cuptiRangeProfilerStart`; synchronize after prepare and after the profiled launch.
- Do not use `cudaCtxResetPersistingL2Cache` / `cuCtxResetPersistingL2Cache` as a general cache-clear primitive. Those APIs reset persisting L2 lines to normal status; they do not evict ordinary L2 contents. Use the benchmark sweep buffer for cache-cleared timing.
- Profile one user range per call. Keep range names deterministic but compact.
- Do not write derived fields such as bandwidth percentages, utilization labels, or read-amplification ratios into the report JSON. Keep `case.cupti` as a raw metric-name map.

## Metric Set

Use a small set that answers the first diagnostic question for decode/prefill attention: "is the kernel moving the required KV bytes, is the GPU being fed enough work, and is the attention core using the BF16 tensor path?"

Default metrics:

```text
gpu__time_duration.sum
sm__cycles_elapsed.avg.per_second
dram__bytes.sum
dram__bytes_op_read.sum
dram__bytes_op_write.sum
lts__t_bytes.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__warps_active.avg.pct_of_peak_sustained_active
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed
sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.per_second
```

Why these SM/tensor counters:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed` is the coarse "are SMs busy over wall time?" signal. It catches low-batch underfill directly.
- `smsp__warps_active.avg.pct_of_peak_sustained_active` says whether active SM partitions have enough resident warps while they are active. It helps separate global grid underfill from per-partition occupancy.
- `sm__cycles_elapsed.avg.per_second` records the profiler-observed SM clock during the range, avoiding a stale external clock assumption.
- `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed` is the direct CUPTI/NVPerf answer for BF16 HMMA math ops versus peak sustained elapsed time. Prefer this over runner-owned MFU calculations when the question is kernel-level tensor utilization.
- `sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.per_second` gives the same BF16 HMMA path as a raw operation-rate metric.

Do not grow this into a full NCU replacement. If the question is stall reason, issue mix, tensor-core use, or scheduler detail, take an NCU profile.

On the CUDA validation host, `ncu` is available at `/usr/local/cuda/bin/ncu`; non-interactive shells may not have that directory in `PATH`.

## Verified Minimal Run

Verified on the CUDA validation host after switching to short range names and adding the unprofiled pre-measure launch:

```bash
cd <validation-checkout>
PEGAINFER_CUDA_SM=120 cargo run --release -p pegainfer-qwen3-4b --features kernel-report --bin qwen3_kernel_report -- run --contexts 1024 --batch-sizes 1 --variants non_partition,split_kv_256x64 --iters 4 --out /tmp/qwen3_kernel_report_cupti_min.json
```

Result:

```text
running 2 qwen3 paged decode attention kernel cases
case variant=non_partition bs=1 ctx=1024
case variant=split_kv_256x64 bs=1 ctx=1024
wrote /tmp/qwen3_kernel_report_cupti_min.json
```

Representative verified snapshot shape:

```text
cupti_enabled True
schema 4
case.cupti keys:
  gpu__time_duration.sum
  sm__cycles_elapsed.avg.per_second
  dram__bytes.sum
  dram__bytes_op_read.sum
  dram__bytes_op_write.sum
  lts__t_bytes.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  smsp__warps_active.avg.pct_of_peak_sustained_active
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed
  sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.per_second
```

Derived analysis of these values should happen in a separate report or notebook, not in `qwen3_kernel_report`.

## Next Step

Keep CUPTI in the kernel report path and add new metrics only when they answer a concrete kernel-maintenance question.
