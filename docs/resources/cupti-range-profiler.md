# CUPTI Range Profiler Notes

**Status**: Active
**TL;DR**: CUPTI Range Profiler works for pegainfer kernel snapshots, but the NVPerf stack on the 5090/CUDA 12.9 box is sensitive to user range names. Keep range names short and treat them as profiler IDs only; store full model/op/shape metadata in the JSON snapshot. Run one unprofiled warmup launch before `cuptiRangeProfilerStart` so the first measured range does not include CUDA lazy initialization. The default Qwen3 attention snapshot records time, DRAM/L2 traffic, and two minimal SM counters; it is still a decode-attention kernel snapshot, not a full decode-step profile.

## Current Use

- Rust wrapper: `crates/pegainfer-cupti`
- C++ Range Profiler bridge: `crates/pegainfer-cupti/csrc/range_profiler.cpp`
- Qwen3 paged decode snapshot: `crates/pegainfer-qwen3-4b/benches/qwen3_kernel_snapshot.rs`

The snapshot runner should enable CUPTI by default. Use `--no-cupti` only for latency-only smoke runs or when the host profiler stack is unavailable.

## 5090 Finding

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
qwen3_kernel_snapshot/paged_decode_attention/non_partition/bs1/kv128
```

Observed stable range name pattern:

```text
qk/non_partition/b1/k128
```

Operational rule until we test more CUDA/NVPerf versions: keep CUPTI range names under roughly 48 ASCII bytes, avoid verbose path-like names, and put full semantics in structured output fields instead of the range name.

## Measurement Rules

- Run one unprofiled `launch_once(path)` plus stream sync before the profiled range. Without this, the first profiled case can include lazy CUDA/module initialization. On 5090, `bs=1,ctx=1024,non_partition` reported about `3528us` before this warmup and `87us` after it.
- Clear L2 in the prepare callback before `cuptiRangeProfilerStart`; synchronize after prepare and after the profiled launch.
- Profile one user range per call. Keep range names deterministic but compact.
- Interpret `kv_read_over_dram_read_pct` as a sanity check for read amplification, not as a correctness oracle.

## Metric Set

Use a small set that answers the first diagnostic question for decode attention: "is the kernel moving the required KV bytes, and is the GPU being fed enough work?"

Default metrics:

```text
gpu__time_duration.sum
dram__bytes.sum
dram__bytes_op_read.sum
dram__bytes_op_write.sum
lts__t_bytes.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
smsp__warps_active.avg.pct_of_peak_sustained_active
```

Why only these two SM counters:

- `sm__throughput.avg.pct_of_peak_sustained_elapsed` is the coarse "are SMs busy over wall time?" signal. It catches low-batch underfill directly.
- `smsp__warps_active.avg.pct_of_peak_sustained_active` says whether active SM partitions have enough resident warps while they are active. It helps separate global grid underfill from per-partition occupancy.

Do not grow this into a full NCU replacement. If the question is stall reason, issue mix, tensor-core use, or scheduler detail, take an NCU profile.

## Verified Smoke

Verified on 5090 after switching to short range names and adding the unprofiled warmup launch:

```bash
ssh 5090 'bash -ic "cd /root/develop/xingming/pegainfer && PEGAINFER_CUDA_SM=120 cargo bench -p pegainfer-qwen3-4b --bench qwen3_kernel_snapshot -- run --contexts 1024 --batch-sizes 1 --variants non_partition,split_kv_256x64 --iters 4 --out /tmp/qwen3_kernel_snapshot_cupti_smoke.json"'
```

Result:

```text
running 2 qwen3 paged decode attention kernel cases
case variant=non_partition bs=1 ctx=1024
case variant=split_kv_256x64 bs=1 ctx=1024
wrote /tmp/qwen3_kernel_snapshot_cupti_smoke.json
```

Representative verified snapshot summary:

```text
cupti_enabled True
non_partition 1 1024 True 75.776us 4.236MB_dram_read 0.75%_sm 8.27%_warps_active
split_kv_256x64 1 1024 True 48.736us 4.249MB_dram_read 1.31%_sm 11.77%_warps_active
```

The same metric set showed the intended signal at `bs=1,ctx=10000`: non-partition used only `1.19%` SM throughput and `6.59%` DRAM peak, while split-K used `8.74%` SM throughput and `41.06%` DRAM peak for essentially the same KV read bytes. That confirms the long-context low-batch issue is underfill, not KV read amplification.

## Next Step

Keep CUPTI in the kernel snapshot path and add new metrics only when they answer a concrete kernel-maintenance question.
