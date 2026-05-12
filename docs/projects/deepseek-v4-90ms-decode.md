# DeepSeek V4 90ms Decode

**Created**: 2026-05-12
**Status**: complete

## TL;DR

Goal achieved. DeepSeek V4 decode now caches grouped MoE FP4 expert weight/scale pointer arrays per rank worker instead of rebuilding and H2D-copying them inside every W1/W3/W2 grouped linear call. Exact E2E remains `20/20`. The fixed-token `1x160 warmup=2 iters=3 seed=42` long bench improved from `107.92ms` / `112.61ms` to `83.37ms` / `89.65ms`, with generated-token hash `6346f03343d75a65` throughout. No new `bs=1` or `seq_len=1` specialization was added.

## Preparation

- **Read**:
  - `docs/index.md` - identified the active DeepSeek MoE AG/RS work, the completed non-MoE decode kernel audit, and the earlier TileLang MoE review.
  - `docs/projects/deepseek-moe-ag-rs.md` - confirmed current decode MoE uses GPU AG/RS, expert-major compaction, grouped TileLang FP4 expert GEMM, and still over-launches grouped expert CTAs by capacity.
  - `docs/projects/deepseek-v4-decode-kernel-audit.md` - confirmed small non-MoE launch fusions were mostly rejected or noisy; stable long-bench baseline remains around `106ms/token`.
  - `docs/projects/deepseek-moe-tilelang-review.md` - confirmed earlier direct top-k MoE reached `80.49ms/token`, but the current AG/RS path likely pays repeated rank skew and routed expert scheduling overhead.
  - `pegainfer-deepseek-v4/src/runtime/moe.rs` - current MoE path builds `expert_indptr`, expands to expert-major rows, then runs three grouped FP4 linears over `rows = routed.seq_len * topk`.
  - `pegainfer-kernels/csrc/deepseek_v4/deepseek_moe.cu` - route mapping already computes per-local-expert prefix sums and local counts on GPU.
  - `pegainfer-kernels/csrc/deepseek_v4/deepseek_quant.cu` - grouped FP4 wrapper currently quantizes all expanded rows and launches grouped GEMM over `(out tiles) x ceil(rows/32) x local_experts`.
  - `pegainfer-kernels/tools/tilelang/deepseek_v4/generate.py` - grouped FP4 kernels currently skip empty expert tiles inside the launched CTA after reading `expert_indptr`.
- **Relevant history**:
  - `docs/projects/deepseek-v4-decode-kernel-audit.md` records that final-head, KV prep, indexer, FP8 two-linear, and h8 sparse-attention local fusions did not produce stable TPOT wins.
  - `docs/projects/deepseek-moe-ag-rs.md` records the next MoE follow-up as a GPU active-expert/tile list to stop launching empty expert tiles.
  - Existing code has historical `seq_len == 1` HC fused paths, but this project must not add new `bs=1` or `seq_len=1` specialization.
- **Plan**:
  1. Establish a fresh local/remote baseline with fixed seed and token hash: exact E2E, two `1x160 warmup=2 iters=3 seed=42` benches, and a short nsys kernel summary.
  2. Add generalized MoE active tile metadata derived from `expert_indptr`: count non-empty 32-row expert tiles and build a GPU `tile_to_expert`/`tile_to_m_block` list without D2H.
  3. Change grouped FP4 GEMM launch to use active tiles instead of `ceil(rows/32) * local_experts`, preserving the same math and avoiding new batch-size-specific branches.
  4. Verify with local release checks, 5090 exact E2E, two long benches with matching generated-token hashes, and profile evidence that grouped FP4 total/average kernel time drops. Nsight Systems `Instances` counts kernel launches, not CTAs, so CTA over-launch reductions will not necessarily change that column.
  5. Record any failed experiments and revert them; only keep changes that survive correctness and reproducible long-bench gates.
- **Risks / open questions**:
  - TileLang generated source is transformed by string rewriting, so changing grouped launch indexing must be done carefully and regenerated through the existing generator.
  - Active tile-list construction adds kernels; it only wins if saved grouped GEMM CTAs exceed list-building overhead on realistic batch decode.
  - NCCL kernel time remains synchronization-window evidence, not pure communication time.
  - Historical `seq_len == 1` decode fuses are outside this first MoE slice, but no new optimization in this project should depend on that shape.

## Execution Log

### Step 1: Fresh baseline
- Synced the current local branch contents to `5090:/root/develop/xingming/pegainfer` using path-preserving `rsync -avR`.
- Correctness baseline:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
```

- Result: `All 20 DeepSeek V4 exact cases passed`.
- Long bench command:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42
```

- Round 1: `steady_tpot_ms.avg = 107.919291`, `p50 = 111.874663`, `p95 = 119.201647`, token hash `6346f03343d75a65` for all measured iterations.
- Round 2: `steady_tpot_ms.avg = 112.614529`, `p50 = 113.245957`, `p95 = 119.728698`, token hash `6346f03343d75a65` for all measured iterations.
- Baseline interpretation: use `~108-113ms/token` as the current stable band. The `<=90ms/token` goal needs about `18-23ms/token` improvement on this measured shape.
- Short nsys command:

```bash
nsys profile --force-overwrite=true --stats=false --sample=none --trace=cuda,nvtx,cublas --delay=35 --duration=12 -o /tmp/dsv4_90ms_baseline_profile target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1 --seed 42
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/dsv4_90ms_baseline_profile.nsys-rep
```

- Result under nsys: `steady_tpot_ms.avg = 135.058327`, token hash `5f6c64b667f2abf5`.
- Top grouped FP4 rows:
  - `deepseek_tilelang_fp4_grouped_gemm_n2048_k4096_kernel`: `492.30ms`, `11668` instances, `42.19us` avg.
  - `deepseek_tilelang_fp4_grouped_gemm_n4096_k2048_kernel`: `140.77ms`, `5835` instances, `24.13us` avg.
- Interpretation: these instance counts are kernel launches, not CTAs. A GPU active-tile implementation should be judged by reduced grouped FP4 total/avg time, lower CTA work in a lower-level profile, and ultimately long-bench TPOT, not by `Instances` alone.
- NCCL rows remain synchronization-window evidence, not pure communication time:
  - `AllReduce_Sum_f32`: `6158.15ms`, `8683` instances.
  - `ReduceScatter_Sum_f32`: `4456.40ms`, `5834` instances.
  - `AllGather`: `1062.49ms`, `6378` instances.
- Shutdown note: the known rank-7 NCCL abort panic appeared after metrics were emitted and the process exited successfully.

### Step 2: Cache grouped MoE FP4 expert pointers
- Added a per-rank `MoeGroupedPtrCache`:
  - `MoeGroupedLinearPtrs` stores GPU arrays of expert weight pointers and scale pointers for one grouped linear.
  - `MoeLayerGroupedPtrs` stores W1/W2/W3 pointer arrays per layer.
  - `MoeGroupedPtrCache` stores all layers for a rank worker.
- Built the cache once in each persistent rank worker after binding the CUDA context.
- Updated prefill and decode MoE paths to pass the cache into grouped FP4 local expert execution.
- Removed hot-path rebuilding of:
  - 32 expert weight pointer host vectors,
  - 32 expert scale pointer host vectors,
  - two `clone_htod` copies per grouped linear call,
  - repeated per-expert shape/dtype validation.
- This is batch-shape neutral: the grouped GEMM still consumes runtime `expert_indptr` and `rows`; no new `bs=1` or `seq_len=1` branch was added.
- Local verification:

```bash
cargo fmt --check
git diff --check
cargo check --release -p pegainfer-deepseek-v4 --features deepseek-v4
```

- Result: all passed.
- 5090 exact verification:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-deepseek-v4 --features deepseek-v4 --bin deepseek_v4_e2e -- --model-path /data/DeepSeek-V4-Flash
```

- Result: `All 20 DeepSeek V4 exact cases passed`.
- Long bench command:

```bash
PEGAINFER_NVCC_JOBS=8 cargo run --release -p pegainfer-server --bin bench_serving --features deepseek-v4 -- --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 160 --warmup 2 --iters 3 --seed 42
```

- Round 1: `steady_tpot_ms.avg = 83.366315`, `p50 = 80.888628`, `p95 = 97.262140`, token hash `6346f03343d75a65` for all measured iterations.
- Round 2: `steady_tpot_ms.avg = 89.645687`, `p50 = 89.705486`, `p95 = 98.415165`, token hash `6346f03343d75a65` for all measured iterations.
- Performance result: both required long-bench rounds are below `90ms/token`, with identical generated-token hash to the baseline.
- Post-change nsys command:

```bash
nsys profile --force-overwrite=true --stats=false --sample=none --trace=cuda,nvtx,cublas --delay=35 --duration=12 -o /tmp/dsv4_90ms_ptr_cache_profile target/release/bench_serving --model-path /data/DeepSeek-V4-Flash --format json request --prompt-len 1 --output-len 32 --warmup 1 --iters 1 --seed 42
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/dsv4_90ms_ptr_cache_profile.nsys-rep
```

- Result under nsys: `steady_tpot_ms.avg = 119.202172`, token hash `5f6c64b667f2abf5`.
- Key post-change rows:
  - `AllReduce_Sum_f32`: `5803.26ms`, `9048` instances, avg `641.39us`.
  - `ReduceScatter_Sum_f32`: `2040.33ms`, `6075` instances, avg `335.86us`.
  - `AllGather`: `1013.17ms`, `6656` instances, avg `152.22us`.
  - `deepseek_tilelang_fp4_grouped_gemm_n2048_k4096_kernel`: `528.25ms`, `12491` instances, avg `42.29us`.
  - `deepseek_tilelang_fp4_grouped_gemm_n4096_k2048_kernel`: `151.18ms`, `6245` instances, avg `24.21us`.
- Interpretation: the grouped FP4 kernels themselves did not become faster; the main gain shows up as a much shorter reduce-scatter synchronization window. Caching expert pointer arrays removes repeated host/device pointer-array work from each rank's MoE hot path, which appears to reduce rank arrival skew before MoE reduce-scatter.
- Active-tile note: a true GPU active tile list still has value, but reducing the launch grid without D2H requires either a persistent/grouped kernel shape or a different launch strategy. Pulling a device active count back to the host per layer would reintroduce hot-path synchronization and was not used.

## Debrief

- **Outcome**: The `<=90ms/token` long-bench target is met on two consecutive `1x160 warmup=2 iters=3 seed=42` runs with identical generated-token hashes. Exact E2E remains `20/20`.
- **Pitfalls encountered**:
  - Nsight Systems `Instances` for grouped FP4 rows counts kernel launches, not CTAs. CTA over-launch work must be judged through kernel total/avg time, lower-level CTA metrics, or TPOT, not that column alone.
  - A host-sized GPU active tile launch would need a D2H active count unless the grouped GEMM is redesigned around a persistent/device-scheduled loop. That would violate the no-hot-path-sync direction, so this pass avoided it.
- **Lessons learned**:
  - The rank-skew bucket was still vulnerable to CPU/host-side MoE bookkeeping. Removing repeated expert pointer vector construction and H2D copies can move TPOT even when the visible grouped GEMM kernel rows do not improve.
  - Token hash equality remains essential: all baseline and post-change long benches used `6346f03343d75a65`, so the TPOT delta is not from different hash routing or EPLB load.
- **Follow-ups**:
  - Replace the shutdown NCCL abort panic with orderly communicator teardown.
  - Revisit active expert/tile scheduling only with a design that does not require per-layer D2H counts.
  - Profile rank arrival timing directly around MoE AG/RS to confirm the pointer-cache mechanism with NVTX ranges rather than inferring only from NCCL window shrinkage.
