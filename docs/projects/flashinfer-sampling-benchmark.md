# FlashInfer Sampling Benchmark

**Created**: 2026-04-03
**Status**: blocked

> **TL;DR:** Runtime token selection now stays on FlashInfer, greedy uses FlashInfer top-k selection (`k=1`) instead of abusing `top_k=1` sampling, and `bench_serving` is back to greedy-only. The remaining blocker is batched decode: `mini-sglang` and `vLLM` both keep batched greedy and batched per-row sampling metadata, while pegainfer currently falls back to per-request sampling inside batch decode.

## Preparation

- **Read**:
  - `docs/index.md` — identified existing benchmark workflow, FlashInfer reference docs, and optimization documentation to reuse.
  - `docs/areas/bench-regression.md` — confirmed current snapshot/compare workflow and that standard regression tracking centers on TTFT and TPOT snapshots.
  - `docs/resources/bench-vs-vllm.md` — captured the repo's benchmarking discipline: same-GPU comparisons, synthetic prompts, and decode-focused configurations.
  - `docs/resources/model-optimization-pipeline.md` — confirmed the standard optimization workflow and the need to isolate decode-heavy measurement when evaluating a kernel/backend swap.
  - `docs/resources/kernel-technology-reference.md` — confirmed FlashInfer is currently treated as a reference library rather than a runtime dependency, so integration cost must be justified.
  - `docs/resources/flashinfer-reference.md` — mapped FlashInfer's sampling/operator surface and local source layout.
- **Relevant history**:
  - `docs/projects/qwen35-4b-optimization.md` — existing optimization work tracks before/after kernel changes with decode-heavy measurements; this is the right precedent for a sampling backend change.
  - no directly matching past project doc for FlashInfer sampling migration found.
- **Plan**:
  1. Add a sampling-focused benchmark surface that measures current non-greedy sampling cost directly, using the existing `criterion` bench infrastructure and realistic vocab sizes.
  2. Extend `bench_serving` so end-to-end runs can exercise non-greedy sampling instead of only `SamplingParams::default()`, then record a baseline snapshot/report for the current backend.
  3. Implement a minimal FlashInfer-backed non-greedy sampling path behind the existing Rust op boundary, preferring a local CUDA/C++ shim over introducing Python or TVM runtime dependencies.
  4. Re-run the same microbench and end-to-end benchmark shapes, document the delta, and record integration lessons and remaining risks.
- **Risks / open questions**:
  - Current benchmark snapshots are greedy-only, so we need to define a non-greedy profile carefully without confusing regression tracking for existing work.
  - FlashInfer's reusable layer is C++/CUDA header code, but not a drop-in Rust ABI; we may need a thin local shim and should validate dtype/RNG semantics before switching production paths.
  - Even if the kernel is faster, per-token sync/D2H in the current decode loop may cap end-to-end gains for batch size 1.

## Execution Log

### Step 1: Add sampling benchmark surfaces
- Started by auditing the current benchmark code paths in `benches/ops/ops_embedding_sampling_bench.rs` and `src/bin/bench_serving.rs`.
- Confirmed the current end-to-end benchmark always uses `SamplingParams::default()` (`temperature=0`, greedy), which would hide any non-greedy sampling backend change.
- Confirmed the existing ops bench only measures a small-vocab `gpu_sample_into` case and needs realistic vocab sizes plus explicit non-greedy shapes.
- Result: success. Proceeding with code changes to expose configurable sampling and realistic sampling microbench shapes.

### Step 2: Implement benchmark plumbing and record baseline
- Updated `src/bin/bench_serving.rs` to accept `--temperature`, `--top-k`, `--top-p`, and `--ignore-eos` on `request`, `matrix`, `curve`, and `snapshot`, and to persist the chosen sampling config in rendered/JSON reports.
- Updated `benches/ops/ops_embedding_sampling_bench.rs` so `gpu_sample_into` microbench now covers realistic vocab sizes (`128256`, `248320`) and two non-greedy profiles (`top_k_top_p`, `top_p_only`).
- Verified compileability with `cargo check --release --benches --bin bench_serving`.
- Verified CLI surface with `cargo run --release --bin bench_serving -- request --help`.
- Verified runtime path with `cargo run --release --bin bench_serving -- request --prompt-len 1 --output-len 2 --warmup 0 --iters 1 --temperature 0.8 --top-k 50 --top-p 0.95`.
- Recorded current backend baseline with `cargo run --release --bin bench_serving -- snapshot --warmup 1 --iters 3 --temperature 0.8 --top-k 50 --top-p 0.95`.
- Baseline snapshot result (`bench_snapshots/rtx-5070-ti/qwen3-4b.json`): prefill-heavy TTFT p50 1187.10 ms / p99 1187.30 ms; decode-heavy TPOT p50 13.16 ms / p99 13.26 ms.
- Recorded microbench baseline with `cargo bench --bench ops_bench -- gpu_sample_into`.
- Microbench result summary: `top_k_top_p/128256` 753 us, `top_p_only/128256` 486 us, `top_k_top_p/248320` 1.41 ms, `top_p_only/248320` 906 us.
- Result: success. Benchmark and baseline prerequisites are now satisfied for the backend swap.

### Step 3: Wire FlashInfer-backed filtered sampling and compare microbench
- Added `csrc/flashinfer_sampling.cu` and a new Rust FFI entrypoint so filtered non-greedy sampling can route through FlashInfer's sampling templates.
- Kept pure temperature-only multinomial on the legacy kernel because FlashInfer's Philox-driven RNG semantics changed the existing `random_val=0.0` unit-test behavior; the current migration scope is now top-k/top-p filtered sampling only.
- Added one-byte validity scratch buffers where decode/prefill sampling reuses GPU scratch (`Qwen3State`, schedulers, and batch decode buffers).
- Verified compileability with `cargo check --release --benches --bin bench_serving`.
- Verified runtime with `cargo run --release --bin bench_serving -- request --prompt-len 1 --output-len 2 --warmup 0 --iters 1 --temperature 0.8 --top-k 50 --top-p 0.95`.
- Verified compatibility with `cargo test --release test_gpu_sample -- --nocapture` after narrowing the migrated path.
- Compared microbench using `cargo bench --bench ops_bench -- gpu_sample_into`.
- Post-change result summary: `top_k_top_p/128256` 907 us (+20.4%), `top_p_only/128256` 144 us (-70.3%), `top_k_top_p/248320` 1.24 ms (-12.0%), `top_p_only/248320` 388 us (-57.2%).
- Recorded updated end-to-end snapshot with `cargo run --release --bin bench_serving -- snapshot --warmup 1 --iters 3 --temperature 0.8 --top-k 50 --top-p 0.95`: prefill-heavy TTFT p50 1186.34 ms, decode-heavy TPOT p50 12.47 ms.
- Result: partial success. FlashInfer clearly helps top-p-heavy sampling and larger-vocab filtered sampling, but the 128k top-k+top-p microbench regressed, so the integration likely needs another pass before concluding the backend swap is uniformly beneficial.

### Step 4: Compare greedy path against FlashInfer top-k=1
- Added greedy-focused microbench cases in `benches/ops/ops_embedding_sampling_bench.rs`: baseline `argmax` and `flashinfer_greedy_top_k_1`.
- Ran `cargo bench --bench ops_bench -- argmax` and `cargo bench --bench ops_bench -- flashinfer_greedy_top_k_1`.
- Result summary: for `128256`, `argmax` 155.24 us vs FlashInfer top-k=1 154.71 us; for `248320`, `argmax` 197.91 us vs FlashInfer top-k=1 196.22 us.
- Result: effectively no meaningful difference. Greedy is already cheap enough that the FlashInfer top-k=1 path is not a compelling optimization target relative to the existing argmax path.

### Step 5: Complete the runtime switch to FlashInfer
- Updated `src/ops/sampling.rs` so all single-row token selection now routes through `gpu_sample_flashinfer_cuda`; greedy is encoded as `top_k=1`.
- Updated `src/ops/tests.rs` to stop asserting the legacy kernel's exact `random_val` semantics for pure temperature sampling, since FlashInfer's RNG behavior is different.
- Removed the all-greedy batch fast path from `src/model/qwen3/batch_decode.rs` and `src/model/qwen35/batch_decode.rs` so batched decode also uses the FlashInfer-backed selection path instead of `argmax_batched`.
- Verified with `cargo check --release --benches --bin bench_serving`, `cargo test --release test_gpu_sample -- --nocapture`, and `cargo run --release --bin bench_serving -- request --prompt-len 1 --output-len 2 --warmup 0 --iters 1`.
- Result: success. Runtime token selection is now unified on the FlashInfer backend; the remaining decision is whether to keep that simplification or reintroduce a separate batched-greedy specialization later if throughput regressions show up.

### Step 6: Remove retired kernels and roll back `bench_serving` sampling knobs
- Reverted `src/bin/bench_serving.rs` and `bench_snapshots/rtx-5070-ti/qwen3-4b.json` to `HEAD`, dropping the temporary `--temperature/--top-k/--top-p/--ignore-eos` benchmark surface and restoring the original greedy-only snapshot schema.
- Split `argmax` into `csrc/argmax.cu`, deleted `csrc/sampling.cu`, and removed the dead Rust/FFI batched-argmax surface so the runtime no longer carries the retired legacy sampling kernel.
- Kept sampling performance coverage in `benches/ops/ops_embedding_sampling_bench.rs`, which is now the intended place to compare FlashInfer sampling behavior.
- Verified symbol cleanup with `rg -n "argmax_batched|gpu_sample_cuda|gpu_sample_kernel|argmax_batched_cuda" src csrc benches` (no matches).
- Result: success. The codebase now reflects the final architecture decision instead of keeping compatibility scaffolding that is no longer wanted.

### Step 7: Move greedy back onto a better FlashInfer primitive
- Profiled the decode-heavy greedy path with `nsys` and confirmed the regression came from using `logits_to_probs_kernel + TopKSamplingFromProbKernel` for greedy token selection.
- Added `csrc/flashinfer_top1.cu`, which uses FlashInfer's radix top-k dispatch with `k=1` to select the greedy token directly, and threaded the needed scratch buffers through the Rust sampling state/buffer structs.
- Kept non-greedy sampling on the existing FlashInfer probability-sampling path; only greedy dispatch changed.
- Verified compileability with `cargo check --release --bin bench_serving --benches`.
- Re-ran greedy microbench: `flashinfer_greedy_top_k_1/128256` now measures about `152-154 us`, effectively matching `argmax`.
- Re-ran end-to-end decode-heavy serving: `cargo run --release --bin bench_serving -- request --prompt-len 1 --output-len 128 --warmup 1 --iters 5` now reports `steady_tpot p50 11.59 ms`.
- Re-generated snapshot with `cargo run --release --bin bench_serving -- snapshot`: prefill-heavy TTFT p50 `1189.99 ms`; decode-heavy TPOT p50 `12.28 ms`, p99 `12.38 ms`.
- Result: success. Greedy stays on FlashInfer primitives, but the extra decode overhead from the `top_k=1` probability-sampling route is gone.

### Step 8: Cross-check `mini-sglang` and `vLLM` batch sampling design
- Audited `../mini-sglang/python/minisgl/engine/sample.py` and `../vllm/vllm/v1/{worker/gpu/sample, sample}/...` to compare how other runtimes handle per-request sampling parameters inside a decode batch.
- `mini-sglang` batches per-request `temperature/top_k/top_p` into device tensors, keeps an all-greedy fast path with `torch.argmax(logits, dim=-1)`, and only calls FlashInfer for batched non-greedy rows.
- `vLLM` keeps per-request sampling metadata in batch state (`temperature/top_k/top_p/min_p/seed`), computes batched greedy tokens first, and only routes the random rows through its top-k/top-p sampler before merging results with a batch-wise `torch.where`.
- `vLLM` also keeps FlashInfer usage narrow: the FlashInfer sampler is only used when top-k/top-p is active, while the no-top-k/no-top-p path stays on its native random sampler because FlashInfer introduces a CPU-GPU sync boundary.
- Result: success. External references agree that per-request sampling parameters do not require per-request sampling launches; the expected design is batched metadata plus a preserved batched greedy path.

### Step 9: Incorporate independent review findings
- Spawned a sub-agent review and asked it to stress the current local diff from a hostile reviewer perspective.
- The review identified one high-severity issue: `src/model/qwen3/batch_decode.rs` and `src/model/qwen35/batch_decode.rs` now sample each row separately, while the serving schedulers still call them on the hot path. That removes the old batched greedy behavior and turns all-greedy batch decode into `O(batch)` GPU launches plus one sync/D2H per row.
- The review also flagged a lower-confidence integration risk: `src/ops/sampling.rs` currently hard-codes a `1MB` FlashInfer row-state scratch allocation, which is a brittle dependency on upstream internal layout.
- Result: blocked. Single-request greedy regressions are fixed, but the batch decode token-selection path still needs a real batched design before this work is ready to land.

## Debrief
- **Outcome**: FlashInfer sampling stayed in the runtime path; legacy `gpu_sample_cuda` and batched argmax were removed; `bench_serving` returned to its original greedy-only role; greedy token selection now uses FlashInfer top-k selection instead of FlashInfer probability sampling. The remaining blocker is batch decode token selection, which currently regresses to per-request launches.
- **Pitfalls encountered**:
  - The temporary `bench_serving` sampling knobs spread into JSON snapshot shape and compare-time validation, so rolling them back cleanly was better handled as a file-level revert than by piecemeal edits.
  - The old `csrc/sampling.cu` file still contained the only `argmax` implementation, so deleting "old sampling kernels" required first extracting the still-used greedy benchmark helper into its own translation unit.
- **Lessons learned**:
  - Serving benchmarks and operator microbenches should stay separated: use `bench_serving` for stable end-to-end service tracking and `ops_bench` for backend-specific sampling experiments.
  - Once FlashInfer becomes the runtime sampling backend, lingering fallback kernels become maintenance debt quickly unless they still serve a distinct benchmark or compatibility purpose.
  - "Using FlashInfer" is not specific enough for performance work. The exact primitive matters: `top_k=1` probability sampling regressed decode TPOT, while FlashInfer top-k selection (`k=1`) restored it.
  - Competing runtimes solve per-request sampling diversity with batched metadata, not with per-request sampling launches. `mini-sglang` and `vLLM` both preserve a batched greedy path and batch the random path around per-row tensors.
- **Follow-ups**:
  - Rework `src/model/qwen3/batch_decode.rs` and `src/model/qwen35/batch_decode.rs` so batch token selection stays batched instead of iterating row by row.
  - Replace the hard-coded FlashInfer row-state scratch size with a source-backed contract or a tighter wrapper that owns the sizing rule.
