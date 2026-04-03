# Refactor Embedding Token IDs To U32

**Created**: 2026-04-03
**Status**: complete

## Preparation

- **Read**:
  - `docs/index.md` — identified the existing docs map and the closest prior kernel/runtime boundary work.
  - `docs/projects/pegainfer-kernels-boundary.md` — confirmed the codebase is already trying to keep operator interfaces explicit and avoid leaking mismatched runtime details across layers.
  - `docs/areas/coding-style.md` — confirmed testing should stay minimal and focus on kernel/operator correctness rather than adding ceremony.
- **Relevant history**:
  - `docs/projects/pegainfer-kernels-boundary.md` — this refactor should tighten the operator boundary instead of adding another type-conversion seam in model code.
  - no existing project doc found for token-id type cleanup in embedding paths.
- **Plan**:
  1. Audit the embedding call chain across `src/model/*/{prefill,batch_decode}.rs`, `src/ops/embedding.rs`, `src/ffi.rs`, and `csrc/elementwise.cu` to identify every `i32` token-id assumption that is part of embedding lookup rather than generic decode metadata.
  2. Change the embedding token-id path to `u32` end-to-end where it represents token indices, keeping unrelated signed metadata untouched unless the code forces coupling.
  3. Update affected benches/tests and run targeted validation for the embedding ops and any compile-time breakage introduced by the signature changes.
  4. Record the execution details and lessons learned here, then update `docs/index.md` if this project doc remains relevant after completion.
- **Risks / open questions**:
  - `embedding_decode` currently reads from `decode_meta: *const i32`; if token id and positional metadata share one buffer today, we may need a minimal interface split instead of a superficial type swap.
  - CUDA-side indexing and Rust FFI signatures must stay layout-compatible; the safest refactor is the smallest one that preserves current buffer structure where semantics are mixed.

## Execution Log

### Step 1: Audit the embedding token-id boundary
- Read the embedding path across `src/model/qwen3/prefill.rs`, `src/model/qwen35/prefill.rs`, `src/model/qwen3/batch_decode.rs`, `src/model/qwen35/batch_decode.rs`, `src/ops/embedding.rs`, `src/ffi.rs`, `csrc/elementwise.cu`, and the embedding ops tests/benches.
- Confirmed the `i32` usage was limited to the embedding token-id buffers and FFI/kernel signatures; position and paged-attention metadata remain separate `i32` buffers and do not need to move.
- Result: success.

### Step 2: Change embedding token-id buffers/signatures to `u32`
- Updated Rust model code to upload `u32` token ids directly instead of materializing temporary `Vec<i32>` conversions.
- Updated `src/ops/embedding.rs`, `src/ffi.rs`, `csrc/elementwise.cu`, and batch decode buffer structs so embedding token-id buffers are `CudaSlice<u32>` end-to-end.
- Split the single-token embedding helper semantics from generic "decode meta": the embedding op now takes a GPU token-id buffer, and the benchmark helper was renamed from `decode_meta` to `decode_token_id`.
- Result: success pending compile/test validation.

### Step 3: Validate the changed embedding path
- First ran `cargo test --release embedding_variants -- --nocapture`.
- Build initially failed before Rust test execution because `build.rs` always compiles `csrc/flashinfer_norm.cu`, and the local checkout had not initialized `third_party/flashinfer` yet.
- Ran `git submodule update --init third_party/flashinfer` to restore the expected FlashInfer headers, then reran `cargo test --release embedding_variants -- --nocapture`.
- Result: success. `ops::tests::test_embedding_variants` passed in release mode.

### Step 4: Run E2E and performance validation
- Ran `cargo test --release --test e2e -- --nocapture` against `models/Qwen3-4B`.
- Ran `cargo test --release --test e2e_qwen35 -- --nocapture` against `models/Qwen3.5-4B`.
- Ran `cargo run --release --bin bench_serving -- request --output-len 32 --iters 3 --warmup 1` as the benchmark smoke test.
- Ran `cargo test --release --test bench_prefill -- bench_prefill_1024 --nocapture` for a focused prefill benchmark.
- Result: success. Both E2E suites passed; smoke benchmark reported `ttft_ms=12.37`, `steady_tpot_ms=11.55`, `request_tok_s=86.34`, `decode_tok_s=86.53`; prefill benchmark reported `TTFT=95.90ms` and `prefill_throughput=10677 tok/s` at `seq_len=1024`.

### Unexpected
- The single-token embedding helper was no longer carrying generic decode metadata in practice; the cleanest minimal refactor was to rename that helper boundary to a token-id buffer instead of preserving the misleading `decode_meta` name.
- Local validation initially failed for an environment reason, but the fix was just initializing the existing `third_party/flashinfer` submodule.

## Debrief

- **Outcome**: Refactored embedding token ids to stay `u32` from model code through GPU buffers, Rust op wrappers, FFI, CUDA kernels, tests, and benches. Release validation passed for targeted ops tests, both model E2E suites, and the local performance smoke benchmarks.
- **Pitfalls encountered**:
  - The previous single-token helper name (`decode_meta`) obscured that the embedding kernel only consumes a token id; leaving the old name would preserve the same semantic mismatch in a different place.
  - Release test validation depends on `third_party/flashinfer` being initialized, because `build.rs` always compiles `csrc/flashinfer_norm.cu`.
- **Lessons learned**:
  - The embedding path is already decoupled enough from attention-position metadata that `u32` token ids can move independently with a small focused patch.
  - When a CUDA build unexpectedly fails on missing FlashInfer headers, check submodule initialization before touching code.
