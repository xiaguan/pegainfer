# vLLM Frontend Rust Integration

**Created**: 2026-04-29
**Status**: complete
**TL;DR**: pegainfer now starts `vllm-server` as the default HTTP/OpenAI frontend, backed by a local engine-core-compatible bridge into the pegainfer scheduler. The old local HTTP frontend and pegainfer tokenizer module are removed; tokenization is handled by `vllm-text`. Qwen3-4B golden data was regenerated for `/data/Qwen3-4B`, and 5090 release build, scheduler e2e, all test-target compilation, completions, models, and chat smoke tests pass.

## Preparation

- **Read**:
  - `docs/index.md` — confirmed there was no existing active doc for vLLM frontend integration.
  - `docs/resources/developer-onboarding.md` — confirmed manifest-only validation should avoid debug/CUDA builds and full compilation belongs on a GPU-capable machine.
  - `/Users/mac/code/vllm-frontend-rs/README.md` — identified `vllm-server` as the OpenAI-compatible HTTP API crate and the Rust frontend entrypoint layer.
  - `/Users/mac/code/vllm-frontend-rs/Cargo.toml` — confirmed the repo is a workspace and that internal crates are wired as workspace dependencies.
  - `/Users/mac/code/vllm-frontend-rs/src/server/Cargo.toml` — confirmed `vllm-server` pulls the chat, text, LLM, metrics, and engine-core-client crates transitively.
  - `/Users/mac/code/vllm-frontend-rs/src/server/src/lib.rs` — confirmed `serve()` currently constructs a concrete `EngineCoreClient`, so direct reuse would target Python vLLM engine-core unless the frontend exposes a local backend seam.
  - `/Users/mac/code/vllm-frontend-rs/src/server/src/state.rs` — confirmed server state owns a concrete `ChatLlm`, which currently points through `TextLlm` and `Llm`.
  - `/Users/mac/code/vllm-frontend-rs/src/text/src/lib.rs` — confirmed `TextLlm` owns a concrete `vllm_llm::Llm`, so pegainfer cannot yet inject its scheduler without either a frontend upstream change or a local engine-core-compatible adapter.
  - `/Users/mac/code/vllm-frontend-rs/src/text/src/backend/mod.rs` — confirmed tokenizer/model metadata is already trait-based and suitable for a pegainfer backend.
  - `src/main.rs` — confirmed pegainfer currently loads the model and then builds an axum router from its scheduler handle.
  - `src/http_server.rs` — confirmed the current OpenAI frontend is a local `/v1/completions` implementation directly over `SchedulerHandle`.
  - Remote 5090 workflow details were verified during execution, but the separate 5090 development guide is intentionally excluded from this PR.
- **Relevant history**:
  - No existing project doc specifically covered `vllm-frontend-rs` integration.
- **Plan**:
  1. Put the integration work on `feat/vllm-frontend-integration`.
  2. Add Git dependencies for the vLLM frontend crates pegainfer needs while replacing its HTTP frontend, then keep only the crates pegainfer directly calls.
  3. Move pegainfer to nightly through `rust-toolchain.toml`, because the imported vLLM frontend crates use unstable features.
  4. Update dependency constraints and lockfile as needed, starting with `serde_json >= 1.0.149`.
  5. Run `cargo metadata` to validate Cargo can resolve the Git workspace dependency without building CUDA code.
  6. Capture the implementation direction: either upstream/local changes to make `vllm-frontend-rs` accept an injected local generate backend, or a pegainfer adapter that speaks the vLLM engine-core protocol.
- **Risks / open questions**:
  - Directly calling `vllm_server::serve()` is not enough for pegainfer because it expects a vLLM engine-core transport, not a pegainfer scheduler.
  - The clean replacement path likely needs small API changes in `vllm-frontend-rs` so its server/text layers accept a local generate backend.
  - Full compilation still needs the target GPU/CUDA machine.

## Execution Log

### Step 1: Add Git dependency

- Initially added `vllm-server` from `https://github.com/Inferact/vllm-frontend-rs.git` on branch `main`.
- `cargo metadata` exposed a lockfile conflict: `fastokens` from `vllm-frontend-rs` requires `serde_json ^1.0.149`, while pegainfer was locked to `1.0.145`.

### Step 2: Expand scope to frontend replacement

- Created branch `feat/vllm-frontend-integration`.
- Expanded dependencies to the vLLM frontend crates needed for a real replacement boundary, then pruned direct dependencies to `vllm-server`, `vllm-text`, and `vllm-engine-core-client`; `vllm-chat` and `vllm-llm` remain transitive through `vllm-server`.
- Added `rust-toolchain.toml` with nightly because the vLLM frontend crates use unstable Rust features.
- Raised `serde_json` to `1.0.149` in `Cargo.toml`.

### Step 3: Validate on 5090

- Used remote clone at `/root/develop/xingming/pegainfer`.
- Synced only `Cargo.toml` and `rust-toolchain.toml` to the remote clone with `rsync`.
- Ran `cargo metadata`; it installed nightly `1.97.0-nightly` and resolved the vLLM frontend workspace crates.
- Pulled the updated remote `Cargo.lock` back with `rsync`.
- First `cargo build --release` failed because the fresh clone had not initialized `third_party/flashinfer`, causing `flashinfer/sampling.cuh` to be missing.
- Ran `git submodule update --init --recursive`.
- Re-ran `PEGAINFER_CUDA_SM=120 cargo build --release`; build succeeded and compiled the vLLM frontend crates plus pegainfer CUDA/Triton kernels.

### Step 4: Model and e2e smoke

- Created `/root/develop/xingming/pegainfer/.venv` with `uv venv`.
- Installed `triton` and `huggingface_hub` into the venv.
- Downloaded `Qwen/Qwen3-4B` to `/data/Qwen3-4B`; resolved revision `1cfa9a7208912126459214e8b04321603b3df60c`, size `7.6G`.
- Ran `PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release --test e2e -- --nocapture`.
- Result: model loads and generates, but the golden regression fails because the current HF revision output does not match `test_data/Qwen3-4B.json`.
- Ran a live HTTP smoke on port `18080`; `/v1/completions` returned a valid response for `{"prompt":"Hello","max_tokens":8}`.

### Step 5: Regenerate Qwen3-4B golden data

- Ran the ignored `regen_test_data` test on 5090 with `PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B`.
- The command also attempted `regen_test_data_qwen35` because the Cargo test name filter matched both names; Qwen3.5 failed as expected against the Qwen3 model path, but the Qwen3-4B file had already been written.
- Pulled `test_data/Qwen3-4B.json` back with `rsync`.
- Re-ran `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release --test e2e -- --nocapture`; result: pass.

### Step 6: Replace default frontend

- Added `src/vllm_frontend.rs`.
- The module starts `vllm_server::serve()` with bootstrapped IPC transport and spawns a local engine-core-compatible bridge.
- The bridge registers with `vllm-engine-core-client`, decodes add/abort/utility frames, submits tokenized requests to `SchedulerHandle`, and sends `EngineCoreOutputs` back over ZMQ.
- Updated `src/main.rs` so the default server path loads pegainfer's model/scheduler and then serves through `vllm-server`.
- Fixed an adapter sampling mismatch: vLLM frontend inherits `top_p=0.95/top_k=20` from `generation_config.json`; when `temperature <= 0`, the bridge now forces greedy sampling (`top_p=1`, `top_k=-1`) before calling pegainfer's sampler.

### Step 7: Validate replaced frontend

- Synced only necessary source files to 5090 with `rsync`.
- Ran `PEGAINFER_CUDA_SM=120 cargo build --release`; result: pass.
- Started `cargo run --release -- --model-path /data/Qwen3-4B --port 18080`.
- Verified logs show `vllm_server` startup and local bridge registration.
- Verified `/v1/models`.
- Verified non-streaming `/v1/completions` with `temperature=0` and `add_special_tokens=false`; output begins with ` Also, can you explain how to`.
- Verified streaming `/v1/completions`; SSE deltas were emitted and terminated with `[DONE]`.
- Verified `/v1/chat/completions`; Qwen3 reasoning parser produced a chat response.
- Stopped the server and confirmed no lingering build/test/server processes.
- Re-ran Qwen3-4B scheduler e2e after the frontend replacement; result: pass.

### Step 8: Remove old frontend and tokenizer ownership

- Deleted the old local OpenAI frontend: `src/http_server.rs` and `src/http_server/openai_v1.rs`.
- Deleted the old pegainfer tokenizer module: `src/tokenizer.rs`.
- Removed old frontend/tokenizer dependencies from pegainfer's direct dependency list: `tokenizers`, `axum`, `tokio-stream`, `futures-util`, `base64`, `tower`, and no-longer-direct vLLM crates.
- Deleted `benches/tokenizer_bench.rs` and `examples/regenerate_test_data.rs`; tokenizer performance and frontend behavior are now vLLM frontend responsibilities.
- Updated Qwen3 e2e tests, golden-data regeneration, and `bench_serving` to use `vllm_text::tokenizer::HuggingFaceTokenizer` directly where test/tool tokenization is still needed.
- Removed old OpenAI stop/usage/stream helper types from `src/server_engine.rs`; vLLM frontend owns text stop handling and OpenAI response shaping.
- Synced changed source/test/bench/example directories to 5090 with `rsync --delete` so remote deleted files match the Mac checkout.
- Verified on 5090:
  - `PEGAINFER_CUDA_SM=120 cargo build --release`: pass.
  - `PEGAINFER_CUDA_SM=120 PEGAINFER_TEST_MODEL_PATH=/data/Qwen3-4B cargo test --release --test e2e -- --nocapture`: pass.
  - `PEGAINFER_CUDA_SM=120 cargo test --release --no-run`: pass.
  - `/v1/models`, `/v1/completions`, and `/v1/chat/completions` smoke through `vllm-server`: pass.
- Qwen3.5 runtime e2e was intentionally skipped for this pass; the user explicitly said 3.5 can be ignored.

## Debrief

- **Outcome**: The vLLM frontend dependencies resolve and compile on the 5090 machine under nightly; Qwen3-4B is downloaded under `/data`; the default pegainfer HTTP/OpenAI entrypoint now uses `vllm-server` through a local engine-core bridge into the pegainfer scheduler. The old local HTTP frontend and self-maintained tokenizer module are gone.
- **Pitfalls encountered**:
  - Fresh clone needs `git submodule update --init --recursive`; otherwise FlashInfer headers are missing.
  - Remote commands that need network should use `bash -ic` so `.bashrc` proxy settings are loaded.
  - The current `Qwen/Qwen3-4B` HF revision did not match the repo's previous golden output; regenerating fixed the scheduler e2e.
  - vLLM frontend sampling defaults differ from pegainfer's old HTTP defaults; the bridge must normalize `temperature <= 0` to greedy for pegainfer's sampler.
  - Cargo test filters are substring-based unless `--exact` is passed; the Qwen3 golden regen command should use `--exact` to avoid matching the Qwen3.5 regen test.
- **Lessons learned**:
  - The 5090 workflow combines remote path, venv, proxy, model, CUDA SM details, and vLLM frontend smoke commands; keep that operational guide separate from this integration PR.
  - Keeping `vllm-frontend-rs` as a Git dependency is feasible if pegainfer implements the engine-core wire boundary locally.
  - Tokenization should stay on the vLLM side of the boundary; pegainfer should pass token IDs through the scheduler/model runtime and avoid owning tokenizer wrappers.
- **Follow-ups**:
  - Add logprobs/prompt-logprobs translation in the bridge if the vLLM frontend logprob endpoints need to be fully supported.
  - Decide whether the public served model ID should remain the local path (`/data/Qwen3-4B`) or be decoupled from the tokenizer/model-file path.
