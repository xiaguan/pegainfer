This file provides guidance to Coding Agent when working with code in this repository.

## What is pegainfer

Pure Rust + CUDA LLM inference engine (~7K Rust, ~3.4K CUDA). No PyTorch, no frameworks. Supports Qwen3-4B, Qwen3-8B, and Qwen3.5-4B (hybrid linear + full attention). OpenAI-compatible `/v1/completions` API.

## Build & Run

**Always use `--release`** — debug builds are extremely slow for GPU/CUDA and will timeout.

```bash
cargo run --release -- --model-path models/Qwen3.5-4B
```

**Key env vars:**
- `PEGAINFER_CUDA_SM` — GPU SM target override when `nvidia-smi` unavailable (e.g. `120` or `120,80`)
- `PEGAINFER_TRITON_PYTHON` — Python with Triton for build-time AOT kernel generation
- `PEGAINFER_TEST_MODEL_PATH` — override test model path (default: `models/Qwen3-4B`)

## Tests

```bash
# Unit tests (~9s)
cargo test --release

# E2E greedy regression — requires GPU + model weights
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35

# Single test
cargo test --release embedding_variants -- --nocapture
```

E2E tests compare against JSON baselines in `test_data/`. Regenerate baselines after any change that affects numerical output.

## Architecture

```
HTTP Request → GenericServerEngine<M> → tokenizer.encode() → M.generate() → tokenizer.decode() → SSE/JSON Response
                                                                  │
                                          ┌───────────────────────┤
                                          │                       │
                                    Qwen3Model              Qwen35Model
                                   (full attention)    (24 linear + 8 full attn)
                                          │                       │
                                          └───────────────────────┘
                                                    │
                                    Prefill (batch GEMM) → Decode loop (GEMV, CUDA Graph)
                                                    │
                                              ops.rs → ffi.rs → CUDA/Triton kernels
```

**Key abstractions:**

- **`GenerativeModel` trait** — implemented by `Qwen3Model` and `Qwen35Model`. `GenericServerEngine<M: GenerativeModel>` provides shared request handling (stop sequences, streaming, sampling).
- **`ops.rs`** — high-level GPU operators (gemv, rms_norm, rope, fused_mlp, fused_attention, sampling). Calls unsafe C FFI in `ffi.rs`.
- **Tensor types** (`tensor.rs`) — `DeviceVec` (1D bf16), `DeviceMatrix` (2D bf16), `HiddenStates` (batched hidden states). All GPU-resident, accessed via cudarc.
- **CUDA Graph** — decode path captured on first token, replayed on subsequent tokens. `embedding_decode_cuda()` reads token_id from GPU memory (not CPU) so the graph is stable. Pre-allocated `DecodeBuffers` ensure pointer stability.
- **KVCache** (`kv_cache.rs`) — per-layer contiguous K/V buffers, seq_len tracking, reset between requests (keeps allocations).
- **RecurrentState** (`recurrent_state.rs`) — Qwen3.5 linear attention state, persists across decode steps within a generation.

**Build system** (`build.rs`) does two things:
1. Compiles `csrc/*.cu` with nvcc (auto-detects GPU SM targets)
2. Runs Triton AOT compilation via `tools/triton/gen_triton_aot.py` for 7 kernels (silu_mul, add, embedding variants, split-KV attention decode/reduce)

---

# Team Documentation Workflow

Collaboration centered on the `docs/` directory.

## Knowledge Architecture (PARA)

Based on the PARA methodology. Classify information at the point of capture — no staging area.

```
docs/
├── index.md           # Document index
├── projects/          # Time-bound efforts with clear deliverables
├── areas/             # Ongoing responsibilities requiring maintained standards
├── resources/         # Topics and references with potential future value
└── archives/          # Completed, abandoned, or shelved inactive items
```

## Documentation Style

- Docs cover what `--help` and code can't: pitfalls, diagnostic paths, decision context. Don't restate CLI reference.
- Every command in a doc must be run and verified before committing. Unverified commands are technical debt.

## Core Principles (CODE)

Documentation exists to advance work, not to hoard information. Four steps when handling information:

1. **Capture**: Only record what materially advances the project. When in doubt, leave it out.
2. **Organize**: Action-oriented. Resist the urge to organize for organization's sake — structure should be just enough.
3. **Distill**: Refactor over append. When you learn something new or hit a pitfall, integrate it into the document body — don't pile a changelog at the bottom.
4. **Express**: Every document must point to a next step. Split unwieldy documents proactively. Active documents must note the current blocker or next action.

## Collaboration Lifecycle

**Sync**

At the start of each session, you must read `index.md` and load the documents needed for the task at hand.

**Execute**
- Update relevant documents as you go. When a new problem or idea arises, create a document in the appropriate PARA directory.
- Record *why* a decision was made, not just *what* was done.

**Commit**

When a session wraps up:
- Update the TL;DR and status at the top of each modified document.
- Update `index.md` to keep the global routing table current.

## index.md Specification

The document index — nothing more:

| Path | TL;DR |
| --- | --- |

---

# Git Conventions

Commit messages use Commitizen format: `<type>(<scope>): <subject>`. Never commit directly to `main` — create a `feat/`/`fix/`/`chore/`/… branch first.
