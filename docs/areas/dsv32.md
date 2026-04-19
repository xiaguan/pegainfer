# DeepSeek-V3.2 on pegainfer

> **Status**: Prefill logits aligned with reference. Decode currently runs dense FlashMLA (implementation gap vs NSA sparse design). H20 baseline for the **teacher-forced top-K logprob harness against SGLang** has completed (`44/44` cases, `6888` positions, `3.03%` argmax mismatches, `5` argmax-outside-topk). The committed regression path is now the **small greedy JSON E2E** over the same 44 prompts; the larger SGLang manifest stays local/H20-only and is gitignored. The old vLLM accuracy path has been retired because its H20 reference runtime never initialized reliably.
> **Last updated**: 2026-04-19
>
> Read this document to know: what's supported, how each operator is implemented, what the performance is, and what's next.

---

## 1. Goals & Non-goals

### Goals

- Run DSV3.2 (671B MoE, FP8 checkpoint) with output logits matching reference.
- Support three target parallelism shapes, in priority order:
  1. **TP1 + DP1 + EP8** (current) — one replica, attention and dense FFN are replicated per rank, routed MoE is sharded 8-way.
  2. **TP1 + DP8 + EP8** — replicate the current EP8 runtime across 8 DP groups.
  3. **TP2 + DP4 + EP8** — add TP2 inside each DP group while keeping EP8 for MoE.
- OpenAI-compatible `/v1/completions` serving on par with existing Qwen3 path.

### Non-goals (short-term)

- **No CUDA Graph capture** for decode (Qwen3 path has it; DSV3.2 doesn't yet).
- **No internode** (cross-machine) communication. Single-box 8-card only.
- **No MTP speculative decoding**.

---

## 2. Parallelism Matrix

Below, `DP` denotes service-level replica count, while `TP` and `EP` describe the per-replica model-parallel shape. The current runtime is `TP1 + DP1 + EP8`.

| Shape | Attention | Dense FFN (layers 0-2) | Routed MoE | Shared Expert | Status |
|-------|-----------|-----------------------|------------|---------------|--------|
| TP1 + DP1 + EP8 | Each rank computes its own tokens | Duplicated per rank | DeepEP dispatch: 32 experts/rank | Duplicated | ✅ current |
| TP1 + DP8 + EP8 | Same as current inside each DP group | Duplicated per rank | DeepEP dispatch: 32 experts/rank within each EP8 group | Duplicated | ⏳ planned |
| TP2 + DP4 + EP8 | 2 ranks share one TP group | TP2 row/col shard | Same EP8 dispatch, 4 DP groups | Duplicated | ⏳ planned |

All shapes use DeepEP intranode (NVLink) for MoE All-to-All.

---

## 3. Hardware & Environment

- Target: single-box 8-card node with NVLink intra-node interconnect.
- FP8 weights ~671 GB fit on 8 cards (BF16 ~1.34 TB does not — FP8 is mandatory).
- Model path / venv / SSH details: see the private gitignored development guide.

---

## 4. Model Architecture

| Field | Value |
|-------|-------|
| Total layers | 61 |
| Dense FFN layers | 0..3 (first 3) |
| MoE layers | 3..61 |
| hidden_size | 7168 |
| intermediate_size (dense) | 18432 |
| moe_intermediate_size | 2048 |
| num_attention_heads | 128 |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| qk_nope_head_dim / qk_rope_head_dim / v_head_dim | 128 / 64 / 128 |
| n_routed_experts / num_experts_per_tok | 256 / 8 |
| NSA indexer heads / head_dim / topk | 64 / 128 / 2048 |
| RoPE | YaRN |

The **NSA (Native Sparse Attention) indexer** is DSV3.2-specific: a lightweight 64-head projection selects top-2048 KV positions per token; FlashMLA sparse kernel then attends only to those.

---

## 5. Per-layer DAG

### Prefill (sparse, seq_len > 1)

```
Full-layer forward:
  RMSNorm(hidden)
  ├── Q path:   FP8 quantize → q_a_proj [1536,7168] → RMSNorm → q_b_proj [24576,1536]
  │             → split per-head (nope 128 + rope 64) → Q absorption (cuBLAS batched GEMM, W_UK) → q_mla [128,576]
  ├── KV path:  FP8 quantize → kv_a_proj [576,7168] → partial RMSNorm(c_kv) + RoPE(k_rope)
  │             → write to MLA paged KV cache (page_size=64)
  ├── Indexer:  wq_b/wk FP8 GEMMs → similarity → topk-2048 indices
  └── Attention: FlashMLA sparse prefill (SM90, d_qk=576, d_v=512)
                → V de-absorption (cuBLAS batched GEMM, W_UV)
                → o_proj [7168,16384] (FP8)
                → residual

Dense layer (0..3):  RMSNorm → gate/up/down FP8 GEMMs + SiLU*Mul → residual
MoE layer (3..61):   RMSNorm
                     ├── Shared expert FFN (local, FP8)
                     ├── Router: gate GEMM (bf16) → sigmoid + group-limited TopK-8
                     ├── DeepEP dispatch (NVLink All-to-All)
                     ├── Routed expert FFN (per-expert sequential FP8 GEMMs)
                     └── DeepEP combine → residual

Final:  RMSNorm → lm_head (bf16 GEMM, tied with embedding)
```

### Decode (bs=1 per request, current implementation)

Same DAG as prefill except:
- seq_len=1, batch GEMM collapses to GEMV-ish shapes.
- FlashMLA dense decode kernel (3-phase: metadata → decode → combine).
- Implementation gap: DSV3.2 is NSA sparse-attention architecture, but current decode skips indexer top-k selection and attends over the full KV cache.

---

## 6. Operator Coverage

Columns: **DAG node**, **Provider** (where the kernel comes from), **Source file**, **Status**, **Notes**.

`flashinfer` accounting note: FlashInfer is header-only, but still counted as an external operator provider whenever our local `.cu` wrapper directly instantiates `flashinfer::...` templates. In the current DSV3.2 runtime, this applies to RMSNorm-family kernels.

### Attention path

| DAG node | Provider | File | Status | Notes |
|----------|----------|------|--------|-------|
| FP8 activation quantize (1×128 block-scale) | Self-written (extracted from TRT-LLM) | `csrc/fp8_quantize.cu` | ✅ | |
| FP8 block-scale GEMM (q_a/q_b/kv_a/o_proj) | DeepGEMM SM90 1D2D | `csrc/fp8_gemm.cu` + `third_party/DeepGEMM` | ✅ | AOT-compiled, 2 tile configs (64×128/8s, 128×128/5s) |
| RMSNorm (full + fused add, incl. final norm) | FlashInfer header-only templates + local C wrapper | `csrc/flashinfer_norm.cu`, `src/ops/norm.rs` | ✅ | `RMSNorm` / `FusedAddRMSNorm` / `GemmaRMSNorm` |
| Partial RMSNorm (c_kv-only, in-place) | Self-written | `csrc/mla.cu` | ✅ | Only normalizes first `kv_lora_rank` dims in `kv_a` |
| MLA RoPE (q_rope extract+apply+copy, k_rope) | Self-written | `csrc/mla.cu` | ✅ | YaRN cos/sin cache pre-computed at load |
| Q absorption / V de-absorption (bf16 batched GEMM) | cuBLAS `cublasGemmStridedBatchedEx` | `csrc/linear.cu` | ✅ | W_UK/W_UV dequant CPU-side at load (~2 GB total) |
| KV cache write (scatter to paged buffer) | Self-written | `csrc/mla.cu` | ✅ | Per-layer paged, page_size=64 |
| FlashMLA dense decode (3-phase) | FlashMLA SM90 | `csrc/flash_mla.cu` + `third_party/FlashMLA` | ✅ (current impl) | d_qk=576, d_v=512, MQA; runtime decode is dense (NSA indexer top-k not wired) |
| FlashMLA sparse prefill (NSA) | FlashMLA SM90 | `csrc/flash_mla_prefill.cu` | ✅ | |
| NSA indexer (wq_b, wk projections + topk) | Self-written + DeepGEMM | `csrc/mla.cu`, `forward_indexer` | ✅ | |

### MoE path

| DAG node | Provider | File | Status | Notes |
|----------|----------|------|--------|-------|
| Router gating (sigmoid + group-limited TopK-8) | Self-written | `csrc/moe.cu` | ✅ | |
| DeepEP `get_dispatch_layout` / `notify_dispatch` / `dispatch` / `combine` | DeepEP intranode (NVLink) | `csrc/deep_ep.cu` + `third_party/DeepEP` | ✅ | `-DDISABLE_NVSHMEM -DTOPK_IDX_BITS=64`, 8-way IPC |
| Routed expert FFN (per-expert sequential FP8 GEMM) | DeepGEMM | `src/model/dsv32/forward.rs:forward_moe_ep` | ✅ | ⚠️ sequential; grouped GEMM is the obvious next optimization |
| Shared expert FFN | DeepGEMM FP8 | `forward_moe*` | ✅ | |
| SiLU * Up | Self-written | `src/ops.rs` | ✅ | |

### Model-level

| DAG node | Provider | File | Status | Notes |
|----------|----------|------|--------|-------|
| Embedding lookup (batched) | Self-written | `src/ops.rs` | ✅ | |
| Final RMSNorm + lm_head (bf16) | FlashInfer (norm) + cuBLAS (lm_head) + local glue | `forward_final`, `csrc/flashinfer_norm.cu`, `csrc/linear.cu` | ✅ | `tie_word_embeddings` shares embedding matrix |
| KV cache pool (paged, per-layer) | Self-written | `src/model/dsv32/mla_kv.rs` | ✅ | `MlaKvPool`, `MlaKvState` |
| Weight loader (safetensors → FP8/bf16 sharded) | Self-written | `src/model/dsv32/weights.rs` | ✅ | Per-rank parallel load, 8 threads ~56 s |
| Multi-GPU executor (per-rank ctx + NCCL + DeepEP) | Self-written | `src/model/dsv32/executor.rs` | ✅ | |
| Scheduler (serial) | Self-written | `scheduler_dsv32` | ✅ | One request at a time, no continuous batching |
| Token sampling (service path) | Self-written CPU sampler | `src/scheduler_dsv32.rs` | ✅ | Current DSV3.2 path samples on CPU; FlashInfer sampling kernels are in-tree but not wired into this scheduler yet |

### Parallelism & communication

| Need | Provider | Status |
|------|----------|--------|
| NCCL AllReduce (for TP≥2) | cudarc NCCL bindings | ⏳ plumbed in Qwen3 path; not exercised for DSV3.2 since current TP=1 |
| DeepEP intranode All-to-All | DeepEP | ✅ |
| CUDA IPC for DeepEP buffer exchange | cudarc + custom Rust glue | ✅ |

---

## 7. Benchmarks

### Micro benchmarks (`benches/`)

**TODO — none exist for DSV3.2 yet.** Planned additions:

- `benches/dsv32_fp8_gemm.rs` — DeepGEMM 1D2D at the exact DSV3.2 shapes (q_a, q_b, kv_a, kv_b, o_proj, gate/up/down, expert gate/up/down).
- `benches/dsv32_flash_mla.rs` — FlashMLA dense decode and sparse prefill at DSV3.2 dims.
- `benches/dsv32_deep_ep.rs` — DeepEP dispatch/combine latency and bandwidth at token_count sweep.

Priority: FP8 GEMM first (largest time share), then DeepEP (EP8 communication), then FlashMLA.

### End-to-end benchmarks

Follow `docs/resources/model-optimization-pipeline.md`:

| Profile | Input | Output | Isolates |
|---------|-------|--------|----------|
| prefill-heavy | 2048 | 1 | TTFT, prefill kernels |
| decode-heavy | 1 | 128 | TPOT, decode kernels |

Commands (**not yet run**):

```bash
# pegainfer
cargo run -r --bin bench_serving -- \
  --model-path /data/models/DeepSeek-V3.2 \
  --dsv32-device-ordinals 0,1,2,3,4,5,6,7 \
  request --prompt-len 2048 --output-len 1

cargo run -r --bin bench_serving -- \
  --model-path /data/models/DeepSeek-V3.2 \
  --dsv32-device-ordinals 0,1,2,3,4,5,6,7 \
  request --prompt-len 1 --output-len 128

# vLLM baseline (on the same 8-card NVLink node)
vllm bench serve --model /data/models/DeepSeek-V3.2 --input-len 2048 --output-len 1
vllm bench serve --model /data/models/DeepSeek-V3.2 --input-len 1   --output-len 128
```

Single concurrency. See `docs/resources/bench-vs-vllm.md` for vLLM setup specifics.

---

## 8. E2E Dashboard

GPU: 8-card NVLink node. Model: DeepSeek-V3.2 FP8.

| Profile | Metric | pegainfer | vLLM | delta |
|---------|--------|-----------|------|-------|
| prefill-heavy (2048,1) | TTFT median | TBD | TBD | — |
| prefill-heavy (2048,1) | TTFT p99 | TBD | TBD | — |
| decode-heavy (1,128) | TPOT median | TBD | TBD | — |
| decode-heavy (1,128) | TPOT p99 | TBD | TBD | — |

Current smoke timings (single request, TP1+DP1+EP8, not a benchmark):

| Prompt | max_tokens | Wall time |
|--------|-----------|-----------|
| `hello world` | 8 | ~1.78 s |
| `1+1=` | 8 | ~1.81 s |
| reasoning-style prompt | 16 | ~5.84 s |

Exploratory benchmark on H20 (single request, synthetic prompt, `prompt_len=64`, `output_len=64`):

| Run | warmup / iters | load_ms | TTFT | first decode step | steady TPOT | E2E | decode tok/s |
|-----|----------------|---------|------|-------------------|-------------|-----|--------------|
| #0 baseline benchmark | `1 / 1` | `57738.68` | `7084.10 ms` | `192.15 ms` | `180.66 ms` | `18477.46 ms` | `5.53` |
| #0 baseline + `nsys` | `0 / 1` | `58434.80` | `7343.73 ms` | `211.01 ms` | `189.83 ms` | `19323.90 ms` | `5.26` |
| #1 grouped-local-expert benchmark | `1 / 1` | `60358.22` | `758.10 ms` | `65.37 ms` | `64.13 ms` | `4799.54 ms` | `15.59` |
| #1 grouped-local-expert + `nsys` | `0 / 1` | `56093.27` | `968.26 ms` | `83.60 ms` | `75.51 ms` | `5733.46 ms` | `13.22` |
| #2 batch gather/scatter benchmark | `1 / 1` | `57609.95` | `283.72 ms` | `57.61 ms` | `54.51 ms` | `3720.82 ms` | `18.33` |
| #2 batch gather/scatter + `nsys` | `0 / 1` | `56648.36` | `426.08 ms` | `72.44 ms` | `58.12 ms` | `4101.88 ms` | `17.14` |
| #3 order-preserving gather/scatter benchmark | `1 / 1` | `58647.12` | `452.17 ms` | `54.46 ms` | `53.46 ms` | `3821.30 ms` | `18.70` |

Notes:
- `load_ms` is reported separately by `bench_serving`; it is **not** part of TTFT/TPOT.
- The benchmark prompt is synthetic (`token_id = 100 + (idx % 1000)`), so these numbers isolate runtime cost rather than tokenizer/prompt-content effects.
- The #1 change keeps `recv_x` immutable, batches local expert GEMMs per expert in `max_bs`-sized chunks, and accumulates into a separate combine input buffer.
- The #2 change replaces per-token `DtoD gather` and per-token `moe_weighted_add` in the EP local-expert path with batched gather/scatter kernels plus chunk metadata upload.
- The raw #2 numbers are **not** the accepted baseline because that version regressed greedy correctness.
- At this size, the current accepted headline after #3 is: **TTFT is now ~0.45 s for 64 tokens, and steady decode is ~53 ms/token, with `e2e_dsv32_small` back to `44/44` passing.**

### Generation alignment harness (teacher-forced top-K)

Ground truth generation (SGLang, top-20 logprobs):

```bash
.venv/bin/python tools/dsv32_sglang_ref/gen_ref.py \
  --model-path /data/models/DeepSeek-V3.2 \
  --output-dir test_data/dsv32_sglang_ref \
  --prompts-file tools/dsv32_sglang_ref/prompts_generation.json \
  --tensor-parallel-size 8 \
  --top-k 20 \
  --seed 42
```

Teacher-forced top-K regression (pegainfer):

```bash
PEGAINFER_DSV32_MODEL_PATH=/data/models/DeepSeek-V3.2 \
PEGAINFER_DSV32_SGLANG_REF_MANIFEST=test_data/dsv32_sglang_ref/manifest.json \
PEGAINFER_DSV32_DEVICE_ORDINALS=0,1,2,3,4,5,6,7 \
cargo test --release --test e2e_dsv32 -- --ignored --nocapture
```

Small greedy JSON regression (pegainfer, string match only):

```bash
PEGAINFER_DSV32_MODEL_PATH=/data/models/DeepSeek-V3.2 \
PEGAINFER_DSV32_DEVICE_ORDINALS=0,1,2,3,4,5,6,7 \
cargo test --release --test e2e_dsv32_small -- --ignored --nocapture
```

Notes:
- Manifest schema `dsv32_sglang_ref.v1` carries `prompt_token_ids`, `generated_token_ids`, and per-output-position top-20 `(token_id, logprob)` pairs; token-level regression is a strict corollary since `output_top_logprobs[i][0][0] == generated_token_ids[i]`.
- `test_data/DeepSeek-V3.2.json` is the checked-in regression fixture. It is generated by running pegainfer greedy decoding over the committed 44-prompt corpus; regenerate it with `cargo test --release --test regen_test_data_dsv32 -- --ignored --nocapture` on H20.
- `test_data/dsv32_sglang_ref/manifest.json` is intentionally **not** checked in. Regenerate it locally on H20 when deeper logprob alignment work is needed.
- The teacher-forced test is report-only by default (prints argmax mismatches + |Δlogprob| stats). Thresholds `PEGAINFER_DSV32_LOGPROB_MAX_ABS` and `PEGAINFER_DSV32_ARGMAX_MISMATCHES` are opt-in.
- Prompt set lives in `tools/dsv32_sglang_ref/prompts_generation.json` (44 cases, cross-domain + music/creative).
- `max_new_tokens` is intentionally high for thinking-style prompts; sglang still stops early on EOS/stop.
- Local `pegainfer/.venv` must first install sglang: `uv pip install -p .venv/bin/python -e ../sglang/python`.
- H20 result (2026-04-19): `44/44` cases, `6888` positions, `argmax_mismatches=209 (3.03%)`, `argmax_outside_topk=5`, `top-K overlap mean=18.59/20`, runtime `1368.02s`.
- H20 greedy JSON regression result after the order-preserving EP fix (2026-04-19): `44/44` cases passed in `394.98s`.

---

## 9. Optimization Log

Append-only. See `docs/resources/model-optimization-pipeline.md` for entry format.

### #0 Baseline — H20 `64 -> 64` benchmark + `nsys` (2026-04-19)

Command:

```bash
cargo run --release --bin bench_serving -- \
  --model-path /data/models/DeepSeek-V3.2 \
  --dsv32-device-ordinals 0,1,2,3,4,5,6,7 \
  request --prompt-len 64 --output-len 64 --warmup 1 --iters 1
```

Profile command:

```bash
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --force-overwrite=true --export=sqlite \
  -o target/profiling/dsv32_64_64 \
  target/release/bench_serving \
  --model-path /data/models/DeepSeek-V3.2 \
  --dsv32-device-ordinals 0,1,2,3,4,5,6,7 \
  request --prompt-len 64 --output-len 64 --warmup 0 --iters 1
```

Observed metrics:
- TTFT: `7.08-7.34 s`
- first decode step: `192-211 ms`
- steady TPOT: `180.66-189.83 ms/token`
- E2E: `18.48-19.32 s`
- decode throughput: `5.26-5.53 tok/s`
- model load: `57.7-58.4 s` (reported separately from TTFT)

`nsys` kernel summary (`target/profiling/dsv32_64_64.sqlite`):
- `deep_ep::intranode::cached_notify_combine`: `59.8%`
- `deep_gemm::sm90_fp8_gemm_1d2d_impl`: `33.6%`
- `scale_1x128_kernel`: `1.2%`
- `deep_ep::intranode::combine`: `0.7%`
- `deep_ep::intranode::notify_dispatch`: `0.6%`
- `deep_ep::intranode::dispatch`: `0.5%`
- `FlashMLA sparse prefill` kernel: `0.1%`
- `indexer_fused_score_topk_kernel`: `0.2%`

`nsys` CUDA API summary:
- `cuMemcpyHtoDAsync_v2`: `49.3%`
- `cuStreamSynchronize`: `30.1%`
- `cuMemAllocAsync`: `8.0%`
- `cudaLaunchKernel`: `7.5%`
- `cuMemcpyDtoHAsync_v2`: `1.3%`

Performance findings:
- **TTFT at `prompt_len=64` is not attention-bound.** Sparse prefill attention (`flash_mla_sparse_prefill`) and NSA indexer top-k are present, but each is only a small fraction of total GPU time. The large wall time is not explained by FlashMLA itself.
- **Prefill MoE is currently reusing the decode-oriented EP path.** In the sparse prefill layer path, `forward_layer_prefill_sparse()` ends up in `forward_moe()`, which for EP8 immediately calls `forward_moe_ep()` (`src/model/dsv32/forward.rs:1252`, `src/model/dsv32/forward.rs:1744`).
- **`forward_moe_ep()` is structurally serialized for prefill.** The function does a host-visible sync to read `num_recv_tokens`, copies routing metadata back to host, then iterates `for tok in 0..num_recv_tokens` and runs expert FFN with `seq_len = 1` (`src/model/dsv32/forward.rs:1959`, `src/model/dsv32/forward.rs:2030`, `src/model/dsv32/forward.rs:2045`, `src/model/dsv32/forward.rs:2090`). The inline comment even says: “For decode (small num_recv_tokens), we process per-token sequentially.” The same function is used during prefill, so batched prefill MoE is effectively falling back to per-token expert execution.
- **DeepEP overhead is paid on every MoE layer in both prefill and decode.** The `cached_notify_combine`/`notify_dispatch`/`dispatch`/`combine` kernels each appear `29,696` times in the `64 -> 64` trace. That count matches `58 MoE layers * (1 prefill + 63 decode steps) * 8 ranks`, i.e. the communication path is exercised for every MoE layer of the entire request.
- **The current bottleneck hierarchy is therefore:**
  1. EP8 communication/control (`cached_notify_combine` especially)
  2. FP8 GEMM volume in MoE/dense projections
  3. Host sync / DTOH control traffic in `forward_moe_ep`
  4. Attention/indexer, which are not the first-order explanation for `TTFT ~7 s` at `64` tokens

### #1 Grouped Local Experts In Prefill EP Path — H20 `64 -> 64` recheck (2026-04-19)

Code change:
- `forward_moe_ep()` no longer overwrites `recv_x` in-place while computing local experts.
- The received activations stay immutable in `ep_recv_x`.
- Local routed outputs accumulate into a separate `ep_local_out`.
- Local experts are executed per expert in `max_bs`-sized chunks instead of `seq_len=1` per token.

Command:

```bash
cargo run --release --bin bench_serving -- \
  --model-path /data/models/DeepSeek-V3.2 \
  --dsv32-device-ordinals 0,1,2,3,4,5,6,7 \
  request --prompt-len 64 --output-len 64 --warmup 1 --iters 1
```

Profile command:

```bash
/usr/local/cuda/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --force-overwrite=true --export=sqlite \
  -o target/profiling/dsv32_64_64_grouped \
  target/release/bench_serving \
  --model-path /data/models/DeepSeek-V3.2 \
  --dsv32-device-ordinals 0,1,2,3,4,5,6,7 \
  request --prompt-len 64 --output-len 64 --warmup 0 --iters 1
```

Observed metrics:
- TTFT: `0.76-0.97 s`
- first decode step: `65-84 ms`
- steady TPOT: `64.13-75.51 ms/token`
- E2E: `4.80-5.73 s`
- decode throughput: `13.22-15.59 tok/s`
- model load: `56.1-60.4 s` (reported separately from TTFT)

Delta vs #0:
- TTFT: `7.5x-9.3x` faster
- steady TPOT: `2.5x-2.9x` faster
- E2E: `3.4x-3.9x` faster

`nsys` kernel summary (`target/profiling/dsv32_64_64_grouped.sqlite`):
- `deep_ep::intranode::cached_notify_combine`: `45.9%`
- `deep_gemm::sm90_fp8_gemm_1d2d_impl`: `32.5%`
- `deep_ep::intranode::notify_dispatch`: `3.0%`
- `deep_ep::intranode::combine`: `2.5%`
- `moe_weighted_add_kernel`: `2.4%`
- `deep_ep::intranode::dispatch`: `2.1%`
- `FlashMLA dense decode`: `1.4%`
- `FlashMLA sparse prefill`: `0.5%`
- `indexer_fused_score_topk_kernel`: `0.7%`

`nsys` CUDA API summary:
- `cuMemcpyHtoDAsync_v2`: `76.8%`
- `cuStreamSynchronize`: `8.9%`
- `cuMemAllocAsync`: `4.3%`
- `cudaLaunchKernel`: `3.1%`
- `cuMemcpyDtoDAsync_v2`: `1.7%`
- `cuMemcpyDtoHAsync_v2`: `1.5%`

Performance findings:
- **The first-order TTFT regression was prefill MoE structure, and this change removed most of it.** The old path ran routed experts token-by-token inside `forward_moe_ep()`. After switching to grouped per-expert batches, `64 -> 64` TTFT fell from `~7.1-7.3 s` to `~0.76-0.97 s`.
- **DeepEP control/communication is still the top GPU cost.** `cached_notify_combine` remains the largest kernel at `45.9%`, with `notify_dispatch`/`dispatch`/`combine` still present on every MoE layer. The optimization improved compute structure, but it did not change the communication shape.
- **FP8 GEMM is now a clearer steady-state compute hotspot.** `deep_gemm::sm90_fp8_gemm_1d2d_impl` still takes `32.5%` of GPU kernel time; this is now a more meaningful next compute target because the per-token prefill serialization is no longer dominating wall time.
- **API-side H2D traffic is still too large.** `cuMemcpyHtoDAsync_v2` grows to `76.8%` in the profiled run, while `cuMemcpyDtoHAsync_v2` drops to `1.5%`. Directionally, the next likely wins are host-built metadata / per-step uploads rather than another round of MoE local batching.
- **Attention remains secondary at this shape.** FlashMLA sparse prefill is still only `0.5%` of kernel time in the profiled `64 -> 64` run; this is not where the remaining seconds are hiding.

### #2 Batch Gather/Scatter In EP Local Experts — H20 `64 -> 64` recheck (2026-04-19)

Change:
- Keep the host-side `tokens_per_expert` bucketing for now, but replace the per-token `memcpy_dtod()` gather loop and per-token `moe_weighted_add_cuda()` accumulation loop inside `run_grouped_local_experts_from_recv()` with batched CUDA helpers:
  - `moe_gather_rows_cuda(dst, src, token_indices, hidden_size, num_rows)`
  - `moe_scatter_weighted_add_rows_cuda(dst, src, token_indices, weights, hidden_size, num_rows)`
- Chunk token indices and weights are uploaded once per expert chunk, then each chunk runs:
  1. one batched gather
  2. shared FP8 quantize + gate/up/down GEMMs
  3. one batched weighted scatter-add

Observed metrics:
- benchmark (`warmup=1`, `iters=1`):
  - TTFT: `283.72 ms`
  - first decode step: `57.61 ms`
  - steady TPOT: `54.51 ms/token`
  - E2E: `3720.82 ms`
  - decode throughput: `18.33 tok/s`
- benchmark + `nsys` (`warmup=0`, `iters=1`):
  - TTFT: `426.08 ms`
  - first decode step: `72.44 ms`
  - steady TPOT: `58.12 ms/token`
  - E2E: `4101.88 ms`
  - decode throughput: `17.14 tok/s`
  - model load: `56.6-57.6 s` (reported separately from TTFT)

Compared with #1:
- TTFT: `2.3x-3.4x` faster
- steady TPOT: `1.1x-1.3x` faster
- decode throughput: `~1.2x-1.3x` higher

`nsys` kernel summary (`target/profiling/dsv32_64_64_gather_scatter_real.sqlite`):
- `deep_gemm::sm90_fp8_gemm_1d2d_impl`: `42.5%`
- `deep_ep::intranode::cached_notify_combine`: `33.5%`
- `deep_ep::intranode::combine`: `3.3%`
- `deep_ep::intranode::notify_dispatch`: `2.6%`
- `deep_ep::intranode::dispatch`: `2.3%`
- `moe_scatter_weighted_add_rows_kernel`: `0.3%` (`36,989` instances)
- `moe_gather_rows_kernel`: `0.3%` (`36,989` instances)
- previous per-token `moe_weighted_add_kernel` is no longer part of the hot path for EP8 runs

`nsys` CUDA API summary:
- `cuMemcpyHtoDAsync_v2`: `82.4%` (`271,930` calls)
- `cuStreamSynchronize`: `8.1%` (`45,669` calls)
- `cuMemAllocAsync`: `3.7%` (`105,280` calls)
- `cudaLaunchKernel`: `2.4%` (`1,011,544` calls)
- `cuMemcpyDtoHAsync_v2`: `1.3%` (`31,418` calls)
- `cuMemcpyDtoDAsync_v2`: effectively gone from the hot path (`8` calls total)

What this means:
- **This is the first `64 -> 64` profile where GEMM is clearly the top kernel family.** `fp8_gemm` moved ahead of `cached_notify_combine` (`42.5%` vs `33.5%`), which matches the expected direction for a healthier compute path.
- **The local-expert batching overhead is now much smaller.** The old EP path paid hundreds of thousands of tiny `DtoD` copies plus per-token weighted-add kernels. After replacing them with batched gather/scatter, the EP local compute path stopped dominating both TTFT and TPOT.
- **The next likely limiter is chunk metadata H2D, not DtoD anymore.** `cuMemcpyDtoDAsync_v2` collapsed to `8` calls, but `cuMemcpyHtoDAsync_v2` grew to `271,930` calls because every expert chunk now uploads token indices and weights. The wall-clock win is still strongly positive, but the next cleanup target is obvious.
- **DeepEP combine is still substantial, but it is no longer the first-order blocker.** With GEMM already at the top, the remaining work is more about reducing chunk-control overhead and then revisiting deeper EP routing / grouped-GEMM integration.

### #3 Preserve top-k accumulation order in EP local expert batching (2026-04-19)

Root cause:
- The first gather/scatter version changed routed-expert accumulation from legacy **token-major, top-k-slot-major** order to **expert-major** order.
- In the old path, each recv token was updated as `k=0 -> k=1 -> ... -> k=topk-1`.
- In the new expert-major path, the same token's contributions were revisited later as each expert bucket was processed.
- The scatter kernel updates `dst` with a bf16 read-modify-write on every contribution (`moe_scatter_weighted_add_rows_kernel`), so this reordering is not numerically equivalent. The decisive line is `dst = bf16(dst + weight * src)` in [moe.cu](/data/code/workspace-rustllm/pegainfer/csrc/moe.cu:226).
- Result: the raw #2 path was fast, but it caused widespread greedy drift (`40/44` mismatches in `e2e_dsv32_small`).

Fix:
- Keep batched gather/scatter, but rebuild the host bucketing loop to preserve the legacy accumulation order.
- The accepted path now iterates **top-k slot first**, then batches all tokens for the local expert inside that slot.
- This preserves per-token accumulation as `slot 0 -> slot 1 -> ...`, while still batching GEMMs and gather/scatter within each slot.

Observed metrics:
- benchmark (`warmup=1`, `iters=1`):
  - TTFT: `452.17 ms`
  - first decode step: `54.46 ms`
  - steady TPOT: `53.46 ms/token`
  - E2E: `3821.30 ms`
  - decode throughput: `18.70 tok/s`
- correctness:
  - single-case A/B on `en_capital_fr`: `legacy` passed, raw expert-major `grouped` failed, order-preserving `grouped` passed
  - full `e2e_dsv32_small`: `44/44` passing on H20 in `394.98s`

What this means:
- **The bug was accumulation-order drift, not an FFI or layout bug.**
- **The final accepted batching path keeps almost all of the #2 speedup.** Compared with the broken raw #2 benchmark, TTFT rose from `283.72 ms` to `452.17 ms`, but steady TPOT stayed slightly better at `53.46 ms/token`.
- **Correctness is back at the committed regression level.** The accepted baseline is now the #3 row above, not the raw #2 row.

---

## 10. Key Design Decisions

Decisions that still shape the current codebase. Historical bring-up decisions are in `docs/archives/dsv32-bringup-log.md`.

| Decision | Reason |
|----------|--------|
| FP8 is mandatory | BF16 1.34 TB > 8-card 1.1 TB total HBM |
| TP1 + DP1 + EP8 as current shape | Attention compute is small vs MoE; duplicating attention/dense weights is simpler than sharding at this stage. DP scale-out and TP≥2 remain follow-up work. |
| DeepGEMM SM90 1D2D for all FP8 GEMMs | Block-scale FP8 matches DSV3.2 checkpoint layout natively; SGLang and vLLM both use this path on Hopper |
| FlashMLA for attention | DeepSeek-native MLA kernel, dimensions (d_qk=576, d_v=512) match DSV3.2 exactly; dense decode + sparse prefill both supported |
| Decode currently uses dense attention | Bring-up chose FlashMLA dense decode first. This does not yet match model-native NSA sparse decode semantics (indexer top-k), and remains a correctness/quality risk item. |
| DeepEP intranode for MoE All-to-All | Single-box 8-card over NVLink; `-DDISABLE_NVSHMEM` drops RDMA entirely |
| Absorption path (not unabsorbed MLA) | 128× smaller K/V footprint in cache; required to fit long context; cost is a CPU-side dequant of kv_b_proj at load (W_UK/W_UV, ~32 MB/layer × 61 ≈ 2 GB, one-time) |
| cuBLAS strided batched GEMM for absorption / de-absorption | Plain bf16 batched GEMM, no TMA or FP8; cuBLAS already initialized and performs well at these shapes |
| Per-layer paged KV cache with page_size=64 | FlashMLA dense decode requires exactly this layout |
| MLA RoPE uses transformers-style `rotate_half` (NOT interleaved pairs) | The bring-up bug: we initially applied interleaved RoPE and logits didn't align. Matching the reference implementation fixed it. |
| Serial scheduler | Current scope is correctness, not throughput |

---

## 11. Implementation Notes

Reference material for anyone touching the DSV3.2 kernel integrations. Kept in the area doc (not archive) because this is what you need to read *before* editing the corresponding code.

### 11.1 MLA dimensions quick-reference

| Name | Value | Meaning |
|------|-------|---------|
| hidden_size | 7168 | |
| num_heads | 128 | |
| q_lora_rank | 1536 | Q low-rank compressed dim |
| kv_lora_rank | 512 | KV low-rank compressed dim (== d_v in FlashMLA) |
| qk_nope_head_dim | 128 | Per-head Q/K non-RoPE dim |
| qk_rope_head_dim | 64 | Per-head Q/K RoPE dim |
| v_head_dim | 128 | Per-head V dim |
| q_head_dim | 192 | nope + rope |
| kv_a_proj_dim | 576 | kv_lora_rank + qk_rope_head_dim (== d_qk in FlashMLA) |

### 11.2 MLA absorption — the math

Standard MLA attention per head `h`:

```
score_h = (q_nope_h @ k_nope_h^T) + (q_rope_h @ k_rope^T)
        = (q_nope_h @ (W_UK_h @ c_kv)^T) + (q_rope_h @ k_rope^T)
        = (q_nope_h @ W_UK_h) @ c_kv^T + q_rope_h @ k_rope^T
```

Define `q_absorbed_h = [q_nope_h @ W_UK_h, q_rope_h]` (576d), `kv_cache = [c_kv, k_rope]` (576d). Then:

```
score_h = q_absorbed_h @ kv_cache^T    ← one dot product, FlashMLA computes directly
```

V side: FlashMLA outputs `attn_out_h` of dim `d_v = kv_lora_rank = 512`; we recover per-head V with `v_out_h = attn_out_h @ W_UV_h^T` → 128d.

**W_UK / W_UV extraction** from FP8 `kv_b_proj [32768, 512]`:

```
kv_b_proj_bf16 [32768, 512] → reshape [128 heads, 256, 512]
W_UK_h = kv_b_proj_bf16[h, 0:128,   :]   # [128, 512] — K nope
W_UV_h = kv_b_proj_bf16[h, 128:256, :]   # [128, 512] — V
```

### 11.3 FlashMLA call parameters

```
flash_mla_decode(
    q:      [bs, 128, 1, 576],    // q_seq_per_hk = 1 * (128/1) = 128
    kcache: [num_blocks, 64, 1, 576],
    o:      [bs, 1, 128, 512],
    h_q=128, h_k=1, d_k=576, d_v=512,
    softmax_scale = (192)^(-0.5) * yarn_mscale^2,
    is_causal = 0
)
```

Key strides (see `csrc/flash_mla.cu`):
```
q_batch_stride = 128 * 1 * 576 = 73728
q_row_stride   = 576
q_head_stride  = 576
o_batch_stride = 1 * 128 * 512 = 65536
o_row_stride   = 512
o_head_stride  = 128 * 512 = 65536
```

### 11.4 DeepGEMM 1D2D vs 1D1D

| | 1D1D | 1D2D |
|---|---|---|
| Scale A (activation) | 1D per-token `[ceil(K/128), padded(M,4)]` | same |
| Scale B (weight) | 1D per-channel | **2D per-block `[ceil(N/128), ceil(K/128)]`** |
| SFB load | TMA | **global `__ldg()` from math warp** |
| Output | FP32 | **BF16** |
| D TMA desc | no swizzle, FP32 | **128B swizzle, BF16** |

**DSV3.2 uses 1D2D** exclusively. Checkpoint scales are already 2D block-scale, which matches `kMajorSFB = Major::K` natively. SGLang and vLLM follow the same choice on Hopper.

Two AOT tile configs in `csrc/fp8_gemm.cu`:
- `block_m=64, block_n=128, 8 stages` — small-M (decode-ish).
- `block_m=128, block_n=128, 5 stages` — large-M (prefill).

Both fit SM90 smem budget (232448 bytes); 1D2D has one more stage than 1D1D at the same block size because D halves (bf16) and there is no per-stage SFB buffer.

### 11.5 DeepEP intranode — what `dispatch` / `combine` do

```
Step 1: get_dispatch_layout(topk_idx, num_experts=256)
        → num_tokens_per_rank [8]
        → is_token_in_rank [num_tokens, 8]
        → num_tokens_per_expert [256]

Step 2: notify_dispatch(...)
        NVLink barrier + per-rank size exchange via IPC buffer.
        CPU busy-waits on mapped host memory for recv count
        → not CUDA Graph compatible by design.

Step 3: intranode::dispatch(x, topk_idx, topk_weights, is_token_in_rank, ...)
        → recv_x, recv_topk_idx, recv_topk_weights, recv_src_idx, handle

Step 4: [local expert FFN compute]

Step 5: intranode::combine(x_out, topk_weights, src_idx, handle, ...)
        → combined_x (weighted reduce back to original ranks)
```

Buffer management (host side):
- Each rank `cudaMalloc`s an NVLink buffer (size from `Config::get_nvl_buffer_size_hint()`, ~300 MB/card at hidden=7168).
- `cudaIpcGetMemHandle` → exchange across ranks (we use Rust thread channels, not `torch.distributed`) → `cudaIpcOpenMemHandle` → `buffer_ptrs[8]`.
- Same pattern for barrier signal buffers.
- Mapped host memory (`cudaHostAlloc` + `cudaHostGetDevicePointer`) for CPU-GPU sync counters.

Build flags: `-DDISABLE_NVSHMEM -DTOPK_IDX_BITS=64`, SM90a. `kNumRanks=8` AOT-instantiated via `LAUNCH_DISPATCH` macro.

Config for EP8:
```
num_sms = 20   (10 channels, 2 SMs each)
num_max_nvl_chunked_send_tokens = 6
num_max_nvl_chunked_recv_tokens = 256
```

### 11.6 Infrastructure reuse map

| Operation | Function | Location |
|-----------|----------|----------|
| Embedding lookup (batched) | `ops::embedding_batch` | `src/ops.rs` |
| RMSNorm batched / fused-add | `ops::rms_norm_batch_into` / `fused_add_rms_norm_batch_into` | `src/ops/norm.rs`, `csrc/flashinfer_norm.cu` |
| SiLU × up | `ops::silu_mul_batch_into` | `src/ops.rs` |
| BF16 GEMM | `ops::gemm_into` | `src/ops.rs` |
| FP8 quantize + GEMM | `ops::fp8::fp8_linear_into` / `fp8_quantize_into` + `fp8_gemm_into` | `src/ops/fp8.rs` |
| FlashMLA 3-phase | `ffi::flash_mla_{get_metadata,decode,combine}` | `src/ffi.rs` |
| FlashMLA sparse prefill | `ffi::flash_mla_sparse_prefill` | `src/ffi.rs` |
| Strided batched GEMM | `ffi::gemm_strided_batched_cuda` | `csrc/linear.cu` |
| MLA RoPE (q_rope extract+copy, k_rope) | `ffi::mla_rope_q_copy_cuda` / `mla_rope_kv_cuda` | `csrc/mla.cu` |
| Partial RMSNorm (c_kv only) | `ffi::rms_norm_partial_cuda` | `csrc/mla.cu` |
| KV cache write (scatter) | `ffi::mla_kv_cache_write_cuda` | `csrc/mla.cu` |
| Token sampling (DSV3.2 service path) | `sample_from_logits` | `src/scheduler_dsv32.rs` |
| YaRN RoPE cos/sin precompute | `precompute_yarn_rope` | `src/model/dsv32/weights.rs` |
| DeepEP 5 functions + IPC helpers | `ffi::deep_ep_*` | `src/ffi.rs`, `src/model/dsv32/deep_ep.rs` |

---

## 12. Open Work

- [ ] `benches/dsv32_*` — micro benchmarks (FP8 GEMM, FlashMLA, DeepEP).
- [ ] `#0 Baseline` — nsys kernel breakdown for both profiles.
- [ ] Decide default pass/fail thresholds for `e2e_dsv32` from the current H20 baseline instead of leaving it report-only.
- [ ] Cross-machine recheck: regenerate `dsv32_sglang_ref` on one H20 node and replay `e2e_dsv32` on a second node.
- [ ] Grouped GEMM for routed experts (DeepGEMM `GroupedContiguous`) — replace per-expert sequential.
- [ ] CUDA Graph capture for decode.
- [ ] TP2 attention sharding.
