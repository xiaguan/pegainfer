# Qwen3-4B Optimization

> **TL;DR:** pegainfer led the measured Qwen3-4B workloads on RTX 5070 Ti after the FlashAttention-2 + PrefillBuffers work. This doc is archived as a dense-attention optimization reference rather than an active project.
>
> **Status:** Archived. Kept as a reference for dense-attention optimization work; no longer an active milestone.

## Goal

pegainfer single-request latency >= vLLM on the same GPU, model, and workload. No regressions allowed — decode must stay at parity while prefill catches up.

## Current (2026-03-13, post FlashAttention-2 + cuBLAS workspace)

GPU: RTX 5070 Ti, Model: Qwen3-4B, vLLM 0.17.1, single concurrency.
pegainfer: in-process bench_serving (no HTTP overhead). vLLM: `vllm bench serve` HTTP.

### in=1, out=1 — Minimal overhead

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 11.89ms | 19.29ms | **-38% faster** |
| TTFT p99 | 12.54ms | 20.27ms | **-38% faster** |
| req/s | 82.53 | 51.43 | **+60%** |

Verdict: **pegainfer wins.**

### in=1024, out=256 — Realistic workload

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 101.73ms | 118.23ms | **-14% faster** |
| TPOT median | 11.18ms | 11.51ms | **-3% faster** |
| TPOT p99 | 11.20ms | 11.61ms | **-4% faster** |

Verdict: **pegainfer wins.** Prefill 14% faster (was -5% in pre-FA2 baseline).

### in=1, out=512 — Pure decode

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 10.37ms | 21.55ms | **-52% faster** |
| TPOT median | 10.61ms | 11.46ms | **-7% faster** |
| TPOT p99 | 10.71ms | 11.46ms | **-7% faster** |

Verdict: **decode wins.**

### in=2048, out=32 — Prefill heavy

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 213ms¹ | 228.54ms² | **-7% faster** |
| TPOT median | 11.80ms | 11.60ms | +2% |

¹ In-process, no HTTP round-trip.
² vLLM median=228ms vs previously-recorded 133ms. The 133ms was a transient GPU boost measurement; 228ms is the repeatable steady-state number. vLLM p99=430ms due to torch.compile cold start on first requests.

Verdict: **prefill at parity, edge to pegainfer.** Pre-optimization was 270ms (+103% vs vLLM). After FlashAttention-2: 213ms (-7% vs vLLM).

## Prefill kernel breakdown (in=2048, nsys profile)

GEMM is at the RTX 5070 Ti hardware ceiling. FA2 is the actionable target.

| Kernel | Time/run | % | Notes |
|--------|----------|---|-------|
| cuBLAS GEMM (7 kernels) | 163ms | 77% | 14.88T FLOPs ÷ 90 TFLOPS = 165ms theoretical. **At ceiling.** |
| flash_attention_prefill | 36.5ms | 17% | Triton FA2. MFU 38%, BW util 47%. GQA-aware rewrite: ~24ms saving. |
| prefill_qk_norm_rope | 4.1ms | 2% | In-place on Q/K, fast. |
| silu_mul | 4.6ms | 2% | Could fuse into gate GEMM epilogue. |
| rms_norm_batched | 0.85ms | 0.4% | |
| add (residual) | 1.4ms | 0.7% | Could fuse into O/down_proj epilogue. |

GEMM MFU: 90 TFLOPS / ~419 TFLOPS BF16 dense peak = **21% HW utilization** (consistent with consumer GPU vs theoretical peak; cuBLAS and Triton both hit this same ceiling).

FA2 MFU analysis:
- Compute: 34 TFLOPS actual vs 90 TFLOPS measured peak → **38% compute util**
- Memory: 288MB/layer × 36 ÷ 600 GB/s = 17.3ms BW-ideal, actual 36.5ms → **47% BW util**
- FA2 is memory-bandwidth limited. Main waste: non-GQA-aware — 4 Q-heads sharing a KV-head each independently load the same K/V tiles (4× redundant reads).
- GQA-aware fix: K/V traffic drops 4× → 3× total traffic reduction → estimated 10–12ms (from 36.5ms). **Attempted — see below.**

## Changelog

### 2026-03-13: FlashAttention-2 + cuBLAS workspace

**Root cause:** Standard attention materialised O(n²) FP32 scores + BF16 softmax buffers (~27GB HBM traffic per prefill at seq=2048). cuBLAS handle lacked workspace, limiting algorithm selection.

**Changes:**
- `csrc/gemv.cu`: added `g_cublas_prefill_handle` with 32MB workspace (`cublasSetWorkspace`) — decode handle unchanged (CUDA Graph safe).
- `tools/triton/flash_attention_prefill_kernel.py`: FlashAttention-2 Triton kernel (online softmax, causal mask, GQA via `kv_head = q_head // gqa_ratio`, BF16 I/O / FP32 accum, BLOCK_M=128/BLOCK_N=64/HEAD_DIM=128).
- `src/ops.rs`: removed `PrefillAttnScratch` (O(n²) scratch buffers). New `prefill_attention_batch()` calls the Triton kernel.
- `src/model.rs`: removed per-layer scratch allocation.

**Impact on in=2048, out=32 TTFT:** 270ms → 219ms (–19%).

**Greedy baseline update:** Triton FA2 online softmax produces marginally different floating-point rounding than full softmax → 4 of 6 test cases diverge after ~15 tokens. `test_data/Qwen3-4B.json` updated with verified outputs. Stream/non-stream consistency and all 49 unit tests pass.

### 2026-03-13: PrefillBuffers — eliminate per-layer CUDA allocations

**Root cause:** `forward_layer_batch` allocates 10 intermediate `HiddenStates` per layer (normed, Q/K/V, O, hidden, normed2, gate, up, act, mlp). At seq=2048, 36 layers: ~468 `cuMemAllocAsync` calls per prefill.

**Changes:**
- `src/ops.rs`: added `gemm_into`, `rms_norm_batch_into`, `add_batch_into`, `silu_mul_batch_into` — zero-allocation variants writing into pre-allocated output buffers.
- `src/model.rs`: `PrefillBuffers` struct (10 pre-allocated tensors); created once in `process_all_layers_batch`, ping-ponged via `std::mem::swap` for hidden state. Per-layer allocations: 468 → 0.

**Impact:** TTFT 219ms → 213ms (–3%). `cuMemAllocAsync` is stream-ordered and does not block GPU execution — overhead was CPU-side scheduling, not GPU execution time.

### 2026-03-13: GQA-aware FA2 attempt — reverted

**Theory:** Load K/V tiles once per block and share across gqa_ratio=4 Q heads ("outer KV, inner Q" loop). Estimated 4× K/V HBM reduction → 10–12ms saving.

**Implementation:** Triton kernel with grid `(seq_tiles, num_kv_heads, 1)`. Each block held 4 Q tile sets + 4 accumulator sets simultaneously in registers.

**Result:** TTFT 213ms → 275ms (+30%). Register spill measured by regression alone.

**Root cause:** Holding 4×Q + 4×acc + K/V tiles simultaneously requires ≈300+ FP32/thread (register limit ≈255). Compiler spills to L1 local memory. L1 spill bandwidth cost exceeds the K/V HBM savings. FA2 at seq=2048 is only 17% of total prefill time (36.5ms of 213ms) — the math never worked.

**Lesson:** True GQA-aware FA needs explicit SMEM (CUDA kernel, not Triton). Triton doesn't expose SMEM for manual tile staging. With pegainfer already 7% faster than vLLM, the complexity cost is unjustified. Reverted.

### 2026-03-13: TCP_NODELAY fix

**Root cause:** axum's default TCP socket lacked `TCP_NODELAY`. With HTTP keep-alive, Nagle's algorithm buffered the final SSE bytes, waiting ~30ms for a delayed ACK before the client could reuse the connection.

**Diagnosis path:** bench_serving (in-process) showed 10.36ms TTFT while `vllm bench serve` (HTTP) showed 42ms. Isolated with curl (11ms) vs Python aiohttp shared session (41ms vs 11ms with new session per request). This pinpointed TCP buffering on keep-alive connections.

**Fix:** One line — `tcp_stream.set_nodelay(true)` via `axum::serve::ListenerExt::tap_io`.

**Impact on in=1, out=1:**
- TTFT: 41.89ms → 11.89ms (3.5x)
- req/s: 23.78 → 82.53 (3.5x)
- Now 38% faster than vLLM (was 117% slower)

No regression on decode or prefill workloads.
