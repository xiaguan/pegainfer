# Batch Optimization

> **TL;DR:** Unified forward pass implemented — prefill and decode tokens in one forward pass, decode GEMMs ride free on prefill compute. With varied-length workloads (n=500, c=8), throughput gap vs warmed vLLM is **2%** (previously 28% with fixed-length). Remaining gap is per-token decode efficiency: TPOT 18.28ms vs 17.38ms (+5%), partly from serial sampling overhead (~0.2ms/step recoverable). ITL p99 improved 17% (97→81ms).
>
> **Status:** Active. Unified forward landed. Next: batch sampling to close remaining TPOT gap.

## Baseline (2026-03-30, n=50)

GPU: RTX 5070 Ti, Model: Qwen3-4B, vLLM 0.18.x, `vllm bench serve` as unified client.
pegainfer: HTTP server with continuous batching scheduler + batch prefill. vLLM: default config (`--max-model-len 4096 --gpu-memory-utilization 0.9`).
n=50 per config, `--request-rate inf --ignore-eos`.

**Note on vLLM cold-start:** torch.compile triggers on new shapes. At c=8, mean TTFT can be 10–100× median. Throughput is also affected when cold-start requests dominate elapsed time. Read **median** for steady-state latency; throughput noted where polluted.

### in=1, out=128 — Decode-heavy

| Metric | c | pegainfer | vLLM | delta |
|--------|---|-----------|------|-------|
| TTFT median (ms) | 1 | 13.64 | 20.12 | **−32%** |
| TTFT p99 (ms) | 1 | 14.12 | 126.93 | −89% |
| TPOT median (ms) | 1 | 10.94 | 11.31 | −3% |
| Output tok/s | 1 | 91.17 | 87.33 | +4% |
| Req/s | 1 | 0.71 | 0.68 | +4% |
| Total tok/s | 1 | 91.89 | 88.01 | +4% |
| TTFT median (ms) | 4 | 36.51 | 33.32 | +10% |
| TTFT p99 (ms) | 4 | 40.61 | 67.12 | −40% |
| TPOT median (ms) | 4 | 11.81 | 11.52 | +3% |
| Output tok/s | 4 | 320.45 | 328.76 | −3% |
| Req/s | 4 | 2.50 | 2.57 | −3% |
| Total tok/s | 4 | 322.95 | 331.33 | −3% |
| TTFT median (ms) | 8 | 39.89 | 43.15 | −8% |
| TTFT p99 (ms) | 8 | 45.92 | 7985.39¹ | — |
| TPOT median (ms) | 8 | 12.12 | 11.42 | +6% |
| Output tok/s | 8 | 581.77 | 348.01¹ | +67%¹ |
| Req/s | 8 | 4.55 | 2.72¹ | +67%¹ |
| Total tok/s | 8 | 586.32 | 350.73¹ | +67%¹ |
| ITL p99 (ms) | 8 | 12.74 | 12.41 | +3% |

¹ vLLM c=8 throughput polluted by torch.compile cold-start (p99 TTFT = 7985ms). Estimated steady-state ~610 tok/s based on median TPOT.

**Takeaway:** Decode at parity (TPOT within ±6%). pegainfer TTFT 32% faster at c=1 (lower fixed overhead, no torch.compile). Decode throughput scales 6.4× at c=8 (91→582 tok/s). TPOT increases 11% from c=1→c=8 (10.94→12.12ms) — scheduling overhead, not GPU saturation.

### in=1024, out=1 — Prefill-only

| Metric | c | pegainfer | vLLM | delta |
|--------|---|-----------|------|-------|
| TTFT median (ms) | 1 | 108.21 | 116.65 | **−7%** |
| TTFT p99 (ms) | 1 | 110.83 | 157.04 | −29% |
| Req/s | 1 | 9.25 | 8.49 | +9% |
| Total tok/s | 1 | 9,486 | 8,699 | +9% |
| TTFT median (ms) | 4 | 376.44 | 381.67 | −1% |
| TTFT p99 (ms) | 4 | 384.52 | 399.25 | −4% |
| Req/s | 4 | 10.59 | 10.40 | +2% |
| Total tok/s | 4 | 10,857 | 10,657 | +2% |
| TTFT median (ms) | 8 | 731.18 | 760.19 | **−4%** |
| TTFT p99 (ms) | 8 | 743.33 | 823.51 | −10% |
| Req/s | 8 | 10.88 | 10.45 | +4% |
| Total tok/s | 8 | 11,157 | 10,714 | +4% |

**Takeaway:** Prefill at parity — pegainfer 4–9% faster across all concurrencies. Both engines show linear TTFT scaling with concurrency (108→731ms pega, 117→760ms vLLM), confirming prefill is compute-bound. The previous n=20 data showing vLLM at 65ms/c=4 and 103ms/c=8 was a cold-start artifact — with n=50 both engines behave identically.

### in=512, out=64 — Mixed workload

| Metric | c | pegainfer | vLLM | delta |
|--------|---|-----------|------|-------|
| TTFT median (ms) | 1 | 55.58 | 30.63 | +81% |
| TTFT p99 (ms) | 1 | 61.00 | 76.71 | −20% |
| TPOT median (ms) | 1 | 11.68 | 11.35 | +3% |
| Output tok/s | 1 | 80.85 | 84.02 | −4% |
| Req/s | 1 | 1.26 | 1.31 | −4% |
| Total tok/s | 1 | 728 | 756 | −4% |
| TTFT median (ms) | 4 | 200.01 | 51.92 | **+285%** |
| TTFT p99 (ms) | 4 | 209.36 | 67.71 | +209% |
| TPOT median (ms) | 4 | 12.49 | 11.96 | +4% |
| Output tok/s | 4 | 251.21 | 304.53 | −18% |
| Req/s | 4 | 3.93 | 4.76 | −17% |
| Total tok/s | 4 | 2,261 | 2,741 | −17% |
| ITL p99 (ms) | 4 | 13.00 | 15.15 | **−14%** |
| TTFT median (ms) | 8 | 380.96 | 76.66 | **+397%** |
| TTFT p99 (ms) | 8 | 391.81 | 101.80 | +285% |
| TPOT median (ms) | 8 | 12.92 | 12.21 | +6% |
| Output tok/s | 8 | 396.70 | 548.38 | −28% |
| Req/s | 8 | 6.20 | 8.57 | −28% |
| Total tok/s | 8 | 3,570 | 4,935 | −28% |
| ITL p99 (ms) | 8 | 13.76 | 17.45 | **−21%** |

**Takeaway:** TTFT is the main gap under concurrency — vLLM's chunked prefill holds TTFT nearly flat (31→77ms) while pegainfer scales linearly (56→381ms). Decode throughput: vLLM leads 18–28% at c≥4 because earlier TTFT means earlier decode start. ITL p99: pegainfer is tighter (13–14ms vs 15–17ms) — batch prefill eliminated decode stalls.

## Batch Prefill Impact (before → after)

Comparison of pegainfer before and after batch prefill. "Before" data from n=20 run (2026-03-29, serial prefill). "After" data from n=50 run above.

### in=1024, out=1 — Prefill-only

| Metric | c | before | after | delta |
|--------|---|--------|-------|-------|
| TTFT median (ms) | 1 | 108.03 | 108.21 | 0% |
| TTFT median (ms) | 4 | 382.45 | 376.44 | −2% |
| TTFT median (ms) | 8 | 765.23 | 731.18 | −4% |
| Req/s | 1 | 9.25 | 9.25 | 0% |
| Req/s | 4 | 10.39 | 10.59 | +2% |
| Req/s | 8 | 10.39 | 10.88 | +5% |

Marginal improvement. Prefill is compute-bound — batching into one GEMM doesn't reduce total compute, just scheduling overhead.

### in=512, out=64 — Mixed workload

| Metric | c | before | after | delta |
|--------|---|--------|-------|-------|
| TTFT median (ms) | 1 | 52.97 | 55.58 | +5% |
| TPOT median (ms) | 1 | 11.67 | 11.68 | 0% |
| Output tok/s | 1 | 81.1 | 80.85 | 0% |
| TTFT median (ms) | 4 | 136.16 | 200.01 | **+47%** |
| TPOT median (ms) | 4 | 13.64 | 12.49 | **−8%** |
| Output tok/s | 4 | 257.2 | 251.21 | −2% |
| ITL p99 (ms) | 4 | 61.67 | 13.00 | **−79%** |
| TTFT median (ms) | 8 | 216.79 | 380.96 | **+76%** |
| TPOT median (ms) | 8 | 14.63 | 12.92 | **−12%** |
| Output tok/s | 8 | 372.5 | 396.70 | +6% |
| ITL p99 (ms) | 8 | 83.22 | 13.76 | **−83%** |

**ITL p99 is the headline**: 83→14ms at c=8. Before, each new prefill blocked all in-flight decode for ~50ms. Now, pending prefills are batched together and decode runs uninterrupted between batches.

**TTFT median regressed** because all requests in a batch wait for the entire batch to complete. Before, earlier requests started prefilling immediately (serial FIFO). This is the expected tradeoff of batch-all-then-decode scheduling — fixable with chunked prefill.

### in=1, out=128 — Decode-heavy

| Metric | c | before | after | delta |
|--------|---|--------|-------|-------|
| TTFT median (ms) | 8 | 49.28 | 39.89 | **−19%** |
| TPOT median (ms) | 8 | 12.32 | 12.12 | −2% |
| Output tok/s | 8 | 531.6 | 581.77 | +9% |

Decode-heavy barely affected (prefill is trivial at in=1). Slight improvement from batching 8 trivial prefills into one call.

## Key Observations

1. **Prefill at parity with vLLM.** 4–9% faster across all concurrencies at in=1024. Both scale linearly (compute-bound). Previous n=20 data showing vLLM at 65ms/c=4 was a cold-start artifact — corrected with n=50.

2. **Decode at parity.** TPOT within ±6% at all concurrencies. pegainfer has lower fixed overhead (no torch.compile) giving a 32% TTFT advantage at c=1 for decode-heavy loads.

3. **Batch prefill fixed ITL stalls.** ITL p99 dropped from 83→14ms at c=8. Decode steps no longer blocked by incoming prefills. pegainfer ITL p99 is now tighter than vLLM's (13.76ms vs 17.45ms at c=8).

4. **TTFT under concurrency is the biggest gap.** Mixed workload at c=8: pegainfer 381ms vs vLLM 77ms. Batch-all-then-decode means all requests wait for the full batch prefill to finish. vLLM's chunked prefill issues first tokens incrementally.

5. **Mixed-workload throughput gap follows from TTFT.** At c=8: pega 397 tok/s vs vLLM 548 tok/s (−28%). Earlier TTFT → earlier decode start → more tokens generated in the same wall time.

6. **Decode throughput scales well.** Output tok/s: 91→582 at c=1→c=8 (6.4×). TPOT increases only 11% (10.94→12.12ms) — GPU not saturated at c=8.

## Unified Forward Pass (2026-03-30)

### Why not chunked prefill

The fixed-length benchmark (in=512, out=64, c=8, `request-rate inf`) creates an artificial worst case: all 8 requests finish simultaneously → perfect waves → prefill and decode never overlap. The 28% throughput gap was a scheduling artifact, not a fundamental deficiency. With varied-length requests, waves break up naturally.

The real optimization is what vLLM does at the **kernel level**: mixing prefill and decode tokens in a single forward pass. Decode tokens (batch=8) are memory-bandwidth-bound; when folded into a compute-bound prefill GEMM (512 tokens), the 8 decode rows add <2% FLOPs and ride free.

### Implementation

New `unified_step` on `Qwen3Model`: when the scheduler has both pending prefill and active decode requests, it runs one forward pass for all tokens:

- **GEMM ops** (QKV, O proj, MLP): all tokens in one batch — decode rides free
- **QK norm + RoPE**: unified per-token positions array, same kernel for prefill and decode
- **KV cache write**: scatter for prefill tokens, append for decode tokens (two kernel calls)
- **Attention**: split — FlashInfer BatchPrefill for prefill portion, BatchDecode for decode portion, outputs concatenated
- **Logits**: extract last token per prefill sequence + all decode tokens

Scheduler policy: `pending && active → unified_step`, `pending only → prefill_batch`, `active only → decode_step (CUDA Graph)`. Pure decode (the common case) still uses CUDA Graph.

### Results: varied-length workload

in~256–768, out~32–96, c=8, n=500, `request-rate inf`. vLLM warmed (torch.compile cached), same seed.

| Metric | pega baseline | pega unified | vLLM (warmed) | unified vs vLLM |
|--------|--------------|-------------|---------------|----------------|
| req/s | 6.27 | 6.41 | 6.54 | **−2.0%** |
| output tok/s | 404.68 | 412.79 | 421.66 | **−2.1%** |
| TTFT median (ms) | 71.21 | 78.60 | 108.09 | **−27.3%** ✓ |
| TPOT median (ms) | 18.75 | 18.28 | 17.38 | +5.2% |
| ITL p99 (ms) | 97.70 | 81.04 | 67.59 | +19.9% |

Throughput gap collapsed from 28% (fixed-length) to **2%** (varied-length). TTFT 27% faster than vLLM (no torch.compile overhead).

### Remaining gap: TPOT +5%

nsys profile of pure decode (in=1, out=128, c=8) shows `argmax_kernel` at 8.3% of GPU time — 8 serial sampling calls per step (extract logits → argmax → sync, one per request). Batching into one call would save ~0.2ms/step (1.9%), closing TPOT gap from 5.9% to ~4%.

The remaining ~4% is kernel-level efficiency: vLLM's torch.compile fuses elementwise ops (residual add + RMSNorm → one memory pass), reducing bandwidth pressure. CUDA Graph eliminates launch overhead but not redundant memory traffic from unfused kernels.

## Next Target: Batch Sampling

Current `select_tokens_batch_varied` does 8 serial `extract_vec → gpu_sample_into → sync` calls. Replace with a single batched argmax/sampling kernel + one sync. Estimated TPOT improvement: ~0.2ms (1.9%).
