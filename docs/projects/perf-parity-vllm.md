# Performance Parity With vLLM

> **TL;DR:** Decode is at parity. Prefill is 2x behind vLLM. Goal: match or exceed vLLM on all workloads.
>
> **Status:** Active. Next action: investigate prefill GEMM strategy.

## Goal

pegainfer single-request latency >= vLLM on the same GPU, model, and workload. No regressions allowed — decode must stay at parity while prefill catches up.

## Baseline (2026-03-13)

GPU: RTX 5070 Ti, Model: Qwen3-4B, vLLM 0.17.1, single concurrency.

### in=1024, out=256 — Realistic workload

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 124.66ms | 118.23ms | +5% slower |
| TTFT p99 | 350.95ms | 290.27ms | +21% |
| TPOT median | 11.18ms | 11.51ms | **-3% faster** |
| TPOT p99 | 11.19ms | 11.61ms | **-4% faster** |
| Output tok/s | 85.61 | 83.50 | **+2.5%** |

Verdict: parity. Decode slightly ahead, prefill slightly behind.

### in=1, out=512 — Pure decode

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 42.45ms | 21.55ms | +97% slower |
| TPOT median | 10.56ms | 11.46ms | **-8% faster** |
| TPOT p99 | 10.58ms | 11.46ms | **-8% faster** |
| Output tok/s | 94.13 | 87.33 | **+7.8%** |

Verdict: **decode wins.** pegainfer 8% faster TPOT on sustained decode. TTFT overhead same as in=1/out=1.

### in=2048, out=32 — Prefill heavy

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 269.12ms | 133.00ms | **+102% slower** |
| TTFT p99 | 276.03ms | 230.96ms | +20% |
| TPOT median | 11.83ms | 11.59ms | +2% |
| TPOT p99 | 11.84ms | 11.64ms | +2% |

Verdict: **prefill is 2x behind.** This is the main gap.

### in=1, out=1 — Minimal overhead

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 41.89ms | 19.29ms | **+117% slower** |
| TTFT p99 | 42.89ms | 20.27ms | +112% |
| req/s | 23.78 | 51.43 | -54% |

Verdict: **fixed overhead is 2x higher.** Likely CUDA Graph capture or first-token path cost.

