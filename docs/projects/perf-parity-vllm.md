# Performance Parity With vLLM

> **TL;DR:** Decode and fixed overhead both ahead of vLLM. Prefill is 2x behind. Goal: match or exceed vLLM on all workloads.
>
> **Status:** Active. Next action: investigate prefill GEMM strategy.

## Goal

pegainfer single-request latency >= vLLM on the same GPU, model, and workload. No regressions allowed — decode must stay at parity while prefill catches up.

## Current (2026-03-13, post TCP_NODELAY fix)

GPU: RTX 5070 Ti, Model: Qwen3-4B, vLLM 0.17.1, single concurrency.

### in=1, out=1 — Minimal overhead

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 11.89ms | 19.29ms | **-38% faster** |
| TTFT p99 | 12.54ms | 20.27ms | **-38% faster** |
| req/s | 82.53 | 51.43 | **+60%** |

Verdict: **pegainfer wins.** Fixed overhead eliminated by TCP_NODELAY.

### in=1024, out=256 — Realistic workload

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 124.74ms | 118.23ms | +5% slower |
| TTFT p99 | 308.71ms | 290.27ms | +6% |
| TPOT median | 11.18ms | 11.51ms | **-3% faster** |
| TPOT p99 | 11.19ms | 11.61ms | **-4% faster** |
| Output tok/s | 85.69 | 83.50 | **+2.6%** |

Verdict: parity. Decode slightly ahead, prefill slightly behind.

### in=1, out=512 — Pure decode

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 12.20ms | 21.55ms | **-43% faster** |
| TPOT median | 10.62ms | 11.46ms | **-7% faster** |
| TPOT p99 | 10.62ms | 11.46ms | **-7% faster** |
| Output tok/s | 94.14 | 87.33 | **+7.8%** |

Verdict: **decode wins.** pegainfer 7% faster TPOT, 43% faster TTFT.

### in=2048, out=32 — Prefill heavy

| Metric | pegainfer | vLLM | delta |
|--------|-----------|------|-------|
| TTFT median | 270.69ms | 133.00ms | **+103% slower** |
| TTFT p99 | 274.11ms | 230.96ms | +19% |
| TPOT median | 11.86ms | 11.59ms | +2% |
| TPOT p99 | 11.87ms | 11.64ms | +2% |

Verdict: **prefill is 2x behind.** This is the main gap.

## Changelog

### 2026-03-13: TCP_NODELAY fix

**Root cause:** axum's default TCP socket lacked `TCP_NODELAY`. With HTTP keep-alive, Nagle's algorithm buffered the final SSE bytes, waiting ~30ms for a delayed ACK before the client could reuse the connection.

**Diagnosis path:** bench_serving (in-process) showed 10.36ms TTFT while `vllm bench serve` (HTTP) showed 42ms. Isolated with curl (11ms) vs Python aiohttp shared session (41ms vs 11ms with new session per request). This pinpointed TCP buffering on keep-alive connections.

**Fix:** One line — `tcp_stream.set_nodelay(true)` via `axum::serve::ListenerExt::tap_io`.

**Impact on in=1, out=1:**
- TTFT: 41.89ms → 11.89ms (3.5x)
- req/s: 23.78 → 82.53 (3.5x)
- Now 38% faster than vLLM (was 117% slower)

No regression on decode or prefill workloads.
