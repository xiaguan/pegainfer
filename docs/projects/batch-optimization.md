# Batch Optimization

> **TL;DR:** Decode batching works — 5.9× throughput at c=8 vs c=1. But prefill is serial, causing TTFT to scale linearly with concurrency (7× at c=8). vLLM batches prefill and holds flat TTFT under load. Prefill batching and per-token latency under batch are the two main targets.
>
> **Status:** Active. Baseline established. Next: identify optimization targets from these numbers.

## Baseline (2026-03-29)

GPU: RTX 5070 Ti, Model: Qwen3-4B, vLLM 0.18.x, `vllm bench serve` as unified client.
pegainfer: HTTP server with continuous batching scheduler. vLLM: default config (`--max-model-len 4096 --gpu-memory-utilization 0.9`).
n=20 per config, `--request-rate inf --ignore-eos`.

**Caveat:** vLLM torch.compile cold-start inflates the first 1–3 requests per shape. With n=20, mean/p99 can be polluted. Read **median** for steady-state. Throughput (tok/s) is affected when cold-start requests dominate elapsed time — noted where relevant.

### in=1, out=128 — Decode-heavy

| Metric | c | pegainfer | vLLM | delta |
|--------|---|-----------|------|-------|
| TTFT median (ms) | 1 | 12.43 | 20.30 | **−39%** |
| TPOT median (ms) | 1 | 10.94 | 11.27 | −3% |
| Output tok/s | 1 | 90.7 | 87.5 | +4% |
| TTFT median (ms) | 4 | 34.99 | 34.19 | +2% |
| TPOT median (ms) | 4 | 11.95 | 11.49 | +4% |
| Output tok/s | 4 | 329.9 | 335.6 | −2% |
| TTFT median (ms) | 8 | 49.28 | 44.40 | +11% |
| TPOT median (ms) | 8 | 12.32 | 11.41 | +8% |
| Output tok/s | 8 | 531.6 | 208.0¹ | +156%¹ |

¹ vLLM c=8 throughput is polluted by torch.compile cold-start (p99 TTFT = 7867ms). Steady-state throughput is likely ~700 tok/s based on median TPOT.

**Takeaway:** Single-request latency: pegainfer wins on TTFT (−39%), decode at parity. Under batch, decode throughput scales well (5.9× at c=8). TPOT increases ~13% from c=1 to c=8 (10.94 → 12.32ms) — scheduling overhead, not GPU saturation.

### in=1024, out=1 — Prefill-only

| Metric | c | pegainfer | vLLM | delta |
|--------|---|-----------|------|-------|
| TTFT median (ms) | 1 | 108.03 | 115.89² | **−7%** |
| TTFT median (ms) | 4 | 382.45 | 65.33 | **+486%** |
| TTFT median (ms) | 8 | 765.23 | 103.33 | **+641%** |
| Req/s | 1 | 9.25 | 0.40² | — |
| Req/s | 4 | 10.39 | 62.54 | **−83%** |
| Req/s | 8 | 10.39 | 67.40 | **−85%** |

² vLLM c=1 mean TTFT = 2527ms (cold-start), median = 115ms (steady-state). Req/s of 0.40 is cold-start-polluted; steady-state ~9 req/s.

**Takeaway:** pegainfer prefill is serial — TTFT scales linearly with queue depth (108 → 382 → 765ms). vLLM batches prefill and holds TTFT nearly flat (115 → 65 → 103ms). At c=8, vLLM is 7.4× faster on TTFT. This is the biggest gap.

### in=512, out=64 — Mixed workload

| Metric | c | pegainfer | vLLM | delta |
|--------|---|-----------|------|-------|
| TTFT median (ms) | 1 | 52.97 | 28.55 | +86% |
| TPOT median (ms) | 1 | 11.67 | 11.32 | +3% |
| Output tok/s | 1 | 81.1 | 86.3 | −6% |
| TTFT median (ms) | 4 | 136.16 | 50.90 | +167% |
| TPOT median (ms) | 4 | 13.64 | 11.93 | +14% |
| Output tok/s | 4 | 257.2 | 317.7 | −19% |
| TTFT median (ms) | 8 | 216.79 | 81.12 | +167% |
| TPOT median (ms) | 8 | 14.63 | 12.14 | +21% |
| Output tok/s | 8 | 372.5 | 509.3 | −27% |
| ITL p99 (ms) | 4 | 61.67 | 14.00 | +340% |
| ITL p99 (ms) | 8 | 83.22 | 15.08 | +452% |

**Takeaway:** Mixed workload exposes both problems. TTFT gap widens with concurrency (serial prefill queuing). TPOT degrades 25% from c=1 to c=8 (11.67 → 14.63ms) — worse than the decode-only case. ITL p99 spikes to 61–83ms under batch, indicating scheduling stalls (prefill blocking decode steps). vLLM's ITL p99 stays tight (14–15ms).

## Key Observations

1. **Decode batching works.** Throughput scales 3.6× at c=4, 5.9× at c=8 for decode-heavy load. The CUDA Graph bucket system is functional.

2. **Prefill is serial and the #1 bottleneck.** TTFT = `queue_depth × single_prefill_time`. At c=8 with 1024-token prefill, users wait 765ms vs vLLM's 103ms.

3. **Prefill blocks decode.** ITL p99 = 61–83ms under mixed load. When a prefill arrives, in-flight decode requests stall until it completes. vLLM interleaves chunked prefill with decode to keep ITL tight.

4. **Per-token TPOT under batch is 8–21% higher than vLLM.** At c=1 they're at parity, but under batch pegainfer degrades faster. Likely scheduler overhead or suboptimal kernel dispatch for batch sizes between CUDA Graph bucket sizes.

5. **vLLM's compiled prefill is fast.** At in=512 c=1, vLLM's 28.55ms vs pegainfer's 52.97ms suggests torch.compile gives vLLM nearly 2× prefill speed at this sequence length. At in=1024, the gap is smaller (108 vs 115ms) — could be shape-specific compilation effects or warm-up order artifact.
