# Project: Continuous Batching

> **TL;DR:** Serve N requests concurrently so each 7.67GB weight read produces N tokens instead of 1. Phase 1: PagedAttention (memory management). Phase 2: Scheduler + batch decode. Phase 3: Multi-request server engine.
>
> **Status:** Active. Phase 1 (PagedAttention) in progress.

## Motivation

Single-request decode is at parity with vLLM (TPOT 11.81ms). 91.6% of decode time is GEMV/MLP, all bandwidth-bound at 82-87% DRAM utilization. No single-operator optimization will yield meaningful gains. The only lever for throughput is amortizing weight reads across multiple requests:

```
Current:   7.67GB weight read → 1 token    → ~85 tok/s
Batch 8:   7.67GB weight read → 8 tokens   → ~680 tok/s (throughput)
Batch 32:  7.67GB weight read → 32 tokens  → ~2700 tok/s (throughput)
```

## VRAM Budget (RTX 5070 Ti, 16GB)

| Component | Size |
|-----------|------|
| Model weights (Qwen3.5-4B, bf16) | ~8 GB |
| Available for KV cache + recurrent state | ~8 GB |
| Per request: KV cache (8 full attn layers, seq=2048) | ~64 MB |
| Per request: recurrent state (24 linear attn layers) | ~48 MB |
| **Per request total** | **~112 MB** |
| **Max concurrent requests** | **~70** |

## Phases

### Phase 1: PagedAttention (current)

Replace contiguous KV cache with paged virtual memory. Enables dynamic per-request allocation/deallocation without fixed `max_seq_len` pre-allocation.

Key components:
- **Page pool**: shared GPU buffer divided into 16-token pages, free list
- **Page table**: per-request logical→physical page mapping (CSR format)
- **KV append**: write new K/V to paged layout
- **Paged attention decode**: FlashInfer `decode.cuh` header (zero external deps, supports SM120, partial RoPE confirmed with `rope_dim=64`)
- **Paged prefill**: gather pages → contiguous buffer → existing Triton FA kernel

Applies to full attention layers only (8 in Qwen3.5, 36 in Qwen3). Linear attention layers use `RecurrentState` (unchanged).

### Phase 2: Scheduler + Batch Decode

- Request queue with admission control
- Batch decode: process N requests' tokens in one forward pass
- GEMV → batched GEMM when batch_size > 1 (cuBLAS handles this)
- Recurrent state pool for Qwen3.5 linear attention layers
- CUDA Graph per batch-size or FlashInfer's `block_valid_mask` pattern

### Phase 3: Multi-Request Server Engine

- Replace single `Mutex<State>` with scheduler-driven request pipeline
- Async prefill interleaving with ongoing decode batches
- Preemption (pause low-priority requests, free their pages)
- Streaming output per request

## Architecture Reference

```
Phase 1 (PagedAttention):
  PagePool (shared GPU buffer)
    ├── alloc_pages(n) → Vec<page_id>
    └── free_pages(Vec<page_id>)

  PagedKVCache (per-request)
    ├── page_tables: [num_full_attn_layers] → page indices
    ├── append_token(layer, k, v)
    ├── build_gpu_metadata() → (indices, indptr, last_page_len)
    └── reset() → return pages to pool

Phase 2 (Scheduler):
  Scheduler
    ├── add_request(prompt, params) → request_id
    ├── step() → run one decode iteration for all active requests
    └── poll_output(request_id) → Option<token>

Phase 3 (Server):
  HTTP handler → Scheduler.add_request()
  Background loop → Scheduler.step() → broadcast tokens to SSE streams
```
