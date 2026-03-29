# Project: Continuous Batching

> **TL;DR:** Serve N requests concurrently so each 7.67GB weight read produces N tokens instead of 1. Phase 1 starts with a generic RAII `PagePool` allocator, then layers paged KV layout and kernels on top. Phase 2: Scheduler + batch decode. Phase 3: Multi-request server engine.
>
> **Status:** Active. Phase 1: FlashInfer paged attention landed for Qwen3 decode (bs=1). Exact greedy parity with baseline, TPOT ~11.26ms. Next: CUDA Graph re-enable, batch decode (Phase 2).

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

#### Decision: page-first memory layout

Each physical page is a contiguous chunk containing **all layers'** K/V for `page_size` tokens:

```
page_i: [L0_K | L0_V | L1_K | L1_V | ... | L7_K | L7_V]
         └─ page_size × num_kv_heads × head_dim each ─┘
```

Single `cudaMalloc` backing buffer. Kernel addressing: `base + page_id × page_stride + layer × layer_kv_offset`.

Why not layer-first (per-layer buffer, pages indexed within each):
- layer-first requires 2×N separate allocations (72 for Qwen3)
- "freeing a page" only recycles an ID, not a contiguous memory region
- kernel needs a per-layer pointer array instead of base + strides
- paged attention already does random access by page index, so cross-page contiguity within a layer buys nothing

Page-first gives: true allocation atomicity, single buffer, pool fully decoupled from KV semantics, simpler kernel interface.

#### Decision: naming convention

No "paged" leaks to callers — page management is an internal implementation detail.

| Struct | Role | Parallels |
|--------|------|-----------|
| `KvPool` | Shared GPU backing + page allocation. Engine owns it. | `PagePool` (generic) → `KvPool` (KV-specific) |
| `KvState` | Per-request KV state: pages held, seq_len, capacity growth. | Parallels `RecurrentState` — both are per-request, both live in `GenerationState` |
| `KvDesc` | Kernel-facing metadata bundle (base_ptr, strides, page_indices, last_page_len). | cuDNN tensor descriptor convention |

Caller-facing API (forward code):
```rust
kv.ensure_capacity(kv.seq_len() + 1)?;
let desc = kv.desc();                      // → KvDesc
ffi::paged_attention_decode(q, k, v, &desc, layer, ...);
kv.advance(1);
```

Internal pointer arithmetic (`base + page_id × page_stride + layer × layer_kv_offset`) is hidden inside `KvDesc` construction.

#### Progress

Done:
- `PagePool`: generic fixed-page allocator, RAII `OwnedPagePermit`, `try_acquire_many(n)`, `try_grow(n)`.
- `KvPool` + `KvState` + `KvDesc` in `src/kv_pool.rs`. Data structures + unit tests (geometry, lifecycle, OOM, drop). Page-first layout with `KvLayout` stride geometry. `KvPool` is `Clone` via `Arc` for trait-compatible state ownership.
- FlashInfer submodule at `third_party/flashinfer` (header-only C++, `include/flashinfer/`).
- `csrc/paged_attention.cu`: thin C wrapper around FlashInfer's `BatchDecodeWithPagedKVCacheDispatched` (decode) and `AppendPagedKVCacheDecode` (KV write). bf16, HEAD_DIM=128, NHD layout, no RoPE (applied externally). Non-partition path (no split-KV) for Phase 1.
- `qk_norm_rope_cuda`: standalone QK RMSNorm + RoPE kernel for decode, reuses `prefill_qk_norm_rope_kernel` with seq_len=1.
- Rust FFI bindings (`ffi.rs`) and ops wrappers (`ops/attention.rs`) for all three kernels.

#### Decision: page-first ↔ FlashInfer stride bridge

FlashInfer's `paged_kv_t` uses separate `k_data`/`v_data` pointers with custom strides. Our page-first layout stores all layers interleaved in one buffer. The bridge: for layer L, `k_data = base + L × layer_stride`, `v_data = base + L × layer_stride + kv_block_len`, `stride_page = page_stride`. NHD within each K/V block matches FlashInfer's NHD mode. No transpose or copy needed.

#### Decision: BatchDecode for bs=1 (not SingleDecode)

FlashInfer's `SingleDecodeWithKVCache` only supports contiguous KV — no paged layout. For paged KV at bs=1, we use `BatchDecodeWithPagedKVCacheDispatched` with `batch_size=1`, `partition_kv=false`. The kernel template always accesses `request_indices`/`kv_tile_indices` (even when not partitioning), so we provide trivial GPU arrays `[0]`.

- Paged attention wired into Qwen3 decode path as the **only** decode attention. `KvPool` always created, `KvState` per-request, CUDA Graph disabled (Phase 1: metadata arrays allocated per-call).
- FlashInfer decode kernel validated end-to-end: 10-token single-prompt generation passes, determinism verified. 50-token multi-token prompt generation produces coherent output (confirms kernel correctness), but differs from baseline (prefill→paged gap, see below).

- Prefill→paged scatter: after prefill writes to contiguous KV cache, `scatter_kv_to_paged` copies all layers into paged layout via FlashInfer's `AppendPagedKVCache` kernel. Bridges HND (contiguous) → NHD (paged) per-layer.

**Result:** Full e2e greedy parity with baseline on Qwen3-4B. All 45 tests pass. TPOT ~11.26ms (88.8 tok/s).

Next:
- Re-enable CUDA Graph with pre-allocated metadata buffers.
- Phase 2: batch decode (multiple requests sharing KvPool).

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
  PagePool (generic allocator, no GPU knowledge)
    ├── try_acquire_many(n) → OwnedPagePermit
    └── OwnedPagePermit::try_grow(n) → bool

  KvPool (shared GPU backing, page-first layout)
    ├── pool: PagePool
    ├── buffer: CudaSlice<bf16>
    └── alloc() → KvState

  KvState (per-request, parallels RecurrentState)
    ├── ensure_capacity(token_count) → grow permit if needed
    ├── desc() → KvDesc (kernel-facing metadata)
    ├── advance(count)
    └── reset()

Phase 2 (Scheduler):
  Scheduler
    ├── add_request(prompt, params) → request_id
    ├── step() → run one decode iteration for all active requests
    └── poll_output(request_id) → Option<token>

Phase 3 (Server):
  HTTP handler → Scheduler.add_request()
  Background loop → Scheduler.step() → broadcast tokens to SSE streams
```
