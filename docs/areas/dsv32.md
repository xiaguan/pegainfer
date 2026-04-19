# DeepSeek-V3.2 on pegainfer

> **Status**: Prefill logits aligned with reference. Decode currently runs dense FlashMLA (implementation gap vs NSA sparse design), and generation quality is not verified end-to-end. Serving endpoint up.
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

### Attention path

| DAG node | Provider | File | Status | Notes |
|----------|----------|------|--------|-------|
| FP8 activation quantize (1×128 block-scale) | Self-written (extracted from TRT-LLM) | `csrc/fp8_quantize.cu` | ✅ | |
| FP8 block-scale GEMM (q_a/q_b/kv_a/o_proj) | DeepGEMM SM90 1D2D | `csrc/fp8_gemm.cu` + `third_party/DeepGEMM` | ✅ | AOT-compiled, 2 tile configs (64×128/8s, 128×128/5s) |
| RMSNorm (full / partial c_kv-only / fused add) | Self-written | `csrc/mla.cu`, `src/ops.rs` | ✅ | |
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
| Final RMSNorm + lm_head (bf16) | Self-written + cuBLAS | `forward_final` | ✅ | `tie_word_embeddings` shares embedding matrix |
| KV cache pool (paged, per-layer) | Self-written | `src/model/dsv32/mla_kv.rs` | ✅ | `MlaKvPool`, `MlaKvState` |
| Weight loader (safetensors → FP8/bf16 sharded) | Self-written | `src/model/dsv32/weights.rs` | ✅ | Per-rank parallel load, 8 threads ~56 s |
| Multi-GPU executor (per-rank ctx + NCCL + DeepEP) | Self-written | `src/model/dsv32/executor.rs` | ✅ | |
| Scheduler (serial) | Self-written | `scheduler_dsv32` | ✅ | One request at a time, no continuous batching |

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
bench_serving request --prompt-len 2048 --output-len 1 --model dsv32
bench_serving request --prompt-len 1   --output-len 128 --model dsv32

# vLLM baseline (on the same 8-card NVLink node)
vllm bench serve --model /data/models/DeepSeek-V3.2 --input-len 2048 --output-len 1
vllm bench serve --model /data/models/DeepSeek-V3.2 --input-len 1   --output-len 128
```

Single concurrency. See `docs/resources/bench-vs-vllm.md` for vLLM setup specifics.

---

## 8. E2E Dashboard

GPU: 8-card NVLink node. Model: DeepSeek-V3.2 FP8. vLLM version: TBD.

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

---

## 9. Optimization Log

Append-only. See `docs/resources/model-optimization-pipeline.md` for entry format.

### #0 Baseline — TBD

No kernel breakdown yet. Will record once `benches/dsv32_*` + nsys capture are in place.

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
| RMSNorm batched / fused-add | `ops::rms_norm_batch_into` / `fused_add_rms_norm_batch_into` | `src/ops.rs` |
| SiLU × up | `ops::silu_mul_batch_into` | `src/ops.rs` |
| BF16 GEMM | `ops::gemm_into` | `src/ops.rs` |
| FP8 quantize + GEMM | `ops::fp8::fp8_linear_into` / `fp8_quantize_into` + `fp8_gemm_into` | `src/ops/fp8.rs` |
| FlashMLA 3-phase | `ffi::flash_mla_{get_metadata,decode,combine}` | `src/ffi.rs` |
| FlashMLA sparse prefill | `ffi::flash_mla_sparse_prefill` | `src/ffi.rs` |
| Strided batched GEMM | `ffi::gemm_strided_batched_cuda` | `csrc/linear.cu` |
| MLA RoPE (q_rope extract+copy, k_rope) | `ffi::mla_rope_q_copy_cuda` / `mla_rope_kv_cuda` | `csrc/mla.cu` |
| Partial RMSNorm (c_kv only) | `ffi::rms_norm_partial_cuda` | `csrc/mla.cu` |
| KV cache write (scatter) | `ffi::mla_kv_cache_write_cuda` | `csrc/mla.cu` |
| YaRN RoPE cos/sin precompute | `precompute_yarn_rope` | `src/model/dsv32/weights.rs` |
| DeepEP 5 functions + IPC helpers | `ffi::deep_ep_*` | `src/ffi.rs`, `src/model/dsv32/deep_ep.rs` |

---

## 12. Open Work

- [ ] `benches/dsv32_*` — micro benchmarks (FP8 GEMM, FlashMLA, DeepEP).
- [ ] `#0 Baseline` — nsys kernel breakdown for both profiles.
- [ ] Decode generation quality sanity check (greedy vs Python reference, 128+ tokens).
- [ ] Grouped GEMM for routed experts (DeepGEMM `GroupedContiguous`) — replace per-expert sequential.
- [ ] CUDA Graph capture for decode.
- [ ] TP2 attention sharding.
