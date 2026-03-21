# Qwen3.5-4B Optimization

> **TL;DR:** Hybrid architecture (24 linear + 8 full attention). Decode TPOT 12.55ms vs vLLM 11.64ms (+8%). Prefill 14.5s vs vLLM 222ms (65× slower). Bottleneck is now per-token attention (80% GPU) and GDR (17% GPU) — both still using decode kernels in a per-token loop.
>
> **Status:** Active. RMSNorm_offset batched (#1 done). Next: FA2 for full attention prefill.

## Goal

pegainfer single-request latency >= vLLM on Qwen3.5-4B, same GPU/workload. Prefill is 76× behind vLLM — this is the only thing that matters right now.

## E2E Dashboard

GPU: RTX 5070 Ti, Model: Qwen3.5-4B, vLLM 0.17.1, single concurrency.
pegainfer: in-process bench_serving (no HTTP overhead). vLLM: `vllm bench serve` HTTP.

| Profile | Metric | pegainfer | vLLM | delta |
|---------|--------|-----------|------|-------|
| prefill-heavy (2048,1) | TTFT median | 14,500ms | 222ms¹ | **+65× slower** |
| prefill-heavy (2048,1) | TTFT p99 | 14,501ms | 245ms¹ | **+59× slower** |
| decode-heavy (1,128) | TPOT median | 12.55ms | 11.64ms² | +8% |
| decode-heavy (1,128) | TPOT p99 | 12.95ms | 11.76ms² | +10% |

¹ vLLM enforce-eager (no torch.compile/CUDA Graph). Compiled mode OOM'd on 2048-token prefill on this GPU.
² vLLM with torch.compile + CUDA Graph (default production config).

Reference — Qwen3-4B on same GPU: TTFT(2048,1)=213ms, TPOT(1,128)≈10.6ms.

## Architecture

- Layers: 32 (24 linear attention + 8 full attention)
- Full attention layer indices: 3, 7, 11, 15, 19, 23, 27, 31 (every 4th)
- hidden_dim: 2560
- MLP intermediate_size: 9216
- RMSNorm: (1 + weight) offset variant, eps=1e-6
- Tied word embeddings (embed_tokens = lm_head)
- Vocab size: 248,320

**Full attention (8 layers):**
- 16 q_heads, 4 kv_heads (GQA ratio 4), head_dim=256
- Partial RoPE: rotary_dim=64 (25% of head_dim), theta=1e7
- Q projection includes output gate: [8192, 2560] (q + gate interleaved)
- QK norm (per-head, broadcast)

**Linear attention (24 layers):**
- 16 q_heads (k_dim=128), 16 k_heads (k_dim=128), 32 v_heads (v_dim=128)
- Conv1d: kernel_dim=4, depthwise on QKV (dim=8192)
- Gated delta rule: recurrent state [32 x 128 x 128] f32 per layer
- Output gating: Z projection [4096, 2560] → SiLU gate on RMSNorm'd output
- Parameters: A_log [32] f32, dt_bias [32] bf16, norm_weight [128] f32

### Full Attention Layer — prefill

```
RMSNorm_offset [2560,seq]
  → GEMM Q [2560→8192,seq]     ← batched
  → GEMM K [2560→1024,seq]     ← batched
  → GEMM V [2560→1024,seq]     ← batched
  → per-token: fused_attention_hd256_single_token (QK norm, partial RoPE, KV write, attention, output gate)
  → GEMM O [4096→2560,seq]     ← batched
  → Residual + RMSNorm_offset [2560,seq]
  → GEMM Gate [2560→9216,seq]  ← batched
  → GEMM Up [2560→9216,seq]    ← batched
  → SiLU*Mul [9216,seq]
  → GEMM Down [9216→2560,seq]  ← batched
  → Residual
```

**Critical:** attention is per-token — no FlashAttention-2, no batched prefill attention kernel. Each token launches a separate fused_attention_hd256_single_token (designed for decode). At seq=2048, that's 2048 tiny kernel launches per full-attention layer × 8 layers = 16,384 kernels.

### Linear Attention Layer — prefill

```
RMSNorm_offset [2560,seq]
  → GEMM QKV [2560→8192,seq]   ← batched
  → GEMM Z [2560→4096,seq]     ← batched
  → GEMM B [2560→32,seq]       ← batched
  → GEMM A [2560→32,seq]       ← batched
  → per-token: conv1d_decode (update conv_state, SiLU)
  → per-token: gated_delta_rule_decode (update recurrent state [32×128×128])
  → per-token: rms_norm_gated (norm * SiLU(z))
  → GEMM O [4096→2560,seq]     ← batched
  → Residual + RMSNorm_offset [2560,seq]
  → GEMM Gate [2560→9216,seq]
  → GEMM Up [2560→9216,seq]
  → SiLU*Mul [9216,seq]
  → GEMM Down [9216→2560,seq]
  → Residual
```

**Critical:** conv1d + GDR + gated_norm are inherently sequential (state depends on previous token). At seq=2048: 2048 × 3 kernel launches × 24 layers = 147,456 kernels. Additionally, RMSNorm_offset is per-token (no batched kernel) — adds 2048 × 2 × 32 = 131,072 norm kernels across all layers.

### Full Attention Layer — decode

```
RMSNorm_offset [2560]
  → GEMV Q [2560→8192]
  → GEMV K [2560→1024]
  → GEMV V [2560→1024]
  → fused_attention_hd256_decode (QK norm, partial RoPE, KV write, split-KV attention, output gate)
  → GEMV O [4096→2560]
  → fused_add_rms_norm_offset (residual + next layer norm)
  → fused_mlp (gate+up+SiLU+down)
  → fused_add_rms_norm_offset
```

### Linear Attention Layer — decode

```
RMSNorm_offset [2560]
  → GEMV QKV [2560→8192]
  → GEMV Z [2560→4096]
  → GEMV B [2560→32]
  → GEMV A [2560→32]
  → conv1d_decode (update conv_state [8192×3], SiLU)
  → gated_delta_rule_decode (update state [32×128×128] f32)
  → rms_norm_gated (norm * SiLU(z))
  → GEMV O [4096→2560]
  → fused_add_rms_norm_offset (residual + next layer norm)
  → fused_mlp (gate+up+SiLU+down)
  → fused_add_rms_norm_offset
```

Decode is fully CUDA Graph'd. Zero GPU allocation after first token. conv1d and GDR are single-token operations — no per-token loop penalty.

## Operator Performance

### Decode (1,128) — nsys kernel breakdown per decode step

Total GPU kernel time: ~12.5ms/step (matches TPOT 12.55ms — fully GPU-bound, near-zero CPU overhead thanks to CUDA Graph).

| Kernel | Time/step | % | Count/step | Avg each | Notes |
|--------|-----------|---|------------|----------|-------|
| gemv (non-MLP) | 4.96ms | 39.6% | 153 | 32μs | QKV/Z/B/A/O projections + LM head |
| fused_mlp_intermediate | 3.73ms | 29.8% | 32 | 117μs | gate+up GEMV + SiLU*mul |
| fused_mlp_output | 1.91ms | 15.2% | 32 | 60μs | down GEMV |
| gated_delta_rule | 1.09ms | 8.7% | 24 | 46μs | recurrent state update [32×128×128] f32 |
| fused_attention_hd256 | 0.48ms | 3.8% | 8 | 60μs | split-KV full attention decode |
| fused_add_rms_norm_offset | 0.22ms | 1.7% | 64 | 3.4μs | residual + norm |
| conv1d_decode | 0.05ms | 0.4% | 24 | 2.1μs | |
| argmax | 0.05ms | 0.4% | 1 | 48μs | |
| rms_norm_gated | 0.03ms | 0.2% | 24 | 1.2μs | |
| other | <0.01ms | ~0% | 2 | — | embedding + first norm |

GEMV + fused_mlp dominate at 84.6%. This is pure memory-bandwidth work (matrix-vector products). GDR is 8.7% — the main "exotic" cost of the hybrid architecture.

### Prefill (128 tokens) — nsys kernel breakdown per prefill

Wall clock: 669ms. GPU kernel time: 222ms. **CPU kernel launch overhead: 447ms (67%).**

| Kernel | Time/prefill | % of GPU | Count/prefill | Avg each | Notes |
|--------|-------------|----------|---------------|----------|-------|
| gated_delta_rule | 138ms | 62.2% | 3,072 | 45μs | 128 tokens × 24 layers, sequential |
| fused_attention_hd256_single_token | 46ms | 20.8% | 1,024 | 45μs | 128 × 8 layers, decode kernel reused |
| rms_norm_offset | 16ms | 7.1% | 8,193 | 1.9μs | per-token (no batched kernel) |
| cuBLAS GEMM (batched projections) | 12ms | 5.3% | ~250 | 48μs | well-utilized, not a bottleneck |
| conv1d_decode | 4.4ms | 2.0% | 3,072 | 1.4μs | |
| rms_norm_gated | 3.7ms | 1.7% | 3,072 | 1.2μs | |
| other | ~2ms | ~1% | — | — | silu_mul, add, embedding, argmax |

**Key insight:** GPU only works 222ms out of 669ms wall clock. The remaining 447ms is CPU overhead from launching ~20,000 tiny kernels in a sequential per-token loop. At seq=2048, this scales to ~160K kernel launches → 16.8s wall clock.

The bottleneck is not kernel speed — it's the per-token loop architecture. Batching the sequential operations (or eliminating the per-token loop entirely) would bring prefill from seconds to sub-second.

## Optimization Log

### #1 Batched RMSNorm_offset (2026-03-21)

**Bottleneck:** rms_norm_offset — 8,193 kernel launches per prefill at seq=128 (~131K at seq=2048), costing ~2.3s of CPU launch overhead + per-token extract/write memcpys.

**Approach:** New `rms_norm_batched_offset_kernel` with `<<<seq_len, 256>>>` grid — one block per token, single launch. Replaced per-token loop in `Qwen35Model::batched_rms_norm_offset`.

**Changes:** `csrc/norm.cu`, `src/ffi.rs`, `src/ops.rs`, `src/qwen35_model.rs`

**Result:** Kernel time seq=2048: 38.7ms → 17.7μs (2186×). RMSNorm now 0% of GPU time (64 launches vs 131K).

**E2E impact:** TTFT (2048,1): 16,846ms → 14,500ms (−14%). Bottleneck shifts entirely to per-token attention (80%) and GDR (17%).

| Kernel | GPU% | Instances | Avg |
|--------|------|-----------|-----|
| fused_attention_hd256_single_token | 80.4% | 16,384 | 630μs |
| gated_delta_rule | 17.2% | 49,152 | 45μs |
| rms_norm_batched_offset ← new | 0.0% | 64 | 13μs |

### #0 Baseline (2026-03-14)

**E2E numbers:**
- Prefill-heavy (2048,1): pegainfer 16,846ms vs vLLM 222ms (**76× slower**)
- Decode-heavy (1,128): pegainfer 12.55ms vs vLLM 11.64ms (+8%)
- Supplementary: prefill (128,1) TTFT 669ms → extrapolation confirms superlinear scaling due to O(n) attention per token

**Decode verdict:** Close to parity. 12.55ms TPOT vs vLLM 11.64ms (+8%), fully GPU-bound, CUDA Graph'd. Slower than Qwen3-4B (10.6ms) due to 153 GEMV/step (vs ~109 for Qwen3-4B) — linear attention's extra projections (Z, B, A). GDR kernel adds 1.09ms (8.7%). Not worth optimizing until prefill is fixed.

**Prefill verdict:** Catastrophic. Two independent problems:

1. **CPU launch overhead (67% of wall time):** The per-token loop launches ~20K tiny kernels at seq=128. Each launch costs ~22μs CPU overhead. At seq=2048, this alone accounts for ~10s.

2. **No batched kernels for attention/recurrent ops (33% of wall time):** GDR, attention, conv1d, and rms_norm_offset all use single-token decode kernels. Even if launch overhead were zero, processing 2048 tokens through 32 layers of single-token kernels would take ~3.5s (vs 213ms for Qwen3-4B with FA2).

**Next steps:**

1. ~~**Batched RMSNorm_offset kernel**~~ — done in #1.
2. **FlashAttention-2 for full attention layers during prefill** — replace 16,384 single-token attention calls with 8 FA2 launches (already implemented for Qwen3-4B, needs adaptation for HD256 + partial RoPE + output gate). **This is now the dominant bottleneck (80% GPU time).**
3. **Chunk-parallel GDR prefill** — reduce 49,152 per-token GDR launches to ~24 (one per layer) via chunk-parallel recurrence. 17% GPU time. Note: WY decomposition tried and failed (FP rounding → incoherent output). Needs a different approach.
