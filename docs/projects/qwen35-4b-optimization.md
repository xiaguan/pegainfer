# Qwen3.5-4B Optimization

> **TL;DR:** Hybrid architecture (24 linear + 8 full attention). After replacing linear-attention prefill's per-token GDR loop with a Triton fused recurrent kernel, prefill-heavy TTFT dropped from `3.89s` to `~378ms` at `(2048,1)`; decode-heavy TPOT remains `12.53ms` at `(1,128)`. `nsys` now shows prefill is dominated by real compute rather than orchestration: `gated_delta_rule_prefill_kernel` is `48.2%` of GPU time and batched GEMM is another `~44%`, while DtoD-copy / alloc-free churn has effectively been removed.
>
> **Status:** Active. RMSNorm_offset batched (#1), standalone HD256 Triton attention (#2), full-attention prefill wiring (#3), E2E refresh (#4), reprofile after full-attention rewrite (#5), and Triton GDR prefill (#6) are all done. Current blocker before commit: decide whether to accept the refreshed Qwen3.5 greedy baseline (`6/13` prompts changed, one materially) or investigate that longstanding drift first. Perf-wise, the next target is reducing real linear-prefill compute inside `gated_delta_rule_prefill_kernel` / GEMM rather than more host-side cleanup.

## Goal

pegainfer single-request latency >= vLLM on Qwen3.5-4B, same GPU/workload. Prefill is still the only meaningful gap, but the problem has changed: the old host-side per-token orchestration disaster is gone, and the remaining gap is now real linear-prefill compute.

## Known Caveat

Refreshing `test_data/Qwen3.5-4B.json` on the current code changes `6/13` prompts relative to the old baseline. Most changes are small tail whitespace / newline drift, but `python_prime` changes more substantially. The focused `gated_delta_rule_prefill` reference check passes, so the current suspicion is integration-level numerical drift rather than a gross recurrent-state bug.

## E2E Dashboard

GPU: RTX 5070 Ti, Model: Qwen3.5-4B, vLLM 0.17.1, single concurrency.
pegainfer: in-process bench_serving (no HTTP overhead). vLLM: `vllm bench serve` HTTP.

| Profile | Metric | pegainfer | vLLM | delta |
|---------|--------|-----------|------|-------|
| prefill-heavy (2048,1) | TTFT median | 377.20ms | 222ms¹ | **+1.7× slower** |
| prefill-heavy (2048,1) | TTFT p99 | 379.85ms | 245ms¹ | **+1.6× slower** |
| decode-heavy (1,128) | TPOT median | 12.54ms | 11.64ms² | +8% |
| decode-heavy (1,128) | TPOT p99 | 12.94ms | 11.76ms² | +10% |

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
  → prefill_attention_hd256_prep (Q/K norm, partial RoPE, KV write)   ← batched CUDA helper
  → flash_attention_prefill_hd256_into (attention core)               ← batched Triton
  → attention_gate_batch_hd256 (apply sigmoid(gate))                  ← batched CUDA helper
  → GEMM O [4096→2560,seq]     ← batched
  → Residual + RMSNorm_offset [2560,seq]
  → GEMM Gate [2560→9216,seq]  ← batched
  → GEMM Up [2560→9216,seq]    ← batched
  → SiLU*Mul [9216,seq]
  → GEMM Down [9216→2560,seq]  ← batched
  → Residual
```

**Current state:** full-attention prefill is no longer a meaningful TTFT bottleneck. The current end-to-end profile shows the remaining cost sits in linear-attention prefill compute.

### Linear Attention Layer — prefill

```
RMSNorm_offset [2560,seq]
  → GEMM QKV [2560→8192,seq]   ← batched
  → GEMM Z [2560→4096,seq]     ← batched
  → GEMM B [2560→32,seq]       ← batched
  → GEMM A [2560→32,seq]       ← batched
  → conv1d_prefill [8192,seq]  ← one launch/layer, updates conv_state across seq
  → gated_delta_rule_prefill [4096,seq]  ← one Triton fused-recurrent launch/layer
  → rms_norm_gated [4096,seq]  ← one launch/layer
  → GEMM O [4096→2560,seq]     ← batched
  → Residual + RMSNorm_offset [2560,seq]
  → GEMM Gate [2560→9216,seq]
  → GEMM Up [2560→9216,seq]
  → SiLU*Mul [9216,seq]
  → GEMM Down [9216→2560,seq]
  → Residual
```

**Critical:** the recurrence is still real, but it now lives inside `conv1d_prefill` / `gated_delta_rule_prefill` instead of a host-side token loop. For one request, the linear stack now pays `24` conv1d launches + `24` GDR launches + `24` gated-norm launches, instead of `2048 × 3 × 24 = 147,456` tiny sequential launches. `RMSNorm_offset` was already fixed in #1 and is no longer part of the prefill hotspot.

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

### Prefill (128 tokens) — baseline before batched Triton full-attention wiring

Wall clock: 669ms. GPU kernel time: 222ms. **CPU kernel launch overhead: 447ms (67%).**

| Kernel | Time/prefill | % of GPU | Count/prefill | Avg each | Notes |
|--------|-------------|----------|---------------|----------|-------|
| gated_delta_rule | 138ms | 62.2% | 3,072 | 45μs | 128 tokens × 24 layers, sequential |
| fused_attention_hd256_single_token | 46ms | 20.8% | 1,024 | 45μs | 128 × 8 layers, old decode-style kernel reused during prefill |
| rms_norm_offset | 16ms | 7.1% | 8,193 | 1.9μs | per-token (no batched kernel) |
| cuBLAS GEMM (batched projections) | 12ms | 5.3% | ~250 | 48μs | well-utilized, not a bottleneck |
| conv1d_decode | 4.4ms | 2.0% | 3,072 | 1.4μs | |
| rms_norm_gated | 3.7ms | 1.7% | 3,072 | 1.2μs | |
| other | ~2ms | ~1% | — | — | silu_mul, add, embedding, argmax |

**Key insight (baseline):** GPU only works 222ms out of 669ms wall clock. The remaining 447ms is CPU overhead from launching ~20,000 tiny kernels in a sequential per-token loop. At seq=2048, this scales to ~160K kernel launches → 16.8s wall clock.

This section is retained as the pre-#3 baseline. Full-attention prefill has since been moved to a batched Triton path; the updated end-to-end `nsys` capture is summarized below.

### Prefill (2048 tokens) — before Triton GDR prefill (#5 historical snapshot)

`nsys` command used:

```bash
nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/qwen35_prefill_2048 \
  cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1
```

Trace note: this capture includes one warmup request plus one measured request, so instance counts below are doubled relative to a single `(2048,1)` run. Percentages are still representative.

| Kernel | GPU% | Instances in trace | Avg each | Notes |
|--------|------|--------------------|----------|-------|
| gated_delta_rule_decode | 87.6% | 98,304 | 45.0us | 24 linear-attn layers × 2048 tokens × 2 runs |
| conv1d_decode | 2.8% | 98,304 | 1.44us | linear-attn sequential prefilter |
| rms_norm_gated | 2.4% | 98,304 | 1.21us | linear-attn post-GDR gating |
| batched GEMM kernels (CUTLASS) | ~6.3% | 416 | 122us to 1.04ms | projections + MLP, no longer dominant |
| flash_attention_prefill_hd256_kernel | 0.4% | 16 | 1.41ms | 8 full-attn layers × 2 runs |
| prefill_attention_hd256_prep + gate + v-cache helpers | <0.1% | 48 | 15us to 122us | Q/K norm + partial RoPE + gate are negligible |
| rms_norm_batched_offset | 0.0% | 128 | 12.5us | fixed in #1, no longer relevant |

**Kernel conclusion:** full-attention prefill is no longer the problem. The remaining GPU bottleneck is the linear-attention sequential loop, and specifically `gated_delta_rule_decode`.

`cuda_api_sum` from the same trace shows the host/runtime side is now dominated by bookkeeping rather than full-attention launches:

| API | Total calls in trace | Avg each | Interpretation |
|-----|----------------------|----------|----------------|
| cuMemsetD8Async | 689,163 | 4.65us | heavy temporary-buffer churn |
| cuMemcpyDtoDAsync_v2 | 491,522 | 5.92us | row extraction / writeback style device copies still dominate |
| cudaLaunchKernel | 295,094 | 5.25us | still a very large sequential launch count |
| cuMemAllocAsync | 689,545 | 0.97us | temporary allocations remain excessive |
| cuMemFreeAsync | 689,545 | 0.75us | same churn on the free side |

**API conclusion:** after deleting the old HD256 attention loop, prefill was still paying for a massive amount of linear-attention per-token orchestration: launches, DtoD copies, memsets, and alloc/free traffic. That is the exact problem addressed in #6 below.

### Prefill (2048 tokens) — after Triton GDR prefill (#6 current snapshot)

Stable request bench:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3
```

Result:

- TTFT avg `377.89ms`
- TTFT p50 `377.20ms`
- TTFT p99 `379.85ms`

`nsys` command used:

```bash
PEGAINFER_TRITON_PYTHON=./.venv/bin/python \
nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node \
  --export=sqlite -o target/profiling/qwen35_prefill_2048_gdr \
  cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B \
  request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1
```

Trace note: this capture includes one warmup request plus one measured request, so instance counts below are doubled relative to a single `(2048,1)` run. The profile run measured TTFT `377.27ms`, matching the 3-iteration bench.

| Kernel | GPU% | Instances in trace | Avg each | Notes |
|--------|------|--------------------|----------|-------|
| gated_delta_rule_prefill | 48.2% | 48 | 7.39ms | 24 linear-attn layers × 2 runs; now the main compute kernel |
| batched GEMM kernels (CUTLASS) | ~44.1% | 496 | 14.9us to 1.04ms | projections + MLP are now co-dominant with GDR |
| flash_attention_prefill_hd256_kernel | 3.1% | 16 | 1.41ms | 8 full-attn layers × 2 runs |
| conv1d_prefill | 1.3% | 48 | 205us | one launch per linear-attn layer per run |
| rms_norm_gated | 0.6% | 48 | 95us | one launch per linear-attn layer per run |
| prefill qk/rope + gate + v-cache helpers | ~0.5% | 48 | 14.9us to 122us | still negligible |
| rms_norm_batched_offset | 0.2% | 128 | 12.6us | unchanged from #1; not relevant |

**Kernel conclusion:** the bottleneck has finally moved to genuine compute. Full-attention prefill is small, host-side per-token orchestration is gone, and the remaining cost is `gated_delta_rule_prefill_kernel` plus batched GEMM.

`cuda_api_sum` from the same trace:

| API | Total calls in trace | Avg each | Interpretation |
|-----|----------------------|----------|----------------|
| cudaLaunchKernel | 278 | 1.28ms | request-level launch count has collapsed from the old per-token regime |
| cuLaunchKernel | 754 | 15.5us | includes driver-level kernel entry points |
| cuMemcpyDtoDAsync_v2 | 2 | 14.4us | row extraction / writeback style copies are effectively gone |
| cuMemAllocAsync | 1,513 | 37.0us | far lower than the old per-token churn, though still not zero |
| cuMemFreeAsync | 1,513 | 1.12us | tracks the same temporary allocation pattern |
| cuMemsetD8Async | 1,131 | 9.44us | no longer a dominant cost center |

Note: `cuMemcpyHtoDAsync_v2` dominates total API time in this whole-process trace because it includes model-load / one-time setup traffic, so it is not the useful signal for request-level comparison here.

**API conclusion:** compared with the historical #5 snapshot, the big wins are structural: `cudaLaunchKernel` went from `~295k` to `278`, `cuMemcpyDtoDAsync_v2` from `~492k` to `2`, and `cuMemAllocAsync` from `~690k` to `1.5k`. Prefill is no longer spending its life in launch/copy/allocation bookkeeping.

## Optimization Log

### #6 Triton GDR prefill for Qwen3.5 (2026-03-21)

**Goal:** Remove the Qwen3.5 linear-attention prefill host-side token loop by replacing per-token GDR launches with a single Triton fused-recurrent kernel per layer.

**Changes:** Added `tools/triton/gated_delta_rule_prefill_kernel.py` and its AOT build wiring, exposed `gated_delta_rule_prefill_cuda` through FFI, added `conv1d_prefill_batch_into()`, `gated_delta_rule_prefill_into()`, and `rms_norm_gated_batch_into()` in `ops.rs`, rewired `Qwen35Model` linear prefill to use the batched `conv1d -> GDR -> gated_norm` path, and added a `gated_delta_rule_prefill_into` microbench entry.

**Validated commands:**
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release test_gated_delta_rule_prefill_matches_decode_reference -- --ignored --nocapture`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 3`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 128 --warmup 1 --iters 3`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python cargo test --release --test gen_test_data_35 -- --nocapture`
- `PEGAINFER_TRITON_PYTHON=./.venv/bin/python nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite -o target/profiling/qwen35_prefill_2048_gdr cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1`
- `nsys stats --report cuda_gpu_kern_sum target/profiling/qwen35_prefill_2048_gdr.sqlite`
- `nsys stats --report cuda_api_sum target/profiling/qwen35_prefill_2048_gdr.sqlite`

**Results:**
- Prefill-heavy `(2048,1)`: TTFT avg `377.89ms`, p50 `377.20ms`, p99 `379.85ms`
- Decode-heavy `(1,128)`: TTFT avg `12.27ms`, steady TPOT avg `12.53ms`, p50 `12.54ms`, p99 `12.94ms`
- `gated_delta_rule_prefill_kernel`: `48.2%` of GPU time, `48` instances, `7.39ms` avg each
- Batched GEMM kernels: another `~44.1%` of GPU time
- Launch/copy/allocation churn collapsed:
  - `cudaLaunchKernel`: `~295k` -> `278`
  - `cuMemcpyDtoDAsync_v2`: `~492k` -> `2`
  - `cuMemAllocAsync`: `~690k` -> `1,513`

**Correctness note:** Regenerating `test_data/Qwen3.5-4B.json` changes `6/13` prompts relative to the old baseline. Most changes are small tail drift, but `python_prime` changes materially. The focused GDR-prefill-vs-decode reference test passes, so this looks like an existing integration-level drift rather than an obvious Triton kernel failure.

**Interpretation:** This is the first Qwen3.5 profile where prefill is no longer dominated by orchestration. TTFT is now within about `1.7x` of vLLM at `(2048,1)`, and the remaining work is real recurrent/GEMM compute rather than launch bookkeeping.

### #3 Batched Triton full-attention prefill for Qwen3.5 (2026-03-21)

**Goal:** Replace the Qwen3.5 HD256 full-attention prefill loop with a batched Triton path and remove the legacy single-token CUDA prefill implementation.

**Changes:** Added `prefill_attention_hd256_prep_cuda` and `attention_gate_batch_hd256_cuda`, wired `Qwen35Model` full-attention prefill to `prefill_attention_hd256_batch()`, updated the `qwen35_prefill_attn` microbench to measure the runtime path, regenerated the Qwen3.5 greedy baseline, and deleted the dead `fused_gqa_attention_hd256_single_token` CUDA implementation.

**Validated commands:**
- `cargo test --release test_prefill_attention_hd256_batch_matches_cpu_reference -- --ignored --nocapture`
- `cargo test --release --test e2e_qwen35 -- --nocapture`
- `cargo bench --bench ops_bench -- qwen35_prefill_attn`

**Microbench result (runtime subpath: prep + Triton attention + gate):**
- seq128: 50.7us
- seq512: 365us
- seq2048: 1.64ms

**Interpretation:** Qwen3.5 full-attention prefill is no longer dominated by per-token HD256 attention launches. That follow-up profile is now in #5 and confirms the remaining TTFT cost sits in the linear-attention sequential kernels.

### #4 End-to-end refresh after Triton prefill wiring (2026-03-21)

**Goal:** Measure real request-level impact before running another profiling pass.

**Validated commands:**
- `cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 5`
- `cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 1 --output-len 128 --warmup 1 --iters 5`

**Results:**
- Prefill-heavy `(2048,1)`: TTFT avg `3889.14ms`, p50 `3889.25ms`, p99 `3890.67ms`
- Decode-heavy `(1,128)`: TTFT avg `12.28ms`, steady TPOT avg `12.53ms`, p50 `12.54ms`, p99 `12.94ms`

**Interpretation:** Removing the old HD256 per-token prefill kernel improved TTFT by about `3.7x` versus the previous `14.5s` baseline while leaving decode essentially unchanged. The remaining gap to vLLM is now much more likely to sit in the 24 linear-attention layers and their sequential prefill kernels.

### #5 nsys reprofile after Triton prefill wiring (2026-03-21)

**Goal:** Confirm where prefill time moved after removing the old HD256 per-token full-attention path.

**Validated commands:**
- `nsys profile --force-overwrite=true --trace=cuda,nvtx --cuda-graph-trace=node --export=sqlite -o target/profiling/qwen35_prefill_2048 cargo run --release --bin bench_serving -- --model-path models/Qwen3.5-4B request --prompt-len 2048 --output-len 1 --warmup 1 --iters 1`
- `nsys stats --report cuda_gpu_kern_sum target/profiling/qwen35_prefill_2048.sqlite`
- `nsys stats --report cuda_api_sum target/profiling/qwen35_prefill_2048.sqlite`

**Key findings:**
- `gated_delta_rule_decode_kernel`: `87.6%` of GPU time
- `conv1d_decode_kernel` + `rms_norm_gated_kernel`: another `5.2%`
- Batched Triton full-attention core: `0.4%`
- HD256 prep/gate helpers combined: `<0.1%`
- API activity still includes `~295k` kernel launches, `~492k` DtoD copies, and `~690k` alloc/free pairs in the warmup+iter trace

**Interpretation:** the full-attention prefill rewrite worked. The dominant bottleneck is now the linear-attention prefill loop and the large amount of per-token runtime bookkeeping around it. The next meaningful win will come from batching or restructuring linear-attention prefill, especially GDR.

### #2 Standalone HD256 Triton attention (2026-03-21)

**Goal:** Validate the Qwen3.5 full-attention prefill core in Triton before touching model wiring, and drop the old CUDA-vs-Triton microbench duplication.

**Changes:** Added `tools/triton/flash_attention_prefill_hd256_kernel.py`, `flash_attention_prefill_hd256_into()` in Rust/FFI, a focused CPU-reference test, and a Triton-only `qwen35_prefill_attn` microbench. Removed the legacy CUDA attention microbench entries so attention coverage now tracks only the Triton path.

**Validated commands:**
- `cargo test --release test_flash_attention_prefill_hd256_matches_cpu_reference -- --ignored --nocapture`
- `cargo bench --bench ops_bench -- qwen35_prefill_attn`

**Microbench result (attention only; excludes QK norm, partial RoPE, KV write, and output gate):**
- seq128: 30.9us
- seq512: 151us
- seq2048: 1.43ms

**Interpretation:** The batched HD256 attention core is no longer the blocking concern. The remaining work is the Qwen3.5-specific prep and gate wiring around it.

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
