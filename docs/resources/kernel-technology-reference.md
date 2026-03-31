# Kernel Technology Reference

> **TL;DR:** pegainfer uses a three-tier operator stack: cuBLAS for dense GEMM, Triton AOT for regular fused and prefill kernels, and handwritten CUDA for decode-critical hotpaths. FlashAttention and FlashInfer serve as algorithm and serving-architecture references, not direct dependencies. Specialist tools (CUTLASS/CuTe, ThunderKittens, TileLang, Gluon) are reserved for proven, durable hot kernels where the default stack has clearly plateaued.
>
> **Status:** Active. Next step: apply this framework to the next operator decision instead of choosing ad hoc.

---

## 1. Current Stack

pegainfer (~7K Rust, ~3.4K CUDA) already mixes three operator backends in production:

| Backend | Role | Scale |
|---------|------|-------|
| **Handwritten CUDA** | GEMV, fused MLP, RMSNorm, sampling, GDR recurrence, Conv1D, fused attention | ~2.6K LOC in `csrc/` |
| **Triton AOT** | SiLU\*gate, Add, Embedding, FlashAttention prefill, GDR chunkwise, decode split-KV attention | ~1.4K LOC in `tools/triton/` |
| **cuBLAS** (via cudarc) | Prefill GEMM, decode GEMV projections | Call-site only |

The build pipeline (`build.rs`) compiles both handwritten CUDA and Triton AOT artifacts, linking them through `src/ffi.rs` into a unified Rust operator surface (`src/ops.rs`). Python is a build-time dependency only; the runtime is pure Rust + GPU.

### Handwritten CUDA kernels

| Kernel | File | Purpose |
|--------|------|---------|
| `gemv.cu` | BF16x4-vectorized GEMV | Row-major M\*V with warp-shuffle + smem reduction |
| `fused_mlp.cu` | Gated MLP | `silu(gate @ x) * (up @ x)` fused into one kernel |
| `norm.cu` | RMSNorm | BF16x4, warp-shuffle, single-block decode path |
| `fused_attention.cu` | GQA decode attention | Tiled online softmax over KV cache, no `MAX_SEQ_LEN` cap |
| `prefill_attention.cu` | QK RMSNorm + RoPE | In-place on Q/K batches before prefill GEMM |
| `prefill_attention_hd256.cu` | HD=256 prefill prep | Four 64-wide chunks to manage register pressure |
| `gated_delta_rule.cu` | GDR recurrent decode | One block per value head, fused recurrence step |
| `conv1d.cu` | Causal depthwise Conv1D | Parallel causal convolution for GDR linear attention |
| `sampling.cu` | Argmax | GPU-side top-1 selection |
| `pos_enc.cu` | RoPE | Rotary position embedding application |

### Triton AOT kernels

| Kernel | File | Purpose |
|--------|------|---------|
| `flash_attention_prefill_hd256_kernel.py` | HD=256 prefill | Adapted tile shapes for larger head dimension |
| `gated_delta_rule_chunkwise_kernels.py` | GDR chunkwise | Linear attention state updates for Qwen3.5 prefill |

---

## 2. Technology Landscape

The following diagram orders kernel authoring technologies from highest abstraction to lowest:

```
  FlashAttention / FlashInfer        Ready-made operator libraries
       |
  Triton (standard @triton.jit)      Python DSL, compiler-managed optimization
       |
  Gluon (triton.experimental)        Triton's low-level DSL: explicit layout, smem, async
       |
  TileLang                           TVM-based tile DSL: explicit smem + pipeline
       |
  ThunderKittens / CUTLASS           C++ template libraries, tile-level abstraction
       |
  Handwritten CUDA                   Full hardware control
```

Each technology is assessed below with source-level detail where the local workspace contains the code.

### 2.1 Triton (Standard Mode)

**When to use:** Elementwise ops (SiLU, Add, Embedding), prototyping, prefill attention.

**Strengths:** The compiler manages layout, shared memory, and vectorization. Python authoring with AOT compilation to C launchers that Rust calls through FFI.

**Weaknesses:** When the compiler cannot optimize well (e.g., M=1 GEMV in decode), there is no manual override. AOT has known bugs (fp32 parameters mapped to double).

**pegainfer experience:** SiLU/Add/Embedding work well in Triton. Prefill FlashAttention performs at parity with vLLM. Decode attention via Triton achieves ~10.4ms TPOT (95.9 tok/s), acceptable but not best-in-class.

Example: pegainfer's prefill attention kernel signature:

```python
@triton.jit
def flash_attention_prefill_kernel(
    Q_ptr,           # [num_q_heads * HEAD_DIM, seq_len] col-major
    K_cache_ptr,     # [num_kv_heads * max_seq * HEAD_DIM] row-major per head
    V_cache_ptr,     # same layout as K
    Output_ptr,      # [num_q_heads * HEAD_DIM, seq_len] col-major
    num_q_heads, num_kv_heads, gqa_ratio,
    seq_len, start_pos, q_dim,
    BLOCK_M: tl.constexpr,    # 128
    BLOCK_N: tl.constexpr,    # 64
    HEAD_DIM: tl.constexpr,   # 128
):
```

### 2.2 Gluon (Triton Experimental DSL)

> Source: `triton/python/triton/experimental/gluon/`
> Tutorials: `triton/python/tutorials/gluon/01-intro.py` through `14-multicta.py`

Gluon shares Triton's compiler but exposes hardware details that standard Triton hides:

| Aspect | Standard Triton | Gluon |
|--------|----------------|-------|
| Tensor layout | Compiler decides | User specifies `BlockedLayout`, `SliceLayout`, `NVMMADistributedLayout` |
| Shared memory | Compiler manages | `gl.allocate_shared_memory()` explicit allocation |
| Async operations | `num_stages` implicit pipeline | TMA + mbarrier explicit producer-consumer |
| Warp specialization | `tl.range(warp_specialize=True)` limited | `gl.warp_specialize()` full multi-partition |
| Tensor Core | `tl.dot()` | `warpgroup_mma()` (Hopper), `tcgen05_mma()` (Blackwell) explicit |

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def kernel(in_ptr, out_ptr, XBLOCK: gl.constexpr, layout: gl.constexpr):
    indices = gl.arange(0, XBLOCK, layout=layout)  # layout must be specified
    value = gl.load(in_ptr + indices)
    gl.store(out_ptr + indices, value)
```

**Performance reference (GB200 tutorials):** memcpy 6.6 TB/s (~82% peak), elementwise add 5.98 TB/s (warp-specialized), GEMM 1317 TFLOPS vs cuBLAS 1432 TFLOPS.

**Assessment for pegainfer:** Gluon is the natural escalation path when standard Triton plateaus. However: the API is experimental, AOT support is unverified, and RTX 5070 Ti (SM120) compatibility needs testing. Tutorials target GB200 datacenter GPUs.

### 2.3 TileLang

> GitHub: `tile-ai/tilelang` (~5.4k stars). Backend: Apache TVM TIR.

```python
@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def kernel(A, B, C):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by*block_M, ko*block_K], A_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by*block_M, bx*block_N])
    return kernel
```

**Assessment for pegainfer: Not recommended.** Adds a third compiler stack (CUDA + Triton + TVM). The decode path is memory-bound GEMV where TileLang's tile-GEMM optimization does not apply. Revisit only if AMD compatibility becomes a requirement.

### 2.4 ThunderKittens

> GitHub: `HazyResearch/ThunderKittens` (~3.3k stars). Paper: arXiv:2410.20399.

ThunderKittens treats the GPU as a matrix-multiply machine where the 16x16 tile is the atomic unit. Everything is built around four types: `rt` (register tile), `st` (shared tile), `rv` (register vec), `sv` (shared vec).

```cpp
kittens::rt_bf<32,16> reg_tile;           // register tile: 32x16 bf16
__shared__ kittens::st_hf<32,64> smem;    // shared memory tile: 32x64 fp16
kittens::warpgroup::mma_AB(acc, A, B);    // tensor core MMA
```

**Hardware support:** Hopper H100/H200 (SM90, primary), Blackwell B200 (SM100, TK 2.0). Ampere deprecated. **RTX 5070 Ti (SM120) is not supported** -- confirmed by build config (`kernels/common.mk` only lists SM80/90/100/103a).

The H100 MHA kernel (`mha_h100.cu`) demonstrates the producer/consumer warpgroup pattern central to ThunderKittens:

```cuda
constexpr int CONSUMER_WARPGROUPS = 3;
constexpr int PRODUCER_WARPGROUPS = 1;

__shared__ kittens::semaphore k_smem_arrived[K::stages],
                               v_smem_arrived[K::stages],
                               compute_done[K::stages];

// Producer warpgroup: fetches K/V tiles via TMA
if (warpgroupid == NUM_WARPGROUPS - 1) {
    warpgroup::decrease_registers<32>();
    for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
        coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
        warp::tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx,
                              k_smem_arrived[(kv_idx+1)%K::stages]);
        // ... load V similarly
        wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
    }
}
// Consumer warpgroups: compute QK^T, softmax, PV
else {
    warpgroup::increase_registers<160>();
    rt_fl<16, K::kv_height> att_block;
    // ... online softmax + MMA loop
}
```

Key design observations:
- Explicit register budget management (`increase_registers` / `decrease_registers`)
- Semaphore-based producer-consumer pipeline coordination
- TMA async loads issued directly from kernel code
- Warpgroup-level parallelism as a first-class concern

**Performance:** Matches cuBLAS on GEMM, 10-40% faster than FA3 on attention backward, 14x speedup on linear attention, >85% peak FLOPs on H100.

**Assessment for pegainfer: Not recommended as a dependency.** SM120 is unsupported, the decode bottleneck is memory-bound GEMV (not compute-bound MMA), and C++ template integration requires an extra FFI layer. However, its linear attention kernel's scan pattern is a valuable **reference** for GDR optimization, and the producer/consumer pipeline style is worth learning for future hot kernels.

### 2.5 CUTLASS / CuTe

> GitHub: `NVIDIA/cutlass` -- NVIDIA official.

CUTLASS 3.x is built on CuTe (Layout algebra), providing a full stack from MMA atoms to complete GEMM:

1. **Atom** -- PTX MMA/Copy instructions
2. **TiledMMA / TiledCopy** -- Cross-thread/warp tiling
3. **Layout algebra** -- (Shape, Stride) composition, complement, product
4. **Collective** -- Multi-CTA cooperation
5. **Kernel** -- GEMM mainloop + epilogue

**vs ThunderKittens:** CUTLASS has broader architecture support (Ampere through Blackwell), is production-grade (powers cuBLAS/cuDNN internally), but has a steeper learning curve due to its abstract layout algebra. ThunderKittens is more intuitive for tile-level programming but targets only datacenter GPUs.

**Assessment for pegainfer:** Not needed now. cuBLAS already uses CUTLASS internally. Becomes relevant when FP8/FP4 quantized GEMM or fused epilogues are needed.

### 2.6 Helion

> Path: `vllm/vllm/kernels/helion/`

A Python kernel framework used by vLLM for managing pre-tuned kernel configs with autotuning infrastructure. **Not applicable to pegainfer** -- Python-only ecosystem, incompatible with Rust + CUDA/Triton AOT.

---

## 3. Reference Libraries: What to Learn, What Not to Copy

These projects are references, not dependencies. The goal is selective borrowing of ideas, not wholesale adoption.

### 3.1 FlashAttention: Attention Algorithm Reference

> GitHub: `Dao-AILab/flash-attention`. Versions: FA2 (SM80+), FA3 (SM90, WGMMA+TMA), FA4 (SM90/100, CuTeDSL).

**Source structure** (local: `flash-attention/csrc/flash_attn/src/`):
- `flash.h` -- runtime params struct (`Flash_fwd_params`)
- `kernel_traits.h` -- compile-time tile shapes, shared memory layout, copy atoms, MMA layout
- `flash_fwd_kernel.h` -- forward kernel mainloop
- `softmax.h` -- online softmax helpers

The forward kernel iterates KV blocks against one Q row-block in reverse order (for causal masking efficiency), maintaining online softmax state throughout:

```cuda
// flash_fwd_kernel.h -- simplified skeleton
template<typename Kernel_traits, bool Is_causal, ...>
__device__ void compute_attn_1rowblock(const Params &params, int bidb, int bidh, int m_block) {
    // Load Q tile: (kBlockM x kHeadDim) from global memory via CuTe
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    // Setup K/V access with GQA ratio
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    // Main loop: iterate KV blocks in reverse for causal efficiency
    for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
        // 1. Load K/V tiles into shared memory
        // 2. Compute QK^T (attention scores)
        // 3. Apply causal/local masking
        // 4. Online softmax rescale (see below)
        // 5. Accumulate O = softmax(QK^T) @ V
    }
    // Epilogue: normalize by row_sum, write output
}
```

The core numerical insight is in `softmax.h` -- online max/sum accumulation using `exp2` scaling with output rescaling on max change:

```cuda
// softmax.h -- online softmax rescale
template<bool Is_first, bool Check_inf, typename Tensor0, typename Tensor1>
__device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &acc_o, float scale_log2) {
    if (Is_first) {
        reduce_max</*zero_init=*/true>(scores, row_max);
        scale_apply_exp2(scores, row_max, scale_log2);
        reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
        Tensor scores_max_prev = make_fragment_like(row_max);
        copy(row_max, scores_max_prev);
        reduce_max</*zero_init=*/false>(scores, row_max);

        // Rescale previous accumulator when max changes
        for (int mi = 0; mi < size(row_max); ++mi) {
            float scores_scale = exp2f((scores_max_prev(mi) - row_max(mi)) * scale_log2);
            row_sum(mi) *= scores_scale;
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
                acc_o_rowcol(mi, ni) *= scores_scale;
        }
        scale_apply_exp2(scores, row_max, scale_log2);
        reduce_sum</*zero_init=*/false>(scores, row_sum);
    }
};
```

**What to learn:**
- Separate runtime params from compile-time tile traits
- Isolate online softmax into reusable helpers
- Keep Q/K/V movement independent from mask/RoPE/dropout policy
- Treat GQA as a first-class kernel concern

**What not to copy:**
- The full CuTe/CUTLASS template stack (too heavy for pegainfer's needs)
- Training-oriented complexity irrelevant to inference
- FA4's CuTeDSL -- even harder to extract as standalone C

**Limitations for pegainfer:** Only covers softmax attention (8/32 layers in Qwen3.5). The 24 GDR linear attention layers still require custom kernels. Deep PyTorch coupling makes standalone C API extraction impractical.

### 3.2 FlashInfer: Serving Architecture Reference

> GitHub: `flashinfer-ai/flashinfer` (~5.2k stars). Header-only C++ (`include/flashinfer/*.cuh`). Supports SM 7.5 through SM 12.1 including consumer Blackwell. Used by vLLM, SGLang, TRT-LLM.

FlashInfer is not just a kernel collection but a serving-oriented operator system organized around the `plan() -> run()` execution model.

**Attention state** (`state.cuh`) -- the mergeable online softmax state that enables split-KV and multi-stage execution:

```cuda
template <size_t vec_size>
struct state_t {
    vec_t<float, vec_size> o;  // weighted sum: exp(logit - m) * v / d
    float m;                    // running max of pre-softmax logits
    float d;                    // sum of exp(logit - m)

    __device__ void init() { o.fill(0.f); m = -inf; d = 1.f; }

    __device__ void merge(const vec_t<float, vec_size>& other_o,
                          float other_m, float other_d) {
        float m_prev = m, d_prev = d;
        m = max(m_prev, other_m);
        d = d_prev * ptx_exp2(m_prev - m) + other_d * ptx_exp2(other_m - m);
        for (size_t i = 0; i < vec_size; ++i)
            o[i] = o[i] * ptx_exp2(m_prev - m) + other_o[i] * ptx_exp2(other_m - m);
    }

    __device__ void normalize() {
        for (size_t i = 0; i < vec_size; ++i) o[i] = __fdividef(o[i], d);
    }
};
```

**Paged KV cache** (`page.cuh`) -- the storage abstraction that makes continuous batching practical:

```cuda
template <typename DType, typename IdType>
struct paged_kv_t {
    uint_fastdiv page_size;
    uint32_t num_heads, head_dim, batch_size;
    uint32_t stride_page, stride_n, stride_h;

    DType *k_data, *v_data;     // all pages in contiguous storage
    IdType *indices;             // page index mapping
    IdType *indptr;              // [batch_size+1] per-sequence page offsets
    IdType *last_page_len;       // [batch_size] tokens in last page

    __device__ uint32_t get_length(uint32_t batch_idx) const {
        return (indptr[batch_idx+1] - indptr[batch_idx] - 1) * page_size
               + last_page_len[batch_idx];
    }

    __device__ DType* get_k_ptr(IdType page_iter, uint32_t head, uint32_t entry, uint32_t feat) {
        return k_data + get_elem_offset(__ldg(indices + page_iter), head, entry, feat);
    }
};
```

**Decode kernel pipeline** (`decode.cuh`) -- multi-stage `cp_async` prefetch with online state updates:

```cuda
// Stage 1: Preload K/V tiles into shared memory ring buffer
for (uint32_t iter = 0; iter < num_stages_smem; ++iter) {
    cp_async::pred_load<vec_bits, PrefetchMode::kPrefetch, SharedMemFillMode::kNoFill>(
        k_smem + offset, k + global_offset, pred);
    cp_async::commit_group();
    // ... same for V
    cp_async::commit_group();
}

// Stage 2: Pipelined compute -- overlap next tile load with current tile compute
state_t<vec_size> st_local;
for (uint32_t iter = 0; iter < num_iters; ++iter) {
    cp_async::wait_group<2 * num_stages_smem - 1>();
    __syncthreads();

    compute_qk(..., k_smem + stage_offset, q_vec, s, st_local);
    __syncthreads();

    // Prefetch next K tile while computing with current V
    cp_async::pred_load<...>(k_smem + next_stage, k + next_offset, pred);
    cp_async::commit_group();

    cp_async::wait_group<2 * num_stages_smem - 1>();
    __syncthreads();
    update_local_state(v_smem + stage_offset, s, st_local);
    __syncthreads();

    // Prefetch next V tile
    cp_async::pred_load<...>(v_smem + next_stage, v + next_offset, pred);
    cp_async::commit_group();
    stage_idx = (stage_idx + 1) % num_stages_smem;
}
```

FlashInfer also provides GDN (Gated Delta Network) support in the submodule (`flashinfer/gdn_prefill.py`, `flashinfer/gdn_decode.py`), but the current pegainfer integration boundary is still not ready to consume it directly:

- The GDN code exists in the submodule, but pegainfer does not yet expose a matching C++/FFI entrypoint. The currently usable FlashInfer path is the header-only paged-attention integration via thin C wrappers.
- FlashInfer GDN's expected state layout is not directly compatible with pegainfer's current `RecurrentState` / `LayerRecurrentState` representation, so plugging it in would require a separate state-pool / layout adaptation layer.
- Short-term strategy: keep Qwen3.5 linear attention on the existing GDR path, and evaluate FlashInfer GDN as a separate workstream instead of mixing it into the critical-path batch/scheduler PRs.

**What to learn:**
- Prefill and decode as separate optimization problems
- KV layout as a first-class abstraction before adding batching
- Plan-run phase separation when launch metadata becomes nontrivial
- Split-KV scheduling in a dedicated planner, not ad hoc call sites
- Mergeable attention state enabling distributed/split execution
- CUDA-graph-friendly metadata design

**What not to copy:**
- The full Python wrapper API surface and JIT compilation system
- Architecture-matrix coverage for every GPU generation
- TVM-FFI module loading (pegainfer uses Rust FFI)

**Integration path when needed:** Header-only C++ -> thin C wrapper -> Rust FFI. Compatible with pegainfer's existing architecture. Most valuable when continuous batching and PagedAttention become requirements.

---

## 4. Decision Framework

### 4.1 Decision Ladder

When adding or rewriting an operator, choose the highest tier that meets the performance target:

| Tier | Default Tool | Use For | Escalate When |
|------|-------------|---------|---------------|
| 0 | cuBLAS / cuBLASLt | Dense GEMM, standard GEMV | Bottleneck is fusion, irregular layout, or launch structure |
| 1 | Triton AOT | Elementwise, embedding, prefill attention, chunkwise tiled kernels, prototypes | Need warpgroup/TMA/cluster control or Triton plateaus on target GPU |
| 2 | Handwritten CUDA | Decode-critical kernels, unusual sync, runtime-state-coupled kernels | Kernel proves stable and architecture-specific enough for templates |
| 3 | CUTLASS / CuTe C++ | Permanent hot kernels needing TMA, WGMMA, cluster control | Authoring cost blocks experimentation |
| 4 | Gluon / CuTe DSL | Bridge between Triton and full C++ template stacks | Kernel becomes core infrastructure needing conservative maintenance |
| 5 | ThunderKittens / TileLang | Targeted experiments only | Project adopts their ecosystem as strategic dependency |

**The rule: do not jump to a lower tier because it is more "hardcore".**

### 4.2 Per-Operator Recommendations

| Operator Family | Default | Rationale |
|----------------|---------|-----------|
| **Prefill GEMM** | cuBLAS | Standard dense math, not worth replacing |
| **Decode GEMV** | cuBLAS + handwritten CUDA | Memory-bound; needs bf16x2 vectorization, register accumulation |
| **Elementwise** (SiLU, Add) | Triton AOT | Fast to write, performance sufficient |
| **Embedding** | Triton AOT | Memory-bound gather, Triton handles well |
| **RMSNorm / fused add+norm** | Handwritten CUDA | Needs fused variants; CUDA provides flexibility |
| **Prefill attention** | Triton AOT | FA2 algorithm in Triton, at parity with vLLM |
| **Decode attention** | Handwritten CUDA / Triton split-KV | Latency-critical path, CUDA Graph stability matters |
| **GDR recurrence** | Handwritten CUDA + Triton AOT | No library support; must be custom |
| **Sampling** | Handwritten CUDA | GPU-side top-k/argmax |
| **PagedAttention** (future) | FlashInfer patterns | Do not reinvent; use header-only C++ integration |
| **FP8/FP4 GEMM** (future) | CUTLASS via FlashInfer | Standard quantized GEMM path |

### 4.3 Performance Escalation Path

```
1. Write in Triton AOT -> verify correctness
   |
   v  Performance gap?
2. Profile: compute-bound or memory-bound?
   |                          |
   v (memory-bound)           v (compute-bound)
3. Handwritten CUDA:          4. Triton + tuning
   manual vectorization,         -> Gluon (explicit control)
   register accumulation,           -> CUTLASS/CuTe (last resort)
   kernel fusion
```

### 4.4 Maturity Tiers

```
Production-ready:  cuBLAS, FlashInfer, CUTLASS, Handwritten CUDA
Stable:            Triton standard, Triton AOT (with known workarounds)
Experimental:      Gluon, TileLang, ThunderKittens
Not applicable:    Helion (Python-only)
```

---

## 5. Roadmap

### Phase 0: Formalize the Existing Stack (current)

- Maintain `cuBLAS + handwritten CUDA + Triton AOT` as the production baseline
- Tag each hot operator with its backend in benchmark logs
- Require any new backend adoption to beat the current path on both speed and maintenance cost

### Phase 1: Expand Triton Where Cost/Benefit Is Best

Prioritize Triton for:
- Additional memory-bound glue kernels
- Sampling experiments
- Nonstandard attention subkernels
- Recurrent/chunkwise kernels where the algorithm is still evolving

Do not spend Phase 1 effort rewriting mature GEMM-class math.

### Phase 2: Import FlashInfer Serving Patterns

Trigger: pegainfer begins work on continuous batching, paged KV cache, shared-prefix reuse, or batched decode.

Borrow:
- Paged KV cache layout and `paged_kv_t` abstraction
- Plan-run metadata preparation
- Split-KV scheduling
- CUDA-graph-friendly decode wrappers
- Mergeable `state_t` for distributed softmax

Integration: header-only C++ -> thin C wrapper -> Rust FFI.

### Phase 3: Specialist Pilot (one kernel, one GPU)

Only after Phases 1 and 2. Run a focused bake-off:
- One permanently hot kernel (from `nsys` profile)
- One hardware target
- One specialist stack (Gluon, CUTLASS/CuTe, or ThunderKittens)

Success criteria:
- Measurable end-to-end gain, not just microbenchmark win
- Manageable build and FFI story
- No accuracy regression
- Acceptable code ownership cost

If the pilot fails any criterion, keep the simpler backend.

---

## 6. Maintenance Rules

Every kernel in pegainfer must satisfy:

1. **One Rust API surface** -- regardless of whether the backend is cuBLAS, CUDA, or Triton
2. **One correctness story** -- reference path, parity test, and failure triage must be explicit
3. **One benchmark owner** -- every hot kernel maps to a reproducible shape and profiling workflow
4. **One reason to exist** -- no new kernel lands without a measured bottleneck and clear target metric
5. **One-way escalation** -- DSL to CUDA, or CUDA to CuTe, must be driven by measured ceilings, not aesthetics

---

## 7. Local Source Reference Paths

| Project | Path |
|---------|------|
| Triton (including Gluon) | `/data/code/workspace-rustllm/triton/` |
| Gluon tutorials (01-14) | `triton/python/tutorials/gluon/` |
| Gluon language core | `triton/python/triton/experimental/gluon/language/` |
| FlashInfer | `/data/code/workspace-rustllm/flashinfer/` |
| FlashInfer C++ headers | `flashinfer/include/flashinfer/` |
| FlashAttention | `/data/code/workspace-rustllm/flash-attention/` |
| ThunderKittens | `/data/code/workspace-rustllm/ThunderKittens/` |
| TK kernels | `ThunderKittens/kernels/` |
| vLLM (Helion etc.) | `/data/code/workspace-rustllm/vllm/` |
