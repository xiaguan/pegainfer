// csrc/flash_mla_prefill.cu
//
// Torch-free C wrapper for FlashMLA SM90 sparse prefill attention (NSA).
// Fills SparseAttnFwdParams from raw pointers and calls the kernel.
//
// DSV3.2 MLA absorbed dimensions: d_qk=576, d_v=512.
// Kernel source: third_party/FlashMLA/csrc/sm90/prefill/sparse/

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "params.h"
#include "sm90/prefill/sparse/fwd.h"

static constexpr float LOG_2_E = 1.4426950408889634f;

extern "C" {

/// FlashMLA sparse prefill (NSA) for V3.2 MLA.
///
/// q:       [s_q, h_q, d_qk]  bf16  — absorbed Q (after W_UK + RoPE concat)
/// kv:      [s_kv, h_kv, d_qk] bf16 — KV cache (compressed latent + RoPE)
/// indices: [s_q, h_kv, topk]  i32  — per-query block indices for sparse attention
/// out:     [s_q, h_q, d_v]    bf16 — output
/// max_logits: [s_q, h_q]      f32  — per-head max logits (for combine)
/// lse:     [s_q, h_q]         f32  — log-sum-exp (for combine)
void flash_mla_sparse_prefill(
    void* q,
    void* kv,
    int* indices,
    void* out,
    float* max_logits,
    float* lse,
    int s_q,
    int s_kv,
    int h_q,
    int h_kv,
    int d_qk,
    int d_v,
    int topk,
    float sm_scale,
    int num_sm,
    cudaStream_t stream)
{
    using bf16 = cutlass::bfloat16_t;

    SparseAttnFwdParams params = {};
    params.s_q = s_q;
    params.s_kv = s_kv;
    params.h_q = h_q;
    params.h_kv = h_kv;
    params.d_qk = d_qk;
    params.d_v = d_v;
    params.topk = topk;
    params.sm_scale = sm_scale;
    params.sm_scale_div_log2 = sm_scale * LOG_2_E;

    params.q = reinterpret_cast<bf16*>(q);
    params.kv = reinterpret_cast<bf16*>(kv);
    params.indices = indices;
    params.attn_sink = nullptr;
    params.topk_length = nullptr;

    // Strides: contiguous layout [s, h, d]
    params.stride_q_s_q = h_q * d_qk;
    params.stride_q_h_q = d_qk;
    params.stride_kv_s_kv = h_kv * d_qk;
    params.stride_kv_h_kv = d_qk;
    params.stride_indices_s_q = h_kv * topk;
    params.stride_indices_h_kv = topk;

    params.out = reinterpret_cast<bf16*>(out);
    params.max_logits = max_logits;
    params.lse = lse;

    params.num_sm = num_sm;
    params.stream = stream;

    sm90::run_fwd_kernel(params);
}

}  // extern "C"
