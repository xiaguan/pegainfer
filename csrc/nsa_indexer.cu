// csrc/nsa_indexer.cu
//
// CUDA kernels for the NSA (Native Sparse Attention) indexer.
//
// 1. LayerNorm with bias (for indexer k_norm)
// 2. Fused indexer score + causal topk (avoids materializing [H, T, T] scores)
// 3. RoPE for indexer q/k (layout: rope first, then nope)

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================================
// LayerNorm with bias (bf16, per-token)
//
// x:      [dim, bs] bf16 column-major (each column = one token)
// weight: [dim] bf16
// bias:   [dim] bf16
// out:    [dim, bs] bf16
// eps:    typically 1e-5
// ============================================================================

__global__ void layernorm_bias_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ bias,
    __nv_bfloat16 *__restrict__ out,
    int dim,
    int bs,
    float eps)
{
    int token = blockIdx.x;
    if (token >= bs) return;

    const __nv_bfloat16 *in_ptr = x + (int64_t)token * dim;
    __nv_bfloat16 *out_ptr = out + (int64_t)token * dim;

    extern __shared__ float smem[];

    // Pass 1: compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += __bfloat162float(in_ptr[i]);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)dim;

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __bfloat162float(in_ptr[i]) - mean;
        local_var += v * v;
    }
    smem[threadIdx.x] = local_var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / (float)dim + eps);

    // Pass 3: normalize with weight and bias
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (__bfloat162float(in_ptr[i]) - mean) * inv_std;
        float w = __bfloat162float(weight[i]);
        float b = __bfloat162float(bias[i]);
        out_ptr[i] = __float2bfloat16(v * w + b);
    }
}

// ============================================================================
// Fused indexer score + causal topk.
//
// Computes: logit[m][n] = sum_h relu(q[m,h,:] · k[n,:]) * weights[m,h] * weight_scale
// for n in [0, m] (causal), then selects topk per row.
//
// q:       [T, H, D] bf16 — indexer query (after RoPE)
// k:       [T, D] bf16    — indexer key (after LN + RoPE)
// weights: [T, H] bf16    — head-mixing weights (from weights_proj GEMM)
// indices: [T, topk] i32  — output: selected KV positions per query
//
// One block per query. Shared memory holds q_m[H*D] + logits[T].
// ============================================================================

__global__ void indexer_fused_score_topk_kernel(
    const __nv_bfloat16 *__restrict__ q,       // [T, H, D]
    const __nv_bfloat16 *__restrict__ k,       // [T, D]
    const __nv_bfloat16 *__restrict__ weights, // [T, H]
    int *__restrict__ indices,                  // [T, topk]
    int T,
    int H,
    int D,
    int topk,
    float weight_scale)
{
    int m = blockIdx.x;
    if (m >= T) return;

    // Shared memory: q_shared[H * D] bf16 | logits[T] f32
    extern __shared__ char smem_raw[];
    __nv_bfloat16 *q_shared = (__nv_bfloat16 *)smem_raw;
    float *logits = (float *)(smem_raw + H * D * sizeof(__nv_bfloat16));

    // Load q[m, :, :] into shared memory
    const __nv_bfloat16 *q_m = q + (int64_t)m * H * D;
    for (int i = threadIdx.x; i < H * D; i += blockDim.x) {
        q_shared[i] = q_m[i];
    }
    __syncthreads();

    // Load weights[m, :] into registers
    // H is typically 64, fits in registers
    const __nv_bfloat16 *w_m = weights + m * H;

    // Compute logit for each key n <= m
    for (int n = threadIdx.x; n < T; n += blockDim.x) {
        if (n > m) {
            logits[n] = -FLT_MAX;
        } else {
            const __nv_bfloat16 *k_n = k + (int64_t)n * D;
            float logit = 0.0f;
            for (int h = 0; h < H; h++) {
                // Dot product q[m, h, :] · k[n, :]
                float dot = 0.0f;
                const __nv_bfloat16 *q_h = q_shared + h * D;
                for (int d = 0; d < D; d++) {
                    dot += __bfloat162float(q_h[d]) * __bfloat162float(k_n[d]);
                }
                // relu
                if (dot > 0.0f) {
                    logit += dot * __bfloat162float(w_m[h]);
                }
            }
            logits[n] = logit * weight_scale;
        }
    }
    __syncthreads();

    // TopK selection (thread 0 only — practical for T <= ~4K)
    if (threadIdx.x == 0) {
        int causal_len = m + 1;
        int k_actual = topk < causal_len ? topk : causal_len;
        int *out = indices + m * topk;

        // Selection sort for top-k
        for (int i = 0; i < k_actual; i++) {
            float best_val = -FLT_MAX;
            int best_idx = 0;
            for (int n = 0; n < causal_len; n++) {
                if (logits[n] > best_val) {
                    best_val = logits[n];
                    best_idx = n;
                }
            }
            out[i] = best_idx;
            logits[best_idx] = -FLT_MAX;
        }

        // Pad with -1 so FlashMLA sparse prefill treats these entries as invalid.
        // Padding with 0 would incorrectly re-attend the first token.
        for (int i = k_actual; i < topk; i++) {
            out[i] = -1;
        }
    }
}

// ============================================================================
// RoPE for indexer: apply to first rope_dim dimensions of q and k.
//
// Indexer layout: rope(rope_dim) + nope(head_dim - rope_dim)
// This is DIFFERENT from main MLA which is nope + rope.
//
// q:   [T, n_heads, head_dim] bf16 — in-place RoPE on q[:, :, 0:rope_dim]
// k:   [T, head_dim] bf16          — in-place RoPE on k[:, 0:rope_dim]
// cos: [max_seq_len, rope_dim] bf16
// sin: [max_seq_len, rope_dim] bf16
// positions: [T] i32
//
// rotate_half style: pairs are (x[i], x[i + half_dim]).
// ============================================================================

__global__ void indexer_rope_kernel(
    __nv_bfloat16 *__restrict__ q,
    __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ cos_cache,
    const __nv_bfloat16 *__restrict__ sin_cache,
    const int *__restrict__ positions,
    int T,
    int n_heads,
    int head_dim,
    int rope_dim)
{
    int t = blockIdx.x;
    if (t >= T) return;

    int pos = positions[t];
    int half_rope = rope_dim / 2;
    const __nv_bfloat16 *cos_ptr = cos_cache + pos * rope_dim;
    const __nv_bfloat16 *sin_ptr = sin_cache + pos * rope_dim;

    // RoPE on q: for each head h, apply to q[t, h, 0:rope_dim]
    for (int idx = threadIdx.x; idx < n_heads * half_rope; idx += blockDim.x) {
        int h = idx / half_rope;
        int i = idx % half_rope;

        int base = t * n_heads * head_dim + h * head_dim;
        float x0 = __bfloat162float(q[base + i]);
        float x1 = __bfloat162float(q[base + i + half_rope]);
        float c = __bfloat162float(cos_ptr[i]);
        float s = __bfloat162float(sin_ptr[i]);

        q[base + i]             = __float2bfloat16(x0 * c - x1 * s);
        q[base + i + half_rope] = __float2bfloat16(x1 * c + x0 * s);
    }

    // RoPE on k: single head, apply to k[t, 0:rope_dim]
    for (int i = threadIdx.x; i < half_rope; i += blockDim.x) {
        int base = t * head_dim;
        float x0 = __bfloat162float(k[base + i]);
        float x1 = __bfloat162float(k[base + i + half_rope]);
        float c = __bfloat162float(cos_ptr[i]);
        float s = __bfloat162float(sin_ptr[i]);

        k[base + i]             = __float2bfloat16(x0 * c - x1 * s);
        k[base + i + half_rope] = __float2bfloat16(x1 * c + x0 * s);
    }
}

// ============================================================================
// Public C API
// ============================================================================

extern "C" {

void nsa_layernorm_bias_cuda(
    const void *x,
    const void *weight,
    const void *bias,
    void *out,
    int dim,
    int bs,
    float eps,
    cudaStream_t stream)
{
    int threads = 256;
    if (dim < 256) threads = 128;
    int smem = threads * sizeof(float);
    layernorm_bias_kernel<<<bs, threads, smem, stream>>>(
        (const __nv_bfloat16 *)x,
        (const __nv_bfloat16 *)weight,
        (const __nv_bfloat16 *)bias,
        (__nv_bfloat16 *)out,
        dim, bs, eps);
}

void nsa_indexer_fused_score_topk_cuda(
    const void *q,
    const void *k,
    const void *weights,
    int *indices,
    int T,
    int H,
    int D,
    int topk,
    float weight_scale,
    cudaStream_t stream)
{
    // Shared memory: q[H * D] bf16 + logits[T] f32
    int smem = H * D * sizeof(__nv_bfloat16) + T * sizeof(float);
    int threads = 256;
    indexer_fused_score_topk_kernel<<<T, threads, smem, stream>>>(
        (const __nv_bfloat16 *)q,
        (const __nv_bfloat16 *)k,
        (const __nv_bfloat16 *)weights,
        indices,
        T, H, D, topk, weight_scale);
}

void nsa_indexer_rope_cuda(
    void *q,
    void *k,
    const void *cos_cache,
    const void *sin_cache,
    const int *positions,
    int T,
    int n_heads,
    int head_dim,
    int rope_dim,
    cudaStream_t stream)
{
    int threads = 256;
    indexer_rope_kernel<<<T, threads, 0, stream>>>(
        (__nv_bfloat16 *)q,
        (__nv_bfloat16 *)k,
        (const __nv_bfloat16 *)cos_cache,
        (const __nv_bfloat16 *)sin_cache,
        positions,
        T, n_heads, head_dim, rope_dim);
}

}  // extern "C"
