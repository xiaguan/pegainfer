#include "common.cuh"

#define HEAD_DIM 128

// ============================================================================
// Kernel 1: Per-head QK RMSNorm + RoPE (in-place on Q and K batches)
//
// Grid: (num_q_heads + num_kv_heads, seq_len)
// Block: head_dim (128) threads
// Each block normalizes one head of one token, then applies RoPE.
// ============================================================================
__global__ void prefill_qk_norm_rope_kernel(
    __nv_bfloat16* __restrict__ q,        // [q_dim, seq_len] modified in-place
    __nv_bfloat16* __restrict__ k,        // [kv_dim, seq_len] modified in-place
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [head_dim]
    const __nv_bfloat16* __restrict__ cos_cache,      // [max_pos * head_dim]
    const __nv_bfloat16* __restrict__ sin_cache,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int q_dim, int kv_dim, int start_pos,
    const int* start_pos_d,  // if non-null, *start_pos_d overrides start_pos (CUDA Graph safe)
    float eps
) {
    int head_global = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    bool is_q = (head_global < num_q_heads);
    int head_local = is_q ? head_global : (head_global - num_q_heads);
    __nv_bfloat16* data = is_q ? q : k;
    int dim_stride = is_q ? q_dim : kv_dim;
    const __nv_bfloat16* norm_w = is_q ? q_norm_weight : k_norm_weight;

    int offset = head_local * head_dim + d + token * dim_stride;
    float val = __bfloat162float(data[offset]);

    // RMSNorm: sum of squares via warp reduction
    float sq = val * val;
    sq = warp_reduce_sum(sq);

    int warp_id = d / WARP_SIZE;
    int lane_id = d % WARP_SIZE;
    __shared__ float warp_sums[4];  // head_dim/32 = 4 warps
    if (lane_id == 0) warp_sums[warp_id] = sq;
    __syncthreads();

    __shared__ float s_inv_rms;
    if (warp_id == 0) {
        float v = (lane_id < 4) ? warp_sums[lane_id] : 0.0f;
        float total = warp_reduce_sum(v);
        if (lane_id == 0) s_inv_rms = rsqrtf(total / head_dim + eps);
    }
    __syncthreads();

    // Match HF precision: round to bf16 after norm, then multiply weight
    __nv_bfloat16 normed = __float2bfloat16(val * s_inv_rms);
    float normed_f = __bfloat162float(normed) * __bfloat162float(norm_w[d]);

    // RoPE via shared memory exchange
    __shared__ __nv_bfloat16 smem[HEAD_DIM];
    smem[d] = __float2bfloat16(normed_f);
    __syncthreads();

    int half = head_dim / 2;
    int actual_start_pos = start_pos_d ? __ldg(start_pos_d) : start_pos;
    int pos = actual_start_pos + token;

    __nv_bfloat16 result;
    if (d < half) {
        float lo = __bfloat162float(smem[d]);
        float hi = __bfloat162float(smem[d + half]);
        float c = __bfloat162float(cos_cache[pos * head_dim + d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + d]);
        result = __float2bfloat16(lo * c - hi * s);
    } else {
        int pair_d = d - half;
        float lo = __bfloat162float(smem[pair_d]);
        float hi = __bfloat162float(smem[d]);
        float c = __bfloat162float(cos_cache[pos * head_dim + pair_d]);
        float s = __bfloat162float(sin_cache[pos * head_dim + pair_d]);
        result = __float2bfloat16(lo * s + hi * c);
    }

    data[offset] = result;
}

// ============================================================================
// Kernel 2: Batch write K and V to KV cache
//
// Grid: (num_kv_heads, seq_len), Block: head_dim
// K is already normed+RoPE'd, V is raw.
// ============================================================================
__global__ void prefill_kv_cache_write_kernel(
    const __nv_bfloat16* __restrict__ k,   // [kv_dim, seq_len] normed+RoPE'd
    const __nv_bfloat16* __restrict__ v,   // [kv_dim, seq_len]
    __nv_bfloat16* __restrict__ k_cache,   // [num_kv_heads * max_seq * head_dim]
    __nv_bfloat16* __restrict__ v_cache,
    int head_dim, int kv_dim, int max_seq_len, int start_pos
) {
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    int d = threadIdx.x;

    int src_offset = kv_head * head_dim + d + token * kv_dim;
    int dst_offset = kv_head * max_seq_len * head_dim + (start_pos + token) * head_dim + d;

    k_cache[dst_offset] = k[src_offset];
    v_cache[dst_offset] = v[src_offset];
}


// ============================================================================
// C API: Prefill attention preparation (QK norm + RoPE + KV cache write)
//
// Steps 1-2 of the prefill attention pipeline:
//   1. Per-head QK norm + RoPE (custom kernel)
//   2. KV cache batch write (custom kernel)
//
// Step 3 (attention computation) is handled by the Triton FlashAttention-2
// kernel, called separately from Rust as flash_attention_prefill_cuda().
// ============================================================================
extern "C" {

void prefill_attention_prep_cuda(
    __nv_bfloat16* q_batch,          // [q_dim, seq_len] modified in-place (normed+RoPE'd)
    __nv_bfloat16* k_batch,          // [kv_dim, seq_len] modified in-place (normed+RoPE'd)
    const __nv_bfloat16* v_batch,    // [kv_dim, seq_len]
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* k_cache,          // [num_kv_heads * max_seq * head_dim]
    __nv_bfloat16* v_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    float rms_eps,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    int max_seq_len = 4096;

    // Step 1: QK norm + RoPE (in-place)
    dim3 norm_grid(num_q_heads + num_kv_heads, seq_len);
    prefill_qk_norm_rope_kernel<<<norm_grid, head_dim, 0, stream>>>(
        q_batch, k_batch, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim,
        seq_len, q_dim, kv_dim, start_pos, /*start_pos_d=*/nullptr, rms_eps
    );

    // Step 2: Write K, V to cache
    dim3 cache_grid(num_kv_heads, seq_len);
    prefill_kv_cache_write_kernel<<<cache_grid, head_dim, 0, stream>>>(
        k_batch, v_batch, k_cache, v_cache,
        head_dim, kv_dim, max_seq_len, start_pos
    );
}

// ============================================================================
// C API: QK norm + RoPE only (no cache write) for decode with paged attention.
//
// CUDA Graph safe: reads position from decode_meta[1] on device.
// decode_meta layout: [token_id, position, seq_len] as int32 on GPU.
// ============================================================================
void qk_norm_rope_cuda(
    __nv_bfloat16* q,                    // [num_q_heads * head_dim] in-place
    __nv_bfloat16* k,                    // [num_kv_heads * head_dim] in-place
    const __nv_bfloat16* q_norm_weight,  // [head_dim]
    const __nv_bfloat16* k_norm_weight,  // [head_dim]
    const __nv_bfloat16* cos_cache,      // [max_pos * head_dim]
    const __nv_bfloat16* sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    const int* decode_meta,              // GPU: [token_id, position, seq_len]
    float rms_eps,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    dim3 grid(num_q_heads + num_kv_heads, 1);  // seq_len=1
    prefill_qk_norm_rope_kernel<<<grid, head_dim, 0, stream>>>(
        q, k, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim,
        /*seq_len=*/1, q_dim, kv_dim, /*start_pos=*/0,
        /*start_pos_d=*/decode_meta + 1,  // points to position field
        rms_eps
    );
}

// ============================================================================
// C API: QK norm + RoPE only (no cache write).
//
// Same as prefill_attention_prep_cuda but skips the KV cache write kernel.
// Used when KV is written to paged layout separately (via AppendPagedKVCache).
// ============================================================================
void prefill_qk_norm_rope_only_cuda(
    __nv_bfloat16* q_batch,          // [q_dim, seq_len] modified in-place (normed+RoPE'd)
    __nv_bfloat16* k_batch,          // [kv_dim, seq_len] modified in-place (normed+RoPE'd)
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int seq_len,
    int start_pos,
    float rms_eps,
    cudaStream_t stream
) {
    int q_dim = num_q_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;

    dim3 norm_grid(num_q_heads + num_kv_heads, seq_len);
    prefill_qk_norm_rope_kernel<<<norm_grid, head_dim, 0, stream>>>(
        q_batch, k_batch, q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        num_q_heads, num_kv_heads, head_dim,
        seq_len, q_dim, kv_dim, start_pos, /*start_pos_d=*/nullptr, rms_eps
    );
}

} // extern "C"
