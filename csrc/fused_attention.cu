#include "common.cuh"
#include <cstdio>

// ============================================================================
// Fused GQA Attention Kernel (bf16 version) — Tiled Online Softmax
//
// Processes KV cache in tiles of TILE_SIZE from global memory using the
// online softmax algorithm. No MAX_SEQ_LEN cap — supports full causal
// attention up to max_seq_len (4096).
//
// Architecture:
// - Each block processes 1 KV head + gqa_ratio Q heads (passed as param)
// - Tiles of K/V loaded from global cache into shared memory
// - Online softmax merges partial results across tiles
// - bf16 storage, fp32 accumulators
// ============================================================================

#define TILE_SIZE 64
#define HEAD_DIM 128
#define THREADS_PER_BLOCK 128
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)  // 4

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ __nv_bfloat16 rms_norm_elem(
    __nv_bfloat16 x,
    float rms_inv,
    __nv_bfloat16 weight
) {
    float val = __bfloat162float(x) * rms_inv * __bfloat162float(weight);
    return __float2bfloat16(val);
}

__device__ __forceinline__ void apply_rope_pair(
    __nv_bfloat16& x0,
    __nv_bfloat16& x1,
    __nv_bfloat16 cos_val,
    __nv_bfloat16 sin_val
) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);

    float temp = fx0;
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(temp * fs + fx1 * fc);
}

// ============================================================================
// Tiled attention for a single Q head using online softmax.
//
// All shared memory buffers are allocated by the caller (kernel) and passed in.
// No __shared__ declarations inside this function.
// ============================================================================
__device__ void tiled_attention(
    const __nv_bfloat16* __restrict__ smem_q,
    const __nv_bfloat16* __restrict__ k_cache_base,
    const __nv_bfloat16* __restrict__ v_cache_base,
    __nv_bfloat16* __restrict__ smem_k,       // [TILE_SIZE * HEAD_DIM]
    __nv_bfloat16* __restrict__ smem_v,       // [TILE_SIZE * HEAD_DIM]
    float* __restrict__ smem_scores,           // [TILE_SIZE]
    float* __restrict__ warp_partial,          // [NUM_WARPS * (TILE_SIZE + 1)]
    float* __restrict__ smem_scratch,          // [NUM_WARPS] scratch for reductions
    float& smem_running_max,
    float& smem_running_sum,
    __nv_bfloat16* __restrict__ output_buf,
    int q_head_idx,
    int seq_len,
    int max_seq_len,
    int head_dim,
    float scale,
    int tid,
    int warp_id,
    int lane_id
) {
    // Initialize online softmax state
    float o_acc = 0.0f;  // output accumulator for dimension tid (register)

    if (tid == 0) {
        smem_running_max = -INFINITY;
        smem_running_sum = 0.0f;
    }
    __syncthreads();

    // Tile loop over KV cache
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, seq_len - tile_start);

        // --- Load K/V tile from global cache into shared memory ---
        for (int i = tid; i < tile_len * HEAD_DIM; i += THREADS_PER_BLOCK) {
            int pos_in_tile = i / HEAD_DIM;
            int dim = i % HEAD_DIM;
            int abs_pos = tile_start + pos_in_tile;
            smem_k[pos_in_tile * HEAD_DIM + dim] = k_cache_base[abs_pos * head_dim + dim];
            smem_v[pos_in_tile * HEAD_DIM + dim] = v_cache_base[abs_pos * head_dim + dim];
        }
        __syncthreads();

        // --- Compute scores: Q · K^T * scale ---
        // Thread-per-dimension dot product, warp reduce, cross-warp combine
        float q_val = __bfloat162float(smem_q[tid]);

        for (int pos = 0; pos < tile_len; pos++) {
            float partial = q_val * __bfloat162float(smem_k[pos * HEAD_DIM + tid]);
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                warp_partial[warp_id * (TILE_SIZE + 1) + pos] = partial;
            }
        }
        __syncthreads();

        // Combine warp partials into final scores
        if (tid < tile_len) {
            float score = 0.0f;
            for (int w = 0; w < NUM_WARPS; w++) {
                score += warp_partial[w * (TILE_SIZE + 1) + tid];
            }
            smem_scores[tid] = score * scale;
        }
        __syncthreads();

        // --- Find tile max ---
        float tile_max_local = -INFINITY;
        if (tid < tile_len) {
            tile_max_local = smem_scores[tid];
        }
        tile_max_local = warp_reduce_max(tile_max_local);

        if (lane_id == 0) {
            smem_scratch[warp_id] = tile_max_local;
        }
        __syncthreads();

        // Thread 0: compute tile_max, online softmax merge, broadcast scale_old
        if (tid == 0) {
            float tile_max = smem_scratch[0];
            for (int i = 1; i < NUM_WARPS; i++) {
                tile_max = fmaxf(tile_max, smem_scratch[i]);
            }
            float old_max = smem_running_max;
            float new_max = fmaxf(old_max, tile_max);
            float scale_old = expf(old_max - new_max);
            smem_running_sum *= scale_old;
            smem_running_max = new_max;
            // Broadcast scale_old via smem_scratch[0]
            smem_scratch[0] = scale_old;
        }
        __syncthreads();

        // ALL threads rescale their output accumulator
        float scale_old = smem_scratch[0];
        o_acc *= scale_old;

        // --- Exp weights + V accumulation ---
        float local_sum = 0.0f;
        float current_max = smem_running_max;
        for (int pos = 0; pos < tile_len; pos++) {
            float w = expf(smem_scores[pos] - current_max);
            local_sum += w;
            o_acc += w * __bfloat162float(smem_v[pos * HEAD_DIM + tid]);
        }

        // local_sum is identical across all threads (smem_scores is shared)
        if (tid == 0) {
            smem_running_sum += local_sum;
        }
        __syncthreads();
    }

    // --- Final normalize ---
    float final_sum = smem_running_sum;
    float result = (final_sum > 0.0f) ? (o_acc / final_sum) : 0.0f;
    output_buf[q_head_idx * head_dim + tid] = __float2bfloat16(result);
}

// ============================================================================
// Main kernel — supports arbitrary GQA ratio via gqa_ratio parameter
// ============================================================================
__global__ void fused_gqa_attention_single_token_kernel(
    const __nv_bfloat16* __restrict__ q_full,
    const __nv_bfloat16* __restrict__ k_full,
    const __nv_bfloat16* __restrict__ v_full,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_cache,
    const __nv_bfloat16* __restrict__ sin_cache,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ output,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int current_pos,
    int seq_len,
    int max_seq_len,
    float scale,
    float rms_eps
) {
    int kv_head_idx = blockIdx.x;

    int tid = threadIdx.x;  // 0..127
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Shared memory layout — all buffers declared here, passed to device functions
    __shared__ __nv_bfloat16 smem_k[TILE_SIZE * HEAD_DIM];       // 16,384 B
    __shared__ __nv_bfloat16 smem_v[TILE_SIZE * HEAD_DIM];       // 16,384 B
    __shared__ __nv_bfloat16 smem_q[HEAD_DIM];                   // 256 B (reused per Q head)
    __shared__ float smem_scores[TILE_SIZE];                      // 256 B
    __shared__ float warp_partial[NUM_WARPS * (TILE_SIZE + 1)];   // 1,040 B
    __shared__ float smem_scratch[NUM_WARPS];                     // 16 B
    __shared__ float smem_rms[2];                                 // 8 B
    __shared__ float smem_running_max;                            // 4 B
    __shared__ float smem_running_sum;                            // 4 B
    // Total: ~34.0 KB (fits 48 KB limit)

    int cache_base_offset = kv_head_idx * max_seq_len * head_dim;

    // ========================================================================
    // Phase 1: K head — slice → norm → rope → write to global cache
    // ========================================================================
    __nv_bfloat16 k_elem = k_full[kv_head_idx * head_dim + tid];

    float k_sq = __bfloat162float(k_elem);
    k_sq = k_sq * k_sq;
    float k_sq_sum = warp_reduce_sum(k_sq);

    if (lane_id == 0) {
        smem_scratch[warp_id] = k_sq_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS; i++) {
            total += smem_scratch[i];
        }
        smem_rms[1] = 1.0f / sqrtf(total / head_dim + rms_eps);
    }
    __syncthreads();

    __nv_bfloat16 k_normed = rms_norm_elem(k_elem, smem_rms[1], k_norm_weight[tid]);

    // Half-split RoPE: pair (tid, tid + half_dim), only threads 0..half_dim-1 rotate
    int half_dim = head_dim / 2;
    // Store normed K in shared memory so paired thread can read it
    smem_k[tid] = k_normed;
    __syncthreads();

    if (tid < half_dim) {
        __nv_bfloat16 k_lo = smem_k[tid];
        __nv_bfloat16 k_hi = smem_k[tid + half_dim];

        apply_rope_pair(k_lo, k_hi, cos_cache[tid], sin_cache[tid]);

        int cache_offset = cache_base_offset + current_pos * head_dim;
        k_cache[cache_offset + tid] = k_lo;
        k_cache[cache_offset + tid + half_dim] = k_hi;
    }
    __syncthreads();

    // ========================================================================
    // Phase 2: V head — slice → write to global cache
    // ========================================================================
    __nv_bfloat16 v_elem = v_full[kv_head_idx * head_dim + tid];
    v_cache[cache_base_offset + current_pos * head_dim + tid] = v_elem;
    __syncthreads();

    // ========================================================================
    // Phase 3: Loop over all Q heads for this KV head
    // ========================================================================
    for (int q = 0; q < gqa_ratio; q++) {
        int q_head_idx = kv_head_idx * gqa_ratio + q;
        if (q_head_idx >= num_qheads) break;

        // Q head — slice → norm → rope → smem_q
        __nv_bfloat16 q_elem = q_full[q_head_idx * head_dim + tid];

        float q_sq = __bfloat162float(q_elem);
        q_sq = q_sq * q_sq;
        float q_sq_sum = warp_reduce_sum(q_sq);

        if (lane_id == 0) {
            smem_scratch[warp_id] = q_sq_sum;
        }
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < NUM_WARPS; i++) {
                total += smem_scratch[i];
            }
            smem_rms[0] = 1.0f / sqrtf(total / head_dim + rms_eps);
        }
        __syncthreads();

        __nv_bfloat16 q_normed = rms_norm_elem(q_elem, smem_rms[0], q_norm_weight[tid]);

        // Half-split RoPE: pair (tid, tid + half_dim)
        smem_q[tid] = q_normed;
        __syncthreads();

        if (tid < half_dim) {
            __nv_bfloat16 q_lo = smem_q[tid];
            __nv_bfloat16 q_hi = smem_q[tid + half_dim];

            apply_rope_pair(q_lo, q_hi, cos_cache[tid], sin_cache[tid]);

            smem_q[tid] = q_lo;
            smem_q[tid + half_dim] = q_hi;
        }
        __syncthreads();

        // Tiled attention for this Q head
        tiled_attention(
            smem_q,
            k_cache + cache_base_offset,
            v_cache + cache_base_offset,
            smem_k, smem_v,
            smem_scores, warp_partial, smem_scratch,
            smem_running_max, smem_running_sum,
            output, q_head_idx,
            seq_len, max_seq_len, head_dim, scale,
            tid, warp_id, lane_id
        );
        __syncthreads();
    }
}

// ============================================================================
// C API
// ============================================================================
extern "C" {

void fused_gqa_attention_single_token(
    const __nv_bfloat16* q_full,
    const __nv_bfloat16* k_full,
    const __nv_bfloat16* v_full,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache,
    const __nv_bfloat16* sin_cache,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* output,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int head_dim,
    int current_pos,
    int seq_len,
    float scale,
    float rms_eps,
    cudaStream_t stream
) {
    int num_blocks = num_kvheads;
    int threads_per_block = head_dim;  // 128
    int max_seq_len = 4096;

    fused_gqa_attention_single_token_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        q_full, k_full, v_full,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        k_cache, v_cache,
        output,
        num_qheads, num_kvheads, gqa_ratio, head_dim,
        current_pos, seq_len, max_seq_len,
        scale, rms_eps
    );
}

} // extern "C"
