#include "common.cuh"

// ============================================================================
// Fused GQA Attention Kernel for Qwen3.5 — head_dim=256, Tiled Online Softmax
//
// Differences from fused_attention.cu (head_dim=128):
// - head_dim=256, TILE_SIZE=32 (shared memory budget)
// - (1+weight) QK norm (GemmaRMSNorm)
// - Partial RoPE: only first rotary_dim dimensions are rotated
// - Output gating: attn_output *= sigmoid(gate)
// - Interleaved Q+gate layout: q_full[head * 2*HD + 0..HD] = query,
//                                q_full[head * 2*HD + HD..2*HD] = gate
// ============================================================================

#define TILE_SIZE_35 32
#define HD256 256
#define THREADS_256 256
#define NUM_WARPS_256 (THREADS_256 / WARP_SIZE)  // 8

// (1+weight) RMSNorm element
__device__ __forceinline__ __nv_bfloat16 rms_norm_elem_offset(
    __nv_bfloat16 x, float rms_inv, __nv_bfloat16 weight) {
    // GemmaRMSNorm: all float32, only round to bf16 at the end (no intermediate rounding)
    float w = 1.0f + __bfloat162float(weight);
    return __float2bfloat16(__bfloat162float(x) * rms_inv * w);
}

__device__ __forceinline__ void apply_rope_pair_35(
    __nv_bfloat16& x0, __nv_bfloat16& x1,
    __nv_bfloat16 cos_val, __nv_bfloat16 sin_val) {
    float fx0 = __bfloat162float(x0);
    float fx1 = __bfloat162float(x1);
    float fc = __bfloat162float(cos_val);
    float fs = __bfloat162float(sin_val);
    x0 = __float2bfloat16(fx0 * fc - fx1 * fs);
    x1 = __float2bfloat16(fx0 * fs + fx1 * fc);
}

// ============================================================================
// Tiled attention for one Q head — online softmax, head_dim=256
// ============================================================================
__device__ void tiled_attention_256(
    const __nv_bfloat16* __restrict__ smem_q,
    const __nv_bfloat16* __restrict__ k_cache_base,
    const __nv_bfloat16* __restrict__ v_cache_base,
    __nv_bfloat16* __restrict__ smem_k,
    __nv_bfloat16* __restrict__ smem_v,
    float* __restrict__ smem_scores,
    float* __restrict__ warp_partial,
    float* __restrict__ smem_scratch,
    float& smem_running_max,
    float& smem_running_sum,
    __nv_bfloat16* __restrict__ output_buf,
    int q_head_idx,
    int seq_len,
    int max_seq_len,
    float scale,
    int tid, int warp_id, int lane_id
) {
    float o_acc = 0.0f;

    if (tid == 0) {
        smem_running_max = -INFINITY;
        smem_running_sum = 0.0f;
    }
    __syncthreads();

    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE_35) {
        int tile_len = min(TILE_SIZE_35, seq_len - tile_start);

        // Load K/V tile into shared memory
        for (int i = tid; i < tile_len * HD256; i += THREADS_256) {
            int pos_in_tile = i / HD256;
            int dim = i % HD256;
            int abs_pos = tile_start + pos_in_tile;
            smem_k[pos_in_tile * HD256 + dim] = k_cache_base[abs_pos * HD256 + dim];
            smem_v[pos_in_tile * HD256 + dim] = v_cache_base[abs_pos * HD256 + dim];
        }
        __syncthreads();

        // Compute scores: Q · K^T * scale
        float q_val = __bfloat162float(smem_q[tid]);

        for (int pos = 0; pos < tile_len; pos++) {
            float partial = q_val * __bfloat162float(smem_k[pos * HD256 + tid]);
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                warp_partial[warp_id * (TILE_SIZE_35 + 1) + pos] = partial;
            }
        }
        __syncthreads();

        // Combine warp partials
        if (tid < tile_len) {
            float score = 0.0f;
            for (int w = 0; w < NUM_WARPS_256; w++) {
                score += warp_partial[w * (TILE_SIZE_35 + 1) + tid];
            }
            smem_scores[tid] = score * scale;
        }
        __syncthreads();

        // Find tile max
        float tile_max_local = -INFINITY;
        if (tid < tile_len) {
            tile_max_local = smem_scores[tid];
        }
        tile_max_local = warp_reduce_max(tile_max_local);
        if (lane_id == 0) smem_scratch[warp_id] = tile_max_local;
        __syncthreads();

        if (tid == 0) {
            float tile_max = smem_scratch[0];
            for (int i = 1; i < NUM_WARPS_256; i++) {
                tile_max = fmaxf(tile_max, smem_scratch[i]);
            }
            float old_max = smem_running_max;
            float new_max = fmaxf(old_max, tile_max);
            float scale_old = expf(old_max - new_max);
            smem_running_sum *= scale_old;
            smem_running_max = new_max;
            smem_scratch[0] = scale_old;
        }
        __syncthreads();

        float scale_old = smem_scratch[0];
        o_acc *= scale_old;

        // Exp weights + V accumulation
        float local_sum = 0.0f;
        float current_max = smem_running_max;
        for (int pos = 0; pos < tile_len; pos++) {
            float w = expf(smem_scores[pos] - current_max);
            local_sum += w;
            o_acc += w * __bfloat162float(smem_v[pos * HD256 + tid]);
        }

        if (tid == 0) {
            smem_running_sum += local_sum;
        }
        __syncthreads();
    }

    // Final normalize
    float final_sum = smem_running_sum;
    float result = (final_sum > 0.0f) ? (o_acc / final_sum) : 0.0f;
    output_buf[q_head_idx * HD256 + tid] = __float2bfloat16(result);
}

// ============================================================================
// Main kernel — decode variant for Qwen3.5 full attention
// Reads pos/seq_len from decode_meta. CUDA Graph safe.
//
// q_full: interleaved [head0_q(256), head0_gate(256), head1_q(256), ...]
//         total size = num_qheads * 2 * head_dim = 8192
// After attention, output is gated: output *= sigmoid(gate)
// ============================================================================
__global__ void fused_gqa_attention_hd256_decode_kernel(
    const __nv_bfloat16* __restrict__ q_full,    // [num_qheads * 2 * HD256]
    const __nv_bfloat16* __restrict__ k_full,    // [num_kvheads * HD256]
    const __nv_bfloat16* __restrict__ v_full,    // [num_kvheads * HD256]
    const __nv_bfloat16* __restrict__ q_norm_weight,  // [HD256]
    const __nv_bfloat16* __restrict__ k_norm_weight,  // [HD256]
    const __nv_bfloat16* __restrict__ cos_cache_base,  // [max_seq * rotary_dim]
    const __nv_bfloat16* __restrict__ sin_cache_base,
    const int* __restrict__ decode_meta,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ output,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int max_seq_len,
    int rotary_dim,
    float scale,
    float rms_eps
) {
    int current_pos = decode_meta[1];
    int seq_len = decode_meta[2];

    int kv_head_idx = blockIdx.x;
    int tid = threadIdx.x;  // 0..255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Shared memory layout
    __shared__ __nv_bfloat16 smem_k[TILE_SIZE_35 * HD256];       // 16 KB
    __shared__ __nv_bfloat16 smem_v[TILE_SIZE_35 * HD256];       // 16 KB
    __shared__ __nv_bfloat16 smem_q[HD256];                       // 512 B
    __shared__ float smem_scores[TILE_SIZE_35];                    // 128 B
    __shared__ float warp_partial[NUM_WARPS_256 * (TILE_SIZE_35 + 1)];  // 1056 B
    __shared__ float smem_scratch[NUM_WARPS_256];                  // 32 B
    __shared__ float smem_rms[2];
    __shared__ float smem_running_max;
    __shared__ float smem_running_sum;
    // Total: ~33.7 KB

    int cache_base_offset = kv_head_idx * max_seq_len * HD256;

    // Compute cos/sin for this position (partial RoPE)
    int half_rotary = rotary_dim / 2;
    const __nv_bfloat16* cos_cache = cos_cache_base + current_pos * rotary_dim;
    const __nv_bfloat16* sin_cache = sin_cache_base + current_pos * rotary_dim;

    // ========================================================================
    // Phase 1: K head — norm (1+w) → partial RoPE → write cache
    // ========================================================================
    __nv_bfloat16 k_elem = k_full[kv_head_idx * HD256 + tid];

    float k_sq = __bfloat162float(k_elem);
    k_sq *= k_sq;
    float k_sq_sum = warp_reduce_sum(k_sq);
    if (lane_id == 0) smem_scratch[warp_id] = k_sq_sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_256; i++) total += smem_scratch[i];
        smem_rms[1] = 1.0f / sqrtf(total / HD256 + rms_eps);
    }
    __syncthreads();

    __nv_bfloat16 k_normed = rms_norm_elem_offset(k_elem, smem_rms[1], k_norm_weight[tid]);

    // Partial RoPE: only first rotary_dim dimensions
    smem_k[tid] = k_normed;
    __syncthreads();

    if (tid < half_rotary) {
        __nv_bfloat16 k_lo = smem_k[tid];
        __nv_bfloat16 k_hi = smem_k[tid + half_rotary];
        apply_rope_pair_35(k_lo, k_hi, cos_cache[tid], sin_cache[tid]);
        int cache_offset = cache_base_offset + current_pos * HD256;
        k_cache[cache_offset + tid] = k_lo;
        k_cache[cache_offset + tid + half_rotary] = k_hi;
    }
    // Non-rotary dimensions: copy directly to cache
    if (tid >= rotary_dim) {
        k_cache[cache_base_offset + current_pos * HD256 + tid] = smem_k[tid];
    }
    __syncthreads();

    // ========================================================================
    // Phase 2: V head — write to cache
    // ========================================================================
    __nv_bfloat16 v_elem = v_full[kv_head_idx * HD256 + tid];
    v_cache[cache_base_offset + current_pos * HD256 + tid] = v_elem;
    __syncthreads();

    // ========================================================================
    // Phase 3: Loop over Q heads for this KV head
    // ========================================================================
    for (int q = 0; q < gqa_ratio; q++) {
        int q_head_idx = kv_head_idx * gqa_ratio + q;
        if (q_head_idx >= num_qheads) break;

        // Q access: interleaved layout, stride = 2 * HD256
        int q_offset = q_head_idx * 2 * HD256;
        __nv_bfloat16 q_elem = q_full[q_offset + tid];

        // Q norm (1+w)
        float q_sq = __bfloat162float(q_elem);
        q_sq *= q_sq;
        float q_sq_sum = warp_reduce_sum(q_sq);
        if (lane_id == 0) smem_scratch[warp_id] = q_sq_sum;
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < NUM_WARPS_256; i++) total += smem_scratch[i];
            smem_rms[0] = 1.0f / sqrtf(total / HD256 + rms_eps);
        }
        __syncthreads();

        __nv_bfloat16 q_normed = rms_norm_elem_offset(q_elem, smem_rms[0], q_norm_weight[tid]);

        // Partial RoPE for Q
        smem_q[tid] = q_normed;
        __syncthreads();

        if (tid < half_rotary) {
            __nv_bfloat16 q_lo = smem_q[tid];
            __nv_bfloat16 q_hi = smem_q[tid + half_rotary];
            apply_rope_pair_35(q_lo, q_hi, cos_cache[tid], sin_cache[tid]);
            smem_q[tid] = q_lo;
            smem_q[tid + half_rotary] = q_hi;
        }
        __syncthreads();

        // Tiled attention
        tiled_attention_256(
            smem_q,
            k_cache + cache_base_offset,
            v_cache + cache_base_offset,
            smem_k, smem_v,
            smem_scores, warp_partial, smem_scratch,
            smem_running_max, smem_running_sum,
            output, q_head_idx,
            seq_len, max_seq_len, scale,
            tid, warp_id, lane_id
        );
        __syncthreads();

        // Output gating: output[q_head_idx * HD256 + tid] *= sigmoid(gate)
        int gate_offset = q_offset + HD256 + tid;  // gate is right after query in interleaved layout
        float gate_val = __bfloat162float(q_full[gate_offset]);
        float sig_gate = 1.0f / (1.0f + expf(-gate_val));
        float out_val = __bfloat162float(output[q_head_idx * HD256 + tid]);
        output[q_head_idx * HD256 + tid] = __float2bfloat16(out_val * sig_gate);
        __syncthreads();
    }
}

// ============================================================================
// Single-token variant (used for prefill single-token path)
// ============================================================================
__global__ void fused_gqa_attention_hd256_single_token_kernel(
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
    int current_pos,
    int seq_len,
    int max_seq_len,
    int rotary_dim,
    float scale,
    float rms_eps
) {
    int kv_head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ __nv_bfloat16 smem_k[TILE_SIZE_35 * HD256];
    __shared__ __nv_bfloat16 smem_v[TILE_SIZE_35 * HD256];
    __shared__ __nv_bfloat16 smem_q[HD256];
    __shared__ float smem_scores[TILE_SIZE_35];
    __shared__ float warp_partial[NUM_WARPS_256 * (TILE_SIZE_35 + 1)];
    __shared__ float smem_scratch[NUM_WARPS_256];
    __shared__ float smem_rms[2];
    __shared__ float smem_running_max;
    __shared__ float smem_running_sum;

    int cache_base_offset = kv_head_idx * max_seq_len * HD256;
    int half_rotary = rotary_dim / 2;

    // Phase 1: K norm + partial RoPE + cache write
    __nv_bfloat16 k_elem = k_full[kv_head_idx * HD256 + tid];
    float k_sq = __bfloat162float(k_elem); k_sq *= k_sq;
    float k_sq_sum = warp_reduce_sum(k_sq);
    if (lane_id == 0) smem_scratch[warp_id] = k_sq_sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < NUM_WARPS_256; i++) total += smem_scratch[i];
        smem_rms[1] = 1.0f / sqrtf(total / HD256 + rms_eps);
    }
    __syncthreads();

    __nv_bfloat16 k_normed = rms_norm_elem_offset(k_elem, smem_rms[1], k_norm_weight[tid]);
    smem_k[tid] = k_normed;
    __syncthreads();

    if (tid < half_rotary) {
        __nv_bfloat16 k_lo = smem_k[tid];
        __nv_bfloat16 k_hi = smem_k[tid + half_rotary];
        apply_rope_pair_35(k_lo, k_hi, cos_cache[tid], sin_cache[tid]);
        k_cache[cache_base_offset + current_pos * HD256 + tid] = k_lo;
        k_cache[cache_base_offset + current_pos * HD256 + tid + half_rotary] = k_hi;
    }
    if (tid >= rotary_dim) {
        k_cache[cache_base_offset + current_pos * HD256 + tid] = smem_k[tid];
    }
    __syncthreads();

    // Phase 2: V cache write
    v_cache[cache_base_offset + current_pos * HD256 + tid] = v_full[kv_head_idx * HD256 + tid];
    __syncthreads();

    // Phase 3: Q heads
    for (int q = 0; q < gqa_ratio; q++) {
        int q_head_idx = kv_head_idx * gqa_ratio + q;
        if (q_head_idx >= num_qheads) break;

        int q_offset = q_head_idx * 2 * HD256;
        __nv_bfloat16 q_elem = q_full[q_offset + tid];

        float q_sq = __bfloat162float(q_elem); q_sq *= q_sq;
        float q_sq_sum = warp_reduce_sum(q_sq);
        if (lane_id == 0) smem_scratch[warp_id] = q_sq_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < NUM_WARPS_256; i++) total += smem_scratch[i];
            smem_rms[0] = 1.0f / sqrtf(total / HD256 + rms_eps);
        }
        __syncthreads();

        __nv_bfloat16 q_normed = rms_norm_elem_offset(q_elem, smem_rms[0], q_norm_weight[tid]);
        smem_q[tid] = q_normed;
        __syncthreads();

        if (tid < half_rotary) {
            __nv_bfloat16 q_lo = smem_q[tid];
            __nv_bfloat16 q_hi = smem_q[tid + half_rotary];
            apply_rope_pair_35(q_lo, q_hi, cos_cache[tid], sin_cache[tid]);
            smem_q[tid] = q_lo;
            smem_q[tid + half_rotary] = q_hi;
        }
        __syncthreads();

        tiled_attention_256(
            smem_q,
            k_cache + cache_base_offset, v_cache + cache_base_offset,
            smem_k, smem_v,
            smem_scores, warp_partial, smem_scratch,
            smem_running_max, smem_running_sum,
            output, q_head_idx,
            seq_len, max_seq_len, scale,
            tid, warp_id, lane_id
        );
        __syncthreads();

        // Output gating
        float gate_val = __bfloat162float(q_full[q_offset + HD256 + tid]);
        float sig_gate = 1.0f / (1.0f + expf(-gate_val));
        float out_val = __bfloat162float(output[q_head_idx * HD256 + tid]);
        output[q_head_idx * HD256 + tid] = __float2bfloat16(out_val * sig_gate);
        __syncthreads();
    }
}

// ============================================================================
// C API
// ============================================================================
extern "C" {

void fused_gqa_attention_hd256_decode(
    const __nv_bfloat16* q_full,
    const __nv_bfloat16* k_full,
    const __nv_bfloat16* v_full,
    const __nv_bfloat16* q_norm_weight,
    const __nv_bfloat16* k_norm_weight,
    const __nv_bfloat16* cos_cache_base,
    const __nv_bfloat16* sin_cache_base,
    const int* decode_meta,
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    __nv_bfloat16* output,
    int num_qheads,
    int num_kvheads,
    int gqa_ratio,
    int rotary_dim,
    float scale,
    float rms_eps,
    cudaStream_t stream
) {
    int max_seq_len = 4096;
    fused_gqa_attention_hd256_decode_kernel<<<num_kvheads, THREADS_256, 0, stream>>>(
        q_full, k_full, v_full,
        q_norm_weight, k_norm_weight,
        cos_cache_base, sin_cache_base,
        decode_meta,
        k_cache, v_cache, output,
        num_qheads, num_kvheads, gqa_ratio,
        max_seq_len, rotary_dim,
        scale, rms_eps
    );
}

void fused_gqa_attention_hd256_single_token(
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
    int current_pos,
    int seq_len,
    int rotary_dim,
    float scale,
    float rms_eps,
    cudaStream_t stream
) {
    int max_seq_len = 4096;
    fused_gqa_attention_hd256_single_token_kernel<<<num_kvheads, THREADS_256, 0, stream>>>(
        q_full, k_full, v_full,
        q_norm_weight, k_norm_weight,
        cos_cache, sin_cache,
        k_cache, v_cache, output,
        num_qheads, num_kvheads, gqa_ratio,
        current_pos, seq_len, max_seq_len, rotary_dim,
        scale, rms_eps
    );
}

} // extern "C"
