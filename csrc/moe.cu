// MoE routing kernel for DeepSeek-V3.
//
// DSV3.2 routing algorithm:
//   1. sigmoid(gate_weight @ hidden) → scores [num_experts]
//   2. scores + e_score_correction_bias → biased scores (selection only)
//   3. Group-limited selection: top-2 per group → group scores → top topk_group groups
//   4. Mask non-selected groups, TopK from remaining → expert indices
//   5. Final weights = original sigmoid scores (no bias), normalized, scaled
//
// Also includes weighted-add helper for accumulating expert outputs.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cfloat>

// ---------------------------------------------------------------------------
// MoE routing kernel
// ---------------------------------------------------------------------------

// One block per token, blockDim.x >= num_experts (256 for DSV3.2).
// Each thread handles one expert.
//
// logits:    [num_experts, bs] bf16, column-major (stride = num_experts between tokens)
// bias:      [num_experts] f32
// topk_idx:  [bs * topk] i32 output
// topk_wt:   [bs * topk] f32 output
__global__ void moe_routing_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_wt,
    int num_experts,
    int bs,
    int topk,
    int n_group,
    int topk_group,
    int norm_topk_prob,
    float routed_scaling_factor)
{
    const int token = blockIdx.x;
    if (token >= bs) return;

    const int tid = threadIdx.x;
    const int experts_per_group = num_experts / n_group; // 32 for DSV3.2

    // Shared memory layout (DSV3.2: num_experts=256, n_group=8)
    extern __shared__ char smem[];
    float* s_scores        = (float*)smem;                    // [256] original sigmoid
    float* s_scores_biased = s_scores + num_experts;          // [256] sigmoid + bias
    float* s_group_scores  = s_scores_biased + num_experts;   // [8]
    int*   s_group_sel     = (int*)(s_group_scores + n_group);// [topk_group]
    int*   s_topk_idx      = s_group_sel + topk_group;        // [topk]
    float* s_topk_wt       = (float*)(s_topk_idx + topk);     // [topk]

    // ---- Step 1: sigmoid + bias ----
    float score = 0.0f;
    float score_biased = -FLT_MAX;
    if (tid < num_experts) {
        float logit = __bfloat162float(logits[tid + token * num_experts]);
        score = 1.0f / (1.0f + expf(-logit));
        score_biased = score + bias[tid];
        s_scores[tid] = score;
        s_scores_biased[tid] = score_biased;
    }
    __syncthreads();

    // ---- Step 2: Group scoring (top-2 per group, sum) ----
    if (tid < n_group) {
        int base = tid * experts_per_group;
        float top1 = -FLT_MAX, top2 = -FLT_MAX;
        for (int i = 0; i < experts_per_group; i++) {
            float v = s_scores_biased[base + i];
            if (v > top1) {
                top2 = top1;
                top1 = v;
            } else if (v > top2) {
                top2 = v;
            }
        }
        s_group_scores[tid] = top1 + top2;
    }
    __syncthreads();

    // ---- Step 3: Select top topk_group groups (thread 0) ----
    // n_group and topk_group are small (8, 4), sequential is fine.
    if (tid == 0) {
        unsigned selected_mask = 0; // bitmask for selected groups

        for (int k = 0; k < topk_group; k++) {
            float best = -FLT_MAX;
            int best_g = 0;
            for (int g = 0; g < n_group; g++) {
                if (!(selected_mask & (1u << g)) && s_group_scores[g] > best) {
                    best = s_group_scores[g];
                    best_g = g;
                }
            }
            s_group_sel[k] = best_g;
            selected_mask |= (1u << best_g);
        }

        // Reuse s_group_scores as group mask (1.0 = selected, 0.0 = masked)
        for (int g = 0; g < n_group; g++) {
            s_group_scores[g] = (selected_mask & (1u << g)) ? 1.0f : 0.0f;
        }
    }
    __syncthreads();

    // ---- Step 4: Mask non-selected groups ----
    if (tid < num_experts) {
        int group = tid / experts_per_group;
        if (s_group_scores[group] == 0.0f) {
            s_scores_biased[tid] = -FLT_MAX;
        }
    }
    __syncthreads();

    // ---- Step 5: TopK from masked scores (thread 0) ----
    // 256 experts × topk iterations = ~2K comparisons, fast enough.
    if (tid == 0) {
        for (int k = 0; k < topk; k++) {
            float best = -FLT_MAX;
            int best_e = 0;
            for (int e = 0; e < num_experts; e++) {
                if (s_scores_biased[e] > best) {
                    best = s_scores_biased[e];
                    best_e = e;
                }
            }
            s_topk_idx[k] = best_e;
            // Final weight uses original sigmoid score (no bias)
            s_topk_wt[k] = s_scores[best_e];
            s_scores_biased[best_e] = -FLT_MAX; // exclude
        }

        // ---- Step 6: Normalize ----
        if (norm_topk_prob) {
            float sum = 0.0f;
            for (int k = 0; k < topk; k++) sum += s_topk_wt[k];
            if (sum > 0.0f) {
                float inv = 1.0f / sum;
                for (int k = 0; k < topk; k++) s_topk_wt[k] *= inv;
            }
        }

        // ---- Step 7: Scale ----
        for (int k = 0; k < topk; k++) {
            s_topk_wt[k] *= routed_scaling_factor;
        }
    }
    __syncthreads();

    // ---- Step 8: Write output ----
    if (tid < topk) {
        int out_base = token * topk;
        topk_idx[out_base + tid] = s_topk_idx[tid];
        topk_wt[out_base + tid] = s_topk_wt[tid];
    }
}

// ---------------------------------------------------------------------------
// Cast i32 → i64 (for DeepEP topk_idx compatibility)
// ---------------------------------------------------------------------------

__global__ void cast_i32_to_i64_kernel(
    const int* __restrict__ in,
    int64_t* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<int64_t>(in[i]);
    }
}

// ---------------------------------------------------------------------------
// Weighted add: out[i] += scale * x[i], bf16, element-wise
// ---------------------------------------------------------------------------

__global__ void moe_weighted_add_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ x,
    float scale,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = __bfloat162float(out[i]) + scale * __bfloat162float(x[i]);
        out[i] = __float2bfloat16(val);
    }
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

void moe_routing_cuda(
    const void* logits,     // bf16 [num_experts, bs]
    const float* bias,      // [num_experts]
    int* topk_idx,          // [bs * topk]
    float* topk_wt,         // [bs * topk]
    int num_experts,
    int bs,
    int topk,
    int n_group,
    int topk_group,
    int norm_topk_prob,
    float routed_scaling_factor,
    void* stream)
{
    // Shared memory: 2 * num_experts floats (scores, scores_biased)
    //              + n_group floats (group_scores / group_mask)
    //              + topk_group ints (group_sel)
    //              + topk ints + topk floats (topk output)
    size_t smem = (2 * num_experts + n_group) * sizeof(float)
               + topk_group * sizeof(int)
               + topk * (sizeof(int) + sizeof(float));

    dim3 grid(bs);
    dim3 block(num_experts); // 256 threads for DSV3.2
    moe_routing_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)logits,
        bias,
        topk_idx,
        topk_wt,
        num_experts,
        bs,
        topk,
        n_group,
        topk_group,
        norm_topk_prob,
        routed_scaling_factor);
}

void moe_weighted_add_cuda(
    void* out,          // bf16
    const void* x,      // bf16
    float scale,
    int n,
    void* stream)
{
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    moe_weighted_add_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)out,
        (const __nv_bfloat16*)x,
        scale,
        n);
}

void cast_i32_to_i64_cuda(
    const int* in_data,
    int64_t* out_data,
    int n,
    void* stream)
{
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cast_i32_to_i64_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        in_data, out_data, n);
}

} // extern "C"
