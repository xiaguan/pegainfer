#include "common.cuh"
#include <cmath>

// ============================================================================
// Gated Delta Rule — Recurrent decode step for linear attention
//
// Each block handles one value head (32 blocks total for Qwen3.5-4B).
// State: [key_head_dim, value_head_dim] f32 per head (128×128 = 64KB)
//
// Algorithm per head:
//   g = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
//   beta = sigmoid(b[h])
//   q, k = l2_normalize(q), l2_normalize(k)
//   state *= exp(g)
//   kv_mem = state @ k         // [key_dim] dot per row
//   delta = (v - kv_mem) * beta
//   state += outer(k, delta)   // rank-1 update
//   output = state^T @ q       // [value_dim] dot per column
//
// Thread mapping: tid ∈ [0, key_head_dim). Each thread owns row tid of state.
// GQA: num_key_heads < num_value_heads, so multiple value heads share one key head.
//       gqa_ratio = num_value_heads / num_key_heads.
// ============================================================================

#define GDR_KEY_DIM 128
#define GDR_VAL_DIM 128

__global__ void gated_delta_rule_decode_kernel(
    const __nv_bfloat16* __restrict__ qkv,   // [q_dim + k_dim + v_dim] after conv1d+SiLU
    const __nv_bfloat16* __restrict__ b_proj, // [num_value_heads] (pre-computed by gemv)
    const __nv_bfloat16* __restrict__ a_proj, // [num_value_heads]
    const __nv_bfloat16* __restrict__ dt_bias,// [num_value_heads] bf16
    const float* __restrict__ A_log,          // [num_value_heads] f32
    float* __restrict__ state,                // [num_value_heads, key_dim, val_dim] f32
    __nv_bfloat16* __restrict__ output,       // [num_value_heads * val_dim] bf16
    int num_key_heads,
    int num_value_heads,
    int key_dim,    // 128
    int val_dim     // 128
) {
    int v_head = blockIdx.x;  // value head index
    int tid = threadIdx.x;    // 0..127 (one per key dimension)

    if (tid >= key_dim) return;

    int k_head = v_head * num_key_heads / num_value_heads;  // GQA mapping

    // Layout: qkv = [q(k_dim * num_key_heads), k(k_dim * num_key_heads), v(val_dim * num_value_heads)]
    int q_dim_total = key_dim * num_key_heads;
    int k_dim_total = q_dim_total;

    // Shared memory for q, k, v vectors and reduction scratch
    __shared__ float smem_q[GDR_KEY_DIM];
    __shared__ float smem_k[GDR_KEY_DIM];
    __shared__ float smem_v[GDR_VAL_DIM];
    __shared__ float smem_delta[GDR_VAL_DIM];
    __shared__ float smem_norm[2];  // [q_norm, k_norm]

    // Load q, k for this key head
    float q_val = __bfloat162float(qkv[k_head * key_dim + tid]);
    float k_val = __bfloat162float(qkv[q_dim_total + k_head * key_dim + tid]);

    // Load v for this value head
    float v_val = 0.0f;
    if (tid < val_dim) {
        v_val = __bfloat162float(qkv[q_dim_total + k_dim_total + v_head * val_dim + tid]);
    }

    // ========================================================================
    // L2 normalize q and k
    // ========================================================================
    float q_sq = q_val * q_val;
    q_sq = warp_reduce_sum(q_sq);

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = key_dim / WARP_SIZE;  // 4

    __shared__ float warp_sums[4];
    if (lane_id == 0) warp_sums[warp_id] = q_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) total += warp_sums[i];
        smem_norm[0] = rsqrtf(total + 1e-12f);
    }

    float k_sq = k_val * k_val;
    k_sq = warp_reduce_sum(k_sq);
    if (lane_id == 0) warp_sums[warp_id] = k_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) total += warp_sums[i];
        smem_norm[1] = rsqrtf(total + 1e-12f);
    }
    __syncthreads();

    q_val *= smem_norm[0];
    k_val *= smem_norm[1];

    // Scale query by 1/sqrt(key_dim) — matches HF recurrent_gated_delta_rule
    q_val *= rsqrtf((float)key_dim);

    smem_q[tid] = q_val;
    smem_k[tid] = k_val;
    if (tid < val_dim) smem_v[tid] = v_val;
    __syncthreads();

    // ========================================================================
    // Compute g and beta for this value head
    // ========================================================================
    // g = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
    // beta = sigmoid(b[h])
    __shared__ float s_g;
    __shared__ float s_beta;
    __shared__ float s_exp_g;

    if (tid == 0) {
        float a_val = __bfloat162float(a_proj[v_head]);
        float b_val = __bfloat162float(b_proj[v_head]);
        float bias = __bfloat162float(dt_bias[v_head]);
        float a_log = A_log[v_head];

        float x = a_val + bias;
        // softplus(x) = log(1 + exp(x)), with threshold for numerical stability
        float softplus_x = (x > 20.0f) ? x : logf(1.0f + expf(x));
        s_g = -expf(a_log) * softplus_x;
        s_exp_g = expf(s_g);
        s_beta = 1.0f / (1.0f + expf(-b_val));
    }
    __syncthreads();

    float exp_g = s_exp_g;
    float beta = s_beta;

    // ========================================================================
    // State pointer for this head
    // ========================================================================
    float* my_state = state + v_head * key_dim * val_dim;
    // Thread tid owns row tid: my_state[tid * val_dim + 0..val_dim-1]

    // ========================================================================
    // Step 1: Decay state
    // ========================================================================
    for (int j = 0; j < val_dim; j++) {
        my_state[tid * val_dim + j] *= exp_g;
    }

    // ========================================================================
    // Step 2: kv_mem[tid] = dot(state[tid, :], k[:])
    //   But state row is [val_dim] and k is [key_dim].
    //   Wait — state is [key_dim, val_dim]. Row tid = state[tid, :] has val_dim elements.
    //   k has key_dim elements. The dot product state @ k gives [key_dim] outputs,
    //   where output[i] = sum_j state[i,j] * k[j] — but k is key_dim and state row is val_dim.
    //
    //   Actually, looking at the algorithm more carefully:
    //   kv_mem = state @ k where state is [key_dim, val_dim] and k is [key_dim].
    //   This doesn't make sense dimensionally. Let me re-read the plan.
    //
    //   Plan says: state[128,128], k[128], v[128]
    //   kv_mem[128] = state @ k[128]  → matrix-vector: [128,128] @ [128] → [128]
    //   This means: kv_mem[i] = sum_j(state[i,j] * k[j]) for j in 0..127
    //   So state row i dotted with k gives kv_mem[i]. key_dim == val_dim == 128.
    //   state is [key_dim, val_dim] and k is [key_dim].
    //
    //   Wait, that's [128,128] @ [128]. If state is [row=key_dim, col=val_dim]
    //   and k is [key_dim], then state @ k gives [val_dim]???
    //   No: matrix @ vector where matrix is [M, N] and vector is [N] gives [M].
    //   Here state is [key_dim, val_dim] = [128, 128], k is... hmm.
    //
    //   Actually the delta rule typically has state as [val_dim, key_dim]:
    //   output = state @ q  → [val_dim, key_dim] @ [key_dim] → [val_dim]
    //   state += outer(v, k) → [val_dim] outer [key_dim] → [val_dim, key_dim]
    //   kv_mem = state @ k → [val_dim, key_dim] @ [key_dim] → [val_dim]
    //   delta = (v - kv_mem) * beta → [val_dim]
    //
    //   So actually state should be [val_dim, key_dim]. Let me use that convention.
    //   With 128 threads = val_dim, each thread owns one row of [val_dim, key_dim].
    //   Thread tid handles state[tid, 0..key_dim-1].
    // ========================================================================

    // Reinterpret: state is [val_dim, key_dim], thread tid owns row tid.
    // kv_mem[tid] = dot(state[tid, :], k[:])
    float kv_mem = 0.0f;
    for (int j = 0; j < key_dim; j++) {
        kv_mem += my_state[tid * key_dim + j] * smem_k[j];
    }

    // delta[tid] = (v[tid] - kv_mem) * beta
    float delta_val = (smem_v[tid] - kv_mem) * beta;
    smem_delta[tid] = delta_val;
    __syncthreads();

    // ========================================================================
    // Step 3: Rank-1 update: state[tid, j] += v[tid] * k[j]
    //   Wait, the plan says: state += outer(k, delta)
    //   outer(k, delta) has shape [key_dim, val_dim] if k is [key_dim] and delta is [val_dim].
    //   But we said state is [val_dim, key_dim].
    //   So: state += outer(delta, k)? Or state^T += outer(k, delta)?
    //
    //   Let me use the correct formulation:
    //   state is [val_dim, key_dim]
    //   state[i,j] += delta[i] * k[j]
    //   This is outer(delta, k) = [val_dim, key_dim]. Correct.
    // ========================================================================
    float my_delta = smem_delta[tid];
    for (int j = 0; j < key_dim; j++) {
        my_state[tid * key_dim + j] += my_delta * smem_k[j];
    }

    // ========================================================================
    // Step 4: output[tid] = dot(state[tid, :], q[:])
    //   output = state @ q → [val_dim, key_dim] @ [key_dim] → [val_dim]
    // ========================================================================
    float out_val = 0.0f;
    for (int j = 0; j < key_dim; j++) {
        out_val += my_state[tid * key_dim + j] * smem_q[j];
    }

    output[v_head * val_dim + tid] = __float2bfloat16(out_val);
}

extern "C" {

void gated_delta_rule_decode_cuda(
    const __nv_bfloat16* qkv,
    const __nv_bfloat16* b_proj,
    const __nv_bfloat16* a_proj,
    const __nv_bfloat16* dt_bias,
    const float* A_log,
    float* state,
    __nv_bfloat16* output,
    int num_key_heads,
    int num_value_heads,
    int key_dim,
    int val_dim,
    cudaStream_t stream
) {
    // One block per value head, key_dim threads per block
    gated_delta_rule_decode_kernel<<<num_value_heads, key_dim, 0, stream>>>(
        qkv, b_proj, a_proj, dt_bias, A_log,
        state, output,
        num_key_heads, num_value_heads, key_dim, val_dim
    );
}

} // extern "C"
