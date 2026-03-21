#include "common.cuh"

#define NORM_BLOCK 256
#define NORM_NUM_WARPS (NORM_BLOCK / WARP_SIZE)

// ============================================================================
// RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
// BF16×4 vectorized loads, warp shuffle reduction.
// Single block, 256 threads — suitable for decode (n=2560).
// ============================================================================
__global__ void rms_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                const __nv_bfloat16 *__restrict__ weight,
                                __nv_bfloat16 *__restrict__ out, int n, float eps) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = n / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  // Pass 1: Compute sum of squares (FP32 accumulator)
  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x);
    float v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x);
    float v3 = __bfloat162float(hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    float val = __bfloat162float(x[i]);
    local_sum += val * val;
  }

  // Warp shuffle reduction
  local_sum = warp_reduce_sum(local_sum);

  // Inter-warp reduction via shared memory
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  // First warp reduces
  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  // Broadcast inv_rms to all threads
  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = 1.0f / sqrtf(total / n + eps);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: Normalize and scale (BF16×4 vectorized)
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    // Match HF: round normalized to bf16 before weight multiply
    __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms);
    __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms);
    __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms);
    __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(n0) * __bfloat162float(w_lo.x));
    r_lo.y = __float2bfloat16(__bfloat162float(n1) * __bfloat162float(w_lo.y));
    r_hi.x = __float2bfloat16(__bfloat162float(n2) * __bfloat162float(w_hi.x));
    r_hi.y = __float2bfloat16(__bfloat162float(n3) * __bfloat162float(w_hi.y));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(x[i]) * inv_rms);
    out[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
  }
}

// ============================================================================
// Fused Add + RMSNorm: hidden += residual; out = rms_norm(hidden, weight)
// One kernel replaces two: saves one global read of hidden.
// BF16×4 vectorized, warp shuffle reduction.
// ============================================================================
__global__ void fused_add_rms_norm_kernel(
    __nv_bfloat16 *__restrict__ hidden,          // in/out: hidden state (updated in-place)
    const __nv_bfloat16 *__restrict__ residual,   // in: residual to add
    const __nv_bfloat16 *__restrict__ weight,     // in: rms_norm weight
    __nv_bfloat16 *__restrict__ out,              // out: normalized output
    int n, float eps) {

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = n / 4;

  uint2 *hidden_vec = reinterpret_cast<uint2 *>(hidden);
  const uint2 *res_vec = reinterpret_cast<const uint2 *>(residual);

  // Pass 1: Add residual to hidden, compute sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = hidden_vec[i];
    uint2 rv = res_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 r_lo = *reinterpret_cast<__nv_bfloat162 *>(&rv.x);
    __nv_bfloat162 r_hi = *reinterpret_cast<__nv_bfloat162 *>(&rv.y);

    // Add in FP32 then store back as BF16
    float s0 = __bfloat162float(h_lo.x) + __bfloat162float(r_lo.x);
    float s1 = __bfloat162float(h_lo.y) + __bfloat162float(r_lo.y);
    float s2 = __bfloat162float(h_hi.x) + __bfloat162float(r_hi.x);
    float s3 = __bfloat162float(h_hi.y) + __bfloat162float(r_hi.y);

    // Write updated hidden
    __nv_bfloat162 s_lo, s_hi;
    s_lo.x = __float2bfloat16(s0);
    s_lo.y = __float2bfloat16(s1);
    s_hi.x = __float2bfloat16(s2);
    s_hi.y = __float2bfloat16(s3);
    uint2 sv;
    sv.x = *reinterpret_cast<unsigned int *>(&s_lo);
    sv.y = *reinterpret_cast<unsigned int *>(&s_hi);
    hidden_vec[i] = sv;

    // Accumulate sum of squares (use the bf16-rounded values for consistency)
    float v0 = __bfloat162float(s_lo.x);
    float v1 = __bfloat162float(s_lo.y);
    float v2 = __bfloat162float(s_hi.x);
    float v3 = __bfloat162float(s_hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    float s = __bfloat162float(hidden[i]) + __bfloat162float(residual[i]);
    hidden[i] = __float2bfloat16(s);
    float v = __bfloat162float(hidden[i]);
    local_sum += v * v;
  }

  // Warp shuffle reduction
  local_sum = warp_reduce_sum(local_sum);

  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = 1.0f / sqrtf(total / n + eps);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: Normalize and scale (read updated hidden, write out)
  const uint2 *h_vec_r = reinterpret_cast<const uint2 *>(hidden);
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = h_vec_r[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(h_lo.x) * inv_rms);
    __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(h_lo.y) * inv_rms);
    __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(h_hi.x) * inv_rms);
    __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(h_hi.y) * inv_rms);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(n0) * __bfloat162float(w_lo.x));
    r_lo.y = __float2bfloat16(__bfloat162float(n1) * __bfloat162float(w_lo.y));
    r_hi.x = __float2bfloat16(__bfloat162float(n2) * __bfloat162float(w_hi.x));
    r_hi.y = __float2bfloat16(__bfloat162float(n3) * __bfloat162float(w_hi.y));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  // Scalar tail
  for (int i = n4 * 4 + tid; i < n; i += NORM_BLOCK) {
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(hidden[i]) * inv_rms);
    out[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
  }
}

// ============================================================================
// Batched RMSNorm: each block handles one vector (blockIdx.x = token index)
// BF16×4 vectorized, warp shuffle reduction.
// ============================================================================
__global__ void rms_norm_batched_kernel(const __nv_bfloat16 *__restrict__ x,
                                         const __nv_bfloat16 *__restrict__ weight,
                                         __nv_bfloat16 *__restrict__ out,
                                         int hidden_dim, float eps) {
  const __nv_bfloat16 *x_row = x + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_row = out + blockIdx.x * hidden_dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int n4 = hidden_dim / 4;
  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x_row);

  // Pass 1: sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x);
    float v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x);
    float v3 = __bfloat162float(hi.y);
    local_sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  for (int i = n4 * 4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    float val = __bfloat162float(x_row[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);

  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = 1.0f / sqrtf(total / hidden_dim + eps);
  }
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: normalize and scale
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out_row);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    __nv_bfloat16 n0 = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms);
    __nv_bfloat16 n1 = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms);
    __nv_bfloat16 n2 = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms);
    __nv_bfloat16 n3 = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(n0) * __bfloat162float(w_lo.x));
    r_lo.y = __float2bfloat16(__bfloat162float(n1) * __bfloat162float(w_lo.y));
    r_hi.x = __float2bfloat16(__bfloat162float(n2) * __bfloat162float(w_hi.x));
    r_hi.y = __float2bfloat16(__bfloat162float(n3) * __bfloat162float(w_hi.y));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4 * 4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    __nv_bfloat16 normed = __float2bfloat16(__bfloat162float(x_row[i]) * inv_rms);
    out_row[i] = __float2bfloat16(__bfloat162float(normed) * __bfloat162float(weight[i]));
  }
}

extern "C" {
void rms_norm_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                   float eps, cudaStream_t stream) {
  rms_norm_kernel<<<1, NORM_BLOCK, 0, stream>>>(x, weight, out, n, eps);
}

void fused_add_rms_norm_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                              const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                              float eps, cudaStream_t stream) {
  fused_add_rms_norm_kernel<<<1, NORM_BLOCK, 0, stream>>>(hidden, residual, weight, out, n, eps);
}

void rms_norm_batched_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                            __nv_bfloat16 *out, int hidden_dim, int seq_len,
                            float eps, cudaStream_t stream) {
  rms_norm_batched_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
      x, weight, out, hidden_dim, eps);
}

// ============================================================================
// RMSNorm with (1+weight) offset — Qwen3.5 / Gemma style
// out[i] = x[i] * (1 + weight[i]) / sqrt(mean(x^2) + eps)
// ============================================================================
void rms_norm_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                           __nv_bfloat16 *out, int n, float eps, cudaStream_t stream);

void fused_add_rms_norm_offset_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                                      const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                                      float eps, cudaStream_t stream);

void rms_norm_batched_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                    __nv_bfloat16 *out, int hidden_dim, int seq_len,
                                    float eps, cudaStream_t stream);

void rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int head_dim, float eps, cudaStream_t stream);
} // extern "C"

// ============================================================================
// (1+weight) RMSNorm kernel
// ============================================================================
__global__ void rms_norm_offset_kernel(const __nv_bfloat16 *__restrict__ x,
                                        const __nv_bfloat16 *__restrict__ weight,
                                        __nv_bfloat16 *__restrict__ out, int n, float eps) {
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int n4 = n / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x), v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x), v3 = __bfloat162float(hi.y);
    local_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    float val = __bfloat162float(x[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) s_inv_rms = 1.0f / sqrtf(total / n + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: out[i] = (x[i] * inv_rms * (1 + weight[i])) cast to bf16
  // NOTE: GemmaRMSNorm does ALL computation in float32, only rounds to bf16 at the end.
  // No intermediate bf16 rounding (unlike Llama/Qwen3 RMSNorm).
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms * (1.0f + __bfloat162float(w_lo.x)));
    r_lo.y = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms * (1.0f + __bfloat162float(w_lo.y)));
    r_hi.x = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms * (1.0f + __bfloat162float(w_hi.x)));
    r_hi.y = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms * (1.0f + __bfloat162float(w_hi.y)));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    out[i] = __float2bfloat16(__bfloat162float(x[i]) * inv_rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// ============================================================================
// Fused Add + (1+weight) RMSNorm
// ============================================================================
__global__ void fused_add_rms_norm_offset_kernel(
    __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out, int n, float eps) {

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int n4 = n / 4;

  uint2 *hidden_vec = reinterpret_cast<uint2 *>(hidden);
  const uint2 *res_vec = reinterpret_cast<const uint2 *>(residual);

  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = hidden_vec[i];
    uint2 rv = res_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 r_lo = *reinterpret_cast<__nv_bfloat162 *>(&rv.x);
    __nv_bfloat162 r_hi = *reinterpret_cast<__nv_bfloat162 *>(&rv.y);

    float s0 = __bfloat162float(h_lo.x) + __bfloat162float(r_lo.x);
    float s1 = __bfloat162float(h_lo.y) + __bfloat162float(r_lo.y);
    float s2 = __bfloat162float(h_hi.x) + __bfloat162float(r_hi.x);
    float s3 = __bfloat162float(h_hi.y) + __bfloat162float(r_hi.y);

    __nv_bfloat162 s_lo, s_hi;
    s_lo.x = __float2bfloat16(s0); s_lo.y = __float2bfloat16(s1);
    s_hi.x = __float2bfloat16(s2); s_hi.y = __float2bfloat16(s3);
    uint2 sv;
    sv.x = *reinterpret_cast<unsigned int *>(&s_lo);
    sv.y = *reinterpret_cast<unsigned int *>(&s_hi);
    hidden_vec[i] = sv;

    float v0 = __bfloat162float(s_lo.x), v1 = __bfloat162float(s_lo.y);
    float v2 = __bfloat162float(s_hi.x), v3 = __bfloat162float(s_hi.y);
    local_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    float s = __bfloat162float(hidden[i]) + __bfloat162float(residual[i]);
    hidden[i] = __float2bfloat16(s);
    // Match vectorized path: sum-of-squares on bf16-rounded value
    float v = __bfloat162float(__float2bfloat16(s));
    local_sum += v * v;
  }

  local_sum = warp_reduce_sum(local_sum);
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) s_inv_rms = 1.0f / sqrtf(total / n + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  const uint2 *h_vec_r = reinterpret_cast<const uint2 *>(hidden);
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 hv = h_vec_r[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 h_lo = *reinterpret_cast<__nv_bfloat162 *>(&hv.x);
    __nv_bfloat162 h_hi = *reinterpret_cast<__nv_bfloat162 *>(&hv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    // GemmaRMSNorm: all in float32, only round to bf16 at end
    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(h_lo.x) * inv_rms * (1.0f + __bfloat162float(w_lo.x)));
    r_lo.y = __float2bfloat16(__bfloat162float(h_lo.y) * inv_rms * (1.0f + __bfloat162float(w_lo.y)));
    r_hi.x = __float2bfloat16(__bfloat162float(h_hi.x) * inv_rms * (1.0f + __bfloat162float(w_hi.x)));
    r_hi.y = __float2bfloat16(__bfloat162float(h_hi.y) * inv_rms * (1.0f + __bfloat162float(w_hi.y)));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4*4 + tid; i < n; i += NORM_BLOCK) {
    out[i] = __float2bfloat16(__bfloat162float(hidden[i]) * inv_rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// ============================================================================
// Batched (1+weight) RMSNorm: one block per token.
// Grid: <<<seq_len, NORM_BLOCK>>>
// ============================================================================
__global__ void rms_norm_batched_offset_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ weight,
    __nv_bfloat16 *__restrict__ out,
    int hidden_dim, float eps) {

  const __nv_bfloat16 *x_row = x + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_row = out + blockIdx.x * hidden_dim;

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int n4 = hidden_dim / 4;

  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x_row);

  // Pass 1: sum of squares
  float local_sum = 0.0f;
  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    __nv_bfloat162 lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    float v0 = __bfloat162float(lo.x), v1 = __bfloat162float(lo.y);
    float v2 = __bfloat162float(hi.x), v3 = __bfloat162float(hi.y);
    local_sum += v0*v0 + v1*v1 + v2*v2 + v3*v3;
  }
  for (int i = n4*4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    float val = __bfloat162float(x_row[i]);
    local_sum += val * val;
  }

  local_sum = warp_reduce_sum(local_sum);
  __shared__ float warp_sums[NORM_NUM_WARPS];
  if (lane_id == 0) warp_sums[warp_id] = local_sum;
  __syncthreads();

  float total = 0.0f;
  if (warp_id == 0) {
    float val = (lane_id < NORM_NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
    total = warp_reduce_sum(val);
  }

  __shared__ float s_inv_rms;
  if (tid == 0) s_inv_rms = 1.0f / sqrtf(total / hidden_dim + eps);
  __syncthreads();
  float inv_rms = s_inv_rms;

  // Pass 2: out = x * inv_rms * (1 + weight), all in f32
  const uint2 *w_vec = reinterpret_cast<const uint2 *>(weight);
  uint2 *out_vec = reinterpret_cast<uint2 *>(out_row);

  for (int i = tid; i < n4; i += NORM_BLOCK) {
    uint2 xv = x_vec[i];
    uint2 wv = w_vec[i];
    __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&xv.x);
    __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&xv.y);
    __nv_bfloat162 w_lo = *reinterpret_cast<__nv_bfloat162 *>(&wv.x);
    __nv_bfloat162 w_hi = *reinterpret_cast<__nv_bfloat162 *>(&wv.y);

    uint2 result;
    __nv_bfloat162 r_lo, r_hi;
    r_lo.x = __float2bfloat16(__bfloat162float(x_lo.x) * inv_rms * (1.0f + __bfloat162float(w_lo.x)));
    r_lo.y = __float2bfloat16(__bfloat162float(x_lo.y) * inv_rms * (1.0f + __bfloat162float(w_lo.y)));
    r_hi.x = __float2bfloat16(__bfloat162float(x_hi.x) * inv_rms * (1.0f + __bfloat162float(w_hi.x)));
    r_hi.y = __float2bfloat16(__bfloat162float(x_hi.y) * inv_rms * (1.0f + __bfloat162float(w_hi.y)));
    result.x = *reinterpret_cast<unsigned int *>(&r_lo);
    result.y = *reinterpret_cast<unsigned int *>(&r_hi);
    out_vec[i] = result;
  }
  for (int i = n4*4 + tid; i < hidden_dim; i += NORM_BLOCK) {
    out_row[i] = __float2bfloat16(__bfloat162float(x_row[i]) * inv_rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// ============================================================================
// Gated RMSNorm for linear attention output:
//   out = rms_norm(x, f32_weight) * silu(gate)
// Per-head normalization: x is [num_heads * head_dim], weight is [head_dim] (broadcast).
// Grid: num_heads blocks, head_dim threads.
// ============================================================================
__global__ void rms_norm_gated_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const float *__restrict__ weight,
    const __nv_bfloat16 *__restrict__ gate,
    __nv_bfloat16 *__restrict__ out,
    int head_dim,
    float eps
) {
  int head = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= head_dim) return;

  int offset = head * head_dim + tid;

  // RMSNorm over this head's slice
  float x_val = __bfloat162float(x[offset]);
  float sq = x_val * x_val;
  sq = warp_reduce_sum(sq);

  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  int num_warps = (head_dim + WARP_SIZE - 1) / WARP_SIZE;

  __shared__ float warp_sums[8];  // max 8 warps for head_dim=256
  if (lane_id == 0) warp_sums[warp_id] = sq;
  __syncthreads();

  __shared__ float s_inv_rms;
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < num_warps; i++) total += warp_sums[i];
    s_inv_rms = rsqrtf(total / head_dim + eps);
  }
  __syncthreads();

  float normed = x_val * s_inv_rms;
  // Weight is F32, per head_dim (broadcast across heads)
  float w = weight[tid];
  normed *= w;

  // SiLU gate
  float g = __bfloat162float(gate[offset]);
  float silu_g = g / (1.0f + expf(-g));

  out[offset] = __float2bfloat16(normed * silu_g);
}

// C API implementations
extern "C" {

void rms_norm_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                           __nv_bfloat16 *out, int n, float eps, cudaStream_t stream) {
  rms_norm_offset_kernel<<<1, NORM_BLOCK, 0, stream>>>(x, weight, out, n, eps);
}

void fused_add_rms_norm_offset_cuda(__nv_bfloat16 *hidden, const __nv_bfloat16 *residual,
                                      const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                                      float eps, cudaStream_t stream) {
  fused_add_rms_norm_offset_kernel<<<1, NORM_BLOCK, 0, stream>>>(hidden, residual, weight, out, n, eps);
}

void rms_norm_batched_offset_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                                    __nv_bfloat16 *out, int hidden_dim, int seq_len,
                                    float eps, cudaStream_t stream) {
  rms_norm_batched_offset_kernel<<<seq_len, NORM_BLOCK, 0, stream>>>(
      x, weight, out, hidden_dim, eps);
}

void rms_norm_gated_cuda(const __nv_bfloat16 *x, const float *weight,
                          const __nv_bfloat16 *gate, __nv_bfloat16 *out,
                          int num_heads, int head_dim, float eps, cudaStream_t stream) {
  rms_norm_gated_kernel<<<num_heads, head_dim, 0, stream>>>(x, weight, gate, out, head_dim, eps);
}

} // extern "C"
