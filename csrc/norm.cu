#include "common.cuh"

// RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
__global__ void rms_norm_kernel(const __nv_bfloat16 *__restrict__ x,
                                const __nv_bfloat16 *__restrict__ weight,
                                __nv_bfloat16 *__restrict__ out, int n, float eps) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int stride = blockDim.x;

  // Compute sum of squares (fp32 accumulator for precision)
  float local_sum = 0.0f;
  for (int i = tid; i < n; i += stride) {
    float val = __bfloat162float(x[i]);
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // Reduce within block
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }

  // Compute RMS
  float rms = sqrtf(shared[0] / n + eps);
  float inv_rms = 1.0f / rms;

  // Normalize and scale
  for (int i = tid; i < n; i += stride) {
    float val = __bfloat162float(x[i]) * inv_rms * __bfloat162float(weight[i]);
    out[i] = __float2bfloat16(val);
  }
}

// Batched RMSNorm: each block handles one vector (blockIdx.x = token index)
__global__ void rms_norm_batched_kernel(const __nv_bfloat16 *__restrict__ x,
                                         const __nv_bfloat16 *__restrict__ weight,
                                         __nv_bfloat16 *__restrict__ out,
                                         int hidden_dim, float eps) {
  const __nv_bfloat16 *x_vec = x + blockIdx.x * hidden_dim;
  __nv_bfloat16 *out_vec = out + blockIdx.x * hidden_dim;

  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int stride = blockDim.x;

  float local_sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += stride) {
    float val = __bfloat162float(x_vec[i]);
    local_sum += val * val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
  }

  float rms = sqrtf(shared[0] / hidden_dim + eps);
  float inv_rms = 1.0f / rms;

  for (int i = tid; i < hidden_dim; i += stride) {
    float val = __bfloat162float(x_vec[i]) * inv_rms * __bfloat162float(weight[i]);
    out_vec[i] = __float2bfloat16(val);
  }
}

extern "C" {
void rms_norm_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight, __nv_bfloat16 *out, int n,
                   float eps, cudaStream_t stream) {
  int block_size = 256;
  int shared_mem = block_size * sizeof(float);
  rms_norm_kernel<<<1, block_size, shared_mem, stream>>>(x, weight, out, n, eps);
}

void rms_norm_batched_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *weight,
                            __nv_bfloat16 *out, int hidden_dim, int seq_len,
                            float eps, cudaStream_t stream) {
  int block_size = 256;
  int shared_mem = block_size * sizeof(float);
  rms_norm_batched_kernel<<<seq_len, block_size, shared_mem, stream>>>(
      x, weight, out, hidden_dim, eps);
}
}
