#include "common.cuh"

// Element-wise add: out = a + b
__global__ void add_kernel(const __nv_bfloat16 *__restrict__ a,
                           const __nv_bfloat16 *__restrict__ b, __nv_bfloat16 *__restrict__ out,
                           int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

// Copy kernel (for slicing)
__global__ void copy_kernel(const __nv_bfloat16 *__restrict__ src,
                            __nv_bfloat16 *__restrict__ dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// Softmax: out = softmax(x)
// Single block kernel for small vectors (attention scores)
__global__ void softmax_kernel(const __nv_bfloat16 *__restrict__ x,
                               __nv_bfloat16 *__restrict__ out, int n) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int stride = blockDim.x;

  // Find max (for numerical stability)
  float local_max = -INFINITY;
  for (int i = tid; i < n; i += stride) {
    local_max = fmaxf(local_max, __bfloat162float(x[i]));
  }
  shared[tid] = local_max;
  __syncthreads();

  // Reduce max
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    }
    __syncthreads();
  }
  float max_val = shared[0];
  __syncthreads();

  // Compute exp(x - max) and sum
  float local_sum = 0.0f;
  for (int i = tid; i < n; i += stride) {
    float exp_val = expf(__bfloat162float(x[i]) - max_val);
    out[i] = __float2bfloat16(exp_val);
    local_sum += exp_val;
  }
  shared[tid] = local_sum;
  __syncthreads();

  // Reduce sum
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared[tid] += shared[tid + s];
    }
    __syncthreads();
  }
  float sum = shared[0];
  __syncthreads();

  // Normalize
  float inv_sum = 1.0f / sum;
  for (int i = tid; i < n; i += stride) {
    out[i] = __float2bfloat16(__bfloat162float(out[i]) * inv_sum);
  }
}

extern "C" {
void add_cuda(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, int n,
              cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  add_kernel<<<num_blocks, block_size, 0, stream>>>(a, b, out, n);
}

void copy_cuda(const __nv_bfloat16 *src, __nv_bfloat16 *dst, int n, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  copy_kernel<<<num_blocks, block_size, 0, stream>>>(src, dst, n);
}

void softmax_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block_size = 256;
  int shared_mem = block_size * sizeof(float);
  softmax_kernel<<<1, block_size, shared_mem, stream>>>(x, out, n);
}
}
