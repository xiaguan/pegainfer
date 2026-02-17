#include "common.cuh"

// Argmax: find index of maximum value
// Returns result in out[0]
__global__ void argmax_kernel(const __nv_bfloat16 *__restrict__ x,
                              int *__restrict__ out, int n) {
  extern __shared__ char shared_mem[];
  float *shared_vals = (float *)shared_mem;
  int *shared_idxs = (int *)(shared_mem + blockDim.x * sizeof(float));

  int tid = threadIdx.x;
  int stride = blockDim.x;

  // Find local max
  float local_max = -INFINITY;
  int local_idx = 0;
  for (int i = tid; i < n; i += stride) {
    float val = __bfloat162float(x[i]);
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
  }
  shared_vals[tid] = local_max;
  shared_idxs[tid] = local_idx;
  __syncthreads();

  // Reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (shared_vals[tid + s] > shared_vals[tid]) {
        shared_vals[tid] = shared_vals[tid + s];
        shared_idxs[tid] = shared_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[0] = shared_idxs[0];
  }
}

extern "C" {
void argmax_cuda(const __nv_bfloat16 *x, int *out, int n, cudaStream_t stream) {
  int block_size = 256;
  int shared_mem = block_size * (sizeof(float) + sizeof(int));
  argmax_kernel<<<1, block_size, shared_mem, stream>>>(x, out, n);
}
}
