#include "common.cuh"

// SiLU activation: out[i] = x[i] * sigmoid(x[i])
__global__ void silu_kernel(const __nv_bfloat16 *__restrict__ x,
                            __nv_bfloat16 *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float val = __bfloat162float(x[idx]);
    float result = val / (1.0f + expf(-val));
    out[idx] = __float2bfloat16(result);
  }
}

// SiLU + element-wise multiply: out[i] = silu(gate[i]) * up[i]
__global__ void silu_mul_kernel(const __nv_bfloat16 *__restrict__ gate,
                                const __nv_bfloat16 *__restrict__ up,
                                __nv_bfloat16 *__restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float g = __bfloat162float(gate[idx]);
    float silu_g = g / (1.0f + expf(-g));
    float result = silu_g * __bfloat162float(up[idx]);
    out[idx] = __float2bfloat16(result);
  }
}

extern "C" {
void silu_cuda(const __nv_bfloat16 *x, __nv_bfloat16 *out, int n, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  silu_kernel<<<num_blocks, block_size, 0, stream>>>(x, out, n);
}

void silu_mul_cuda(const __nv_bfloat16 *gate, const __nv_bfloat16 *up, __nv_bfloat16 *out, int n,
                   cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;
  silu_mul_kernel<<<num_blocks, block_size, 0, stream>>>(gate, up, out, n);
}
}
