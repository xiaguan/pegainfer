#include "common.cuh"

// RoPE: Apply rotary position embedding
// x: (head_dim,), cos/sin: (head_dim,) precomputed for this position
__global__ void rope_kernel(const __nv_bfloat16 *__restrict__ x,
                            const __nv_bfloat16 *__restrict__ cos,
                            const __nv_bfloat16 *__restrict__ sin,
                            __nv_bfloat16 *__restrict__ out, int head_dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;

  if (idx < half_dim) {
    int i0 = idx * 2;
    int i1 = idx * 2 + 1;

    float x0 = __bfloat162float(x[i0]);
    float x1 = __bfloat162float(x[i1]);
    float c = __bfloat162float(cos[i0]);
    float s = __bfloat162float(sin[i0]);

    out[i0] = __float2bfloat16(x0 * c - x1 * s);
    out[i1] = __float2bfloat16(x0 * s + x1 * c);
  }
}

extern "C" {
void rope_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *cos, const __nv_bfloat16 *sin, __nv_bfloat16 *out,
               int head_dim, cudaStream_t stream) {
  int half_dim = head_dim / 2;
  int block_size = 128;
  int num_blocks = (half_dim + block_size - 1) / block_size;
  rope_kernel<<<num_blocks, block_size, 0, stream>>>(x, cos, sin, out, head_dim);
}
}
