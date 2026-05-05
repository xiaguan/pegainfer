#include "common.cuh"

// ============================================================================
// Deinterleave QKV: split column-major [qkv_dim, bs] into
//   q [q_dim, bs], k [kv_dim, bs], v [kv_dim, bs]
//
// Column-major layout: element (row, col) at offset col * stride + row.
// Combined buffer has stride = qkv_dim; outputs have stride = their dim.
// ============================================================================

__global__ void deinterleave_qkv_kernel(
    const __nv_bfloat16 *__restrict__ qkv, // [qkv_dim, bs] col-major
    __nv_bfloat16 *__restrict__ q_out,      // [q_dim, bs] col-major
    __nv_bfloat16 *__restrict__ k_out,      // [kv_dim, bs] col-major
    __nv_bfloat16 *__restrict__ v_out,      // [kv_dim, bs] col-major
    int q_dim, int kv_dim, int qkv_dim, int bs) {

  int total = qkv_dim * bs;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int col = idx / qkv_dim;
    int row = idx % qkv_dim;
    __nv_bfloat16 val = qkv[idx];

    if (row < q_dim) {
      q_out[col * q_dim + row] = val;
    } else if (row < q_dim + kv_dim) {
      k_out[col * kv_dim + (row - q_dim)] = val;
    } else {
      v_out[col * kv_dim + (row - q_dim - kv_dim)] = val;
    }
  }
}

// ============================================================================
// Fused SiLU-mul from combined [2*I, bs] gate+up buffer.
// Column-major: token j at offset j * 2*I.
//   gate = combined[j * 2*I + i]     for i in [0, I)
//   up   = combined[j * 2*I + I + i] for i in [0, I)
//   out[j * I + i] = silu(gate) * up
// ============================================================================

__global__ void silu_mul_fused_kernel(
    const __nv_bfloat16 *__restrict__ gate_up, // [2*I, bs] col-major
    __nv_bfloat16 *__restrict__ out,            // [I, bs] col-major
    int intermediate_size, int bs) {

  int total = intermediate_size * bs;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += gridDim.x * blockDim.x) {
    int col = idx / intermediate_size;
    int row = idx % intermediate_size;

    int src_offset = col * 2 * intermediate_size;
    float g = __bfloat162float(gate_up[src_offset + row]);
    float u = __bfloat162float(gate_up[src_offset + intermediate_size + row]);

    float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2bfloat16(silu_g * u);
  }
}

extern "C" {

void deinterleave_qkv_cuda(
    const __nv_bfloat16 *qkv, __nv_bfloat16 *q_out,
    __nv_bfloat16 *k_out, __nv_bfloat16 *v_out,
    int q_dim, int kv_dim, int bs, cudaStream_t stream) {
  int qkv_dim = q_dim + 2 * kv_dim;
  int total = qkv_dim * bs;
  int block = 256;
  int grid = (total + block - 1) / block;
  deinterleave_qkv_kernel<<<grid, block, 0, stream>>>(
      qkv, q_out, k_out, v_out, q_dim, kv_dim, qkv_dim, bs);
}

void silu_mul_fused_cuda(
    const __nv_bfloat16 *gate_up, __nv_bfloat16 *out,
    int intermediate_size, int bs, cudaStream_t stream) {
  int total = intermediate_size * bs;
  int block = 256;
  int grid = (total + block - 1) / block;
  silu_mul_fused_kernel<<<grid, block, 0, stream>>>(
      gate_up, out, intermediate_size, bs);
}

} // extern "C"
