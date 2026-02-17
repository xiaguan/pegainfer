#include "common.cuh"

#define FUSED_MLP_TILE 256
#define FUSED_MLP_INTER_PER_BLOCK 4

// Phase 1: Compute gate, up, and activation
// Each block handles FUSED_MLP_INTER_PER_BLOCK intermediate dimensions
__global__ void fused_mlp_intermediate_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate_proj,
    const __nv_bfloat16 *__restrict__ up_proj,
    __nv_bfloat16 *__restrict__ act,
    int intermediate_size,
    int hidden_size) {

  int inter_base = blockIdx.x * FUSED_MLP_INTER_PER_BLOCK;
  int tid = threadIdx.x;

  __shared__ float gate_acc[FUSED_MLP_INTER_PER_BLOCK];
  __shared__ float up_acc[FUSED_MLP_INTER_PER_BLOCK];
  __shared__ float tile_red[FUSED_MLP_INTER_PER_BLOCK][FUSED_MLP_TILE];

  // Compute gate projections for this block
  for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
    int inter_idx = inter_base + r;
    if (inter_idx >= intermediate_size) break;

    float sum = 0.0f;
    const __nv_bfloat16 *gate_row = gate_proj + inter_idx * hidden_size;

    for (int i = tid; i < hidden_size; i += FUSED_MLP_TILE) {
      sum += __bfloat162float(gate_row[i]) * __bfloat162float(x[i]);
    }
    tile_red[r][tid] = sum;
  }
  __syncthreads();

  // Reduce gate
  for (int stride = FUSED_MLP_TILE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
        tile_red[r][tid] += tile_red[r][tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
      gate_acc[r] = tile_red[r][0];
    }
  }
  __syncthreads();

  // Compute up projections
  for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
    int inter_idx = inter_base + r;
    if (inter_idx >= intermediate_size) break;

    float sum = 0.0f;
    const __nv_bfloat16 *up_row = up_proj + inter_idx * hidden_size;

    for (int i = tid; i < hidden_size; i += FUSED_MLP_TILE) {
      sum += __bfloat162float(up_row[i]) * __bfloat162float(x[i]);
    }
    tile_red[r][tid] = sum;
  }
  __syncthreads();

  // Reduce up
  for (int stride = FUSED_MLP_TILE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
        tile_red[r][tid] += tile_red[r][tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
      up_acc[r] = tile_red[r][0];
    }
  }
  __syncthreads();

  // Compute SiLU(gate) * up and write activation
  if (tid == 0) {
    for (int r = 0; r < FUSED_MLP_INTER_PER_BLOCK; r++) {
      int inter_idx = inter_base + r;
      if (inter_idx < intermediate_size) {
        float g = gate_acc[r];
        float silu_g = g / (1.0f + expf(-g));
        float result = silu_g * up_acc[r];
        act[inter_idx] = __float2bfloat16(result);
      }
    }
  }
}

// Phase 2: Down projection (out = down_proj @ act)
#define FUSED_MLP_OUT_PER_BLOCK 4

__global__ void fused_mlp_output_kernel(
    const __nv_bfloat16 *__restrict__ act,
    const __nv_bfloat16 *__restrict__ down_proj,
    __nv_bfloat16 *__restrict__ out,
    int hidden_size,
    int intermediate_size) {

  int out_base = blockIdx.x * FUSED_MLP_OUT_PER_BLOCK;
  int tid = threadIdx.x;

  __shared__ float tile_red[FUSED_MLP_OUT_PER_BLOCK][FUSED_MLP_TILE];
  float acc[FUSED_MLP_OUT_PER_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int k0 = 0; k0 < intermediate_size; k0 += FUSED_MLP_TILE) {
    int k = k0 + tid;
    float act_val = (k < intermediate_size) ? __bfloat162float(act[k]) : 0.0f;

    for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
      int row = out_base + r;
      float contrib = 0.0f;
      if (row < hidden_size && k < intermediate_size) {
        contrib = __bfloat162float(down_proj[row * intermediate_size + k]) * act_val;
      }
      tile_red[r][tid] = contrib;
    }
    __syncthreads();

    // Reduce
    for (int stride = FUSED_MLP_TILE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
          tile_red[r][tid] += tile_red[r][tid + stride];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
        acc[r] += tile_red[r][0];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int r = 0; r < FUSED_MLP_OUT_PER_BLOCK; r++) {
      int row = out_base + r;
      if (row < hidden_size) {
        out[row] = __float2bfloat16(acc[r]);
      }
    }
  }
}

extern "C" {
void fused_mlp_cuda(const __nv_bfloat16 *x, const __nv_bfloat16 *gate_proj, const __nv_bfloat16 *up_proj,
                    const __nv_bfloat16 *down_proj, __nv_bfloat16 *out, int hidden_size,
                    int intermediate_size, cudaStream_t stream) {
  __nv_bfloat16 *d_act;
  cudaMallocAsync(&d_act, intermediate_size * sizeof(__nv_bfloat16), stream);

  // Phase 1: Compute gate, up, and activation
  int inter_blocks = (intermediate_size + FUSED_MLP_INTER_PER_BLOCK - 1) / FUSED_MLP_INTER_PER_BLOCK;
  fused_mlp_intermediate_kernel<<<inter_blocks, FUSED_MLP_TILE, 0, stream>>>(
      x, gate_proj, up_proj, d_act, intermediate_size, hidden_size);

  // Phase 2: Down projection
  int out_blocks = (hidden_size + FUSED_MLP_OUT_PER_BLOCK - 1) / FUSED_MLP_OUT_PER_BLOCK;
  fused_mlp_output_kernel<<<out_blocks, FUSED_MLP_TILE, 0, stream>>>(
      d_act, down_proj, out, hidden_size, intermediate_size);

  cudaFreeAsync(d_act, stream);
}
}
