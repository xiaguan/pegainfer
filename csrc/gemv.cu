#include "common.cuh"
#include <cublas_v2.h>

// ============================================================================
// Hand-written GEMV: y = A @ x (row-major matrix)
// Each block processes GEMV_ROWS_PER_BLOCK rows.
// 256 threads stride over K, shared-memory reduction to final dot product.
// BF16 inputs, FP32 accumulators, BF16 output.
// Graph-capture safe (no cuBLAS workspace allocation).
// ============================================================================
#define GEMV_BLOCK 256
#define GEMV_ROWS_PER_BLOCK 4

__global__ void gemv_handwritten_kernel(
    const __nv_bfloat16 *__restrict__ A, // (M, K) row-major
    const __nv_bfloat16 *__restrict__ x, // (K,)
    __nv_bfloat16 *__restrict__ y,       // (M,)
    int M, int K) {

  int row_base = blockIdx.x * GEMV_ROWS_PER_BLOCK;
  int tid = threadIdx.x;

  __shared__ float tile_red[GEMV_ROWS_PER_BLOCK][GEMV_BLOCK];

  // Each thread accumulates partial dot products for GEMV_ROWS_PER_BLOCK rows
  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
    int row = row_base + r;
    float sum = 0.0f;
    if (row < M) {
      const __nv_bfloat16 *A_row = A + row * K;
      for (int k = tid; k < K; k += GEMV_BLOCK) {
        sum += __bfloat162float(A_row[k]) * __bfloat162float(x[k]);
      }
    }
    tile_red[r][tid] = sum;
  }
  __syncthreads();

  // Shared memory reduction
  for (int stride = GEMV_BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      #pragma unroll
      for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
        tile_red[r][tid] += tile_red[r][tid + stride];
      }
    }
    __syncthreads();
  }

  // Thread 0 writes final results
  if (tid == 0) {
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
      int row = row_base + r;
      if (row < M) {
        y[row] = __float2bfloat16(tile_red[r][0]);
      }
    }
  }
}

// Attention score: score = q @ k / sqrt(head_dim)
__global__ void attention_scores_kernel(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ scores,
    int seq_len, int head_dim, float scale) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < seq_len) {
    float dot = 0.0f;
    const __nv_bfloat16 *k = k_cache + pos * head_dim;
    for (int i = 0; i < head_dim; i++) {
      dot += __bfloat162float(q[i]) * __bfloat162float(k[i]);
    }
    scores[pos] = __float2bfloat16(dot * scale);
  }
}

// Attention weighted sum: out = sum(weights[i] * v[i])
__global__ void attention_weighted_sum_kernel(
    const __nv_bfloat16 *__restrict__ weights,
    const __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ out,
    int seq_len, int head_dim) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < head_dim) {
    float sum = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
      sum += __bfloat162float(weights[pos]) * __bfloat162float(v_cache[pos * head_dim + d]);
    }
    out[d] = __float2bfloat16(sum);
  }
}

// cuBLAS handle management
static cublasHandle_t g_cublas_handle = nullptr;

extern "C" {

void cublas_init() {
  if (g_cublas_handle == nullptr) {
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
  }
}

void cublas_destroy() {
  if (g_cublas_handle != nullptr) {
    cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
  }
}


void gemv_batched_qkv_cuda(const __nv_bfloat16 *Wq, const __nv_bfloat16 *Wk, const __nv_bfloat16 *Wv,
                           const __nv_bfloat16 *x, __nv_bfloat16 *q_out, __nv_bfloat16 *k_out,
                           __nv_bfloat16 *v_out, int Mq, int Mk, int K,
                           cudaStream_t stream) {
  int blocks_q = (Mq + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
  int blocks_k = (Mk + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;

  gemv_handwritten_kernel<<<blocks_q, GEMV_BLOCK, 0, stream>>>(Wq, x, q_out, Mq, K);
  gemv_handwritten_kernel<<<blocks_k, GEMV_BLOCK, 0, stream>>>(Wk, x, k_out, Mk, K);
  gemv_handwritten_kernel<<<blocks_k, GEMV_BLOCK, 0, stream>>>(Wv, x, v_out, Mk, K);
}

void gemv_cuda(const __nv_bfloat16 *A, const __nv_bfloat16 *x, __nv_bfloat16 *y, int M, int K,
               cudaStream_t stream) {
  int num_blocks = (M + GEMV_ROWS_PER_BLOCK - 1) / GEMV_ROWS_PER_BLOCK;
  gemv_handwritten_kernel<<<num_blocks, GEMV_BLOCK, 0, stream>>>(A, x, y, M, K);
}

// General GEMM: Y = W @ X where W is [M, K] row-major, X is [K, N] col-major, Y is [M, N] col-major
// N=1 is equivalent to GEMV. N>1 enables batched prefill.
void gemm_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
               int M, int N, int K, cudaStream_t stream) {
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasSetStream(g_cublas_handle, stream);
  cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               M, N, K,
               &h_alpha,
               W, CUDA_R_16BF, K,
               X, CUDA_R_16BF, K,
               &h_beta,
               Y, CUDA_R_16BF, M,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void attention_scores_cuda(const __nv_bfloat16 *q, const __nv_bfloat16 *k_cache, __nv_bfloat16 *scores,
                           int seq_len, int head_dim, float scale,
                           cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (seq_len + block_size - 1) / block_size;
  attention_scores_kernel<<<num_blocks, block_size, 0, stream>>>(
      q, k_cache, scores, seq_len, head_dim, scale);
}

void attention_weighted_sum_cuda(const __nv_bfloat16 *weights, const __nv_bfloat16 *v_cache,
                                 __nv_bfloat16 *out, int seq_len, int head_dim,
                                 cudaStream_t stream) {
  int block_size = 128;
  int num_blocks = (head_dim + block_size - 1) / block_size;
  attention_weighted_sum_kernel<<<num_blocks, block_size, 0, stream>>>(
      weights, v_cache, out, seq_len, head_dim);
}

} // extern "C"
