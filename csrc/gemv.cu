#include "common.cuh"
#include <cublas_v2.h>

// Simple GEMV: y = A @ x (row-major matrix)
// Each thread computes one output element
__global__ void gemv_kernel(const __nv_bfloat16 *__restrict__ A, // (M, K) row-major
                            const __nv_bfloat16 *__restrict__ x, // (K,)
                            __nv_bfloat16 *__restrict__ y,       // (M,)
                            int M, int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    float sum = 0.0f;
    const __nv_bfloat16 *A_row = A + row * K;
    for (int k = 0; k < K; k++) {
      sum += __bfloat162float(A_row[k]) * __bfloat162float(x[k]);
    }
    y[row] = __float2bfloat16(sum);
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

void gemv_naive_cuda(const __nv_bfloat16 *A, const __nv_bfloat16 *x, __nv_bfloat16 *y, int M, int K,
                     cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (M + block_size - 1) / block_size;
  gemv_kernel<<<num_blocks, block_size, 0, stream>>>(A, x, y, M, K);
}

void gemv_cublas_cuda(const __nv_bfloat16 *A, const __nv_bfloat16 *x, __nv_bfloat16 *y, int M, int K,
                      cudaStream_t stream) {
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;

  cublasSetStream(g_cublas_handle, stream);
  cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               M, 1, K,
               &h_alpha,
               A, CUDA_R_16BF, K,
               x, CUDA_R_16BF, K,
               &h_beta,
               y, CUDA_R_16BF, M,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void gemv_batched_qkv_cuda(const __nv_bfloat16 *Wq, const __nv_bfloat16 *Wk, const __nv_bfloat16 *Wv,
                           const __nv_bfloat16 *x, __nv_bfloat16 *q_out, __nv_bfloat16 *k_out,
                           __nv_bfloat16 *v_out, int Mq, int Mk, int K,
                           cudaStream_t stream) {
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;

  cublasSetStream(g_cublas_handle, stream);

  cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               Mq, 1, K, &h_alpha,
               Wq, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
               &h_beta, q_out, CUDA_R_16BF, Mq,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               Mk, 1, K, &h_alpha,
               Wk, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
               &h_beta, k_out, CUDA_R_16BF, Mk,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
               Mk, 1, K, &h_alpha,
               Wv, CUDA_R_16BF, K, x, CUDA_R_16BF, K,
               &h_beta, v_out, CUDA_R_16BF, Mk,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void gemv_cuda(const __nv_bfloat16 *A, const __nv_bfloat16 *x, __nv_bfloat16 *y, int M, int K,
               cudaStream_t stream) {
  gemv_cublas_cuda(A, x, y, M, K, stream);
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
