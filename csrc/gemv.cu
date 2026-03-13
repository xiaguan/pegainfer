#include "common.cuh"
#include <cublas_v2.h>

// ============================================================================
// Hand-written GEMV: y = A @ x (row-major matrix)
// Each block processes GEMV_ROWS_PER_BLOCK rows.
// BF16×4 vectorized loads (8 bytes/thread/stride) for memory throughput.
// Warp shuffle reduction + shared memory for inter-warp reduce.
// BF16 inputs, FP32 accumulators, BF16 output.
// Graph-capture safe (no cuBLAS workspace allocation).
// ============================================================================
#define GEMV_BLOCK 256
#define GEMV_ROWS_PER_BLOCK 4
#define GEMV_NUM_WARPS (GEMV_BLOCK / WARP_SIZE)

__global__ void gemv_handwritten_kernel(
    const __nv_bfloat16 *__restrict__ A, // (M, K) row-major
    const __nv_bfloat16 *__restrict__ x, // (K,)
    __nv_bfloat16 *__restrict__ y,       // (M,)
    int M, int K) {

  int row_base = blockIdx.x * GEMV_ROWS_PER_BLOCK;
  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  // Vectorized BF16×4 path: process 4 elements per load
  int K4 = K / 4;  // number of bf16x4 groups
  int K_tail = K - K4 * 4;  // remainder for scalar fallback

  float sums[GEMV_ROWS_PER_BLOCK];
  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) sums[r] = 0.0f;

  // Cast to uint2 for 8-byte aligned loads (4× bf16)
  const uint2 *x_vec = reinterpret_cast<const uint2 *>(x);

  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
    int row = row_base + r;
    if (row < M) {
      const uint2 *A_row_vec = reinterpret_cast<const uint2 *>(A + row * K);
      float sum = 0.0f;

      // Main vectorized loop: 4 bf16 elements per iteration
      for (int k4 = tid; k4 < K4; k4 += GEMV_BLOCK) {
        uint2 a_val = A_row_vec[k4];
        uint2 x_val = x_vec[k4];
        // Unpack 4× bf16 from two uint32
        __nv_bfloat162 a_lo = *reinterpret_cast<__nv_bfloat162 *>(&a_val.x);
        __nv_bfloat162 a_hi = *reinterpret_cast<__nv_bfloat162 *>(&a_val.y);
        __nv_bfloat162 x_lo = *reinterpret_cast<__nv_bfloat162 *>(&x_val.x);
        __nv_bfloat162 x_hi = *reinterpret_cast<__nv_bfloat162 *>(&x_val.y);
        sum += __bfloat162float(a_lo.x) * __bfloat162float(x_lo.x);
        sum += __bfloat162float(a_lo.y) * __bfloat162float(x_lo.y);
        sum += __bfloat162float(a_hi.x) * __bfloat162float(x_hi.x);
        sum += __bfloat162float(a_hi.y) * __bfloat162float(x_hi.y);
      }

      // Scalar tail for K not divisible by 4
      if (K_tail > 0) {
        const __nv_bfloat16 *A_row = A + row * K;
        int k_start = K4 * 4;
        for (int k = k_start + tid; k < K; k += GEMV_BLOCK) {
          sum += __bfloat162float(A_row[k]) * __bfloat162float(x[k]);
        }
      }

      sums[r] = sum;
    }
  }

  // Warp-level reduction via shuffle
  #pragma unroll
  for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
    sums[r] = warp_reduce_sum(sums[r]);
  }

  // Inter-warp reduction via shared memory
  __shared__ float warp_sums[GEMV_ROWS_PER_BLOCK][GEMV_NUM_WARPS];

  if (lane_id == 0) {
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
      warp_sums[r][warp_id] = sums[r];
    }
  }
  __syncthreads();

  // First warp reduces across all warps
  if (warp_id == 0) {
    #pragma unroll
    for (int r = 0; r < GEMV_ROWS_PER_BLOCK; r++) {
      float val = (lane_id < GEMV_NUM_WARPS) ? warp_sums[r][lane_id] : 0.0f;
      val = warp_reduce_sum(val);
      if (lane_id == 0) {
        int row = row_base + r;
        if (row < M) {
          y[row] = __float2bfloat16(val);
        }
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

// cuBLAS handle management (external linkage — shared with prefill_attention.cu)
// g_cublas_handle: workspace-free, safe for CUDA Graph capture (decode path).
// g_cublas_prefill_handle: has 32MB workspace, allows cuBLAS to pick faster algorithms
// for the 252 GEMMs per prefill. Never used under CUDA Graphs.
cublasHandle_t g_cublas_handle = nullptr;
cublasHandle_t g_cublas_prefill_handle = nullptr;

static void *g_cublas_workspace = nullptr;
static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB

extern "C" {

void cublas_init() {
  if (g_cublas_handle == nullptr) {
    cublasCreate(&g_cublas_handle);
    cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH);
  }
  if (g_cublas_prefill_handle == nullptr) {
    cublasCreate(&g_cublas_prefill_handle);
    cublasSetMathMode(g_cublas_prefill_handle, CUBLAS_TENSOR_OP_MATH);
    cudaMalloc(&g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
    cublasSetWorkspace(g_cublas_prefill_handle, g_cublas_workspace, CUBLAS_WORKSPACE_SIZE);
  }
}

void cublas_destroy() {
  if (g_cublas_handle != nullptr) {
    cublasDestroy(g_cublas_handle);
    g_cublas_handle = nullptr;
  }
  if (g_cublas_prefill_handle != nullptr) {
    cublasDestroy(g_cublas_prefill_handle);
    g_cublas_prefill_handle = nullptr;
  }
  if (g_cublas_workspace != nullptr) {
    cudaFree(g_cublas_workspace);
    g_cublas_workspace = nullptr;
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
// Uses prefill handle (with workspace) — only called from prefill path, never under CUDA Graphs.
void gemm_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
               int M, int N, int K, cudaStream_t stream) {
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasSetStream(g_cublas_prefill_handle, stream);
  cublasGemmEx(g_cublas_prefill_handle, CUBLAS_OP_T, CUBLAS_OP_N,
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
