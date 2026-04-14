#include <cuda_bf16.h>
#include <cublas_v2.h>

// cuBLAS handle management.
// Make handles thread-local so each TP rank thread can bind a handle to its own
// CUDA context/device without racing on a process-global singleton.
thread_local cublasHandle_t g_cublas_handle = nullptr;
thread_local cublasHandle_t g_cublas_prefill_handle = nullptr;
thread_local void *g_cublas_workspace = nullptr;
static const size_t CUBLAS_WORKSPACE_SIZE = 32 * 1024 * 1024; // 32MB

extern "C" {

int cuda_set_device(int device_ordinal) { return static_cast<int>(cudaSetDevice(device_ordinal)); }

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

// Graph-safe GEMM: same math as gemm_cuda but uses the workspace-free handle.
// Safe for CUDA Graph capture and decode path.
void gemm_graphsafe_cuda(const __nv_bfloat16 *W, const __nv_bfloat16 *X, __nv_bfloat16 *Y,
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

// Strided batched GEMM: C_i = alpha * op(A_i) * op(B_i) for i in [0, batch_count)
// All bf16 inputs and outputs, fp32 accumulation.
// Used for MLA Q absorption and V de-absorption.
void gemm_strided_batched_cuda(
    int transa,  // 0 = N, 1 = T
    int transb,  // 0 = N, 1 = T
    int m, int n, int k,
    const __nv_bfloat16 *A, int lda, long long stride_a,
    const __nv_bfloat16 *B, int ldb, long long stride_b,
    __nv_bfloat16 *C, int ldc, long long stride_c,
    int batch_count,
    cudaStream_t stream)
{
  const float h_alpha = 1.0f;
  const float h_beta = 0.0f;
  cublasOperation_t opA = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSetStream(g_cublas_prefill_handle, stream);
  cublasGemmStridedBatchedEx(
      g_cublas_prefill_handle,
      opA, opB,
      m, n, k,
      &h_alpha,
      A, CUDA_R_16BF, lda, stride_a,
      B, CUDA_R_16BF, ldb, stride_b,
      &h_beta,
      C, CUDA_R_16BF, ldc, stride_c,
      batch_count,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

} // extern "C"
