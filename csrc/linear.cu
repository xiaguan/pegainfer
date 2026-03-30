#include <cuda_bf16.h>
#include <cublas_v2.h>

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

} // extern "C"
