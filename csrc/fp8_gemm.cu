// csrc/fp8_gemm.cu
//
// Torch-free C wrapper for DeepGEMM SM90 FP8 block-scale GEMM (1D2D variant).
// Constructs TMA descriptors via CUDA driver API and launches pre-instantiated
// DeepGEMM kernel templates. No JIT, no torch dependency.
//
// 1D2D means: 1D per-token scale on A (activation), 2D per-block scale on B (weight).
// This is the standard path for Hopper FP8 block-scale GEMM — used by SGLang, vLLM,
// and DeepGEMM's own default recipe when gran_n != 1.
//
// Key differences from the 1D1D variant we replaced:
//   - Output is bf16 (not fp32) — the kernel does dequant + accumulation + bf16 cast
//   - SFB (weight scales) loaded via global memory reads, not TMA
//   - D (output) TMA descriptor uses bf16 dtype with 128B swizzle
//   - Template adds kMajorSFB and kSwizzleDMode; epilogue type lives in
//     `deep_gemm::epilogue::transform`
//
// Kernel header: deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh
// Compiled with: --std=c++20 -arch=sm_90a
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// DeepGEMM kernel header (pure CUDA + CUTLASS, no torch)
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

// ============================================================================
// Number of SMs — template parameter and grid dimension.
// Passed from build.rs via -DDG_NUM_SMS=<detected_value>.
// Default to 132 (H100) if not specified.
// ============================================================================
#ifndef DG_NUM_SMS
#define DG_NUM_SMS 132
#endif

// ============================================================================
// TMA descriptor construction (replaces DeepGEMM's runtime_utils.hpp torch glue)
// ============================================================================

static CUtensorMapSwizzle swizzle_mode_to_tma(int mode) {
    switch (mode) {
        case 0:
        case 16:  return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default:  return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

// Core 2D TMA descriptor builder — direct cuTensorMapEncodeTiled call.
static CUtensorMap make_tma_2d_desc(
    const void* data_ptr,
    CUtensorMapDataType dtype,
    int elem_size,
    int gmem_inner_dim, int gmem_outer_dim,
    int smem_inner_dim, int smem_outer_dim,
    int gmem_outer_stride,
    int swizzle_mode)
{
    if (swizzle_mode != 0)
        smem_inner_dim = swizzle_mode / elem_size;

    CUtensorMap tensor_map{};
    const cuuint64_t gmem_dims[2] = {
        static_cast<cuuint64_t>(gmem_inner_dim),
        static_cast<cuuint64_t>(gmem_outer_dim)
    };
    const cuuint32_t smem_dims[2] = {
        static_cast<cuuint32_t>(smem_inner_dim),
        static_cast<cuuint32_t>(smem_outer_dim)
    };
    const cuuint64_t gmem_strides[1] = {
        static_cast<cuuint64_t>(gmem_outer_stride * elem_size)
    };
    const cuuint32_t elem_strides[2] = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &tensor_map, dtype,
        2, const_cast<void*>(data_ptr),
        gmem_dims, gmem_strides, smem_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_mode_to_tma(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (err != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(err, &err_str);
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", err_str ? err_str : "unknown");
    }
    return tensor_map;
}

// A descriptor: A[M, K] row-major (Major::K) — inner=K, outer=M
static CUtensorMap make_tma_a_desc(
    const void* ptr, int shape_m, int shape_k,
    int block_m, int block_k, int swizzle_mode)
{
    return make_tma_2d_desc(
        ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, /*elem_size=*/1,
        /*gmem_inner=*/shape_k, /*gmem_outer=*/shape_m,
        /*smem_inner=*/block_k, /*smem_outer=*/block_m,
        /*stride=*/shape_k,
        swizzle_mode);
}

// B descriptor: B[N, K] row-major (Major::K) — inner=K, outer=N
static CUtensorMap make_tma_b_desc(
    const void* ptr, int shape_n, int shape_k,
    int block_n, int block_k, int swizzle_mode)
{
    return make_tma_2d_desc(
        ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, /*elem_size=*/1,
        /*gmem_inner=*/shape_k, /*gmem_outer=*/shape_n,
        /*smem_inner=*/block_k, /*smem_outer=*/block_n,
        /*stride=*/shape_k,
        swizzle_mode);
}

// SFA descriptor: scale_a [ceil(K/128), padded(M, 4)] — M as inner (MN-major), K-chunk as outer.
// TMA loads one [block_m, 1] tile per stage (block_m scales for one K-chunk).
// TMA requires inner dim aligned to 16 bytes (4 floats for float32).
static CUtensorMap make_tma_sfa_desc(
    const void* ptr, int shape_m, int shape_k,
    int block_m, int gran_k)
{
    constexpr int kTMAAlignBytes = 16;
    int aligned_m = ((shape_m + kTMAAlignBytes / sizeof(float) - 1)
                     / (kTMAAlignBytes / sizeof(float)))
                    * (kTMAAlignBytes / sizeof(float));
    int sf_k_dim = (shape_k + gran_k - 1) / gran_k;  // ceil_div(K, 128)

    return make_tma_2d_desc(
        ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, /*elem_size=*/sizeof(float),
        /*gmem_inner=*/aligned_m, /*gmem_outer=*/sf_k_dim,
        /*smem_inner=*/block_m,   /*smem_outer=*/1,
        /*stride=*/aligned_m,
        /*swizzle=*/0);  // no swizzle for scale factors
}

// D descriptor: D[M, N] bf16 row-major, with swizzle for efficient TMA store.
// 1D2D stores full block_m rows (not just wgmma_m=64 like 1D1D).
// Swizzle 128B for bf16: TMA_D_BLOCK_N = 128 / sizeof(bf16) = 64.
// Two TMA stores per block to cover full BLOCK_N=128.
static CUtensorMap make_tma_d_desc(
    const void* ptr, int shape_m, int shape_n,
    int block_m, int block_n, int swizzle_mode)
{
    return make_tma_2d_desc(
        ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, /*elem_size=*/sizeof(__nv_bfloat16),
        /*gmem_inner=*/shape_n, /*gmem_outer=*/shape_m,
        /*smem_inner=*/block_n, /*smem_outer=*/block_m,
        /*stride=*/shape_n,
        swizzle_mode);
}

// ============================================================================
// Shared memory size computation (1D2D variant)
//
// 1D2D layout differs from 1D1D:
//   - D buffer is bf16 (2 bytes), not fp32 (4 bytes) → smaller
//   - No per-stage SFB in smem (SFB loaded via global memory by math warps)
//   - SFB has a fixed-size buffer for the full K range
// ============================================================================

static inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }
static inline int ceil_div_int(int x, int y) { return (x + y - 1) / y; }

static int compute_smem_size(
    int block_m, int block_n, int block_k,
    int num_stages, int k)
{
    // D output buffer (bf16), aligned to 1024 for TMA swizzle
    int smem_d = align_up(block_m * block_n * (int)sizeof(__nv_bfloat16), 1024);

    // Per-stage A/B (FP8, 1 byte per element)
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;

    // Per-stage SFA (FP32), aligned to 128 for TMA
    int smem_sfa_per_stage = align_up(block_m * (int)sizeof(float), 128);

    // No per-stage SFB — loaded via global memory reads
    int per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;

    // SFB buffer (loaded from global memory by math warps)
    // For BLOCK_K % BLOCK_N == 0 (both 128): uniform=true → factor=1
    bool uniform = (block_k % block_n == 0);
    int shape_k_scales = ceil_div_int(k, block_k);
    int smem_sfb = align_up(shape_k_scales * (uniform ? 1 : 2) * (int)sizeof(float), 8);

    // Barriers: num_stages * 2 barriers * 8 bytes each
    int smem_barrier = num_stages * 8 * 2;

    return smem_d + num_stages * per_stage + smem_sfb + smem_barrier;
}

// ============================================================================
// Kernel instantiations (1D2D)
//
// FP8 1D2D Normal GEMM: 1D per-token scale on A, 2D per-block scale on B.
// Both A/B Major::K (row-major), BF16 output.
//
// block_k = 128 (required for FP8 block scaling)
// swizzle_a = swizzle_b = 128 (128 bytes = block_k * 1 byte FP8)
// swizzle_d = 128 (128 bytes for bf16 output)
// num_tma_threads = 128
// num_multicast = 1 (no multicast for simplicity)
//
// kMajorSFB = Major::K — DSV3.2 checkpoint stores weight scales as
//   [ceil(N/128), ceil(K/128)] with K-chunk as inner/contiguous dimension.
//
// Config 1: BLOCK_M=64,  BLOCK_N=128, 8 stages, 128+128 threads (decode/small M)
// Config 2: BLOCK_M=128, BLOCK_N=128, 5 stages, 128+256 threads (prefill/large M)
//
// Stage counts computed from SM90 smem capacity (232448 bytes):
//   Config 1: per_stage=24832, fixed≈16608 → 8 stages → 215264 bytes
//   Config 2: per_stage=33280, fixed≈33072 → 5 stages → 199472 bytes
// ============================================================================

namespace {
    // Config 1: block_m=64, block_n=128, 8 stages
    static auto kernel_64_128 = reinterpret_cast<void*>(
        &sm90_fp8_gemm_1d2d_impl<
            cute::UMMA::Major::K,       // kMajorSFB: weight scales [ceil(N/128), ceil(K/128)] K-major
            0, 0, 0,                    // SHAPE_M/N/K (all dynamic)
            1,                          // kNumGroups (Normal GEMM)
            64, 128, 128,              // BLOCK_M, BLOCK_N, BLOCK_K
            128, 128, 128,             // kSwizzleAMode, kSwizzleBMode, kSwizzleDMode
            8,                         // kNumStages
            128, 128,                  // kNumTMAThreads, kNumMathThreads
            1, false,                  // kNumTMAMulticast, kIsTMAMulticastOnA
            DG_NUM_SMS,                // kNumSMs
            GemmType::Normal,          // kGemmType
            epilogue::transform::EpilogueIdentity  // identity epilogue
        >);

    // Config 2: block_m=128, block_n=128, 5 stages
    static auto kernel_128_128 = reinterpret_cast<void*>(
        &sm90_fp8_gemm_1d2d_impl<
            cute::UMMA::Major::K,
            0, 0, 0,
            1,
            128, 128, 128,
            128, 128, 128,
            5,
            128, 256,                  // block_m > 64 → 2 warpgroups → 256 math threads
            1, false,
            DG_NUM_SMS,
            GemmType::Normal,
            epilogue::transform::EpilogueIdentity
        >);
}

// ============================================================================
// Public C API (called from Rust via FFI)
//
// D[M,N] = dequant(A[M,K]) @ dequant(B[N,K])^T
//
// A: FP8 e4m3 [M, K] row-major (activation, online-quantized)
// scale_a: FP32 [ceil(K/128), padded(M, 4)] K-chunk-major — 1D per-token dequant scales
// B: FP8 e4m3 [N, K] row-major (weight, from checkpoint)
// scale_b: FP32 [ceil(N/128), ceil(K/128)] — 2D per-block dequant scales (K-major)
// D: BF16 [M, N] row-major (output)
// ============================================================================

extern "C" void fp8_gemm_cuda(
    const void* a,           // FP8 e4m3 data [M, K] row-major
    const void* scale_a,     // FP32 dequant scales for A, [ceil(K/128), padded(M, 4)] K-chunk-major (1D per-token)
    const void* b,           // FP8 e4m3 data [N, K] row-major
    const void* scale_b,     // FP32 dequant scales for B, [ceil(N/128), ceil(K/128)] (2D per-block, K-major)
    void* d,                 // BF16 output [M, N] row-major
    int m, int n, int k,
    cudaStream_t stream)
{
    // Select tile config based on M
    void* kernel_func;
    int block_m, block_n, num_stages, num_threads;

    if (m <= 64) {
        kernel_func = kernel_64_128;
        block_m = 64;  block_n = 128;
        num_stages = 8;
        num_threads = 128 + 128;  // TMA + math
    } else {
        kernel_func = kernel_128_128;
        block_m = 128; block_n = 128;
        num_stages = 5;
        num_threads = 128 + 256;
    }

    constexpr int block_k = 128;
    constexpr int swizzle_ab = 128;   // FP8 with block_k=128: 128 * 1 byte = 128B
    constexpr int swizzle_d = 128;    // BF16 with block_n=128: 128 * 2 bytes = 256B → 128B swizzle
    int num_sms = DG_NUM_SMS;

    // Compute dynamic shared memory size
    int smem_size = compute_smem_size(block_m, block_n, block_k, num_stages, k);

    // Set max dynamic shared memory for this kernel (SM90 supports up to 232448 bytes)
    cudaFuncSetAttribute(kernel_func,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Construct TMA descriptors
    CUtensorMap tma_a   = make_tma_a_desc(a, m, k, block_m, block_k, swizzle_ab);
    CUtensorMap tma_b   = make_tma_b_desc(b, n, k, block_n, block_k, swizzle_ab);
    CUtensorMap tma_sfa = make_tma_sfa_desc(scale_a, m, k, block_m, block_k);
    CUtensorMap tma_d   = make_tma_d_desc(d, m, n, block_m, block_n, swizzle_d);

    // Kernel arguments for 1D2D Normal GEMM.
    // sfb: raw float pointer (loaded by math warps from global memory, not TMA).
    // grouped_layout: nullptr for Normal GEMM (only used by GroupedMasked/GroupedContiguous).
    float* sfb_ptr = const_cast<float*>(static_cast<const float*>(scale_b));
    int* null_layout = nullptr;
    uint32_t shape_m = static_cast<uint32_t>(m);
    uint32_t shape_n = static_cast<uint32_t>(n);
    uint32_t shape_k = static_cast<uint32_t>(k);

    void* args[] = {
        &sfb_ptr, &null_layout,
        &shape_m, &shape_n, &shape_k,
        &tma_a, &tma_b, &tma_d, &tma_sfa
    };

    dim3 grid(num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaLaunchKernel(kernel_func, grid, block, args, smem_size, stream);
}
