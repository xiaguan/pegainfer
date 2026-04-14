// csrc/fp8_gemm.cu
//
// Torch-free C wrapper for DeepGEMM SM90 FP8 block-scale GEMM.
// Constructs TMA descriptors via CUDA driver API and launches pre-instantiated
// DeepGEMM kernel templates. No JIT, no torch dependency.
//
// Kernel header path: deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh
// This file is compiled with: --std=c++20 -arch=sm_90a
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

// DeepGEMM kernel header (pure CUDA + CUTLASS, no torch)
#include <deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

// ============================================================================
// Number of SMs — used as both template parameter and grid dimension.
// Passed from build.rs via -DDG_NUM_SMS=<detected_value>.
// Default to 132 (H100) if not specified; works on any GPU with <= 132 SMs
// (excess blocks exit immediately).
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
// Mirrors DeepGEMM's make_tma_2d_desc() but takes raw params instead of torch::Tensor.
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

// Scale factor descriptor: sf layout is MN-major [ceil(MN/gran), ceil(K/gran)] float.
// TMA requires inner dim aligned to 16 bytes (4 floats for float32).
static CUtensorMap make_tma_sf_desc(
    const void* ptr, int shape_mn, int shape_k,
    int block_mn, int gran_k)
{
    constexpr int kTMAAlignBytes = 16;
    int aligned_mn = ((shape_mn + kTMAAlignBytes / sizeof(float) - 1)
                      / (kTMAAlignBytes / sizeof(float)))
                     * (kTMAAlignBytes / sizeof(float));
    int sf_k_dim = (shape_k + gran_k - 1) / gran_k;  // ceil_div(K, 128)

    return make_tma_2d_desc(
        ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, /*elem_size=*/sizeof(float),
        /*gmem_inner=*/aligned_mn, /*gmem_outer=*/sf_k_dim,
        /*smem_inner=*/block_mn,   /*smem_outer=*/1,
        /*stride=*/aligned_mn,
        /*swizzle=*/0);  // no swizzle for scale factors
}

// C/D descriptor: D[M, N] float row-major.
// SM90 stores with single_warpgroup_sync: store_block_m = wgmma_m = 64.
static CUtensorMap make_tma_cd_desc(
    const void* ptr, int shape_m, int shape_n,
    int block_m, int block_n)
{
    constexpr int kWgmmaM = 64;
    int store_block_m = kWgmmaM;  // SM90 single warpgroup sync
    (void)block_m;

    return make_tma_2d_desc(
        ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, /*elem_size=*/sizeof(float),
        /*gmem_inner=*/shape_n, /*gmem_outer=*/shape_m,
        /*smem_inner=*/block_n, /*smem_outer=*/store_block_m,
        /*stride=*/shape_n,
        /*swizzle=*/0);  // no swizzle for FP32 output
}

// ============================================================================
// Shared memory size computation
// Must match what the kernel template expects given BLOCK_M/N/K and NUM_STAGES.
// ============================================================================

static inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }
static inline int ceil_div_int(int x, int y) { return (x + y - 1) / y; }

static int compute_smem_size(
    int block_m, int block_n, int block_k,
    int num_stages, int k)
{
    // C/D output buffer (FP32), aligned to 1024 for TMA
    int smem_cd = align_up(block_m * block_n * (int)sizeof(float), 1024);

    // Per-stage A/B (FP8, 1 byte per element)
    int smem_a_per_stage = block_m * block_k;
    int smem_b_per_stage = block_n * block_k;

    // Per-stage scale factors (FP32), aligned to 128 for TMA
    int smem_sfa_per_stage = align_up(block_m * (int)sizeof(float), 128);
    int smem_sfb_per_stage = align_up(block_n * (int)sizeof(float), 128);

    int per_stage = smem_a_per_stage + smem_b_per_stage
                  + smem_sfa_per_stage + smem_sfb_per_stage;

    // Extra SFB buffer for non-uniform tile sizes
    int use_uniform = (block_k % block_n == 0) ? 1 : 2;
    int smem_extra_sfb = align_up(
        ceil_div_int(k, block_k) * (int)sizeof(float) * use_uniform, 8);

    // M-barriers: num_stages * 2 barriers * 8 bytes each
    int smem_barrier = num_stages * 8 * 2;

    return smem_cd + num_stages * per_stage + smem_extra_sfb + smem_barrier;
}

// ============================================================================
// Kernel instantiations
//
// FP8 1D1D Normal GEMM, both A/B Major::K (row-major), FP32 output.
// block_k = 128 (required for FP8 block scaling)
// swizzle_a = swizzle_b = 128 (128 bytes = block_k * 1 byte)
// swizzle_cd = 0 (FP32 output)
// num_tma_threads = 128
// num_multicast = 1 (no multicast for simplicity)
//
// Config 1: BLOCK_M=64,  BLOCK_N=128, 7 stages, 128 math threads (small M)
// Config 2: BLOCK_M=128, BLOCK_N=128, 4 stages, 256 math threads (large M)
// ============================================================================

// Force template instantiation via DeepGEMM's pattern (taking address)
namespace {
    // Config 1: block_m=64, block_n=128
    static auto kernel_64_128 = reinterpret_cast<void*>(
        &sm90_fp8_gemm_1d1d_impl<
            0, 0, 0,                    // SHAPE_M/N/K (all dynamic)
            1,                          // kNumGroups
            64, 128, 128,               // BLOCK_M, BLOCK_N, BLOCK_K
            128, 128,                   // kSwizzleAMode, kSwizzleBMode
            7,                          // kNumStages
            128, 128,                   // kNumTMAThreads, kNumMathThreads
            1, false,                   // kNumTMAMulticast, kIsTMAMulticastOnA
            DG_NUM_SMS,                 // kNumSMs
            GemmType::Normal,           // kGemmType
            float                       // cd_dtype_t
        >);

    // Config 2: block_m=128, block_n=128
    static auto kernel_128_128 = reinterpret_cast<void*>(
        &sm90_fp8_gemm_1d1d_impl<
            0, 0, 0,
            1,
            128, 128, 128,
            128, 128,
            4,
            128, 256,                   // block_m > 64 → 2 warpgroups
            1, false,
            DG_NUM_SMS,
            GemmType::Normal,
            float
        >);
}

// ============================================================================
// Public C API (called from Rust via FFI)
// ============================================================================

extern "C" void fp8_gemm_cuda(
    const void* a,           // FP8 e4m3 data [M, K] row-major
    const void* scale_a,     // FP32 scale factors for A, MN-major [ceil(M/128), ceil(K/128)]
    const void* b,           // FP8 e4m3 data [N, K] row-major
    const void* scale_b,     // FP32 scale factors for B, MN-major [ceil(N/128), ceil(K/128)]
    float* d,                // FP32 output [M, N] row-major
    int m, int n, int k,
    cudaStream_t stream)
{
    // Select tile config based on M
    void* kernel_func;
    int block_m, block_n, num_stages, num_threads;

    if (m <= 64) {
        kernel_func = kernel_64_128;
        block_m = 64;  block_n = 128;
        num_stages = 7;
        num_threads = 128 + 128;  // TMA + math
    } else {
        kernel_func = kernel_128_128;
        block_m = 128; block_n = 128;
        num_stages = 4;
        num_threads = 128 + 256;
    }

    constexpr int block_k = 128;
    constexpr int swizzle = 128;  // FP8 with block_k=128
    int num_sms = DG_NUM_SMS;

    // Compute dynamic shared memory size
    int smem_size = compute_smem_size(block_m, block_n, block_k, num_stages, k);

    // Set max dynamic shared memory for this kernel
    cudaFuncSetAttribute(kernel_func,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Construct TMA descriptors
    CUtensorMap tma_a   = make_tma_a_desc(a, m, k, block_m, block_k, swizzle);
    CUtensorMap tma_b   = make_tma_b_desc(b, n, k, block_n, block_k, swizzle);
    CUtensorMap tma_sfa = make_tma_sf_desc(scale_a, m, k, block_m, block_k);
    CUtensorMap tma_sfb = make_tma_sf_desc(scale_b, n, k, block_n, block_k);
    CUtensorMap tma_cd  = make_tma_cd_desc(d, m, n, block_m, block_n);

    // Kernel arguments.
    // For GemmType::Normal, gmem_a/b_ptr and grouped_layout are unused (nullptr).
    // tensor_map_buffer is only for KGroupedContiguous (nullptr).
    __nv_fp8_e4m3* null_a_ptr = nullptr;
    __nv_fp8_e4m3* null_b_ptr = nullptr;
    int*           null_layout = nullptr;
    CUtensorMap*   null_tma_buf = nullptr;
    uint32_t shape_m = static_cast<uint32_t>(m);
    uint32_t shape_n = static_cast<uint32_t>(n);
    uint32_t shape_k = static_cast<uint32_t>(k);

    void* args[] = {
        &null_a_ptr, &null_b_ptr, &null_layout, &null_tma_buf,
        &shape_m, &shape_n, &shape_k,
        &tma_a, &tma_b, &tma_sfa, &tma_sfb, &tma_cd
    };

    dim3 grid(num_sms, 1, 1);
    dim3 block(num_threads, 1, 1);

    cudaLaunchKernel(kernel_func, grid, block, args, smem_size, stream);
}
