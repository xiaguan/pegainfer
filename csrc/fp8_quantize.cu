// csrc/fp8_quantize.cu
//
// FP8 block-scale activation quantization: bf16 -> fp8 e4m3 + per-token 1x128 block scale.
//
// Extracted from TRT-LLM's scale_1x128_kernel (Apache 2.0):
//   flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/
//     fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh
//
// Zero external dependencies — only needs CUDA FP8/BF16 intrinsics (SM89+).
//
// Scale output layout: [ceil(K/128), padded(M, 4)] — K-chunk-major, M as inner dim.
// This matches DeepGEMM's TMA scale descriptor layout.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <algorithm>

// ============================================================================
// Helpers
// ============================================================================

__host__ __device__ constexpr int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// Warp-wide max reduction via butterfly shuffle.
template <typename T>
__forceinline__ __device__ T find_max_elem_in_warp(T value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = T(std::max(float(value),
                           __shfl_down_sync(0xFFFFFFFF, float(value), offset)));
    }
    value = T(__shfl_sync(0xFFFFFFFF, float(value), 0));
    return value;
}

// ============================================================================
// Kernel: bf16 -> fp8 e4m3 with per-token 1x128 block scaling
//
// Each warp handles one [1, 128] block:
//   - 32 threads each read 4 bf16 elements (2 x bfloat162)
//   - warp reduce for absmax
//   - scale = 448.0 / absmax  (fp8 e4m3 max representable value)
//   - multiply and cast to fp8 e4m3
//   - store dequant scale = 1/scale = absmax/448.0
//
// Args:
//   output:  [M, K] fp8 e4m3 (same layout as input)
//   scales:  [ceil(K/128), padded(M, 4)] float32 dequant scales (K-chunk-major)
//   input:   [M, K] bf16 row-major
//   dim_x:   K (number of columns)
//   dim_y:   M (number of rows/tokens)
// ============================================================================

__global__ void scale_1x128_kernel(
    __nv_fp8_e4m3* output,
    float* scales,
    const __nv_bfloat16* input,
    int dim_x,
    int dim_y)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
    size_t scales_along_dim_x = div_up(dim_x, 128);
    size_t scales_along_dim_y = (size_t)dim_y;
    size_t stride_scale_dim_y = div_up(dim_y, 4) * 4;

    for (size_t warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
         warp_idx < scales_along_dim_x * scales_along_dim_y;
         warp_idx += gridDim.x * blockDim.x / 32) {

        int scales_idx_y = warp_idx / scales_along_dim_x;  // row (token) index
        int scales_idx_x = warp_idx % scales_along_dim_x;  // K-chunk index

        const __nv_bfloat16* input_line =
            input + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;
        __nv_bfloat16 input_amax = __nv_bfloat16(0);
        int lane_id = threadIdx.x % 32 * 2;

        // Each thread reads 2 x bfloat162 = 4 bf16 elements covering 128 total
        __nv_bfloat162 input_frag2[2] = {
            __nv_bfloat162(0, 0), __nv_bfloat162(0, 0)
        };

#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
                break;
            } else {
                input_frag2[i] = *((__nv_bfloat162*)(input_line) + lane_id / 2);
            }
            input_line += 64;
        }

#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
                break;
            } else {
                input_amax = __nv_bfloat16(
                    __hmax(input_amax,
                           __hmax(__habs(input_frag2[i].x),
                                  __habs(input_frag2[i].y))));
            }
        }

        __nv_bfloat16 amax = find_max_elem_in_warp(input_amax);
        float scale = amax != __nv_bfloat16(0.f) ? 448.f / float(amax) : 1.f;

        // Store dequant scale (1/scale) in K-chunk-major layout
        if (lane_id == 0) {
            scales[(size_t)scales_idx_x * stride_scale_dim_y + scales_idx_y] =
                1.f / scale;
        }

        // Quantize and store fp8 output
        __nv_fp8_e4m3* output_line =
            output + (size_t)scales_idx_y * dim_x + scales_idx_x * 128;

#pragma unroll
        for (int i = 0; i < 2; i++) {
            if (scales_idx_x * 128 + i * 64 + lane_id >= dim_x) {
                break;
            } else {
                float value_1 = float(input_frag2[i].x) * scale;
                float value_2 = float(input_frag2[i].y) * scale;
                output_line[lane_id]     = __nv_fp8_e4m3(value_1);
                output_line[lane_id + 1] = __nv_fp8_e4m3(value_2);
            }
            output_line += 64;
        }
    }
#endif
}

// ============================================================================
// Public C API (called from Rust via FFI)
// ============================================================================

static int g_num_sms = -1;

static int get_num_sms() {
    if (g_num_sms < 0) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, device);
    }
    return g_num_sms;
}

extern "C" void fp8_quantize_1x128_cuda(
    const void* input,       // bf16 [M, K] row-major
    void* output,            // fp8 e4m3 [M, K] row-major
    float* scales,           // float32 [ceil(K/128), padded(M, 4)] K-chunk-major
    int m,                   // number of rows (tokens)
    int k,                   // number of columns (hidden dim)
    cudaStream_t stream)
{
    int num_sms = get_num_sms();
    scale_1x128_kernel<<<num_sms * 8, 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3*>(output),
        scales,
        reinterpret_cast<const __nv_bfloat16*>(input),
        k,   // dim_x = K (columns)
        m);  // dim_y = M (rows)
}
