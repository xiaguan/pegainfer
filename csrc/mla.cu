// csrc/mla.cu
//
// MLA-specific CUDA kernels for DeepSeek-V3:
//   - YaRN RoPE for MLA rope dims (64d)
//   - KV cache write to paged buffer
//   - Q rope copy from q_full interleaved layout to FlashMLA Q buffer

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// MLA RoPE kernel — applies RoPE to k_rope (contiguous 64d) in-place.
//
// kv_a: [kv_a_proj_dim, bs] bf16, token t at offset t * kv_a_proj_dim.
//       k_rope is at [kv_lora_rank..kv_lora_rank+rope_dim] per token.
//
// cos_cache, sin_cache: [max_seq_len, rope_dim] fp32, precomputed.
//   Only the first half_rope entries per row are used (cos[pos, 2i] unused).
// positions: [bs] i32, per-token positions.
//
// **Interleaved (GPT-J) pairing — matches HF DeepSeek-V3 `rope_interleave=True`
// which is the default for DSv3 / DSv3.2**:
//   y[2i]   = x[2i]   * cos[i] - x[2i+1] * sin[i]
//   y[2i+1] = x[2i+1] * cos[i] + x[2i]   * sin[i]
// ============================================================================

__global__ void mla_rope_kv_kernel(
    __nv_bfloat16 *kv_a,           // [kv_a_proj_dim, bs]
    const float *cos_cache,         // [max_seq_len, rope_dim] fp32
    const float *sin_cache,
    const int *positions,           // [bs]
    int kv_a_proj_dim,              // 576
    int kv_lora_rank,               // 512 (offset to k_rope)
    int rope_dim,                   // 64
    int bs)
{
    int half_rope = rope_dim / 2;
    int total = half_rope * bs;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        int i = idx % half_rope;  // pair index
        int token = idx / half_rope;

        int base = token * kv_a_proj_dim + kv_lora_rank;
        int pos = positions[token];

        float x_lo = __bfloat162float(kv_a[base + 2 * i]);
        float x_hi = __bfloat162float(kv_a[base + 2 * i + 1]);

        float c = cos_cache[pos * rope_dim + i];
        float s = sin_cache[pos * rope_dim + i];

        kv_a[base + 2 * i]     = __float2bfloat16(x_lo * c - x_hi * s);
        kv_a[base + 2 * i + 1] = __float2bfloat16(x_hi * c + x_lo * s);
    }
}

// ============================================================================
// MLA Q rope + copy kernel.
//
// Reads q_rope from q_full interleaved layout, applies rotate_half RoPE,
// and writes to FlashMLA Q buffer at the rope slot.
//
// q_full: [q_b_proj_dim, bs] = [24576, bs]
//         Per head h: q_nope at h*q_head_dim, q_rope at h*q_head_dim+nope_dim.
//         q_head_dim = nope_dim + rope_dim = 192.
//
// q_mla: [bs, num_heads, kv_a_proj_dim] FlashMLA Q buffer.
//        Per token t, head h: absorbed at t*num_heads*kv_a_proj_dim + h*kv_a_proj_dim,
//                              rope at  ... + kv_lora_rank.
// ============================================================================

__global__ void mla_rope_q_copy_kernel(
    const __nv_bfloat16 *q_full,    // [q_b_proj_dim, bs]
    __nv_bfloat16 *q_mla,           // [bs, num_heads, kv_a_proj_dim]
    const float *cos_cache,          // [max_seq_len, rope_dim] fp32
    const float *sin_cache,
    const int *positions,
    int q_b_proj_dim,                // 24576
    int q_head_dim,                  // 192
    int nope_dim,                    // 128
    int rope_dim,                    // 64
    int num_heads,                   // 128
    int kv_a_proj_dim,               // 576
    int kv_lora_rank,                // 512
    int bs)
{
    int half_rope = rope_dim / 2;
    int total = half_rope * num_heads * bs;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        int i = idx % half_rope;  // pair index
        int rem = idx / half_rope;
        int head = rem % num_heads;
        int token = rem / num_heads;

        // Read from q_full — q_rope for head h at [h*q_head_dim + nope_dim..+rope_dim]
        int src_base = token * q_b_proj_dim + head * q_head_dim + nope_dim;
        float x_lo = __bfloat162float(q_full[src_base + 2 * i]);
        float x_hi = __bfloat162float(q_full[src_base + 2 * i + 1]);

        // RoPE (interleaved / GPT-J format, DSv3 default rope_interleave=True)
        int pos = positions[token];
        float c = cos_cache[pos * rope_dim + i];
        float s = sin_cache[pos * rope_dim + i];

        // Write to FlashMLA Q buffer rope slot
        int dst_base = token * num_heads * kv_a_proj_dim + head * kv_a_proj_dim + kv_lora_rank;
        q_mla[dst_base + 2 * i]     = __float2bfloat16(x_lo * c - x_hi * s);
        q_mla[dst_base + 2 * i + 1] = __float2bfloat16(x_hi * c + x_lo * s);
    }
}

// ============================================================================
// KV cache write — scatter [c_kv_normed, k_rope_rotated] to paged buffer.
//
// kv_a: [kv_a_proj_dim, bs] bf16 — already layernormed on c_kv, RoPE on k_rope.
//       Token t's 576d vector at offset t * kv_a_proj_dim.
//
// kv_buffer: the full per-layer paged buffer [num_pages, page_size, 1, kv_dim] bf16.
//            We write to kv_buffer + layer_offset.
//
// page_indices: [num_pages_for_this_seq] i32.
// start_pos: starting position in the sequence (existing seq_len before this append).
// num_tokens: number of tokens to write (usually bs for prefill, 1 for decode).
// ============================================================================

__global__ void mla_kv_cache_write_kernel(
    const __nv_bfloat16 *kv_a,       // [kv_a_proj_dim, bs]
    __nv_bfloat16 *kv_buffer,        // paged buffer for this layer
    const int *page_indices,          // page table for this sequence
    int kv_dim,                       // 576
    int page_size,                    // 64
    int start_pos,                    // seq position of first new token
    int num_tokens)                   // number of tokens to write
{
    // Each thread handles one (dim, token) pair.
    int total = kv_dim * num_tokens;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += gridDim.x * blockDim.x) {
        int d = idx % kv_dim;
        int t = idx / kv_dim;

        int seq_pos = start_pos + t;
        int page_idx = seq_pos / page_size;
        int slot = seq_pos % page_size;

        int page_id = page_indices[page_idx];
        // kv_buffer layout: [page_id, slot, 0, d]
        // = page_id * page_size * kv_dim + slot * kv_dim + d
        int dst = page_id * page_size * kv_dim + slot * kv_dim + d;

        kv_a[t * kv_dim + d]; // source
        kv_buffer[dst] = kv_a[t * kv_dim + d];
    }
}

// ============================================================================
// RMSNorm on a sub-range of hidden dims (for kv_a_layernorm on c_kv part only).
//
// Applies RMSNorm to the first `norm_dim` elements of each token's vector,
// leaving the remaining elements (k_rope) untouched.
//
// x: [total_dim, bs], norm applies to x[0..norm_dim] per token.
// ============================================================================

__global__ void rms_norm_partial_kernel(
    __nv_bfloat16 *x,                // [total_dim, bs] in-place
    const __nv_bfloat16 *weight,     // [norm_dim]
    int total_dim,                    // 576
    int norm_dim,                     // 512
    int bs,
    float eps)
{
    // One block per token.
    int token = blockIdx.x;
    if (token >= bs) return;

    __nv_bfloat16 *token_ptr = x + token * total_dim;

    // Compute sum of squares over norm_dim using all threads in block.
    extern __shared__ float smem[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < norm_dim; i += blockDim.x) {
        float v = __bfloat162float(token_ptr[i]);
        local_sum += v * v;
    }

    // Block-wide reduction
    smem[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = rsqrtf(smem[0] / (float)norm_dim + eps);

    // Apply normalization with weight to first norm_dim elements.
    for (int i = threadIdx.x; i < norm_dim; i += blockDim.x) {
        float v = __bfloat162float(token_ptr[i]);
        float w = __bfloat162float(weight[i]);
        token_ptr[i] = __float2bfloat16(v * rms * w);
    }
}

// ============================================================================
// Public C API
// ============================================================================

extern "C" {

void mla_rope_kv_cuda(
    void *kv_a,
    const void *cos_cache,
    const void *sin_cache,
    const int *positions,
    int kv_a_proj_dim,
    int kv_lora_rank,
    int rope_dim,
    int bs,
    cudaStream_t stream)
{
    int half_rope = rope_dim / 2;
    int total = half_rope * bs;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mla_rope_kv_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(kv_a),
        reinterpret_cast<const float *>(cos_cache),
        reinterpret_cast<const float *>(sin_cache),
        positions,
        kv_a_proj_dim, kv_lora_rank, rope_dim, bs);
}

void mla_rope_q_copy_cuda(
    const void *q_full,
    void *q_mla,
    const void *cos_cache,
    const void *sin_cache,
    const int *positions,
    int q_b_proj_dim,
    int q_head_dim,
    int nope_dim,
    int rope_dim,
    int num_heads,
    int kv_a_proj_dim,
    int kv_lora_rank,
    int bs,
    cudaStream_t stream)
{
    int half_rope = rope_dim / 2;
    int total = half_rope * num_heads * bs;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mla_rope_q_copy_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(q_full),
        reinterpret_cast<__nv_bfloat16 *>(q_mla),
        reinterpret_cast<const float *>(cos_cache),
        reinterpret_cast<const float *>(sin_cache),
        positions,
        q_b_proj_dim, q_head_dim, nope_dim, rope_dim,
        num_heads, kv_a_proj_dim, kv_lora_rank, bs);
}

void mla_kv_cache_write_cuda(
    const void *kv_a,
    void *kv_buffer,
    const int *page_indices,
    int kv_dim,
    int page_size,
    int start_pos,
    int num_tokens,
    cudaStream_t stream)
{
    int total = kv_dim * num_tokens;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mla_kv_cache_write_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(kv_a),
        reinterpret_cast<__nv_bfloat16 *>(kv_buffer),
        page_indices,
        kv_dim, page_size, start_pos, num_tokens);
}

void rms_norm_partial_cuda(
    void *x,
    const void *weight,
    int total_dim,
    int norm_dim,
    int bs,
    float eps,
    cudaStream_t stream)
{
    int threads = 256;
    int smem_size = threads * sizeof(float);
    rms_norm_partial_kernel<<<bs, threads, smem_size, stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(x),
        reinterpret_cast<const __nv_bfloat16 *>(weight),
        total_dim, norm_dim, bs, eps);
}

} // extern "C"
