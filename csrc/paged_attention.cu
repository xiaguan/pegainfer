// Thin C wrappers around FlashInfer's paged KV attention decode and append.
//
// We include FlashInfer headers (header-only C++) and instantiate only the
// template variants needed: bf16 Q/KV/O, HEAD_DIM=128, NHD layout, no RoPE.
//
// FlashInfer's BatchDecode dispatcher internally instantiates multiple GQA
// group sizes (1,2,3,4,8) — this covers both Qwen3-4B (GQA=4) and
// Qwen3.5-4B (GQA=8).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>

using namespace flashinfer;

using DType  = __nv_bfloat16;
using IdType = int32_t;
using ParamsT = BatchDecodeParams<DType, DType, DType, IdType>;
using Variant = DefaultAttention</*custom_mask=*/false,
                                 /*sliding_window=*/false,
                                 /*logits_soft_cap=*/false,
                                 /*alibi=*/false>;

// Helper: build paged_kv_t from our page-first layout.
//
// Our KvPool stores all layers interleaved in one buffer. For a given layer L:
//   k_data = base + L * layer_stride
//   v_data = base + L * layer_stride + kv_block_len
//   stride_page = page_stride  (spans all layers — jumps to same layer in next page)
//   NHD within-block: stride_n = num_kv_heads * head_dim, stride_h = head_dim
static paged_kv_t<DType, IdType> make_paged_kv(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int64_t  stride_page)
{
    DType* k_data = reinterpret_cast<DType*>(kv_data) + k_offset_elems;
    DType* v_data = reinterpret_cast<DType*>(kv_data) + v_offset_elems;

    // kv_strides[0] = stride_page, [1] = stride for NHD-n, [2] = stride for NHD-h
    int64_t kv_strides[3] = {
        stride_page,
        static_cast<int64_t>(num_kv_heads) * head_dim,
        static_cast<int64_t>(head_dim),
    };

    return paged_kv_t<DType, IdType>(
        num_kv_heads, page_size, head_dim, batch_size,
        QKVLayout::kNHD,
        k_data, v_data, kv_strides,
        page_indices, page_indptr, last_page_len_d,
        /*rope_pos_offset=*/nullptr);
}

extern "C" {

// ---------------------------------------------------------------------------
// Paged attention decode — wraps FlashInfer BatchDecodeWithPagedKVCache.
//
// Reads Q from `q`, reads K/V from the paged cache, writes output to `output`.
// No RoPE applied inside (caller does RoPE beforehand).
// No partition-KV (single block per batch×head) for Phase 1 simplicity.
// ---------------------------------------------------------------------------
int paged_attention_decode_cuda(
    // Q and output
    void*    q,                    // [num_qo_heads * head_dim] bf16, device
    void*    output,               // [num_qo_heads * head_dim] bf16, device
    // KV pool buffer (entire pool)
    void*    kv_data,
    int64_t  k_offset_elems,       // element offset: base → layer's K in page 0
    int64_t  v_offset_elems,       // element offset: base → layer's V in page 0
    // Paged KV metadata (GPU arrays)
    int32_t* page_indices,         // [num_pages_this_request]
    int32_t* page_indptr,          // [batch_size + 1]
    int32_t* last_page_len_d,      // [batch_size]
    // Plan metadata (GPU arrays — trivial for non-partition bs=1)
    int32_t* request_indices,      // [padded_batch_size], e.g. [0]
    int32_t* kv_tile_indices,      // [padded_batch_size], e.g. [0]
    int32_t* kv_chunk_size_ptr,    // GPU ptr → 1 int32 (kv_len)
    // Dimensions
    int32_t  num_qo_heads,
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,           // 1 for Phase 1
    int64_t  stride_page,          // KvLayout.page_stride
    float    sm_scale,             // typically 1/sqrt(head_dim)
    // Stream
    void*    stream)
{
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    ParamsT params(
        reinterpret_cast<DType*>(q),
        /*q_rope_offset=*/nullptr,
        paged_kv,
        reinterpret_cast<DType*>(output),
        /*lse=*/nullptr,
        /*maybe_alibi_slopes=*/nullptr,
        num_qo_heads,
        /*q_stride_n=*/num_qo_heads * head_dim,
        /*q_stride_h=*/head_dim,
        /*window_left=*/-1,
        /*logits_soft_cap=*/0.0f,
        sm_scale,
        /*rope_scale=*/1.0f,
        /*rope_theta=*/1e6f);

    params.padded_batch_size = batch_size;
    params.request_indices   = request_indices;
    params.kv_tile_indices   = kv_tile_indices;
    params.o_indptr          = nullptr;
    params.kv_chunk_size_ptr = kv_chunk_size_ptr;
    params.block_valid_mask  = nullptr;
    params.partition_kv      = false;

    // tmp_v = nullptr → non-partition path (no merge step)
    return static_cast<int>(
        BatchDecodeWithPagedKVCacheDispatched<
            /*HEAD_DIM=*/128,
            PosEncodingMode::kNone,
            Variant,
            ParamsT>(
            params,
            /*tmp_v=*/nullptr,
            /*tmp_s=*/nullptr,
            /*enable_pdl=*/false,
            reinterpret_cast<cudaStream_t>(stream)));
}

// ---------------------------------------------------------------------------
// Paged KV append — writes one K and one V token per request to paged cache.
//
// Must be called AFTER RMSNorm + RoPE on K, and BEFORE the attention decode.
// V is appended as-is (no norm/RoPE).
// ---------------------------------------------------------------------------
int paged_kv_append_cuda(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    void*    key,                  // [batch_size * num_kv_heads * head_dim] bf16
    void*    value,                // [batch_size * num_kv_heads * head_dim] bf16
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int32_t  batch_size,
    int64_t  stride_page,
    void*    stream)
{
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, batch_size, stride_page);

    return static_cast<int>(AppendPagedKVCacheDecode(
        paged_kv,
        reinterpret_cast<DType*>(key),
        reinterpret_cast<DType*>(value),
        reinterpret_cast<cudaStream_t>(stream)));
}

// ---------------------------------------------------------------------------
// Scatter contiguous KV cache into paged layout (one layer at a time).
//
// Source layout (HND per layer): k[head, pos, dim]
//   stride_n = head_dim, stride_h = max_seq_len * head_dim
//
// Called once after prefill to bridge contiguous → paged.
// ---------------------------------------------------------------------------
int paged_kv_scatter_cuda(
    void*    kv_data,
    int64_t  k_offset_elems,
    int64_t  v_offset_elems,
    int32_t* page_indices,
    int32_t* page_indptr,
    int32_t* last_page_len_d,
    void*    src_k,                // contiguous K for this layer [num_kv_heads, max_seq, head_dim]
    void*    src_v,                // contiguous V for this layer [num_kv_heads, max_seq, head_dim]
    int32_t* batch_indices,        // [nnz] = [0, 0, ..., 0]
    int32_t* positions,            // [nnz] = [0, 1, 2, ..., seq_len-1]
    int32_t  nnz,                  // = seq_len
    int32_t  num_kv_heads,
    int32_t  head_dim,
    int32_t  page_size,
    int64_t  stride_page,
    int64_t  src_stride_n,         // = head_dim
    int64_t  src_stride_h,         // = max_seq_len * head_dim
    void*    stream)
{
    auto paged_kv = make_paged_kv(
        kv_data, k_offset_elems, v_offset_elems,
        page_indices, page_indptr, last_page_len_d,
        num_kv_heads, head_dim, page_size, /*batch_size=*/1, stride_page);

    return static_cast<int>(AppendPagedKVCache(
        paged_kv,
        reinterpret_cast<DType*>(src_k),
        reinterpret_cast<DType*>(src_v),
        batch_indices,
        positions,
        static_cast<uint32_t>(nnz),
        static_cast<size_t>(src_stride_n),
        static_cast<size_t>(src_stride_h),
        static_cast<size_t>(src_stride_n),   // V has same layout as K
        static_cast<size_t>(src_stride_h),
        reinterpret_cast<cudaStream_t>(stream)));
}

} // extern "C"
