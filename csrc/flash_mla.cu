// csrc/flash_mla.cu
//
// Torch-free C wrapper for FlashMLA SM90 dense decode attention.
// Fills FlashMLA param structs from raw pointers and calls the kernels.
//
// Three-phase pipeline (matching FlashMLA's Python API):
//   1. get_mla_metadata  — compute tile scheduler metadata
//   2. flash_mla_decode  — main split-KV MLA attention kernel
//   3. flash_mla_combine — combine split-KV partial results
//
// Kernel sources compiled from third_party/FlashMLA/csrc/:
//   sm90/decode/dense/instantiations/bf16.cu
//   smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu
//   smxx/decode/combine/combine.cu

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include "params.h"
#include "sm90/decode/dense/splitkv_mla.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

extern "C" {

// Phase 1: Compute tile scheduler metadata.
//
// tile_scheduler_metadata: int32 [num_sm_parts, 8]
// num_splits:              int32 [batch_size + 1]
void flash_mla_get_metadata(
    int batch_size,
    int seqlen_q,
    int* seqlens_k,                    // [batch_size] on device
    int* tile_scheduler_metadata,      // [num_sm_parts, 8] on device
    int* num_splits,                   // [batch_size + 1] on device
    int num_sm_parts,
    cudaStream_t stream)
{
    GetDecodeSchedMetaParams params = {};
    params.b = batch_size;
    params.s_q = seqlen_q;
    params.block_size_n = 64; // PAGE_BLOCK_SIZE
    params.fixed_overhead_num_blocks = 5;
    params.topk = -1;        // Dense attention
    params.extra_topk = -1;
    params.topk_length = nullptr;
    params.extra_topk_length = nullptr;
    params.seqlens_k_ptr = seqlens_k;
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata;
    params.num_splits_ptr = num_splits;
    params.num_sm_parts = num_sm_parts;
    params.stream = stream;

    smxx::decode::run_get_decoding_sched_meta_kernel(params);
}

// Phase 2: Main MLA split-KV attention kernel (bf16).
//
// q:       bf16 [batch, q_seq_per_hk, h_k, d_k]  (already reshaped by caller)
// kcache:  bf16 [num_blocks, page_size=64, h_k, d_k]  (one layer's paged KV)
// o:       bf16 [batch, h_k, q_seq_per_hk, d_v]  (output)
// lse:     f32  [batch, h_k, q_seq_per_hk]
// lse_accum: f32 [total_num_splits, h_k, q_seq_per_hk]
// o_accum:   f32 [total_num_splits, h_k, q_seq_per_hk, d_v]
void flash_mla_decode(
    void* q,
    void* kcache,
    void* o,
    float* lse,
    float* lse_accum,
    float* o_accum,
    int* block_table,                  // [batch, max_blocks_per_seq]
    int* seqlens_k,                    // [batch]
    int* tile_scheduler_metadata,
    int* num_splits,
    int batch_size,
    int seqlen_q,
    int q_seq_per_hk,
    int h_q,
    int h_k,
    int d_k,
    int d_v,
    int num_blocks,
    int max_blocks_per_seq,
    int num_sm_parts,
    int total_num_splits,
    float softmax_scale,
    int is_causal,
    cudaStream_t stream)
{
    DenseAttnDecodeParams params = {};
    params.b = batch_size;
    params.s_q = seqlen_q;
    params.q_seq_per_hk = q_seq_per_hk;
    params.h_q = h_q;
    params.h_k = h_k;
    params.num_blocks = num_blocks;
    params.q_head_per_hk = h_q / h_k;
    params.is_causal = (bool)is_causal;
    params.d = d_k;
    params.d_v = d_v;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * 1.44269504f; // * log2(e)

    params.q_ptr = q;
    params.k_ptr = kcache;
    params.o_ptr = o;
    params.softmax_lse_ptr = lse;

    // q: [batch, q_seq_per_hk, h_k, d_k] — contiguous row-major
    params.q_batch_stride = (int64_t)q_seq_per_hk * h_k * d_k;
    params.q_row_stride   = (int64_t)h_k * d_k;
    params.q_head_stride  = (int64_t)d_k;

    // kcache: [num_blocks, 64, h_k, d_k] — contiguous row-major
    params.k_batch_stride = 0; // not used for paged KV
    params.k_row_stride   = (int64_t)h_k * d_k;
    params.k_head_stride  = (int64_t)d_k;

    // o: [batch, h_k, q_seq_per_hk, d_v] — contiguous row-major
    params.o_batch_stride = (int64_t)h_k * q_seq_per_hk * d_v;
    params.o_row_stride   = (int64_t)d_v;
    params.o_head_stride  = (int64_t)q_seq_per_hk * d_v;

    params.block_table = block_table;
    params.block_table_batch_stride = max_blocks_per_seq;
    params.page_block_size = 64;
    params.seqlens_k_ptr = seqlens_k;

    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata;
    params.num_sm_parts = num_sm_parts;
    params.num_splits_ptr = num_splits;

    params.total_num_splits = total_num_splits;
    params.softmax_lseaccum_ptr = lse_accum;
    params.oaccum_ptr = o_accum;

    params.stream = stream;

    sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params);
}

// Phase 3: Combine split-KV partial results.
void flash_mla_combine(
    float* lse,           // [batch, seqlen_q, h_q]
    void*  out,           // [batch, seqlen_q, h_q, d_v]  bf16
    float* lse_accum,     // [total_splits, seqlen_q, h_q]
    float* o_accum,       // [total_splits, seqlen_q, h_q, d_v]
    int* tile_scheduler_metadata,
    int* num_splits,
    int batch_size,
    int seqlen_q,
    int h_q,
    int d_v,
    int num_sm_parts,
    cudaStream_t stream)
{
    CombineParams params = {};
    params.b = batch_size;
    params.s_q = seqlen_q;
    params.h_q = h_q;
    params.d_v = d_v;

    params.lse = lse;
    params.out = out;
    // lse: [batch, seqlen_q, h_q] — but combine kernel expects strides in
    // the reshaped layout [batch, h_k, q_seq_per_hk]
    // For our usage: h_k=1, q_seq_per_hk = seqlen_q * h_q
    // The combine kernel indexes as: batch*stride_lse_b + s_q*stride_lse_s_q + h_q_idx
    params.stride_lse_b = seqlen_q * h_q;
    params.stride_lse_s_q = h_q;

    // out: [batch, seqlen_q, h_q, d_v]
    params.stride_o_b = seqlen_q * h_q * d_v;
    params.stride_o_s_q = h_q * d_v;
    params.stride_o_h_q = d_v;

    params.lse_accum = lse_accum;
    params.o_accum = o_accum;
    // accum: [total_splits, h_k, q_seq_per_hk, ...]
    // With h_k=1: stride_split = q_seq_per_hk * h_q (but q_seq_per_hk already folds h_q)
    // Match the strides from dense_decode.h
    int q_seq_per_hk = seqlen_q * h_q; // h_q / h_k * s_q, with h_k=1
    params.stride_lse_accum_split = q_seq_per_hk;
    params.stride_lse_accum_s_q = h_q;

    params.stride_o_accum_split = q_seq_per_hk * d_v;
    params.stride_o_accum_s_q = h_q * d_v;
    params.stride_o_accum_h_q = d_v;

    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata;
    params.num_splits_ptr = num_splits;
    params.num_sm_parts = num_sm_parts;

    params.attn_sink = nullptr;
    params.stream = stream;

    smxx::decode::run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params);
}

} // extern "C"
