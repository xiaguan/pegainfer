// Thin C wrappers around DeepEP intranode kernel API.
// Only intranode normal kernels — NVSHMEM disabled, kNumRanks=8 for EP8.

#include "api.cuh"

using namespace deep_ep;

// ============================================================================
// Layout: compute dispatch routing metadata
// ============================================================================

extern "C" void deep_ep_get_dispatch_layout(
    const int64_t* topk_idx,    // [num_tokens * num_topk]
    int* num_tokens_per_rank,   // [num_ranks] output
    int* num_tokens_per_expert, // [num_experts] output
    bool* is_token_in_rank,     // [num_tokens * num_ranks] output
    int num_tokens,
    int num_topk,
    int num_ranks,
    int num_experts,
    cudaStream_t stream)
{
    layout::get_dispatch_layout(
        reinterpret_cast<const topk_idx_t*>(topk_idx),
        num_tokens_per_rank,
        nullptr,  // num_tokens_per_rdma_rank — not needed for intranode
        num_tokens_per_expert,
        is_token_in_rank,
        num_tokens,
        num_topk,
        num_ranks,
        num_experts,
        stream);
}

// ============================================================================
// Intranode barrier
// ============================================================================

extern "C" void deep_ep_intranode_barrier(
    int** barrier_signal_ptrs_gpu,
    int rank,
    int num_ranks,
    cudaStream_t stream)
{
    intranode::barrier(barrier_signal_ptrs_gpu, rank, num_ranks, stream);
}

// ============================================================================
// Intranode notify_dispatch: exchange token counts via NVLink IPC
// Returns num_recv_tokens and per-expert counts via mapped host memory.
// ============================================================================

extern "C" void deep_ep_notify_dispatch(
    const int* num_tokens_per_rank,
    int* moe_recv_counter_mapped,
    int num_ranks,
    const int* num_tokens_per_expert,
    int* moe_recv_expert_counter_mapped,
    int num_experts,
    int num_tokens,
    const bool* is_token_in_rank,
    int* channel_prefix_matrix,
    int* rank_prefix_matrix_copy,
    int num_memset_int,
    int expert_alignment,
    void** buffer_ptrs_gpu,
    int** barrier_signal_ptrs_gpu,
    int rank,
    cudaStream_t stream,
    int num_sms)
{
    intranode::notify_dispatch(
        num_tokens_per_rank,
        moe_recv_counter_mapped,
        num_ranks,
        num_tokens_per_expert,
        moe_recv_expert_counter_mapped,
        num_experts,
        num_tokens,
        is_token_in_rank,
        channel_prefix_matrix,
        rank_prefix_matrix_copy,
        num_memset_int,
        expert_alignment,
        buffer_ptrs_gpu,
        barrier_signal_ptrs_gpu,
        rank,
        stream,
        num_sms);
}

// ============================================================================
// Intranode dispatch: send tokens to target ranks via NVLink
// ============================================================================

extern "C" void deep_ep_intranode_dispatch(
    void* recv_x,
    float* recv_x_scales,
    int* recv_src_idx,
    int64_t* recv_topk_idx,
    float* recv_topk_weights,
    int* recv_channel_offset,
    int* send_head,
    const void* x,
    const float* x_scales,
    const int64_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    const int* channel_prefix_matrix,
    int num_tokens,
    int num_worst_tokens,
    int hidden_int4,
    int num_topk,
    int num_experts,
    int num_scales,
    int scale_token_stride,
    int scale_hidden_stride,
    void** buffer_ptrs_gpu,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens)
{
    intranode::dispatch(
        recv_x,
        recv_x_scales,
        recv_src_idx,
        reinterpret_cast<topk_idx_t*>(recv_topk_idx),
        recv_topk_weights,
        recv_channel_offset,
        send_head,
        x,
        x_scales,
        reinterpret_cast<const topk_idx_t*>(topk_idx),
        topk_weights,
        is_token_in_rank,
        channel_prefix_matrix,
        num_tokens,
        num_worst_tokens,
        hidden_int4,
        num_topk,
        num_experts,
        num_scales,
        scale_token_stride,
        scale_hidden_stride,
        buffer_ptrs_gpu,
        rank,
        num_ranks,
        stream,
        num_sms,
        num_max_send_tokens,
        num_recv_buffer_tokens);
}

// ============================================================================
// Intranode cached_notify_combine: barrier + zero buffer + preprocess send_head
// Must be called between dispatch and combine to reset the NVL buffer layout.
// ============================================================================

extern "C" void deep_ep_cached_notify_combine(
    void** buffer_ptrs_gpu,
    int* send_head,
    int num_channels,
    int num_recv_tokens,
    int num_memset_int,
    int** barrier_signal_ptrs_gpu,
    int rank,
    int num_ranks,
    cudaStream_t stream)
{
    intranode::cached_notify_combine(
        buffer_ptrs_gpu,
        send_head,
        num_channels,
        num_recv_tokens,
        num_memset_int,
        barrier_signal_ptrs_gpu,
        rank,
        num_ranks,
        stream);
}

// ============================================================================
// Intranode combine: gather expert outputs back to source ranks
// ============================================================================

extern "C" void deep_ep_intranode_combine(
    void* combined_x,
    float* combined_topk_weights,
    const void* x,
    const float* topk_weights,
    const int* src_idx,
    const int* rank_prefix_matrix,
    const int* channel_prefix_matrix,
    int* send_head,
    int num_tokens,
    int num_recv_tokens,
    int hidden,
    int num_topk,
    void** buffer_ptrs_gpu,
    int rank,
    int num_ranks,
    cudaStream_t stream,
    int num_sms,
    int num_max_send_tokens,
    int num_recv_buffer_tokens)
{
    intranode::combine(
        CUDA_R_16BF,     // bf16 data type
        combined_x,
        combined_topk_weights,
        x,
        topk_weights,
        nullptr,          // bias_0 — not used for DSV3.2
        nullptr,          // bias_1 — not used for DSV3.2
        src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        send_head,
        num_tokens,
        num_recv_tokens,
        hidden,
        num_topk,
        buffer_ptrs_gpu,
        rank,
        num_ranks,
        stream,
        num_sms,
        num_max_send_tokens,
        num_recv_buffer_tokens);
}
