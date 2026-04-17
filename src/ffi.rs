use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub(crate) type Half = u16;

/// cudaIpcMemHandle_t — 64-byte opaque handle for IPC memory sharing.
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct CudaIpcMemHandle {
    pub reserved: [u8; 64],
}

/// cudaMemcpyKind constants
pub(crate) const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub(crate) const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

/// cudaHostAlloc flags
pub(crate) const CUDA_HOST_ALLOC_MAPPED: u32 = 0x02;

/// cudaIpcMemLazyEnablePeerAccess
pub(crate) const CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 0x01;

// CUDA kernels - all use half precision
unsafe extern "C" {
    pub(crate) fn rms_norm_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub(crate) fn rms_norm_batched_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    pub(crate) fn add_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn fused_add_rms_norm_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub(crate) fn fused_add_rms_norm_batched_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        batch_size: i32,
        eps: f32,
        stream: CUstream,
    );

    pub(crate) fn silu_mul_triton_aot_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const u32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream);

    pub(crate) fn flashinfer_top1_cuda(
        logits: *const Half,
        top1_value_scratch: *mut Half,
        row_states_scratch: *mut u8,
        output: *mut i32,
        vocab_size: i32,
        stream: CUstream,
    );

    pub(crate) fn gpu_sample_flashinfer_cuda(
        logits: *const Half,
        probs_scratch: *mut f32,
        valid_scratch: *mut u8,
        output: *mut i32,
        vocab_size: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        seed: u64,
        stream: CUstream,
    );

    pub(crate) fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    );

    pub(crate) fn gemm_graphsafe_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    );

    // Strided batched GEMM: C_i = op(A_i) @ op(B_i), bf16, fp32 accumulation.
    // Used for MLA Q absorption / V de-absorption.
    pub(crate) fn gemm_strided_batched_cuda(
        transa: i32, // 0 = N, 1 = T
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const Half,
        lda: i32,
        stride_a: i64,
        b: *const Half,
        ldb: i32,
        stride_b: i64,
        c: *mut Half,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
        stream: CUstream,
    );

    // Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
    pub(crate) fn embedding_decode_cuda(
        embed: *const Half,
        token_id: *const u32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn silu_mul_fused_cuda(
        gate_up: *const Half,
        out: *mut Half,
        intermediate_size: i32,
        bs: i32,
        stream: CUstream,
    );

    pub(crate) fn cublas_init();
    pub(crate) fn cublas_destroy();
    pub(crate) fn cuda_set_device(device_ordinal: i32) -> i32;

    // Prefill QK norm + RoPE only (no KV cache write). For paged prefill path.
    pub(crate) fn prefill_qk_norm_rope_only_cuda(
        q_batch: *mut Half,
        k_batch: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos: i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Qwen3.5 full-attention prefill prep: Q/K norm + partial RoPE + KV cache write.
    pub(crate) fn prefill_attention_hd256_prep_cuda(
        q_full_batch: *const Half,
        k_batch: *const Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        q_batch_out: *mut Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        rotary_dim: i32,
        rms_eps: f32,
        max_seq_len: i32,
        stream: CUstream,
    );

    // Apply sigmoid(gate) from interleaved q_full onto attention output in-place.
    pub(crate) fn attention_gate_batch_hd256_cuda(
        q_full_batch: *const Half,
        attn_out: *mut Half,
        num_q_heads: i32,
        seq_len: i32,
        stream: CUstream,
    );

    pub(crate) fn qk_norm_partial_rope_batched_decode_hd256_cuda(
        q_full_batch: *const Half,
        k_batch: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        q_batch_out: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        batch_size: i32,
        rotary_dim: i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // ========================================================================
    // Qwen3.5 kernels
    // ========================================================================

    // Batched (1+weight) RMSNorm — one block per token
    pub(crate) fn rms_norm_batched_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    // (1+weight) RMSNorm — Qwen3.5 / Gemma style
    pub(crate) fn rms_norm_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    // Per-head RMSNorm with F32 weight + SiLU gate
    pub(crate) fn rms_norm_gated_cuda(
        x: *const Half,
        weight: *const f32,
        gate: *const Half,
        out: *mut Half,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    );

    // Gated delta rule recurrent decode (single step)
    pub(crate) fn gated_delta_rule_decode_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        A_log: *const f32,
        state: *mut f32,
        output: *mut Half,
        num_key_heads: i32,
        num_value_heads: i32,
        key_dim: i32,
        val_dim: i32,
        stream: CUstream,
    );

    // Causal depthwise conv1d prefill (parallel over sequence)
    pub(crate) fn conv1d_prefill_cuda(
        x_seq: *const Half,
        conv_weight: *const Half,
        conv_state: *mut Half,
        out_seq: *mut Half,
        num_channels: i32,
        seq_len: i32,
        kernel_size: i32,
        stream: CUstream,
    );

    pub(crate) fn gated_delta_rule_prefill_chunk_prepare_cuda(
        qkv: *const Half,
        b_proj: *const Half,
        a_proj: *const Half,
        dt_bias: *const Half,
        a_log: *const f32,
        q_out: *mut Half,
        k_out: *mut Half,
        v_out: *mut Half,
        g_out: *mut f32,
        beta_out: *mut f32,
        num_key_heads: i32,
        num_value_heads: i32,
        qkv_dim: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_cumsum_cuda(
        g_in: *const f32,
        g_out: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_a_cuda(
        k: *const Half,
        g_cumsum: *const f32,
        beta: *const f32,
        a_tril: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_solve_cuda(
        a_tril: *const f32,
        a_inv: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gated_delta_rule_prefill_chunk_recompute_cuda(
        k: *const Half,
        v: *const Half,
        beta: *const f32,
        w: *mut Half,
        u: *mut Half,
        a_inv: *const Half,
        g_cumsum: *const f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // Chunk-wise GDR prefill stage 1 (Triton AOT): recurrent chunk-state update.
    // Expected future inputs:
    //   k / w: [seq_len, num_value_heads, 128] bf16
    //   u / v_new: [seq_len, num_value_heads, 128] bf16
    //   g_cumsum: [seq_len, num_value_heads] fp32
    //   initial_state / final_state: [num_value_heads, 128, 128] fp32 in [H, K, V] (V contiguous)
    //   chunk_state: [num_chunks, num_value_heads, 128, 128] fp32
    pub(crate) fn gated_delta_rule_prefill_chunk_state_cuda(
        k: *const Half,
        w: *const Half,
        u: *const Half,
        g_cumsum: *const f32,
        initial_state: *const f32,
        chunk_state: *mut f32,
        v_new: *mut Half,
        final_state: *mut f32,
        seq_len: i32,
        num_value_heads: i32,
        stream: CUstream,
    ) -> CUresult;

    // Chunk-wise GDR prefill stage 2 (Triton AOT): chunk output accumulation.
    // Expected future inputs:
    //   q / k / v_new: [seq_len, num_value_heads, 128] bf16
    //   chunk_state: [num_chunks, num_value_heads, 128, 128] fp32
    //   g_cumsum: [seq_len, num_value_heads] fp32
    //   output: [seq_len, num_value_heads * 128] bf16
    pub(crate) fn gated_delta_rule_prefill_chunk_o_cuda(
        q: *const Half,
        k: *const Half,
        v_new: *const Half,
        chunk_state: *const f32,
        g_cumsum: *const f32,
        output: *mut Half,
        seq_len: i32,
        num_value_heads: i32,
        scale: f32,
        stream: CUstream,
    ) -> CUresult;

    // ========================================================================
    // Paged attention (FlashInfer)
    // ========================================================================

    // Batched QK RMSNorm + RoPE for decode with per-request positions.
    pub(crate) fn qk_norm_rope_batched_decode_cuda(
        q: *mut Half,
        k: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        batch_size: i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Scatter contiguous KV → paged layout (one layer, FlashInfer prefill append).
    pub(crate) fn paged_kv_scatter_cuda(
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        src_k: *const Half,
        src_v: *const Half,
        batch_indices: *const i32,
        positions: *const i32,
        nnz: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        stride_page: i64,
        src_stride_n: i64,
        src_stride_h: i64,
        stream: CUstream,
    ) -> i32;

    // Return the number of Q tiles for batch prefill (needed to size plan arrays).
    pub(crate) fn batch_prefill_paged_num_tiles(
        seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    // Return the CTA tile size for batch prefill planning.
    pub(crate) fn batch_prefill_cta_tile_q(
        total_seq_len: i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    // Batch prefill with paged KV cache (FlashInfer BatchPrefill, causal, kNone).
    pub(crate) fn batch_prefill_paged_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;
    // Paged attention decode for HEAD_DIM=256 (Qwen3.5-4B full-attention layers).
    pub(crate) fn paged_attention_decode_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // Batch prefill with paged KV for HEAD_DIM=256 (Qwen3.5-4B multi-token prefill).
    pub(crate) fn batch_prefill_paged_cuda_hd256(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        q_indptr: *const i32,
        request_indices: *const i32,
        qo_tile_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        total_num_rows: *const u32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        seq_len: i32,
        batch_size: i32,
        padded_batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

    // ========================================================================
    // MLA kernels (csrc/mla.cu)
    // ========================================================================

    // RoPE on k_rope portion of kv_a (in-place).
    pub(crate) fn mla_rope_kv_cuda(
        kv_a: *mut Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        kv_a_proj_dim: i32,
        kv_lora_rank: i32,
        rope_dim: i32,
        bs: i32,
        stream: CUstream,
    );

    // Copy q_rope from q_full interleaved layout → FlashMLA Q buffer, with RoPE.
    pub(crate) fn mla_rope_q_copy_cuda(
        q_full: *const Half,
        q_mla: *mut Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        positions: *const i32,
        q_b_proj_dim: i32,
        q_head_dim: i32,
        nope_dim: i32,
        rope_dim: i32,
        num_heads: i32,
        kv_a_proj_dim: i32,
        kv_lora_rank: i32,
        bs: i32,
        stream: CUstream,
    );

    // Write kv_a [kv_dim, bs] to paged KV cache for one layer.
    pub(crate) fn mla_kv_cache_write_cuda(
        kv_a: *const Half,
        kv_buffer: *mut Half,
        page_indices: *const i32,
        kv_dim: i32,
        page_size: i32,
        start_pos: i32,
        num_tokens: i32,
        stream: CUstream,
    );

    // RMSNorm on first norm_dim elements of each token's vector (in-place).
    pub(crate) fn rms_norm_partial_cuda(
        x: *mut Half,
        weight: *const Half,
        total_dim: i32,
        norm_dim: i32,
        bs: i32,
        eps: f32,
        stream: CUstream,
    );

    // ========================================================================
    // DeepGEMM FP8 GEMM — 1D2D variant (SM90a)
    // ========================================================================

    // FP8 block-scale GEMM via DeepGEMM 1D2D: D[M,N] = A[M,K] @ B[N,K]^T
    // A, B: FP8 e4m3, row-major.
    // scale_a: FP32 dequant scales [ceil(K/128), padded(M, 4)] K-chunk-major (1D per-token).
    // scale_b: FP32 dequant scales [ceil(N/128), ceil(K/128)] K-major (2D per-block).
    // Output D is BF16 row-major.
    pub(crate) fn fp8_gemm_cuda(
        a: *const u8,
        scale_a: *const f32,
        b: *const u8,
        scale_b: *const f32,
        d: *mut Half,
        m: i32,
        n: i32,
        k: i32,
        stream: CUstream,
    );

    // ========================================================================
    // FP8 activation quantization (SM90a)
    // ========================================================================

    // bf16 [M, K] -> fp8 e4m3 [M, K] + dequant scales [ceil(K/128), padded(M, 4)]
    // Scale layout is K-chunk-major to match DeepGEMM TMA descriptor.
    // Extracted from TRT-LLM's scale_1x128_kernel.
    pub(crate) fn fp8_quantize_1x128_cuda(
        input: *const Half, // bf16 [M, K] (Half = bf16 in our codebase)
        output: *mut u8,    // fp8 e4m3 [M, K]
        scales: *mut f32,   // [ceil(K/128), padded(M, 4)]
        m: i32,
        k: i32,
        stream: CUstream,
    );

    // ========================================================================
    // FlashMLA dense decode (SM90a)
    // ========================================================================

    // Phase 1: Compute tile scheduler metadata for FlashMLA split-KV.
    pub(crate) fn flash_mla_get_metadata(
        batch_size: i32,
        seqlen_q: i32,
        seqlens_k: *const i32,             // [batch_size]
        tile_scheduler_metadata: *mut i32, // [num_sm_parts, 8]
        num_splits: *mut i32,              // [batch_size + 1]
        num_sm_parts: i32,
        stream: CUstream,
    );

    // Phase 2: Main MLA split-KV attention kernel (bf16).
    // q:      bf16 [batch, q_seq_per_hk, h_k, d_k]
    // kcache: bf16 [num_blocks, 64, h_k, d_k]  (one layer's paged KV)
    // o:      bf16 [batch, h_k, q_seq_per_hk, d_v]
    pub(crate) fn flash_mla_decode(
        q: *const Half,
        kcache: *const Half,
        o: *mut Half,
        lse: *mut f32,
        lse_accum: *mut f32,
        o_accum: *mut f32,
        block_table: *const i32, // [batch, max_blocks_per_seq]
        seqlens_k: *const i32,   // [batch]
        tile_scheduler_metadata: *const i32,
        num_splits: *const i32,
        batch_size: i32,
        seqlen_q: i32,
        q_seq_per_hk: i32,
        h_q: i32,
        h_k: i32,
        d_k: i32,
        d_v: i32,
        num_blocks: i32,
        max_blocks_per_seq: i32,
        num_sm_parts: i32,
        total_num_splits: i32,
        softmax_scale: f32,
        is_causal: i32,
        stream: CUstream,
    );

    // Phase 3: Combine split-KV partial results.
    pub(crate) fn flash_mla_combine(
        lse: *mut f32,
        out: *mut Half, // bf16
        lse_accum: *mut f32,
        o_accum: *mut f32,
        tile_scheduler_metadata: *const i32,
        num_splits: *const i32,
        batch_size: i32,
        seqlen_q: i32,
        h_q: i32,
        d_v: i32,
        num_sm_parts: i32,
        stream: CUstream,
    );

    // ========================================================================
    // NSA Indexer kernels (csrc/nsa_indexer.cu)
    // ========================================================================

    /// LayerNorm with bias (bf16). x: [dim, bs] col-major, out: [dim, bs].
    pub(crate) fn nsa_layernorm_bias_cuda(
        x: *const std::ffi::c_void,
        weight: *const std::ffi::c_void,
        bias: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        dim: i32,
        bs: i32,
        eps: f32,
        stream: CUstream,
    );

    /// Fused indexer score + causal topk (no [H,T,T] intermediate).
    /// q: [T, H, D] bf16, k: [T, D] bf16, weights: [T, H] bf16.
    /// indices: [T, topk] i32 output.
    /// weight_scale: D^{-0.5} * H^{-0.5}.
    pub(crate) fn nsa_indexer_fused_score_topk_cuda(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        weights: *const std::ffi::c_void,
        indices: *mut i32,
        t: i32,
        h: i32,
        d: i32,
        topk: i32,
        weight_scale: f32,
        stream: CUstream,
    );

    /// RoPE for indexer q/k. Applies to first rope_dim dims (indexer layout: rope+nope).
    /// q: [T, n_heads, head_dim] bf16 in-place.
    /// k: [T, head_dim] bf16 in-place.
    pub(crate) fn nsa_indexer_rope_cuda(
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        cos_cache: *const std::ffi::c_void,
        sin_cache: *const std::ffi::c_void,
        positions: *const i32,
        t: i32,
        n_heads: i32,
        head_dim: i32,
        rope_dim: i32,
        stream: CUstream,
    );

    // ========================================================================
    // FlashMLA sparse prefill — NSA (csrc/flash_mla_prefill.cu)
    // ========================================================================

    pub(crate) fn flash_mla_sparse_prefill(
        q: *const std::ffi::c_void,  // [s_q, h_q, d_qk] bf16
        kv: *const std::ffi::c_void, // [s_kv, h_kv, d_qk] bf16
        indices: *const i32,         // [s_q, h_kv, topk] i32
        out: *mut std::ffi::c_void,  // [s_q, h_q, d_v] bf16
        max_logits: *mut f32,        // [s_q, h_q] f32
        lse: *mut f32,               // [s_q, h_q] f32
        s_q: i32,
        s_kv: i32,
        h_q: i32,
        h_kv: i32,
        d_qk: i32,
        d_v: i32,
        topk: i32,
        sm_scale: f32,
        num_sm: i32,
        stream: CUstream,
    );

    // ========================================================================
    // MoE routing + helpers (csrc/moe.cu)
    // ========================================================================

    // DSV3.2 MoE routing: sigmoid + bias + group-limited TopK + normalize + scale.
    // logits:    bf16 [num_experts, bs] (gate_weight @ normed output)
    // bias:      f32  [num_experts] (e_score_correction_bias)
    // topk_idx:  i32  [bs * topk] output
    // topk_wt:   f32  [bs * topk] output
    pub(crate) fn moe_routing_cuda(
        logits: *const Half,
        bias: *const f32,
        topk_idx: *mut i32,
        topk_wt: *mut f32,
        num_experts: i32,
        bs: i32,
        topk: i32,
        n_group: i32,
        topk_group: i32,
        norm_topk_prob: i32,
        routed_scaling_factor: f32,
        stream: CUstream,
    );

    // Cast i32 → i64 (for DeepEP topk_idx compatibility).
    pub(crate) fn cast_i32_to_i64_cuda(
        in_data: *const i32,
        out_data: *mut i64,
        n: i32,
        stream: CUstream,
    );

    // Weighted add: out[i] += scale * x[i], bf16, element-wise.
    // Used to accumulate weighted expert outputs into hidden states.
    pub(crate) fn moe_weighted_add_cuda(
        out: *mut Half,
        x: *const Half,
        scale: f32,
        n: i32,
        stream: CUstream,
    );

    // ========================================================================
    // DeepEP intranode All-to-All (csrc/deep_ep.cu)
    // ========================================================================

    // Compute dispatch layout: which tokens go to which ranks/experts.
    pub(crate) fn deep_ep_get_dispatch_layout(
        topk_idx: *const i64,            // [num_tokens * num_topk]
        num_tokens_per_rank: *mut i32,   // [num_ranks] output
        num_tokens_per_expert: *mut i32, // [num_experts] output
        is_token_in_rank: *mut bool,     // [num_tokens * num_ranks] output
        num_tokens: i32,
        num_topk: i32,
        num_ranks: i32,
        num_experts: i32,
        stream: CUstream,
    );

    // Intranode NVLink barrier.
    pub(crate) fn deep_ep_intranode_barrier(
        barrier_signal_ptrs_gpu: *mut *mut i32,
        rank: i32,
        num_ranks: i32,
        stream: CUstream,
    );

    // Exchange token counts via NVLink IPC buffer.
    // Writes num_recv_tokens to moe_recv_counter_mapped (host-mapped memory).
    pub(crate) fn deep_ep_notify_dispatch(
        num_tokens_per_rank: *const i32,
        moe_recv_counter_mapped: *mut i32,
        num_ranks: i32,
        num_tokens_per_expert: *const i32,
        moe_recv_expert_counter_mapped: *mut i32,
        num_experts: i32,
        num_tokens: i32,
        is_token_in_rank: *const bool,
        channel_prefix_matrix: *mut i32,
        rank_prefix_matrix_copy: *mut i32,
        num_memset_int: i32,
        expert_alignment: i32,
        buffer_ptrs_gpu: *mut *mut std::ffi::c_void,
        barrier_signal_ptrs_gpu: *mut *mut i32,
        rank: i32,
        stream: CUstream,
        num_sms: i32,
    );

    // Dispatch tokens to target ranks via NVLink.
    pub(crate) fn deep_ep_intranode_dispatch(
        recv_x: *mut std::ffi::c_void,
        recv_x_scales: *mut f32,
        recv_src_idx: *mut i32,
        recv_topk_idx: *mut i64,
        recv_topk_weights: *mut f32,
        recv_channel_offset: *mut i32,
        send_head: *mut i32,
        x: *const std::ffi::c_void,
        x_scales: *const f32,
        topk_idx: *const i64,
        topk_weights: *const f32,
        is_token_in_rank: *const bool,
        channel_prefix_matrix: *const i32,
        num_tokens: i32,
        num_worst_tokens: i32,
        hidden_int4: i32,
        num_topk: i32,
        num_experts: i32,
        num_scales: i32,
        scale_token_stride: i32,
        scale_hidden_stride: i32,
        buffer_ptrs_gpu: *mut *mut std::ffi::c_void,
        rank: i32,
        num_ranks: i32,
        stream: CUstream,
        num_sms: i32,
        num_max_send_tokens: i32,
        num_recv_buffer_tokens: i32,
    );

    // Barrier + zero NVL buffer + preprocess send_head before combine.
    pub(crate) fn deep_ep_cached_notify_combine(
        buffer_ptrs_gpu: *mut *mut std::ffi::c_void,
        send_head: *mut i32,
        num_channels: i32,
        num_recv_tokens: i32,
        num_memset_int: i32,
        barrier_signal_ptrs_gpu: *mut *mut i32,
        rank: i32,
        num_ranks: i32,
        stream: CUstream,
    );

    // Combine expert outputs back to source ranks via NVLink.
    pub(crate) fn deep_ep_intranode_combine(
        combined_x: *mut std::ffi::c_void,
        combined_topk_weights: *mut f32,
        x: *const std::ffi::c_void,
        topk_weights: *const f32,
        src_idx: *const i32,
        rank_prefix_matrix: *const i32,
        channel_prefix_matrix: *const i32,
        send_head: *mut i32,
        num_tokens: i32,
        num_recv_tokens: i32,
        hidden: i32,
        num_topk: i32,
        buffer_ptrs_gpu: *mut *mut std::ffi::c_void,
        rank: i32,
        num_ranks: i32,
        stream: CUstream,
        num_sms: i32,
        num_max_send_tokens: i32,
        num_recv_buffer_tokens: i32,
    );

    // ========================================================================
    // CUDA runtime helpers for DeepEP buffer management
    // ========================================================================

    // cudaMalloc / cudaFree
    pub(crate) fn cudaMalloc(devptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    pub(crate) fn cudaFree(devptr: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaMemset(devptr: *mut std::ffi::c_void, value: i32, count: usize) -> i32;
    pub(crate) fn cudaMemsetAsync(
        devptr: *mut std::ffi::c_void,
        value: i32,
        count: usize,
        stream: CUstream,
    ) -> i32;

    // IPC memory handle exchange
    pub(crate) fn cudaIpcGetMemHandle(
        handle: *mut CudaIpcMemHandle,
        devptr: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn cudaIpcOpenMemHandle(
        devptr: *mut *mut std::ffi::c_void,
        handle: CudaIpcMemHandle,
        flags: u32,
    ) -> i32;
    pub(crate) fn cudaIpcCloseMemHandle(devptr: *mut std::ffi::c_void) -> i32;

    // Peer device access (intra-process cross-GPU memory access)
    pub(crate) fn cudaDeviceCanAccessPeer(
        can_access: *mut i32,
        device: i32,
        peer_device: i32,
    ) -> i32;
    pub(crate) fn cudaDeviceEnablePeerAccess(peer_device: i32, flags: u32) -> i32;

    // Host-mapped memory for CPU-GPU sync
    pub(crate) fn cudaHostAlloc(phost: *mut *mut std::ffi::c_void, size: usize, flags: u32) -> i32;
    pub(crate) fn cudaHostGetDevicePointer(
        pdevice: *mut *mut std::ffi::c_void,
        phost: *mut std::ffi::c_void,
        flags: u32,
    ) -> i32;
    pub(crate) fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;

    // Device memory copy
    pub(crate) fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;

    // Paged attention decode (FlashInfer BatchDecode, no partition-KV).
    pub(crate) fn paged_attention_decode_cuda(
        q: *const Half,
        output: *mut Half,
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        request_indices: *const i32,
        kv_tile_indices: *const i32,
        kv_chunk_size_ptr: *const i32,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        sm_scale: f32,
        stream: CUstream,
    ) -> i32;

}
