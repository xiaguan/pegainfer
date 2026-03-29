use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub(crate) type Half = u16;

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

    pub(crate) fn silu_mul_triton_aot_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream);

    pub(crate) fn gpu_sample_cuda(
        logits: *const Half,
        probs_scratch: *mut f32,
        output: *mut i32,
        vocab_size: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        random_val: f32,
        stream: CUstream,
    );

    pub(crate) fn gemv_cuda(
        A: *const Half,
        x: *const Half,
        y: *mut Half,
        M: i32,
        K: i32,
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

    pub(crate) fn fused_mlp_cuda(
        x: *const Half,
        gate_proj: *const Half,
        up_proj: *const Half,
        down_proj: *const Half,
        act: *mut Half,
        out: *mut Half,
        hidden_size: i32,
        intermediate_size: i32,
        stream: CUstream,
    );

    // Embedding lookup reading token_id from decode_meta[0] (CUDA Graph safe)
    pub(crate) fn embedding_decode_cuda(
        embed: *const Half,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn cublas_init();

    // Prefill attention preparation: QK norm + RoPE + KV cache write.
    pub(crate) fn prefill_attention_prep_cuda(
        q_batch: *mut Half,
        k_batch: *mut Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos: i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // FlashAttention-2 prefill (Triton AOT) for HEAD_DIM=256.
    // Q/Output are col-major [q_dim, seq_len]. K/V cache are per-head [max_seq, HEAD_DIM].
    pub(crate) fn flash_attention_prefill_hd256_cuda(
        Q: *const Half,
        K_cache: *const Half,
        V_cache: *const Half,
        Output: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        gqa_ratio: i32,
        seq_len: i32,
        start_pos_ptr: *const i32,
        q_dim: i32,
        stream: CUstream,
    ) -> CUresult;

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

    // Fused add + (1+weight) RMSNorm
    pub(crate) fn fused_add_rms_norm_offset_cuda(
        hidden: *mut Half,
        residual: *const Half,
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

    // QK RMSNorm + RoPE for decode (seq_len=1, CUDA Graph safe).
    // Reads position from decode_meta[1] on device.
    pub(crate) fn qk_norm_rope_cuda(
        q: *mut Half,
        k: *mut Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        decode_meta: *const i32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Append one K/V token per request to paged KV cache (FlashInfer).
    pub(crate) fn paged_kv_append_cuda(
        kv_data: *const Half,
        k_offset_elems: i64,
        v_offset_elems: i64,
        page_indices: *const i32,
        page_indptr: *const i32,
        last_page_len_d: *const i32,
        key: *const Half,
        value: *const Half,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        batch_size: i32,
        stride_page: i64,
        stream: CUstream,
    ) -> i32;

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

    // Single-request prefill (FlashInfer SinglePrefill, contiguous HND KV, no RoPE).
    pub(crate) fn single_prefill_cuda(
        q: *const Half,
        output: *mut Half,
        k_cache: *const Half,
        v_cache: *const Half,
        num_qo_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        kv_len: i32,
        max_seq_len: i32,
        sm_scale: f32,
        stream: CUstream,
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
