use cudarc::driver::sys::{CUresult, CUstream};

// Half type (16-bit float) - same layout as CUDA half
pub type Half = u16;

// CUDA kernels - all use half precision
unsafe extern "C" {
    pub fn rms_norm_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn rms_norm_batched_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        hidden_dim: i32,
        seq_len: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn rope_cuda(
        x: *const Half,
        cos: *const Half,
        sin: *const Half,
        out: *mut Half,
        head_dim: i32,
        stream: CUstream,
    );

    pub fn add_cuda(
        a: *const Half,
        b: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn fused_add_rms_norm_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    pub fn silu_mul_triton_aot_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_cuda(
        embed: *const Half,
        token_id: i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream);

    pub fn gpu_sample_cuda(
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

    pub fn attention_scores_cuda(
        q: *const Half,
        k_cache: *const Half,
        scores: *mut Half,
        seq_len: i32,
        head_dim: i32,
        scale: f32,
        stream: CUstream,
    );

    pub fn attention_weighted_sum_cuda(
        weights: *const Half,
        v_cache: *const Half,
        out: *mut Half,
        seq_len: i32,
        head_dim: i32,
        stream: CUstream,
    );

    pub fn gemv_cuda(
        A: *const Half,
        x: *const Half,
        y: *mut Half,
        M: i32,
        K: i32,
        stream: CUstream,
    );

    pub fn gemm_cuda(
        W: *const Half,
        X: *const Half,
        Y: *mut Half,
        M: i32,
        N: i32,
        K: i32,
        stream: CUstream,
    );

    pub fn gemv_batched_qkv_cuda(
        Wq: *const Half,
        Wk: *const Half,
        Wv: *const Half,
        x: *const Half,
        q_out: *mut Half,
        k_out: *mut Half,
        v_out: *mut Half,
        Mq: i32,
        Mk: i32,
        K: i32,
        stream: CUstream,
    );

    pub fn fused_mlp_cuda(
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
    pub fn embedding_decode_cuda(
        embed: *const Half,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn cublas_init();

    // Batched prefill attention (QK norm + RoPE + cuBLAS GEMM + causal softmax)
    pub fn prefill_attention_cuda(
        q_batch: *mut Half,
        k_batch: *mut Half,
        v_batch: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        scores_buf: *mut f32,
        softmax_buf: *mut Half,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        start_pos: i32,
        scale: f32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Fused GQA Attention — prefill variant (scalar pos/seq_len, per-position cos/sin)
    pub fn fused_gqa_attention_single_token(
        q_full: *const Half,
        k_full: *const Half,
        v_full: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        num_qheads: i32,
        num_kvheads: i32,
        gqa_ratio: i32,
        head_dim: i32,
        current_pos: i32,
        seq_len: i32,
        scale: f32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Fused GQA Attention — decode variant (reads pos/seq_len from decode_meta, base cos/sin)
    pub fn fused_gqa_attention_decode(
        q_full: *const Half,
        k_full: *const Half,
        v_full: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache_base: *const Half,
        sin_cache_base: *const Half,
        decode_meta: *const i32,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        num_qheads: i32,
        num_kvheads: i32,
        gqa_ratio: i32,
        head_dim: i32,
        scale: f32,
        rms_eps: f32,
        stream: CUstream,
    );

    // ========================================================================
    // Qwen3.5 kernels
    // ========================================================================

    // (1+weight) RMSNorm — Qwen3.5 / Gemma style
    pub fn rms_norm_offset_cuda(
        x: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    // Fused add + (1+weight) RMSNorm
    pub fn fused_add_rms_norm_offset_cuda(
        hidden: *mut Half,
        residual: *const Half,
        weight: *const Half,
        out: *mut Half,
        n: i32,
        eps: f32,
        stream: CUstream,
    );

    // Per-head RMSNorm with F32 weight + SiLU gate
    pub fn rms_norm_gated_cuda(
        x: *const Half,
        weight: *const f32,
        gate: *const Half,
        out: *mut Half,
        num_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: CUstream,
    );

    // Causal depthwise conv1d decode (single step)
    pub fn conv1d_decode_cuda(
        x: *const Half,
        conv_weight: *const Half,
        conv_state: *mut Half,
        out: *mut Half,
        num_channels: i32,
        kernel_size: i32,
        stream: CUstream,
    );

    // Causal depthwise conv1d prefill (parallel over sequence)
    pub fn conv1d_prefill_cuda(
        x_seq: *const Half,
        conv_weight: *const Half,
        conv_state: *mut Half,
        out_seq: *mut Half,
        num_channels: i32,
        seq_len: i32,
        kernel_size: i32,
        stream: CUstream,
    );

    // Gated delta rule recurrent decode (single step)
    pub fn gated_delta_rule_decode_cuda(
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

    // Fused GQA attention HD256 — decode variant (reads pos/seq_len from decode_meta)
    pub fn fused_gqa_attention_hd256_decode(
        q_full: *const Half,
        k_full: *const Half,
        v_full: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache_base: *const Half,
        sin_cache_base: *const Half,
        decode_meta: *const i32,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        num_qheads: i32,
        num_kvheads: i32,
        gqa_ratio: i32,
        rotary_dim: i32,
        scale: f32,
        rms_eps: f32,
        stream: CUstream,
    );

    // Fused GQA attention HD256 — single token variant (scalar pos/seq_len)
    pub fn fused_gqa_attention_hd256_single_token(
        q_full: *const Half,
        k_full: *const Half,
        v_full: *const Half,
        q_norm_weight: *const Half,
        k_norm_weight: *const Half,
        cos_cache: *const Half,
        sin_cache: *const Half,
        k_cache: *mut Half,
        v_cache: *mut Half,
        output: *mut Half,
        num_qheads: i32,
        num_kvheads: i32,
        gqa_ratio: i32,
        current_pos: i32,
        seq_len: i32,
        rotary_dim: i32,
        scale: f32,
        rms_eps: f32,
        stream: CUstream,
    );
}
