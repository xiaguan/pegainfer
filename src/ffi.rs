use cudarc::driver::sys::CUstream;

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

    pub fn add_cuda(a: *const Half, b: *const Half, out: *mut Half, n: i32, stream: CUstream);

    pub fn copy_cuda(src: *const Half, dst: *mut Half, n: i32, stream: CUstream);

    pub fn silu_mul_cuda(
        gate: *const Half,
        up: *const Half,
        out: *mut Half,
        n: i32,
        stream: CUstream,
    );

    pub fn embedding_cuda(
        embed: *const Half,
        token_id: i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    );

    pub fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    );

    pub fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream);

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

    pub fn gemv_cuda(A: *const Half, x: *const Half, y: *mut Half, M: i32, K: i32, stream: CUstream);

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
        out: *mut Half,
        hidden_size: i32,
        intermediate_size: i32,
        stream: CUstream,
    );

    pub fn cublas_init();

    // Fused GQA Attention (single token generation)
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
}
