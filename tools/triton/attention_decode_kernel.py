import triton
import triton.language as tl


@triton.jit
def fused_attention_decode_kernel(
    q_full_ptr,
    k_full_ptr,
    v_full_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    cos_cache_base_ptr,
    sin_cache_base_ptr,
    decode_meta_ptr,
    k_cache_ptr,
    v_cache_ptr,
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    num_qheads,
    num_kvheads,
    gqa_ratio,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Split-KV fused GQA decode attention.

    Grid: (num_qheads, NUM_KV_SPLITS, 1).
    Each block handles one Q head and one KV chunk.
    All splits load Q/K/V and compute QK-norm + RoPE (cheap: HEAD_DIM register ops).
    Only split 0 writes K/V to KV cache and handles the current token (Stage 2).
    All splits write partial (acc, m, l) to FP32 temp buffers.
    A separate reduce kernel merges partials into final bf16 output.

    Numerical safety for empty splits:
    - m_i initialised to -1e38 (finite) instead of -inf to avoid
      exp2(-inf - (-inf)) = NaN in the alpha computation.
    - p is guarded by pos_mask so masked positions contribute 0 rather than
      exp2(-inf - (-1e38)) = exp2(+inf) = +inf which would corrupt acc.
    """
    HALF: tl.constexpr = HEAD_DIM // 2
    scale = 1.0 / (HEAD_DIM ** 0.5)
    rms_eps = 1e-6

    q_head_idx = tl.program_id(0).to(tl.int32)
    split_id = tl.program_id(1).to(tl.int32)
    kv_head_idx = q_head_idx // gqa_ratio

    current_pos = tl.load(decode_meta_ptr + 1).to(tl.int32)

    offs_d = tl.arange(0, HALF)
    q_base = q_head_idx * HEAD_DIM
    kv_base = kv_head_idx * HEAD_DIM
    cache_base = kv_head_idx * 4096 * HEAD_DIM

    # ---- Load Q, apply RMSNorm + RoPE (every split) ----
    q_lo = tl.load(q_full_ptr + q_base + offs_d).to(tl.float32)
    q_hi = tl.load(q_full_ptr + q_base + HALF + offs_d).to(tl.float32)
    q_sq = tl.sum(q_lo * q_lo, axis=0) + tl.sum(q_hi * q_hi, axis=0)
    q_rms = tl.rsqrt(q_sq / HEAD_DIM + rms_eps)
    qw_lo = tl.load(q_norm_weight_ptr + offs_d).to(tl.float32)
    qw_hi = tl.load(q_norm_weight_ptr + HALF + offs_d).to(tl.float32)
    q_lo = q_lo * q_rms * qw_lo
    q_hi = q_hi * q_rms * qw_hi

    cos = tl.load(cos_cache_base_ptr + current_pos * HEAD_DIM + offs_d).to(tl.float32)
    sin = tl.load(sin_cache_base_ptr + current_pos * HEAD_DIM + offs_d).to(tl.float32)
    q_rot_lo = q_lo * cos - q_hi * sin
    q_rot_hi = q_lo * sin + q_hi * cos

    # ---- Load K, V + RMSNorm + RoPE (every split) ----
    # Loading K/V projection outputs is cheap (HEAD_DIM elements from fast cache).
    # Unconditional load avoids Triton SSA scope issues in the Stage-2 block below.
    k_lo = tl.load(k_full_ptr + kv_base + offs_d).to(tl.float32)
    k_hi = tl.load(k_full_ptr + kv_base + HALF + offs_d).to(tl.float32)
    k_sq = tl.sum(k_lo * k_lo, axis=0) + tl.sum(k_hi * k_hi, axis=0)
    k_rms = tl.rsqrt(k_sq / HEAD_DIM + rms_eps)
    kw_lo = tl.load(k_norm_weight_ptr + offs_d).to(tl.float32)
    kw_hi = tl.load(k_norm_weight_ptr + HALF + offs_d).to(tl.float32)
    k_lo = k_lo * k_rms * kw_lo
    k_hi = k_hi * k_rms * kw_hi

    v_lo = tl.load(v_full_ptr + kv_base + offs_d).to(tl.float32)
    v_hi = tl.load(v_full_ptr + kv_base + HALF + offs_d).to(tl.float32)

    k_rot_lo = k_lo * cos - k_hi * sin
    k_rot_hi = k_lo * sin + k_hi * cos

    # ---- Split 0 only: write current K/V to KV cache ----
    if split_id == 0:
        cur_off = cache_base + current_pos * HEAD_DIM
        tl.store(k_cache_ptr + cur_off + offs_d, k_rot_lo.to(tl.bfloat16))
        tl.store(k_cache_ptr + cur_off + HALF + offs_d, k_rot_hi.to(tl.bfloat16))
        tl.store(v_cache_ptr + cur_off + offs_d, v_lo.to(tl.bfloat16))
        tl.store(v_cache_ptr + cur_off + HALF + offs_d, v_hi.to(tl.bfloat16))

    # ---- Compute this split's KV range ----
    seq_len = current_pos
    tiles_total = tl.cdiv(seq_len, BLOCK_N)
    tiles_per_split = tl.cdiv(tiles_total, NUM_KV_SPLITS)
    split_start = split_id * tiles_per_split * BLOCK_N
    split_end = tl.minimum((split_id + 1) * tiles_per_split * BLOCK_N, seq_len)

    # ---- Online softmax attention over this split's KV chunk ----
    acc_lo = tl.zeros([HALF], dtype=tl.float32)
    acc_hi = tl.zeros([HALF], dtype=tl.float32)
    # Use -1e38 (large finite negative) instead of -inf.
    # Reason: when a split is entirely empty, block_max = -inf from the masked qk
    # values.  With m_i = -inf, alpha = exp2(-inf - (-inf)) = exp2(NaN) = NaN which
    # corrupts the accumulator.  With m_i = -1e38, alpha = exp2(0) = 1 (safe no-op).
    m_i = tl.full([1], -1e38, dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    qk_scale = scale * 1.44269504  # scale * log2(e) for exp2 trick
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in tl.range(0, tiles_per_split * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        abs_pos = split_start + start_n + offs_n
        pos_mask = abs_pos < split_end

        cache_offs = cache_base + abs_pos[:, None] * HEAD_DIM + offs_d[None, :]
        kb_lo = tl.load(k_cache_ptr + cache_offs, mask=pos_mask[:, None], other=0.0).to(tl.float32)
        kb_hi = tl.load(
            k_cache_ptr + cache_offs + HALF, mask=pos_mask[:, None], other=0.0
        ).to(tl.float32)

        qk = (
            tl.sum(kb_lo * q_rot_lo[None, :], axis=1)
            + tl.sum(kb_hi * q_rot_hi[None, :], axis=1)
        ) * qk_scale
        qk = tl.where(pos_mask, qk, float("-inf"))

        block_max = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - m_new)
        # Guard p: masked positions have qk=-inf; with m_i=-1e38 that gives
        # exp2(-inf - (-1e38)) = exp2(+inf) = +inf.  Force them to 0 via pos_mask.
        p = tl.where(pos_mask, tl.math.exp2(qk - m_new), 0.0)
        l_block = tl.sum(p, axis=0)

        acc_lo = acc_lo * alpha
        acc_hi = acc_hi * alpha

        vb_lo = tl.load(v_cache_ptr + cache_offs, mask=pos_mask[:, None], other=0.0).to(tl.float32)
        vb_hi = tl.load(
            v_cache_ptr + cache_offs + HALF, mask=pos_mask[:, None], other=0.0
        ).to(tl.float32)
        acc_lo += tl.sum(p[:, None] * vb_lo, axis=0)
        acc_hi += tl.sum(p[:, None] * vb_hi, axis=0)

        l_i = l_i * alpha + l_block
        m_i = m_new

    # ---- Split 0: handle current token (Stage 2) from registers ----
    # k_rot_lo/hi and v_lo/hi are defined unconditionally above.
    if split_id == 0:
        qk_cur = (
            tl.sum(k_rot_lo * q_rot_lo, axis=0) + tl.sum(k_rot_hi * q_rot_hi, axis=0)
        ) * qk_scale
        m_new = tl.maximum(m_i, qk_cur)
        alpha = tl.math.exp2(m_i - m_new)
        p_cur = tl.math.exp2(qk_cur - m_new)

        acc_lo = acc_lo * alpha + v_lo * p_cur
        acc_hi = acc_hi * alpha + v_hi * p_cur
        l_i = l_i * alpha + p_cur
        m_i = m_new

    # ---- Write partial results (FP32, unnormalized) ----
    partial_base = (q_head_idx * NUM_KV_SPLITS + split_id) * HEAD_DIM
    tl.store(partial_out_ptr + partial_base + offs_d, acc_lo)
    tl.store(partial_out_ptr + partial_base + HALF + offs_d, acc_hi)
    scalar_base = q_head_idx * NUM_KV_SPLITS + split_id
    # tl.sum squeezes [1] tensor -> scalar for scalar-pointer stores
    tl.store(partial_m_ptr + scalar_base, tl.sum(m_i))
    tl.store(partial_l_ptr + scalar_base, tl.sum(l_i))
