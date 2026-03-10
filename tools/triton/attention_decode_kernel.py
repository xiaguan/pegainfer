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
    output_ptr,
    num_qheads,
    num_kvheads,
    gqa_ratio,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Fused GQA decode attention: QK-norm + RoPE + KV-cache write + online-softmax attention.

    Scale and rms_eps are computed from HEAD_DIM to avoid Triton AOT fp32 parameter
    passing issues (the AOT launcher uses double for fp32 params but cuLaunchKernel
    reads float-sized values, causing the lower bytes of the double to be misinterpreted).
    """
    HALF: tl.constexpr = HEAD_DIM // 2
    # Constants derived from HEAD_DIM — avoids passing fp32 through Triton AOT ABI
    scale = 1.0 / (HEAD_DIM ** 0.5)
    rms_eps = 1e-6

    pid = tl.program_id(0)
    q_head_idx = pid.to(tl.int32)
    kv_head_idx = q_head_idx // gqa_ratio

    current_pos = tl.load(decode_meta_ptr + 1).to(tl.int32)

    offs_d = tl.arange(0, HALF)
    q_base = q_head_idx * HEAD_DIM
    kv_base = kv_head_idx * HEAD_DIM
    cache_base = kv_head_idx * 4096 * HEAD_DIM

    # ---- Load Q, apply RMSNorm ----
    q_lo = tl.load(q_full_ptr + q_base + offs_d).to(tl.float32)
    q_hi = tl.load(q_full_ptr + q_base + HALF + offs_d).to(tl.float32)
    q_sq = tl.sum(q_lo * q_lo, axis=0) + tl.sum(q_hi * q_hi, axis=0)
    q_rms = tl.rsqrt(q_sq / HEAD_DIM + rms_eps)
    qw_lo = tl.load(q_norm_weight_ptr + offs_d).to(tl.float32)
    qw_hi = tl.load(q_norm_weight_ptr + HALF + offs_d).to(tl.float32)
    q_lo = q_lo * q_rms * qw_lo
    q_hi = q_hi * q_rms * qw_hi

    # ---- Load K, apply RMSNorm ----
    k_lo = tl.load(k_full_ptr + kv_base + offs_d).to(tl.float32)
    k_hi = tl.load(k_full_ptr + kv_base + HALF + offs_d).to(tl.float32)
    k_sq = tl.sum(k_lo * k_lo, axis=0) + tl.sum(k_hi * k_hi, axis=0)
    k_rms = tl.rsqrt(k_sq / HEAD_DIM + rms_eps)
    kw_lo = tl.load(k_norm_weight_ptr + offs_d).to(tl.float32)
    kw_hi = tl.load(k_norm_weight_ptr + HALF + offs_d).to(tl.float32)
    k_lo = k_lo * k_rms * kw_lo
    k_hi = k_hi * k_rms * kw_hi

    # ---- Load V (no norm) ----
    v_lo = tl.load(v_full_ptr + kv_base + offs_d).to(tl.float32)
    v_hi = tl.load(v_full_ptr + kv_base + HALF + offs_d).to(tl.float32)

    # ---- RoPE for Q and K ----
    cos = tl.load(cos_cache_base_ptr + current_pos * HEAD_DIM + offs_d).to(tl.float32)
    sin = tl.load(sin_cache_base_ptr + current_pos * HEAD_DIM + offs_d).to(tl.float32)
    q_rot_lo = q_lo * cos - q_hi * sin
    q_rot_hi = q_lo * sin + q_hi * cos
    k_rot_lo = k_lo * cos - k_hi * sin
    k_rot_hi = k_lo * sin + k_hi * cos

    # ---- Write K/V to cache at current_pos ----
    cur_off = cache_base + current_pos * HEAD_DIM
    tl.store(k_cache_ptr + cur_off + offs_d, k_rot_lo.to(tl.bfloat16))
    tl.store(k_cache_ptr + cur_off + HALF + offs_d, k_rot_hi.to(tl.bfloat16))
    tl.store(v_cache_ptr + cur_off + offs_d, v_lo.to(tl.bfloat16))
    tl.store(v_cache_ptr + cur_off + HALF + offs_d, v_hi.to(tl.bfloat16))

    # ---- Online softmax attention ----
    acc_lo = tl.zeros([HALF], dtype=tl.float32)
    acc_hi = tl.zeros([HALF], dtype=tl.float32)
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    qk_scale = scale * 1.44269504  # scale * log2(e) for exp2 trick
    offs_n = tl.arange(0, BLOCK_N)

    # Stage 1: past tokens from KV cache
    for start_n in tl.range(0, current_pos, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        pos = start_n + offs_n
        pos_mask = pos < current_pos

        cache_offs = cache_base + pos[:, None] * HEAD_DIM + offs_d[None, :]
        kb_lo = tl.load(k_cache_ptr + cache_offs, mask=pos_mask[:, None], other=0.0).to(tl.float32)
        kb_hi = tl.load(
            k_cache_ptr + cache_offs + HALF, mask=pos_mask[:, None], other=0.0
        ).to(tl.float32)

        # QK dot product per position: [BLOCK_N]
        qk = (
            tl.sum(kb_lo * q_rot_lo[None, :], axis=1)
            + tl.sum(kb_hi * q_rot_hi[None, :], axis=1)
        ) * qk_scale
        qk = tl.where(pos_mask, qk, float("-inf"))

        # Online softmax update
        block_max = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)  # [BLOCK_N]
        l_block = tl.sum(p, axis=0)

        # Rescale running accumulators
        acc_lo = acc_lo * alpha
        acc_hi = acc_hi * alpha

        # Weighted V accumulation (element-wise multiply + reduce, avoids tl.dot)
        vb_lo = tl.load(v_cache_ptr + cache_offs, mask=pos_mask[:, None], other=0.0).to(tl.float32)
        vb_hi = tl.load(
            v_cache_ptr + cache_offs + HALF, mask=pos_mask[:, None], other=0.0
        ).to(tl.float32)
        acc_lo += tl.sum(p[:, None] * vb_lo, axis=0)
        acc_hi += tl.sum(p[:, None] * vb_hi, axis=0)

        l_i = l_i * alpha + l_block
        m_i = m_new

    # Stage 2: current token from registers (no global memory read-back)
    qk_cur = (
        tl.sum(k_rot_lo * q_rot_lo, axis=0) + tl.sum(k_rot_hi * q_rot_hi, axis=0)
    ) * qk_scale
    m_new = tl.maximum(m_i, qk_cur)
    alpha = tl.math.exp2(m_i - m_new)
    p_cur = tl.math.exp2(qk_cur - m_new)

    acc_lo = acc_lo * alpha + v_lo * p_cur
    acc_hi = acc_hi * alpha + v_hi * p_cur
    l_i = l_i * alpha + p_cur

    # Normalize and store
    out_lo = acc_lo / l_i
    out_hi = acc_hi / l_i
    tl.store(output_ptr + q_base + offs_d, out_lo.to(tl.bfloat16))
    tl.store(output_ptr + q_base + HALF + offs_d, out_hi.to(tl.bfloat16))
