import triton
import triton.language as tl


@triton.jit
def attention_reduce_kernel(
    partial_out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    output_ptr,
    num_qheads,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Merge NUM_KV_SPLITS partial attention results per Q head.

    Grid: (num_qheads, 1, 1).
    Uses online softmax merge: for each split, rescale running accumulator
    by exp2(old_max - new_max) and add the new partial result rescaled
    by exp2(split_max - new_max).
    """
    HALF: tl.constexpr = HEAD_DIM // 2
    q_head_idx = tl.program_id(0).to(tl.int32)
    offs_d = tl.arange(0, HALF)

    acc_lo = tl.zeros([HALF], dtype=tl.float32)
    acc_hi = tl.zeros([HALF], dtype=tl.float32)
    m_global = tl.full([1], float("-inf"), dtype=tl.float32)
    l_global = tl.zeros([1], dtype=tl.float32)

    base = q_head_idx * NUM_KV_SPLITS
    for s in tl.static_range(0, NUM_KV_SPLITS):
        m_s = tl.load(partial_m_ptr + base + s)
        l_s = tl.load(partial_l_ptr + base + s)

        m_new = tl.maximum(m_global, m_s)
        alpha_old = tl.math.exp2(m_global - m_new)
        alpha_new = tl.math.exp2(m_s - m_new)

        p_lo = tl.load(partial_out_ptr + (base + s) * HEAD_DIM + offs_d)
        p_hi = tl.load(partial_out_ptr + (base + s) * HEAD_DIM + HALF + offs_d)

        acc_lo = acc_lo * alpha_old + p_lo * alpha_new
        acc_hi = acc_hi * alpha_old + p_hi * alpha_new
        l_global = l_global * alpha_old + l_s * alpha_new
        m_global = m_new

    out_lo = acc_lo / l_global
    out_hi = acc_hi / l_global
    out_base = q_head_idx * HEAD_DIM
    tl.store(output_ptr + out_base + offs_d, out_lo.to(tl.bfloat16))
    tl.store(output_ptr + out_base + HALF + offs_d, out_hi.to(tl.bfloat16))
