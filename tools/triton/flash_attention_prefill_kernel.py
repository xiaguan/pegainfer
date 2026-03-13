import triton
import triton.language as tl


@triton.jit
def flash_attention_prefill_kernel(
    Q_ptr,              # [num_q_heads * HEAD_DIM, seq_len] col-major (already normed+RoPE'd)
    K_cache_ptr,        # [num_kv_heads * max_seq * HEAD_DIM] row-major per head
    V_cache_ptr,        # [num_kv_heads * max_seq * HEAD_DIM] row-major per head
    Output_ptr,         # [num_q_heads * HEAD_DIM, seq_len] col-major
    num_q_heads,
    num_kv_heads,
    gqa_ratio,
    seq_len,            # number of query tokens in this prefill
    start_pos,          # starting position (for multi-turn)
    q_dim,              # num_q_heads * HEAD_DIM (stride between tokens in Q/Output)
    BLOCK_M: tl.constexpr,   # 128 — query tile size
    BLOCK_N: tl.constexpr,   # 64  — KV tile size
    HEAD_DIM: tl.constexpr,  # 128
):
    """FlashAttention-2 for prefill.

    Grid: (cdiv(seq_len, BLOCK_M), num_q_heads, 1)
    Each block computes BLOCK_M output rows for one Q head.

    Q layout: col-major [q_dim, seq_len] — stride between tokens = q_dim.
    K/V cache layout: per-head contiguous [max_seq, HEAD_DIM] with max_seq=4096.
    Output layout: col-major [q_dim, seq_len] — same stride as Q.

    Online softmax (FlashAttention-2 algorithm):
    - Iterate over KV tiles in BLOCK_N chunks
    - For each tile: compute QK^T, apply causal mask, online softmax update
    - Final rescale to produce correct attention output

    GQA note: GQA is handled by kv_head = q_head // gqa_ratio. This non-GQA-aware
    implementation loads the same K/V tiles independently for each Q head in a group.
    A GQA-aware implementation (outer KV loop, inner Q loop sharing K/V in registers)
    would reduce K/V HBM traffic 4× but requires holding 4× accumulators simultaneously.
    At gqa_ratio=4 with BLOCK_M=128, this exceeds register budget causing L1 spilling —
    net effect is slower. True GQA-aware FA would need explicit SMEM (CUDA, not Triton).
    """
    MAX_SEQ: tl.constexpr = 4096
    HALF_HD: tl.constexpr = HEAD_DIM // 2
    scale = 1.44269504 / tl.sqrt(float(HEAD_DIM))  # log2(e) / sqrt(HEAD_DIM)

    tile_m = tl.program_id(0)       # which BLOCK_M tile of queries
    q_head = tl.program_id(1)       # which Q head
    kv_head = q_head // gqa_ratio   # corresponding KV head (GQA)

    # Query row indices for this tile
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    # Which KV positions are valid: 0..total_seq where total_seq = start_pos + seq_len
    total_seq = start_pos + seq_len

    # Pointers into Q: Q is col-major [q_dim, seq_len].
    # Element (head, dim, token) is at: q_head*HEAD_DIM + dim + token * q_dim
    # We'll load Q[offs_m, :] — each query token's HEAD_DIM values.
    # For BLOCK_M queries × HEAD_DIM, we load HEAD_DIM/2 at a time.
    q_head_offset = q_head * HEAD_DIM
    offs_d = tl.arange(0, HALF_HD)  # [HALF_HD]

    # Load Q tile [BLOCK_M, HEAD_DIM] as two halves: lo and hi
    # Q element at (query_idx, d) = Q_ptr[q_head_offset + d + offs_m[i] * q_dim]
    q_ptrs_lo = Q_ptr + q_head_offset + offs_d[None, :] + offs_m[:, None] * q_dim  # [BLOCK_M, HALF_HD]
    q_ptrs_hi = Q_ptr + q_head_offset + HALF_HD + offs_d[None, :] + offs_m[:, None] * q_dim
    q_mask = offs_m[:, None] < seq_len  # [BLOCK_M, 1] broadcast

    q_lo = tl.load(q_ptrs_lo, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, HALF_HD]
    q_hi = tl.load(q_ptrs_hi, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, HALF_HD]

    # KV cache base for this KV head
    kv_cache_base = kv_head * MAX_SEQ * HEAD_DIM

    # Online softmax state: per query row
    m_i = tl.full([BLOCK_M], -1e38, dtype=tl.float32)  # max scores
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)          # sum of exp
    acc_lo = tl.zeros([BLOCK_M, HALF_HD], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, HALF_HD], dtype=tl.float32)

    # Causal bound per query: query i can attend to keys 0..start_pos+i (inclusive).
    # Maximum causal bound in this tile = start_pos + min(tile_m * BLOCK_M + BLOCK_M - 1, seq_len - 1)
    # We iterate KV blocks up to the max causal bound.
    max_query_in_tile = tl.minimum(tile_m * BLOCK_M + BLOCK_M, seq_len) - 1
    max_kv_pos = start_pos + max_query_in_tile  # highest KV position any query in this tile can see
    num_kv_blocks = tl.cdiv(max_kv_pos + 1, BLOCK_N)

    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]

    for kv_block in tl.range(0, num_kv_blocks):
        kv_start = kv_block * BLOCK_N
        kv_pos = kv_start + offs_n   # [BLOCK_N] — absolute KV positions

        # Load K tile [BLOCK_N, HEAD_DIM] from cache
        # K cache element at (pos, d) = K_cache_ptr[kv_cache_base + pos * HEAD_DIM + d]
        k_cache_offs = kv_cache_base + kv_pos[:, None] * HEAD_DIM + offs_d[None, :]  # [BLOCK_N, HALF_HD]
        kv_mask = kv_pos[:, None] < total_seq  # valid cache positions
        k_lo = tl.load(K_cache_ptr + k_cache_offs, mask=kv_mask, other=0.0).to(tl.float32)
        k_hi = tl.load(K_cache_ptr + k_cache_offs + HALF_HD, mask=kv_mask, other=0.0).to(tl.float32)

        # QK^T: [BLOCK_M, BLOCK_N] = Q[BLOCK_M, HEAD_DIM] @ K[BLOCK_N, HEAD_DIM]^T
        # Split into lo/hi halves for computation
        qk = tl.dot(q_lo.to(tl.bfloat16), tl.trans(k_lo.to(tl.bfloat16)), out_dtype=tl.float32)
        qk += tl.dot(q_hi.to(tl.bfloat16), tl.trans(k_hi.to(tl.bfloat16)), out_dtype=tl.float32)
        qk *= scale  # multiply by log2(e)/sqrt(HEAD_DIM) for exp2 trick

        # Causal mask: query at position (start_pos + offs_m[i]) can attend to kv_pos <= start_pos + offs_m[i]
        causal_bound = start_pos + offs_m   # [BLOCK_M] — max KV position each query can see
        causal_mask = kv_pos[None, :] <= causal_bound[:, None]  # [BLOCK_M, BLOCK_N]

        # Also mask out-of-range queries and KV positions beyond total_seq
        valid_mask = (offs_m[:, None] < seq_len) & (kv_pos[None, :] < total_seq)
        full_mask = causal_mask & valid_mask

        qk = tl.where(full_mask, qk, float("-inf"))

        # Online softmax update (FlashAttention-2)
        block_max = tl.max(qk, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.math.exp2(m_i - m_new)  # rescale factor for old accumulators
        p = tl.math.exp2(qk - m_new[:, None])  # attention weights [BLOCK_M, BLOCK_N]
        p = tl.where(full_mask, p, 0.0)  # zero out masked positions

        # Load V tile [BLOCK_N, HEAD_DIM] from cache
        v_lo = tl.load(V_cache_ptr + k_cache_offs, mask=kv_mask, other=0.0).to(tl.float32)
        v_hi = tl.load(V_cache_ptr + k_cache_offs + HALF_HD, mask=kv_mask, other=0.0).to(tl.float32)

        # Update accumulators: rescale old + add new
        acc_lo = acc_lo * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_lo.to(tl.bfloat16), out_dtype=tl.float32)
        acc_hi = acc_hi * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_hi.to(tl.bfloat16), out_dtype=tl.float32)

        l_block = tl.sum(p, axis=1)  # [BLOCK_M]
        l_i = l_i * alpha + l_block
        m_i = m_new

    # Final normalization: divide by l_i
    inv_l = 1.0 / tl.maximum(l_i, 1e-6)  # avoid div-by-zero for padded queries
    acc_lo = acc_lo * inv_l[:, None]
    acc_hi = acc_hi * inv_l[:, None]

    # Store output [BLOCK_M, HEAD_DIM] to col-major output
    # Output element at (query_idx, d) = Output_ptr[q_head_offset + d + offs_m[i] * q_dim]
    out_ptrs_lo = Output_ptr + q_head_offset + offs_d[None, :] + offs_m[:, None] * q_dim
    out_ptrs_hi = Output_ptr + q_head_offset + HALF_HD + offs_d[None, :] + offs_m[:, None] * q_dim
    out_mask = offs_m[:, None] < seq_len

    tl.store(out_ptrs_lo, acc_lo.to(tl.bfloat16), mask=out_mask)
    tl.store(out_ptrs_hi, acc_hi.to(tl.bfloat16), mask=out_mask)
