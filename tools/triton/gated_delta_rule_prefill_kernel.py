import triton
import triton.language as tl


@triton.jit
def gated_delta_rule_prefill_kernel(
    qkv_ptr,         # [seq_len, qkv_dim] token-major
    b_ptr,           # [seq_len, num_value_heads] token-major
    a_ptr,           # [seq_len, num_value_heads] token-major
    dt_bias_ptr,     # [num_value_heads]
    a_log_ptr,       # [num_value_heads]
    state_ptr,       # [num_value_heads, VAL_DIM, KEY_DIM] fp32
    output_ptr,      # [seq_len, out_dim] token-major
    num_key_heads,
    num_value_heads,
    qkv_dim,
    seq_len,
    out_dim,
    BLOCK_V: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VAL_DIM: tl.constexpr,
):
    """Fused recurrent gated delta rule over a full prefill sequence.

    Grid: (ceil_div(VAL_DIM, BLOCK_V), num_value_heads, 1)
    Each program owns one value-head tile of the recurrent state and iterates
    over the sequence inside the kernel, removing the per-token host loop.
    """
    v_tile = tl.program_id(0)
    v_head = tl.program_id(1)

    offs_k = tl.arange(0, KEY_DIM)
    offs_v = v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < VAL_DIM

    qk_dim_total = num_key_heads * KEY_DIM
    k_head = (v_head * num_key_heads) // num_value_heads

    q_offset = k_head * KEY_DIM
    k_offset = qk_dim_total + k_head * KEY_DIM
    v_offset = 2 * qk_dim_total + v_head * VAL_DIM + offs_v

    state_base = state_ptr + v_head * KEY_DIM * VAL_DIM + offs_v[:, None] * KEY_DIM + offs_k[None, :]
    state = tl.load(state_base, mask=mask_v[:, None], other=0.0).to(tl.float32)

    dt_bias = tl.load(dt_bias_ptr + v_head).to(tl.float32)
    a_log = tl.load(a_log_ptr + v_head).to(tl.float32)

    for t in tl.range(0, seq_len):
        token_qkv = qkv_ptr + t * qkv_dim
        q = tl.load(token_qkv + q_offset + offs_k).to(tl.float32)
        k = tl.load(token_qkv + k_offset + offs_k).to(tl.float32)
        v = tl.load(token_qkv + v_offset, mask=mask_v, other=0.0).to(tl.float32)

        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-12)
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-12)
        q *= tl.rsqrt(float(KEY_DIM))

        a = tl.load(a_ptr + t * num_value_heads + v_head).to(tl.float32)
        b = tl.load(b_ptr + t * num_value_heads + v_head).to(tl.float32)

        x = a + dt_bias
        softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        exp_g = tl.exp(-tl.exp(a_log) * softplus_x)
        beta = 1.0 / (1.0 + tl.exp(-b))

        state *= exp_g
        kv_mem = tl.sum(state * k[None, :], axis=1)
        delta = (v - kv_mem) * beta
        state += delta[:, None] * k[None, :]

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(
            output_ptr + t * out_dim + v_head * VAL_DIM + offs_v,
            out.to(tl.bfloat16),
            mask=mask_v,
        )

    tl.store(state_base, state, mask=mask_v[:, None])
