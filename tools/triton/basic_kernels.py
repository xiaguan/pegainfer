import triton
import triton.language as tl


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@triton.jit
def embedding_kernel(embed_ptr, token_id, out_ptr, hidden_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(embed_ptr + token_id * hidden_size + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def embedding_decode_kernel(
    embed_ptr, decode_meta_ptr, out_ptr, hidden_size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    token_id = tl.load(decode_meta_ptr).to(tl.int32)
    values = tl.load(embed_ptr + token_id * hidden_size + offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, values, mask=mask)


@triton.jit
def embedding_batched_kernel(
    embed_ptr, token_ids_ptr, out_ptr, hidden_size, seq_len, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = hidden_size * seq_len
    mask = offsets < total
    token_offsets = offsets // hidden_size
    dim_offsets = offsets % hidden_size
    token_ids = tl.load(token_ids_ptr + token_offsets, mask=mask, other=0).to(tl.int32)
    values = tl.load(embed_ptr + token_ids * hidden_size + dim_offsets, mask=mask, other=0)
    tl.store(out_ptr + offsets, values, mask=mask)
