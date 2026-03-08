import triton
import triton.language as tl


@triton.jit
def silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    out = silu.to(tl.bfloat16) * up
    tl.store(out_ptr + offsets, out, mask=mask)
