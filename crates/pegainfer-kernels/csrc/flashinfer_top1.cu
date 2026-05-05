#include "common.cuh"

#include <flashinfer/sampling.cuh>
#include <flashinfer/topk.cuh>

extern "C" void flashinfer_top1_cuda(const __nv_bfloat16* logits,
                                     __nv_bfloat16* top1_value_scratch,
                                     uint8_t* row_states_scratch, int* output,
                                     int vocab_size, cudaStream_t stream) {
  auto* row_states =
      reinterpret_cast<flashinfer::sampling::RadixRowState*>(row_states_scratch);
  auto* input = const_cast<__nv_bfloat16*>(logits);
  (void)flashinfer::sampling::TopKDispatch<__nv_bfloat16, int>(
      input, output, top1_value_scratch, 1, 1, vocab_size, row_states, stream);
}
