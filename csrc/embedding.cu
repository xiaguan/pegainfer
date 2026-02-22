#include "common.cuh"

// Embedding lookup: out = embed[token_id]
// token_id passed as scalar (for prefill compatibility)
__global__ void
embedding_kernel(const __nv_bfloat16 *__restrict__ embed, // (vocab_size, hidden_size)
                 int token_id, __nv_bfloat16 *__restrict__ out, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hidden_size) {
    out[idx] = embed[token_id * hidden_size + idx];
  }
}

// Embedding lookup reading token_id from device buffer (CUDA Graph safe)
__global__ void
embedding_meta_kernel(const __nv_bfloat16 *__restrict__ embed,
                      const int *__restrict__ decode_meta,
                      __nv_bfloat16 *__restrict__ out, int hidden_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hidden_size) {
    int token_id = decode_meta[0];
    out[idx] = embed[token_id * hidden_size + idx];
  }
}

// Batched embedding lookup: out[i] = embed[token_ids[i]] for i in 0..seq_len
__global__ void embedding_batched_kernel(const __nv_bfloat16 *__restrict__ embed,
                                          const int *__restrict__ token_ids,
                                          __nv_bfloat16 *__restrict__ out,
                                          int hidden_size, int seq_len) {
  int total = hidden_size * seq_len;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int token_pos = idx / hidden_size;
    int dim_idx = idx % hidden_size;
    out[idx] = embed[token_ids[token_pos] * hidden_size + dim_idx];
  }
}

extern "C" {
void embedding_cuda(const __nv_bfloat16 *embed, int token_id, __nv_bfloat16 *out,
                    int hidden_size, cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (hidden_size + block_size - 1) / block_size;
  embedding_kernel<<<num_blocks, block_size, 0, stream>>>(embed, token_id, out, hidden_size);
}

void embedding_decode_cuda(const __nv_bfloat16 *embed, const int *decode_meta,
                            __nv_bfloat16 *out, int hidden_size,
                            cudaStream_t stream) {
  int block_size = 256;
  int num_blocks = (hidden_size + block_size - 1) / block_size;
  embedding_meta_kernel<<<num_blocks, block_size, 0, stream>>>(embed, decode_meta, out, hidden_size);
}

void embedding_batched_cuda(const __nv_bfloat16 *embed, const int *token_ids,
                             __nv_bfloat16 *out, int hidden_size, int seq_len,
                             cudaStream_t stream) {
  int total = hidden_size * seq_len;
  int block_size = 256;
  int num_blocks = (total + block_size - 1) / block_size;
  embedding_batched_kernel<<<num_blocks, block_size, 0, stream>>>(
      embed, token_ids, out, hidden_size, seq_len);
}
}
