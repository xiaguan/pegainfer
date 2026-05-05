# PegaInfer Kernels Index

**Scope**: this crate owns CUDA/Triton build output, FFI declarations, kernel ABI tensor helpers, paged-KV layout metadata, and Rust operator wrappers. Runtime policy objects such as `KvPool`, `PagePool`, and `SamplingParams` stay outside this crate.

Use this file as the LLM entrypoint before editing kernels. Start from `op_id`, then jump to the Rust wrapper, FFI symbol, and source file.

## Qwen3-4B Dense Full-Attention Path

Qwen3-4B uses bf16 dense full attention with `hidden_size=2560`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`, and GQA group size 4. TP shards these head/intermediate dimensions per rank; the kernel IDs remain the same.

| op_id | Phase | Rust wrapper | FFI symbol | Source | Backend | Shape / layout notes |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3_4b.embedding.batch` | prefill/unified | `ops::embedding_batch` | `embedding_batched_cuda` | `csrc/elementwise.cu` | CUDA | token ids u32, output `HiddenStates` column-major `[hidden, tokens]` |
| `qwen3_4b.norm.rms_batch` | prefill/decode/unified | `ops::rms_norm_batch_into` | `rms_norm_batched_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | bf16 hidden states, one row per token |
| `qwen3_4b.norm.rms_vec` | logits | `ops::rms_norm` / `ops::rms_norm_into` | `rms_norm_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | bf16 vector |
| `qwen3_4b.linear.gemm_rows` | qkv projection | `ops::gemm_rows_into` | `gemm_cuda` | `csrc/linear.cu` | cuBLAS | row slices from fused QKV matrix |
| `qwen3_4b.linear.gemm` | o/mlp/lm_head | `ops::gemm_into` / `ops::gemm` | `gemm_cuda` | `csrc/linear.cu` | cuBLAS | weight row-major, hidden column-major |
| `qwen3_4b.attn.qk_norm_rope` | attention prep | `ops::qk_norm_rope_batch_decode_into` or direct FFI in unified path | `qk_norm_rope_batched_decode_cuda` | `csrc/prefill_attention.cu` | CUDA | full RoPE, `head_dim=128`, per-token positions |
| `qwen3_4b.kv.scatter` | prefill/decode/unified | direct FFI from model paths | `paged_kv_scatter_cuda` | `csrc/paged_attention.cu` | FlashInfer-layout CUDA wrapper | page-first `KvLayout`, NHD K/V blocks |
| `qwen3_4b.attn.prefill_paged` | prefill/unified | `ops::prefill_attention_paged_into` or direct FFI in unified path | `batch_prefill_paged_cuda` | `csrc/paged_attention.cu` | FlashInfer CUDA | `HEAD_DIM=128`, causal, paged KV |
| `qwen3_4b.attn.decode_paged` | decode/unified | `ops::paged_attention_batch_decode_into` or direct FFI in unified path | `paged_attention_decode_cuda` | `csrc/paged_attention.cu` | FlashInfer CUDA | `HEAD_DIM=128`, no partition-KV |
| `qwen3_4b.norm.fused_add_rms` | residual | `ops::fused_add_rms_norm_batch_into` | `fused_add_rms_norm_batched_cuda` | `csrc/flashinfer_norm.cu` | FlashInfer CUDA | residual add plus RMSNorm over batch |
| `qwen3_4b.mlp.silu_mul_fused` | MLP | `ops::silu_mul_fused_batch_into` | `silu_mul_fused_cuda` | `csrc/fused_proj.cu` | CUDA | input `[2 * intermediate, batch]`, output `[intermediate, batch]` |
| `qwen3_4b.elementwise.add` | residual/unified | `ops::add_batch_into` | `add_cuda` | `csrc/elementwise.cu` | CUDA | same-shape `HiddenStates` |
| `qwen3_4b.sampling.greedy` | decode output | `ops::gpu_sample_into` | `flashinfer_top1_cuda` | `csrc/flashinfer_top1.cu` | FlashInfer CUDA | top-1 path, uses row-state scratch |
| `qwen3_4b.sampling.random` | decode output | `ops::gpu_sample_into` | `gpu_sample_flashinfer_cuda` | `csrc/flashinfer_sampling.cu` | FlashInfer CUDA | temperature/top-k/top-p path |

## Non-Qwen3 Compatibility

The crate still builds CUDA/Triton symbols needed by the current root binary:

- Qwen3.5 HD256 full-attention kernels: `prefill_attention_hd256.cu`, `paged_attention.cu`.
- Qwen3.5 linear-attention decode kernels: `conv1d.cu`, `gated_delta_rule.cu`.
- Qwen3.5 chunk-wise GDR prefill Triton AOT kernels: `tools/triton/gated_delta_rule_chunkwise_kernels.py`.

These are preserved for build compatibility. They are not part of the Qwen3-4B Phase 1 API surface.

## Editing Rule

When adding or replacing a kernel used by Qwen3-4B, update this routing table.

Do not add model-specific machine-readable manifests here. The kernels crate
owns reusable operator implementations; model crates should own model DAG
metadata. If a Qwen3-4B manifest becomes useful for tracing or simulation, put
it beside the Qwen3-4B model crate and generate or validate it from code.
