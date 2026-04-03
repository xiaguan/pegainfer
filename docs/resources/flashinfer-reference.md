# FlashInfer Reference

**Status**: Complete
**TL;DR**: FlashInfer is a Python-first GPU inference kernel library and kernel generator for serving workloads. Its docs are organized into install/CLI/logging/autotuning guides, two data-model tutorials, and a broad PyTorch API surface spanning attention, GEMM, MoE, sampling, sparse attention, communication, quantization, and normalization.

---

## 1. What FlashInfer Is

FlashInfer positions itself as a high-performance inference-kernel library for LLM serving. The emphasis is not just one attention kernel, but a serving-oriented operator stack:

- attention kernels for decode, prefill, append, paged KV, ragged KV, shared-prefix, and MLA workloads
- GEMM kernels across BF16, FP8, FP4, mixed precision, and grouped/segmented execution
- fused MoE kernels with multiple backends and routing styles
- sampling, top-k, logits processing, RoPE, normalization, and activation kernels
- distributed communication utilities such as allreduce, CUDA IPC, MNNVL, and MoE all-to-all
- JIT compilation, optional precompiled artifacts, autotuning, and logging/debugging support

At the packaging level, FlashInfer is Python-first and PyTorch-oriented, but internally it is also a kernel generator: many operators are selected or specialized at runtime based on hardware, backend, and shape.

## 2. Where Its Docs Are

The main docs entry is `third_party/flashinfer/docs/index.rst`. The structure is straightforward:

### Get Started

- `installation.rst`: package model, prerequisites, install from PyPI/source, nightly builds, verification
- `cli.rst`: `flashinfer` CLI commands for config, modules, cubins, cache, and compile database export
- `logging.rst`: API logging, dump levels, filters, replay-oriented debugging support
- `autotuning.rst`: runtime tuning for GEMM and MoE backends/tactics, cache persistence and reuse

### Tutorials

- `tutorials/recursive_attention.rst`: the attention-state abstraction and recursive merge operator behind cascade/shared-prefix/split-KV style execution
- `tutorials/kv_layout.rst`: NHD vs HND KV layout, ragged tensors, page-table KV cache, MLA KV layout, mask packing

### API Reference

- `api/attention.rst`
- `api/gemm.rst`
- `api/fused_moe.rst`
- `api/cascade.rst`
- `api/comm.rst`
- `api/sparse.rst`
- `api/page.rst`
- `api/sampling.rst`
- `api/topk.rst`
- `api/logits_processor.rst`
- `api/norm.rst`
- `api/rope.rst`
- `api/activation.rst`
- `api/quantization.rst`
- `api/green_ctx.rst`
- `api/fp4_quantization.rst`
- `api/testing.rst`

If the goal is to understand FlashInfer quickly, the fastest route is:

1. `README.md`
2. `docs/index.rst`
3. `tutorials/kv_layout.rst`
4. `api/attention.rst`
5. then the module pages relevant to your workload

## 3. Core Feature Map

FlashInfer's feature set can be grouped into a few major buckets.

### Attention and KV-cache serving

This is the center of gravity of the project.

- single-request decode and prefill
- batch decode and batch prefill
- paged KV cache
- ragged KV cache
- append-to-KV workflows
- CUDA-graph-friendly decode wrappers
- XQA APIs
- MLA paged attention for DeepSeek-style latent attention
- cascade attention for shared-prefix reuse
- POD attention for mixed prefill+decode serving
- sparse attention wrappers

Relevant docs:

- `docs/api/attention.rst`
- `docs/api/cascade.rst`
- `docs/api/sparse.rst`
- `docs/api/page.rst`
- `docs/tutorials/recursive_attention.rst`
- `docs/tutorials/kv_layout.rst`

### GEMM and linear algebra

FlashInfer is not just attention kernels. The GEMM side includes:

- BF16 GEMM and batched GEMM
- FP8 GEMM
- FP4 GEMM
- mixed precision paths such as FP8 x FP4
- grouped and segmented GEMM wrappers
- backend-dependent implementations spanning cuDNN, CUTLASS, TensorRT-LLM, and DeepGEMM-related paths in code

Relevant docs:

- `docs/api/gemm.rst`
- `docs/autotuning.rst`

### Fused MoE

FlashInfer has a real MoE stack rather than a token example API:

- fused expert execution
- multiple routing methods
- TensorRT-LLM-oriented MoE paths
- CUTLASS fused MoE
- FP8 and FP4 quantized MoE variants
- helper utilities for expert-row reorder and block layout conversion
- MoE-related distributed collectives and all-to-all under `comm`

Relevant docs:

- `docs/api/fused_moe.rst`
- `docs/api/comm.rst`
- `docs/autotuning.rst`

### Sampling and post-processing

FlashInfer also covers the token-selection side of inference:

- sampling from logits or probabilities
- top-p, top-k, and min-p sampling
- top-k/top-p combined sampling
- sorting-free top-k selection
- page-table and ragged top-k transforms
- chain speculative sampling
- a declarative logits processor pipeline with fusion/customization hooks

Relevant docs:

- `docs/api/sampling.rst`
- `docs/api/topk.rst`
- `docs/api/logits_processor.rst`

### Common transformer building blocks

There is a useful layer of smaller but production-relevant operators:

- RoPE and Llama 3.1 RoPE variants
- RMSNorm, fused add+RMSNorm, Gemma-specific norm variants, LayerNorm
- gated activations such as SiLU and GELU fused with multiply
- bit packing and segment packing utilities for masks
- FP4/NVFP4 quantization helpers, including KV-cache quantize/dequantize paths

Relevant docs:

- `docs/api/rope.rst`
- `docs/api/norm.rst`
- `docs/api/activation.rst`
- `docs/api/quantization.rst`
- `docs/api/fp4_quantization.rst`

### Distributed and runtime infrastructure

FlashInfer exposes serving/runtime features beyond math kernels:

- CUDA IPC shared-buffer utilities
- TensorRT-LLM allreduce helpers and fused allreduce paths
- vLLM custom allreduce integration points
- MNNVL multi-node NVLink helpers
- MoE A2A helpers
- API logging with dump/replay-oriented levels
- autotuning with persistent config caches
- CLI tools for artifact and module management
- green context utilities for partitioning device execution resources

Relevant docs:

- `docs/api/comm.rst`
- `docs/logging.rst`
- `docs/autotuning.rst`
- `docs/cli.rst`
- `docs/api/green_ctx.rst`

## 4. Operator Families at a Glance

From the public docs and `flashinfer/__init__.py`, the operator surface roughly breaks down like this:

| Family | Examples |
| --- | --- |
| Attention | `single_decode_with_kv_cache`, `BatchDecodeWithPagedKVCacheWrapper`, `single_prefill_with_kv_cache`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchPrefillWithRaggedKVCacheWrapper`, `BatchMLAPagedAttentionWrapper`, `xqa`, `xqa_mla`, `PODWithPagedKVCacheWrapper` |
| KV utilities | `append_paged_kv_cache`, `append_paged_mla_kv_cache`, `get_batch_indices_positions`, `get_seq_lens` |
| Cascade/shared-prefix | `merge_state`, `merge_states`, `MultiLevelCascadeAttentionWrapper`, `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` |
| Sparse attention | `BlockSparseAttentionWrapper`, `VariableBlockSparseAttentionWrapper` |
| GEMM | `mm_bf16`, `bmm_bf16`, `mm_fp8`, `bmm_fp8`, `mm_fp4`, `SegmentGEMMWrapper` |
| MoE | `cutlass_fused_moe`, `trtllm_fp8_block_scale_moe`, `trtllm_fp4_block_scale_moe`, routed variants |
| Sampling/top-k | `sampling_from_logits`, `top_p_sampling_from_probs`, `top_k_sampling_from_probs`, `chain_speculative_sampling`, `top_k`, `top_k_page_table_transform` |
| Logits pipeline | `LogitsPipe`, `Temperature`, `Softmax`, `TopK`, `TopP`, `MinP`, `Sample` |
| Norm/rope/activation | `rmsnorm`, `fused_add_rmsnorm`, `layernorm`, `apply_rope`, `apply_llama31_rope`, `silu_and_mul`, `gelu_and_mul` |
| Quantization | `packbits`, `segment_packbits`, `fp4_quantize`, `nvfp4_quantize`, `nvfp4_kv_quantize`, `mxfp8_quantize` |
| Communication | `trtllm_custom_all_reduce`, `vllm_all_reduce`, `MoeAlltoAll`, `MnnvlMemory` |

## 5. Notable Architectural Ideas

Some FlashInfer ideas matter more than any single API name:

- **Paged KV cache as a first-class serving abstraction**: the docs explicitly teach page-table layout, append semantics, and ragged/paged execution rather than hiding them.
- **Recursive attention states**: attention outputs can be merged through state tuples, which is the conceptual basis for cascade attention and KV partitioning.
- **Multiple backends per operator family**: one API may dispatch across FlashAttention-style kernels, cuDNN, CUTLASS, TensorRT-LLM, or other backend-specific implementations.
- **JIT by default, precompiled optional**: the core package can compile or download on first use; `flashinfer-cubin` and `flashinfer-jit-cache` are acceleration layers, not the only install mode.
- **Autotuning as part of normal execution**: GEMM and MoE operators can profile several runners/tactics and cache the winning config.
- **Serving-debugging support**: API logging goes beyond printf-style tracing and includes tensor-statistics and full dump modes.

## 6. Things the Code Exposes Beyond the Main Docs Index

The docs index gives the main supported surface, but the source tree shows additional modules that are useful to know about:

- `flashinfer.mamba`: state-space / selective-state-update related code
- `flashinfer.gdn_decode` and `flashinfer.gdn_prefill`: gated-delta-rule related code paths
- `flashinfer.deep_gemm` and backend-specific GEMM helpers
- `flashinfer.cudnn` and `flashinfer.triton`: lower-level backend modules
- `flashinfer.cute_dsl` and `flashinfer.fused_moe.cute_dsl`: CuTeDSL-oriented implementation areas

These are worth reading when studying internals, but they are not as central in the public docs navigation as the main API pages above.

## 7. Practical Reading Order

Choose the branch that matches your question.

If you care about serving attention:

1. `README.md`
2. `docs/tutorials/kv_layout.rst`
3. `docs/tutorials/recursive_attention.rst`
4. `docs/api/attention.rst`
5. `docs/api/cascade.rst`
6. `docs/api/page.rst`

If you care about low-precision compute and MoE:

1. `docs/api/gemm.rst`
2. `docs/api/fused_moe.rst`
3. `docs/api/fp4_quantization.rst`
4. `docs/autotuning.rst`
5. `flashinfer/__init__.py` and the corresponding subpackages

If you care about runtime operations and debugging:

1. `docs/installation.rst`
2. `docs/cli.rst`
3. `docs/logging.rst`
4. `docs/api/comm.rst`
5. `docs/api/green_ctx.rst`

## 8. Bottom Line

FlashInfer should be read as a serving-kernel platform, not just an attention library.

The shortest accurate summary is:

- its docs cover install, CLI, logging, autotuning, data-layout tutorials, and a broad PyTorch API reference
- its strongest public surface is attention/KV-cache serving, but it also has substantial GEMM, MoE, sampling, quantization, and distributed-communication coverage
- the source tree exposes even more backend-specific and experimental areas than the docs index highlights, so `README.md` + `docs/index.rst` + `flashinfer/__init__.py` is the fastest way to build a full map
