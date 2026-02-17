<p align="center">
  <img src="logo.png" width="200" alt="pegainfer logo">
</p>

<h1 align="center">pegainfer</h1>

<p align="center">
  Pure Rust + CUDA LLM inference engine. No PyTorch. No frameworks. Just metal.
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#api">API</a>
</p>

---

## What is this?

pegainfer is a from-scratch LLM inference engine written in Rust with hand-written CUDA kernels. It currently runs [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) at **~249 tokens/sec** on a single GPU.

The goal is not to replace vLLM or TensorRT-LLM — it's to understand every layer of the inference stack by building it from the ground up, and to explore what a Rust-native inference engine can look like.

**What's implemented:**

- Full Qwen3 transformer: GQA, RoPE, SwiGLU MLP, RMSNorm
- 11 custom CUDA kernels + cuBLAS GEMV
- BF16 storage, FP32 accumulators
- KV cache with tiled fused attention (online softmax, TILE_SIZE=64)
- OpenAI-compatible `/v1/completions` HTTP API
- Safetensors weight loading, HuggingFace tokenizer

**What's not (yet):**

- Batching, PagedAttention, streaming (SSE)
- FlashAttention-level kernel optimization
- Multi-GPU / tensor parallelism
- Quantization (INT8/INT4)

## Quickstart

### Prerequisites

- Rust (2024 edition)
- CUDA Toolkit (nvcc, cuBLAS)
- A CUDA-capable GPU
- Qwen3-4B model weights in `models/Qwen3-4B/`

### Build & Run

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Build (compiles CUDA kernels via build.rs)
cargo build --release

# Run inference server on port 8000
cargo run --release

# Run tests
cargo test --release
```

### Download Model

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
```

## API

OpenAI-compatible completions endpoint:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'
```

## Architecture

```
Tokenize → Embedding → 28× TransformerBlock → RMSNorm → LM Head → Argmax
                              │
                              ├── RMSNorm → Fused GQA Attention → Residual
                              └── RMSNorm → Fused SwiGLU MLP    → Residual
```

```
src/
├── main.rs           # HTTP server (axum)
├── model.rs          # Qwen3Model, Attention, MLP, TransformerBlock
├── tensor.rs         # DeviceVec, DeviceMatrix — GPU tensor types
├── ops.rs            # GPU operators (linear, rms_norm, rope, fused_mlp, fused_attention)
├── kv_cache.rs       # KV cache for autoregressive generation
├── weight_loader.rs  # Safetensors loading + RoPE precomputation
├── ffi.rs            # FFI bindings to CUDA kernels
├── qwen3_config.rs   # Model config parsing
├── tokenizer.rs      # HuggingFace tokenizers wrapper
└── trace_reporter.rs # Chrome Trace JSON profiling output

csrc/
├── kernels.cu          # RMSNorm, RoPE, SiLU, embedding, GEMV, fused MLP, sampling
├── fused_attention.cu  # Fused GQA attention with tiled online softmax
└── common.cuh          # Shared CUDA utilities
```

### Key design decisions

- **All computation on GPU** — no CPU fallback, no hybrid execution
- **Custom CUDA kernels** for everything except matrix multiplication (cuBLAS)
- **Fused operators** — attention and MLP are each a single kernel launch
- **BF16 storage, FP32 accumulation** — numerical stability without memory overhead
- **Synchronous execution** — simple and debuggable, no overlap optimization yet

## Performance

Measured on RTX 5070 Ti, Qwen3-4B, BF16:

| Metric | Value |
|--------|-------|
| TTFT (prompt_len=9) | ~33 ms |
| TPOT | ~4 ms/token |
| Throughput | ~249 tokens/sec |

Profiling traces are written to `traces/` in Chrome Trace JSON format — open with [Perfetto UI](https://ui.perfetto.dev).

## License

MIT
