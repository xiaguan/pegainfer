<p align="center">
  <img src="logo.png" width="200" alt="pegainfer logo">
</p>

<h1 align="center">pegainfer</h1>

<p align="center">
  Pure Rust + CUDA LLM inference engine. No PyTorch. No frameworks. Just metal.
</p>

<p align="center">
  <a href="#supported-models">Models</a> &middot;
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#performance">Performance</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#architecture">Architecture</a>
</p>

---

## What is this?

pegainfer is a from-scratch LLM inference engine written in **~7K lines of Rust** and **~3.4K lines of hand-written CUDA kernels**. No PyTorch, no ONNX, no frameworks — just Rust + raw CUDA.

The goal is to understand every layer of the inference stack by building it from the ground up, and to explore what a Rust-native inference engine can look like.

## Supported Models

| Model | Architecture | Layers | Params | Status |
|-------|-------------|--------|--------|--------|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Full attention (GQA) | 36 | 4B | Greedy + sampling |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Full attention (GQA) | 36 | 8B | Greedy + sampling |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Hybrid (24 linear + 8 full attention) | 32 | 4B | Greedy + sampling |

Model type is auto-detected from `config.json` — just point `--model-path` at any supported model directory.

## Quickstart

### Prerequisites

- Rust (2024 edition)
- CUDA Toolkit (nvcc, cuBLAS)
- A CUDA-capable GPU (SM target auto-detected at build time)
- Python 3 with Triton installed for build-time AOT kernel generation

### Python Environment

```bash
# Create venv (once, from pegainfer/ root)
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers accelerate pytest
```

This single venv covers everything: build-time Triton AOT kernel compilation, HF reference generation, and Python kernel tests.

### Download a Model

```bash
# Pick one (or more):
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
```

### Build & Run

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PEGAINFER_TRITON_PYTHON=.venv/bin/python  # uses the venv created above

# If `nvidia-smi` is unavailable in your build environment, set the target SM explicitly.
# Example: export PEGAINFER_CUDA_SM=120

# Build (compiles CUDA kernels plus the default Triton AOT `silu_mul` replacement)
cargo build --release

# Start server (defaults to Qwen3-4B on port 8000)
cargo run --release

# Start server with a different model
cargo run --release -- --model-path models/Qwen3.5-4B

# Disable CUDA Graph (useful for debugging)
cargo run --release -- --cuda-graph=false

# Enable performance tracing (Chrome Trace JSON → open with Perfetto UI)
cargo run --release -- --trace-output-path traces/
```

### Windows

```powershell
# Set CUDA path (auto-detected on Linux, required on Windows)
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"

# Install Triton for Windows (official Triton only supports Linux)
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:PEGAINFER_TRITON_PYTHON = ".venv\Scripts\python.exe"

cargo build --release
cargo run --release --bin pegainfer -- --model-path models/Qwen3-4B
```

### Triton AOT Notes

- `silu_mul` is the first default-on Triton AOT replacement; the rest of the kernel stack still builds from `csrc/`.
- Triton is used at build time only. Runtime stays in Rust + CUDA via the generated C wrapper.
- Generated Triton artifacts live under Cargo `OUT_DIR`, typically `target/release/build/pegainfer-*/out/triton_aot/silu_mul/`.
- `PEGAINFER_CUDA_SM` now doubles as the explicit Triton AOT target when GPU auto-detection is unavailable during build.
- If Triton Python setup or GPU SM detection fails, see `tools/triton/README.md` for the exact environment variables and validation commands.

### Try it

```bash
# Non-streaming
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Streaming (SSE)
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'

# With sampling parameters
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 128, "temperature": 0.7, "top_p": 0.9}'
```

### Run Tests

```bash
# Unit tests (tensor, ops, config, tokenizer, sampler)
cargo test --release

# E2E greedy regression (Qwen3-4B, needs GPU + model weights)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e

# E2E greedy regression (Qwen3.5-4B)
cargo test --release --test e2e_qwen35
```

> **Note:** Always use `--release`. Debug builds are extremely slow for GPU/CUDA code and will timeout.

## Performance

Measured on **RTX 5070 Ti** (16 GB), BF16, CUDA Graph enabled:

| Metric | Qwen3-4B | Qwen3.5-4B |
|--------|----------|-------------|
| TTFT (short prompt) | ~14 ms | ~22 ms |
| TPOT (decode) | ~11 ms/tok | ~12.2 ms/tok |
| Throughput | **~91 tok/s** | **~82 tok/s** |

<details>
<summary>What do these metrics mean?</summary>

- **TTFT** (Time To First Token): latency from receiving the prompt to generating the first output token. Includes tokenization, embedding, and the full prefill pass.
- **TPOT** (Time Per Output Token): average time to generate each subsequent token during the decode phase.
- **Throughput**: 1000 / TPOT, i.e. tokens generated per second during decode.

</details>

Profiling traces can be written to `traces/` in Chrome Trace JSON format — open with [Perfetto UI](https://ui.perfetto.dev).

## API

OpenAI-compatible `/v1/completions` endpoint.

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | (required) | Input text |
| `max_tokens` | int | 128 | Maximum tokens to generate |
| `temperature` | float | 0.0 | Sampling temperature (0 = greedy) |
| `top_k` | int | 50 | Top-k sampling |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `stream` | bool | false | Enable SSE streaming |

## Architecture

```
Tokenize → Embedding → N × TransformerBlock → RMSNorm → LM Head → Sample
                              │
                              ├── RMSNorm → Attention → Residual
                              └── RMSNorm → MLP (SwiGLU) → Residual
```

- **Prefill**: batched GEMM for all prompt tokens at once
- **Decode**: single-token GEMV per step, CUDA Graph captured and replayed
- **BF16 storage**, FP32 accumulators in all kernels
- **Qwen3**: 32 Q heads, 8 KV heads (GQA 4:1), head_dim=128
- **Qwen3.5**: hybrid architecture — 24 linear attention layers (Gated Delta Rule) + 8 full attention layers, head_dim=256

### Source Layout

```
src/
├── bin/triton_silu_smoke.rs # Focused Triton-vs-CUDA silu_mul validation binary
├── main.rs              # CLI + HTTP server startup (axum)
├── http_server/         # OpenAI-compatible /v1/completions (streaming + non-streaming)
├── server_engine.rs     # ServerEngine trait, model type detection, engine loading
├── model.rs             # Qwen3Model: attention, MLP, forward, generate
├── qwen35_model.rs      # Qwen3.5Model: hybrid linear + full attention
├── tensor.rs            # DeviceVec, DeviceMatrix, HiddenStates — GPU tensor types
├── ops.rs               # GPU operators (linear, rms_norm, rope, fused_mlp, fused_attention, …)
├── kv_cache.rs          # KV cache for autoregressive generation
├── recurrent_state.rs   # Recurrent state for linear attention (Qwen3.5)
├── weight_loader.rs     # Safetensors loading + RoPE precomputation
├── ffi.rs               # FFI bindings to CUDA kernels
├── qwen3_config.rs      # Qwen3 config parsing
├── qwen35_config.rs     # Qwen3.5 config parsing (mixed layer types)
├── tokenizer.rs         # HuggingFace tokenizers wrapper
├── sampler.rs           # Temperature, top-k, top-p sampling
└── trace_reporter.rs    # Chrome Trace JSON profiling output

csrc/
├── embedding.cu             # Token embedding lookup
├── norm.cu                  # RMSNorm (+ fused Add+RMSNorm)
├── pos_enc.cu               # RoPE
├── gemv.cu                  # GEMV (BF16×2 vectorized)
├── fused_attention.cu       # Fused GQA attention, tiled online softmax (head_dim=128)
├── fused_attention_hd256.cu # Fused attention for head_dim=256 (Qwen3.5)
├── prefill_attention.cu     # Batched prefill attention
├── fused_mlp.cu             # Fused SwiGLU MLP (gate+up→SiLU→down)
├── activation.cu            # SiLU activation
├── elementwise.cu           # Add, copy, softmax
├── conv1d.cu                # Conv1d for linear attention (Qwen3.5)
├── gated_delta_rule.cu      # Gated Delta Rule recurrence (Qwen3.5)
└── sampling.cu              # GPU argmax, top-k/top-p sampling

tools/triton/
├── gen_triton_aot.py            # Triton AOT compilation driver (used by build.rs)
├── silu_mul_kernel.py           # Triton silu_mul kernel
├── attention_decode_kernel.py   # Triton fused decode attention kernel
├── basic_kernels.py             # Triton add / embedding kernels
└── README.md                    # Setup, failure modes, and validation commands
```

### Key Design Decisions

- **All computation on GPU** — no CPU fallback, no hybrid execution
- **Custom CUDA kernels** for everything except matrix multiplication (cuBLAS)
- **Fused operators** — attention and MLP are each a single kernel launch
- **BF16 storage, FP32 accumulation** — numerical stability without memory overhead
- **CUDA Graph** on decode path — eliminates kernel launch overhead
- **Synchronous execution** — simple and debuggable, no CPU-GPU overlap yet

### What's not (yet) implemented

- Batched requests / continuous batching
- PagedAttention
- Multi-GPU / tensor parallelism
- Quantization (INT8/INT4)

## License

MIT
