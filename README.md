<p align="center">
  <img src="logo.png" width="200" alt="pegainfer logo">
</p>

<h1 align="center">pegainfer</h1>

<p align="center">
  Pure Rust + CUDA LLM inference engine. No PyTorch. No frameworks. Just metal.
</p>

<p align="center">
  <a href="#performance">Performance</a> &middot;
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#supported-models">Models</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="#architecture">Architecture</a>
</p>

---

pegainfer is a from-scratch LLM inference engine written in **~9.6K lines of Rust**, **~2.6K lines of CUDA**, and **~1.4K lines of Triton GPU kernels**. No PyTorch, no ONNX, no frameworks — just Rust + raw CUDA/Triton.

The goal is to understand every layer of the inference stack by building it from the ground up, and to explore what a Rust-native inference engine can look like.

## Performance

Measured on **RTX 5070 Ti** (16 GB), BF16, CUDA Graph enabled, single request:

| Metric | Qwen3-4B | Qwen3.5-4B |
|--------|----------|-------------|
| TTFT (short prompt) | ~14 ms | ~22 ms |
| TPOT (decode) | ~11 ms/tok | ~11.8 ms/tok |
| Throughput | **~91 tok/s** | **~85 tok/s** |

<details>
<summary>What do these metrics mean?</summary>

- **TTFT** (Time To First Token): latency from receiving the prompt to generating the first output token. Includes tokenization, embedding, and the full prefill pass.
- **TPOT** (Time Per Output Token): average time to generate each subsequent token during the decode phase.
- **Throughput**: 1000 / TPOT, i.e. tokens generated per second during decode.

</details>

## Quickstart

### Prerequisites

- Rust (2024 edition), CUDA Toolkit (nvcc, cuBLAS), CUDA-capable GPU
- Python 3 + Triton (build-time only — no Python at runtime)

### Build & Run

```bash
# One-time Python setup (for Triton AOT kernel compilation)
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Download a model
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

# Build & start server on port 8000
export CUDA_HOME=/usr/local/cuda
export PEGAINFER_TRITON_PYTHON=.venv/bin/python
cargo run --release
```

```bash
# Try it
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Streaming
curl -N http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about Rust:", "max_tokens": 64, "stream": true}'
```

> Always use `--release`. Debug builds are extremely slow for GPU/CUDA code.

<details>
<summary>More options</summary>

```bash
# Different model
cargo run --release -- --model-path models/Qwen3.5-4B

# Disable CUDA Graph (useful for debugging)
cargo run --release -- --cuda-graph=false

# Performance tracing (Chrome Trace JSON → open with Perfetto UI)
cargo run --release -- --trace-output-path traces/
```

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `CUDA_HOME` | CUDA Toolkit path (default: `/usr/local/cuda`) |
| `PEGAINFER_TRITON_PYTHON` | Python with Triton for build-time AOT compilation |
| `PEGAINFER_CUDA_SM` | GPU SM target override when `nvidia-smi` unavailable (e.g. `120`) |

</details>

<details>
<summary>Windows</summary>

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
uv venv .venv --python 3.12
uv pip install "triton-windows<3.7"
$env:PEGAINFER_TRITON_PYTHON = ".venv\Scripts\python.exe"

cargo build --release
cargo run --release --bin pegainfer -- --model-path models/Qwen3-4B
```

</details>

## Supported Models

| Model | Architecture | Params | Status |
|-------|-------------|--------|--------|
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Full attention (GQA) | 4B | Greedy + sampling |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Full attention (GQA) | 8B | Greedy + sampling |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Hybrid (24 linear + 8 full attention) | 4B | Greedy + sampling |

Model type is auto-detected from `config.json` — just point `--model-path` at any supported model directory.

## API

OpenAI-compatible `/v1/completions` endpoint.

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
HTTP → GenericServerEngine<M: ModelForward> → model.forward(tokens, state)
                                                       │
                                         ┌─────────────┴─────────────┐
                                    Qwen3Model                 Qwen35Model
                                  (full attention)        (24 linear + 8 full attn)
                                         │                         │
                                         └────────────┬────────────┘
                                                      │
                                        Prefill (GEMM) / Decode (GEMV + CUDA Graph)
                                                      │
                                            ops → ffi → CUDA / Triton kernels
```

**Key design decisions:**

- **All computation on GPU** — no CPU fallback, no hybrid execution
- **Custom GPU kernels** — CUDA for decode-critical paths (GEMV, fused MLP, GDR recurrence), Triton AOT for attention, embedding, and prefill. Matrix multiplication via cuBLAS
- **Fused operators** — attention and MLP are each a single kernel launch
- **BF16 storage, FP32 accumulation** — numerical stability without memory overhead
- **CUDA Graph** on decode path — eliminates kernel launch overhead

**Model details:**

- **Qwen3**: 32 Q heads, 8 KV heads (GQA 4:1), head_dim=128
- **Qwen3.5**: hybrid — 24 linear attention layers (Gated Delta Rule) + 8 full attention layers, head_dim=256

### What's not (yet) implemented

- Batched requests / continuous batching
- PagedAttention
- Multi-GPU / tensor parallelism
- Quantization (INT8/INT4)

## Development

### Tests

```bash
# Unit tests
cargo test --release

# E2E greedy regression (needs GPU + model weights)
PEGAINFER_TEST_MODEL_PATH=models/Qwen3-4B cargo test --release --test e2e
cargo test --release --test e2e_qwen35
```

### Triton AOT

Triton compiles **16+ kernels** at build time (silu_mul, add, embedding variants, split-KV attention decode/reduce, FlashAttention prefill, chunk-wise GDR prefill). Runtime has no Python dependency — everything runs through generated C wrappers.

See `tools/triton/README.md` for setup and troubleshooting.

### Source Layout

<details>
<summary>Expand</summary>

```
src/
├── main.rs                # CLI + vLLM/OpenAI server startup
├── vllm_frontend.rs       # vLLM engine-core bridge into pegainfer scheduler
├── server_engine.rs       # Model detection and shared scheduler-facing types
├── scheduler.rs           # Qwen3 continuous batching scheduler
├── scheduler_qwen35.rs    # Qwen3.5 scheduler
├── model_executor.rs      # Shared execution helpers for scheduler paths
├── model.rs               # ModelForward trait
├── model/
│   ├── cuda_graph.rs      # CUDA Graph capture/replay
│   ├── kv_cache.rs        # KV cache
│   ├── qwen3/             # Qwen3: config, weights, forward, prefill, decode
│   └── qwen35/            # Qwen3.5: config, weights, forward, prefill, decode, recurrent_state
├── ops.rs                 # GPU operator dispatch
├── ops/
│   ├── attention.rs       # Fused GQA attention, prefill attention
│   ├── elementwise.rs     # Add, copy, softmax
│   ├── embedding.rs       # Token embedding lookup
│   ├── linear.rs          # GEMV, cuBLAS GEMM
│   ├── norm.rs            # RMSNorm (+ fused Add+RMSNorm)
│   ├── recurrent.rs       # Conv1d, Gated Delta Rule (Qwen3.5)
│   └── sampling.rs        # GPU argmax, top-k/top-p
├── tensor.rs              # GPU tensor types (DeviceVec, DeviceMatrix, HiddenStates)
├── ffi.rs                 # FFI bindings to CUDA/Triton kernels
├── weight_loader.rs       # Safetensors loading + RoPE precomputation
├── sampler.rs             # Temperature, top-k, top-p sampling
└── trace_reporter.rs      # Chrome Trace JSON profiling

csrc/                      # Hand-written CUDA kernels
├── gemv.cu                # GEMV (BF16×2 vectorized)
├── fused_attention.cu     # Fused GQA decode attention (head_dim=128)
├── fused_mlp.cu           # Fused SwiGLU MLP (gate+up→SiLU→down)
├── gated_delta_rule.cu    # GDR decode recurrence (Qwen3.5)
├── norm.cu                # RMSNorm (+ fused Add+RMSNorm)
├── pos_enc.cu             # RoPE
├── prefill_attention.cu   # Batched prefill attention (head_dim=128)
├── prefill_attention_hd256.cu  # Prefill attention (head_dim=256)
├── conv1d.cu              # Conv1d (Qwen3.5)
└── sampling.cu            # GPU argmax, top-k/top-p

tools/triton/              # Triton AOT kernels (build-time compiled)
├── gen_triton_aot.py      # AOT compilation driver
├── silu_mul_kernel.py
├── basic_kernels.py       # add, embedding variants
├── attention_decode_kernel.py
├── attention_reduce_kernel.py
├── flash_attention_prefill_kernel.py
├── flash_attention_prefill_hd256_kernel.py
├── gated_delta_rule_chunkwise_kernels.py
└── README.md
```

</details>

## License

MIT
