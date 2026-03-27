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
```

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.