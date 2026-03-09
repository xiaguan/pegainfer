# Triton AOT Integration

`pegainfer` now builds `silu_mul`, `add`, and the embedding lookup kernels through Triton AOT by default.

## What this covers

- Build-time generation of Triton AOT cubins for:
  - `silu_mul`
  - `add`
  - `embedding`
  - `embedding_decode`
  - `embedding_batched`
- Generated C wrappers linked into the normal Rust build
- Default runtime routing of the corresponding ops onto Triton-generated launchers
- `extract_vec` / `write_vec` now using `cudarc` device-to-device memcpy instead of a custom CUDA copy kernel
- A focused `triton_silu_smoke` binary that compares the Triton path against a CPU reference

`build.rs` now skips compiling the replaced legacy CUDA translation units `csrc/activation.cu`, `csrc/elementwise.cu`, and `csrc/embedding.cu`.

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Bootstrap a repo-local Triton Python once:

```bash
uv venv tools/triton/.venv
uv pip install -p tools/triton/.venv/bin/python triton
```

Then either point the build to that interpreter explicitly:

```bash
export PEGAINFER_TRITON_PYTHON=$PWD/tools/triton/.venv/bin/python
```

or let `build.rs` auto-probe `tools/triton/.venv/bin/python` before trying `python3` / `python`.

If `nvidia-smi` is unavailable where you build, also set the target SM manually.

```bash
export PEGAINFER_CUDA_SM=120
```

`PEGAINFER_CUDA_SM` also drives the explicit Triton AOT compile target, so it is the default escape hatch when the build environment cannot query a live GPU.

## Build

```bash
cargo build --release
```

Generated Triton artifacts are written to Cargo `OUT_DIR`, typically under:

```text
target/release/build/pegainfer-*/out/triton_aot/
```

## Validation

Sanity-check the default `silu_mul` path against a host-side reference:

```bash
cargo run --release --bin triton_silu_smoke -- --seq-len 32 --hidden-dim 4096 --iters 20
```

Run the focused GPU tests for the newly replaced paths:

```bash
cargo test --release embedding_variants -- --nocapture
cargo test --release extract_write_vec_roundtrip -- --nocapture
cargo test --release add_and_add_inplace -- --nocapture
```

## Common failures

- `Could not find a Python interpreter with Triton installed`
  - Set `PEGAINFER_TRITON_PYTHON`, or bootstrap `tools/triton/.venv` with `uv`.
- `GPU detection failed`
  - Set `PEGAINFER_CUDA_SM` explicitly if `nvidia-smi` is not available during build.
- `Triton AOT generator failed`
  - Re-run the build and inspect the generator stderr printed by `build.rs`; the generator accepts an explicit `cuda:<sm>:32` target derived from `PEGAINFER_CUDA_SM`.
- `CUDA_ERROR_NO_BINARY_FOR_GPU` or similar runtime load errors
  - Rebuild on the target GPU environment; the generated Triton cubin is target-specific.
