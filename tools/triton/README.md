# Triton AOT Integration

`pegainfer` now builds the `silu_mul` kernel through Triton AOT by default.

## What this covers

- Build-time generation of a Triton AOT cubin for `silu_mul`
- A generated C wrapper linked into the normal Rust build
- Default runtime routing of `ops::silu_mul_batch` to the Triton-generated kernel
- A focused smoke binary that compares the Triton path against the legacy CUDA implementation

The rest of the kernel stack still builds from `csrc/` exactly as before.

## Prerequisites

```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PEGAINFER_TRITON_PYTHON=/path/to/python-with-triton
```

If `nvidia-smi` is unavailable where you build, also set the target SM manually.

```bash
export PEGAINFER_CUDA_SM=120
```

`PEGAINFER_CUDA_SM` also drives the explicit Triton AOT compile target, so it is the default escape hatch when the build environment cannot query a live GPU.

If `PEGAINFER_TRITON_PYTHON` is unset, the build probes `python3` and then `python`.

## Build

```bash
cargo build --release
```

Generated Triton artifacts are written to Cargo `OUT_DIR`, typically:

```text
target/release/build/pegainfer-*/out/triton_aot/silu_mul/
```

Nothing under `OUT_DIR` is checked into git.

## Validation

Compare the default Triton path with the in-tree CUDA reference implementation:

```bash
cargo run --release --bin triton_silu_smoke -- --seq-len 32 --hidden-dim 4096 --iters 20
```

The smoke binary reports:

- `max_abs_diff`
- `cuda_ms`
- `triton_ms`
- `speedup`

## Common failures

- `Could not find a Python interpreter with Triton installed`
  - Set `PEGAINFER_TRITON_PYTHON` to an interpreter where `import triton` succeeds.
- `GPU detection failed`
  - Set `PEGAINFER_CUDA_SM` explicitly if `nvidia-smi` is not available during build.
- `Triton AOT generator failed`
  - Re-run the build and inspect the generator stderr printed by `build.rs`; the generator now accepts an explicit `cuda:<sm>:32` target derived from `PEGAINFER_CUDA_SM`.
- `CUDA_ERROR_NO_BINARY_FOR_GPU` or similar runtime load errors
  - Rebuild on the target GPU environment; the generated Triton cubin is currently treated as target-specific validation output.
