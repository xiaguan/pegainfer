# Profiling & Benchmarking Guide

> **TL;DR:** Three layers of performance visibility: `bench_serving` for application-level metrics, `fastrace` for per-request tracing, `nsys`/`ncu` for GPU kernel analysis. All outputs viewable in [Perfetto UI](https://ui.perfetto.dev).

## bench_serving — Application-Level Benchmarks

In-process benchmark binary. Loads model once, runs generation directly — no HTTP overhead.

```bash
cd pegainfer/
cargo run -r --bin bench_serving -- [global-opts] <subcommand> [subcommand-opts]
```

### Global Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `models/Qwen3-4B` | Model directory |
| `--cuda-graph` | `true` | Enable/disable CUDA Graph on decode path |
| `--format` | `text` | Output format: `text` or `json` |
| `--label` | — | Annotation tag for the report |
| `--out` | — | Write rendered report to file |

### Subcommand: `request`

Measures a single (prompt_len, output_len) shape. The workhorse for A/B comparisons.

```bash
cargo run -r --bin bench_serving -- request --prompt-len 1024 --output-len 256
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | — | Literal prompt text |
| `--prompt-file` | — | Read prompt from file |
| `--prompt-len` | — | Generate synthetic prompt of N tokens (mutually exclusive with above) |
| `--output-len` | 64 | Max tokens to generate |
| `--warmup` | 5 | Warmup iterations (not measured) |
| `--iters` | 20 | Measured iterations |
| `--seed` | 42 | RNG seed for sampling |

**Metrics reported:** TTFT, first decode step, steady TPOT (excludes first step), E2E latency, decode tok/s. Each with min/avg/p50/p95/p99/max.

**Synthetic tokens:** `token_id = 100 + (idx % 1000)` — deterministic, avoids special tokens.

### Subcommand: `matrix`

Sweeps prompt_len x output_len combinations. Good for finding performance cliffs.

```bash
cargo run -r --bin bench_serving -- matrix --prompt-lens 32,128,512,2048 --output-lens 32,128,256
```

Each cell shows TTFT, E2E, TPOT, and throughput.

### Subcommand: `curve`

Shows how TPOT degrades as KV cache context grows. Reveals attention scaling behavior.

```bash
cargo run -r --bin bench_serving -- curve --prompt-len 1024 --output-len 256 --window 32
```

Groups decode positions into windows (e.g., "tokens 1025–1056") and reports per-window TPOT. `--window` controls granularity.

### Typical Comparisons

```bash
# Before/after kernel change
cargo run -r --bin bench_serving -- --format json --out before.json request --prompt-len 512 --output-len 128
# ... make changes, rebuild ...
cargo run -r --bin bench_serving -- --format json --out after.json request --prompt-len 512 --output-len 128

# CUDA Graph on vs off
cargo run -r --bin bench_serving -- --cuda-graph true  request --prompt-len 512
cargo run -r --bin bench_serving -- --cuda-graph false request --prompt-len 512
```

---

## fastrace — Per-Request Tracing

Application-level spans via the [fastrace](https://github.com/fastrace-rs/fastrace) crate. Outputs Chrome Trace Event Format JSON — one file per request.

### Enable

```bash
RUST_LOG=info cargo run --release -- --trace-output-path traces/
```

Creates `traces/{timestamp_ms}_{trace_id}.json` per request.

### What's Traced

- `get_embeddings_batch` — embedding lookup
- `process_all_layers_batch` — all transformer layers
- Generation-level properties: `ttft_ms`, `tpot_avg_ms`, `generated_tokens`, `tok_per_sec`

### View

Open any `traces/*.json` in [Perfetto UI](https://ui.perfetto.dev). Events are sorted by timestamp, with pid=1/tid=1 (single-threaded inference).

---

## nsys — GPU Kernel Profiling

NVIDIA Nsight Systems. Use when you need kernel-level timing — which kernels dominate, launch overhead, memory transfers.

### Quick Profile

```bash
nsys profile -o bench_trace --force-overwrite \
  cargo run -r --bin bench_serving -- request
```

### With CUDA Graph Trace Modes

```bash
# Default: graph replay shown as single GRAPH_TRACE block (low overhead)
nsys profile -o trace cargo run -r --bin bench_serving -- request

# Expand graph internals: see individual kernels inside replayed graph (has profiling overhead)
nsys profile --cuda-graph-trace=node -o trace cargo run -r --bin bench_serving -- request
```

**Important:** `--cuda-graph-trace=node` adds significant overhead that masks the speedup from CUDA Graphs. Use it to understand kernel composition, not to measure absolute performance.

### Automation Script

`profile_data/run_profile.sh` wraps nsys + summary generation + JSON conversion:

```bash
cd profile_data/
./run_profile.sh ../pegainfer ./output                       # basic
./run_profile.sh ../pegainfer ./output --cuda-graph-trace=node  # expanded graph
```

Produces:
- `.nsys-rep` — native nsys report (open with `nsys-ui`)
- kernel summary (`cuda_gpu_kern_sum`)
- CUDA API summary (`cuda_api_sum`)
- `.json` — for Perfetto

### nsys → Perfetto JSON

```bash
# Manual conversion (nsys exports to SQLite, then convert)
nsys export -t sqlite -o trace.sqlite trace.nsys-rep
python3 scripts/nsys2json.py -f trace.sqlite -o trace.json -t kernel
```

**Known issue:** ns→us conversion causes precision loss. Tightly-spaced kernels (e.g., consecutive fused_mlp stages) appear to overlap, causing Perfetto to hide the second kernel. `nsys2json.py` includes a fix that pushes overlapping events forward. See `profile_data/nsys2json-overlap-bug.md` for details.

---

## ncu — Kernel Micro-Profiling

NVIDIA Nsight Compute. Use for deep-diving a single kernel — occupancy, memory throughput, instruction mix.

```bash
# Profile a specific kernel (slow — replays kernel many times)
/usr/local/cuda/bin/ncu \
  --kernel-name "fused_gqa_attention_decode" \
  --launch-count 3 \
  cargo run -r --bin bench_serving -- --cuda-graph false request --output-len 8
```

**Note:** Disable CUDA Graph (`--cuda-graph false`) and use short output lengths. ncu replays each kernel launch multiple times for accurate counters — CUDA Graph replay complicates this.

Key metrics to look at:
- **Memory throughput** — are you bandwidth-bound? (decode kernels usually are)
- **Occupancy** — enough warps to hide latency?
- **Compute throughput** — are ALUs utilized? (prefill GEMM should be high)

---

## Decision Tree

```
"Is it slow?"
  → Run bench_serving request, check TTFT vs TPOT
    → TTFT high → prefill problem → nsys profile with batch GEMM workload
    → TPOT high → decode problem → nsys profile, look at per-kernel time
      → One kernel dominates → ncu that kernel
      → Launch overhead high → check CUDA Graph is enabled
  → Unclear where time goes → fastrace tracing for span-level breakdown
```
