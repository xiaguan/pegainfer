# pegainfer vs vLLM Comparative Benchmarking

> **TL;DR:** Run both engines on the same GPU sequentially, benchmark with `vllm bench serve` as a unified client, compare TTFT/TPOT/throughput side by side.

## Method

Both servers are tested on the **same GPU, same model, same port** (sequentially — one GPU means no parallel serving). `vllm bench serve` drives both, ensuring identical client behavior and measurement methodology.

Key flags applied to both: `--ignore-eos` (forces exact output length), `--dataset-name random` (synthetic prompts).

## Setup

```
MODEL_PATH=models/Qwen3-4B
VLLM_PYTHON=.venv/bin/python      # pegainfer/.venv with vllm installed
VLLM_CMD=.venv/bin/vllm
PORT=8000
RESULTS_DIR=bench_results/<timestamp>
```

Prerequisites:
- `cargo build --release` — pegainfer up-to-date
- vLLM installed in `.venv` (`uv pip install vllm`)
- Kill any existing process on the port before starting

## Workflow

### 1. Benchmark vLLM

Start server, wait for ready, run benchmarks, kill server:

```bash
# Start
$VLLM_CMD serve $MODEL_PATH --port $PORT --max-model-len 4096 --gpu-memory-utilization 0.9 &

# Poll until ready (up to 120s — vLLM has torch.compile cold start)
curl -s http://localhost:$PORT/v1/models

# Benchmark (repeat for each config)
$VLLM_CMD bench serve \
  --backend openai --model <model_name> --port $PORT \
  --dataset-name random --input-len <in> --output-len <out> \
  --num-prompts <n> --request-rate inf --max-concurrency 1 \
  --ignore-eos --tokenizer $MODEL_PATH \
  --save-result --result-dir $RESULTS_DIR \
  --result-filename vllm-in<in>-out<out>.json

# Cleanup
pkill -f "vllm serve"
```

### 2. Benchmark pegainfer

Same flow, different server:

```bash
# Start
RUST_LOG=warn cargo run --release -- --port $PORT &

# Poll until ready (up to 60s — no torch.compile, much faster startup)

# Benchmark (same vllm bench serve client)
$VLLM_CMD bench serve \
  --backend openai --model Qwen3-4B --port $PORT \
  --dataset-name random --input-len <in> --output-len <out> \
  --num-prompts <n> --request-rate inf --max-concurrency 1 \
  --ignore-eos --tokenizer $MODEL_PATH \
  --save-result --result-dir $RESULTS_DIR \
  --result-filename pega-in<in>-out<out>.json

# Cleanup
pkill -f "target/release/pegainfer"
```

### 3. Compare

Read both JSON results. Key metrics:

| Metric | What it measures |
|--------|-----------------|
| TTFT (mean/median/p99) | Prefill speed. vLLM's mean >> median indicates torch.compile cold start |
| TPOT (mean/median/p99) | Per-token decode latency |
| ITL (mean/median/p99) | Inter-token latency (includes scheduling jitter) |
| Output tok/s | End-to-end decode throughput |
| Request throughput | Requests completed per second |

## Typical Configs

| Config | What it tests |
|--------|---------------|
| `in=1, out=1, n=200` | Minimal overhead — launch latency, framework tax |
| `in=1024, out=256, n=20` | Realistic workload — prefill + sustained decode |
| `in=1, out=512, n=20` | Pure decode — long generation from short prompt |
| `in=2048, out=32, n=20` | Prefill-heavy — TTFT dominates |

## Gotchas

- **vLLM cold start:** First request triggers torch.compile. Mean TTFT >> Median TTFT is expected. Note this in reports.
- **pegainfer empty prompts:** Random dataset may produce empty prompts which pegainfer rejects. Check failed request count.
- **Zombie processes:** Always `pkill` after benchmarking. Leftover servers block the port and hold GPU memory.
- **CUDA Graph:** pegainfer enables CUDA Graph by default. For apples-to-apples decode comparison, note this in the report. vLLM also uses CUDA Graph by default.
- **Concurrency=1:** Default is single-request sequential. pegainfer has no batching yet, so concurrency>1 would just queue on the server side.
