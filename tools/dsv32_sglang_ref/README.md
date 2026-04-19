# DSV3.2 SGLang Ground Truth

Generates a **teacher-forced top-K logprob** ground truth manifest for DSV3.2 regression. The manifest is the only source of truth for `tests/e2e_dsv32.rs`; it carries enough signal to catch both argmax drift and sub-argmax numerical drift along the greedy trajectory.

## Why top-K logprobs instead of text

Token/text-level comparisons collapse the distribution to one bit (argmax match or not). A kernel change that shifts a logit by 0.02 can flip an argmax, and once the trajectories diverge, all downstream positions compare different prefixes. Top-20 logprobs keep the full precision signal while bounding manifest size (~MB per 44-case run).

## Environment

Install sglang into the workspace virtualenv:

```bash
cd /data/code/workspace-rustllm/pegainfer
uv pip install -p .venv/bin/python -e ../sglang/python
```

## Generate manifest (on an 8×H20 node)

```bash
cd /data/code/workspace-rustllm/pegainfer
.venv/bin/python tools/dsv32_sglang_ref/gen_ref.py \
  --model-path /data/models/DeepSeek-V3.2 \
  --output-dir test_data/dsv32_sglang_ref \
  --prompts-file tools/dsv32_sglang_ref/prompts_generation.json \
  --tensor-parallel-size 8 \
  --top-k 20 \
  --seed 42
```

The prompt set lives in `tools/dsv32_sglang_ref/prompts_generation.json` and is shared by both the large teacher-forced harness and the small checked-in greedy JSON regression.

## Manifest schema (`dsv32_sglang_ref.v1`)

```json
{
  "schema_version": "dsv32_sglang_ref.v1",
  "engine": "sglang",
  "meta": {
    "top_k": 20, "seed": 42, "tensor_parallel_size": 8,
    "argmax_mismatches_total": 73,
    "total_output_positions": 6888,
    "..."
  },
  "cases": [{
    "name": "...",
    "prompt": "...",
    "max_new_tokens": 128,
    "prompt_token_ids": [...],
    "generated_token_ids": [...],
    "generated_token_logprobs": [...],     // logprob of sampled token at each position
    "generated_text": "...",
    "finish_reason": "stop" | "length" | null,
    "argmax_mismatches": 1,                // count for this case
    "output_top_logprobs": [
      [[token_id, logprob], ...],          // length = top_k, sorted desc by logprob
      ...
    ]
  }]
}
```

Two reference points per output position:
- `output_top_logprobs[i][0][0]` — logit-space argmax (what the distribution says is most likely)
- `generated_token_ids[i]` — token sglang's FlashInfer sampler actually picked at `temperature=0, top_k=1`

These almost always match, but can disagree on near-ties (softmax-in-FP32 resolves ties differently from logit-space sort). `argmax_mismatches_total` in the meta reports how many positions disagreed during manifest generation — on the 2026-04-19 baseline, 73 out of 6888 positions (1.06%).

`generated_token_logprobs[i]` gives the logprob of the actually-sampled token even when it falls outside the top-K list.

## Run the pegainfer-side regression

```bash
PEGAINFER_DSV32_MODEL_PATH=/data/models/DeepSeek-V3.2 \
PEGAINFER_DSV32_SGLANG_REF_MANIFEST=test_data/dsv32_sglang_ref/manifest.json \
PEGAINFER_DSV32_DEVICE_ORDINALS=0,1,2,3,4,5,6,7 \
cargo test --release --test e2e_dsv32 -- --ignored --nocapture
```

Useful filters (same envs the logits test already uses):

```bash
PEGAINFER_DSV32_CASE_FILTER=music_ PEGAINFER_DSV32_MAX_CASES=4 \
  cargo test --release --test e2e_dsv32 -- --ignored --nocapture
```

The test is **reporting-only by default** — it prints per-case and aggregate statistics (argmax mismatches, max/mean/p50/p99 |Δlogprob|, top-K overlap) and exits successfully. To enforce thresholds once the numeric behavior is understood:

- `PEGAINFER_DSV32_LOGPROB_MAX_ABS=0.05` — fail if any |Δlogprob| across all positions exceeds this
- `PEGAINFER_DSV32_ARGMAX_MISMATCHES=0` — fail if more than this many positions disagree with the manifest's argmax

## How the regression works

For each case pegainfer does a single prefill on `prompt_token_ids`, then steps decode over `generated_token_ids` (teacher forcing — we feed the sglang-chosen token at every step, not pegainfer's own argmax). At every position we compare pegainfer's top-K against the manifest's top-K. Because the input at each step is fixed by the manifest, divergence cannot compound: every position is an independent measurement of the model's distribution.

This exercises the real decode kernel path (not just prefill), so it catches both prefill and decode numerical regressions. The generated `manifest.json` is intentionally local-only and should stay out of git.

## Cross-machine recheck

Generate the manifest on one 8×H20 node, copy it to a different 8×H20 node, and run `e2e_dsv32` there against the same local file. Same argmax and same logprob distribution across nodes means the regression is reproducible.
