# DSV3.2 vLLM Ground Truth Generator

This folder contains the Phase A tooling for generating DSV3.2 logits ground truth
used by the new integration test.

## Environment

Use the workspace Python environment at `pegainfer/.venv`.
Do not create another virtual environment for this tool.

```bash
cd /data/code/workspace-rustllm/pegainfer
.venv/bin/python -c "import vllm, transformers; print('vllm', vllm.__version__)"
```

## Inputs

- Model path (for example `/data/models/DeepSeek-V3.2`)
- Prompt set JSON (`tools/dsv32_vllm_ref/prompts.json`)

## Output

- `test_data/dsv32_vllm_logits_ref/manifest.json`

Each case includes:

- `name`
- `prompt`
- `token_ids`
- `positions`
- `generated_token_id`
- `top10_ids`
- `top10_logprobs`

## Run

```bash
cd /data/code/workspace-rustllm/pegainfer
.venv/bin/python tools/dsv32_vllm_ref/gen_logits_ref.py \
  --model-path /data/models/DeepSeek-V3.2 \
  --output-dir test_data/dsv32_vllm_logits_ref \
  --prompts-file tools/dsv32_vllm_ref/prompts.json \
  --tensor-parallel-size 8
```

## Prompt File Format

`prompts.json` supports either:

```json
{
  "cases": [
    {"name": "short_en", "prompt": "The capital of France is"},
    {"name": "math", "prompt": "What is 17 + 25?"}
  ]
}
```

or:

```json
[
  {"name": "short_en", "prompt": "The capital of France is"},
  {"name": "math", "prompt": "What is 17 + 25?"}
]
```
