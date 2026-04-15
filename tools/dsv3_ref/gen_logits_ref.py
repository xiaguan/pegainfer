"""
Generate reference logits for DSV3.2 full model forward.

Uses vLLM offline inference to get output logits for multiple prompts.
Saves top-K logit values + indices and full logits for the last token
position of each prompt.

Usage:
    /root/develop/xingming/vllm_test/.venv/bin/python gen_logits_ref.py \
        --model-path /data/models/DeepSeek-V3.2 \
        --output-dir ../../test_data/dsv3_logits_ref
"""

import argparse
import json
import os

import torch
from vllm import LLM, SamplingParams

PROMPTS = [
    "Hello",
    "The capital of France is",
    "1 + 1 =",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="../../test_data/dsv3_logits_ref")
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model via vLLM from {args.model_path} (TP={args.tp})...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    # We need logits — use prompt_logprobs to get them.
    # To get the logits for the last token position, we use a trick:
    # append a dummy token and request prompt_logprobs, which gives us
    # logits at every input position. Or we can use the logprobs on the
    # generated output token.
    #
    # Simplest approach: generate 1 token with logprobs for the full vocab.
    # vLLM SamplingParams supports logprobs=N to return top-N logprobs.
    # We want the full vocab — use a large number.

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=20,  # top-20 logprobs for the generated token
    )

    results = llm.generate(PROMPTS, sampling_params)

    manifest = {"prompts": []}

    for i, (prompt, result) in enumerate(zip(PROMPTS, results)):
        subdir = os.path.join(args.output_dir, f"prompt_{i}")
        os.makedirs(subdir, exist_ok=True)

        token_ids = list(result.prompt_token_ids)
        output = result.outputs[0]
        generated_token_id = output.token_ids[0]

        # Get logprobs for the generated (first output) token position.
        # This is the model's prediction at the last input token position.
        logprobs_dict = output.logprobs[0]  # dict: token_id -> Logprob

        # Extract top-K from logprobs
        sorted_logprobs = sorted(
            logprobs_dict.items(), key=lambda x: x[1].logprob, reverse=True
        )

        top10_ids = [int(tid) for tid, _ in sorted_logprobs[:10]]
        top10_logprobs = [lp.logprob for _, lp in sorted_logprobs[:10]]

        print(f"\nPrompt: {prompt!r}")
        print(f"  Token IDs ({len(token_ids)}): {token_ids}")
        print(f"  Generated: id={generated_token_id} {tokenizer.decode([generated_token_id])!r}")
        print(f"  Top-5:")
        for j in range(min(5, len(sorted_logprobs))):
            tid, lp = sorted_logprobs[j]
            tok = tokenizer.decode([int(tid)])
            print(f"    {j}: id={int(tid):6d}  logprob={lp.logprob:+.4f}  {tok!r}")

        meta = {
            "prompt": prompt,
            "token_ids": token_ids,
            "vocab_size": tokenizer.vocab_size,
            "generated_token_id": int(generated_token_id),
            "top10_ids": top10_ids,
            "top10_logprobs": top10_logprobs,
        }
        with open(os.path.join(subdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        manifest["prompts"].append({
            "index": i,
            "dir": f"prompt_{i}",
            **meta,
        })

        print(f"  Saved to {subdir}/")

    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{len(PROMPTS)} prompts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
