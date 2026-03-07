#!/usr/bin/env python3
"""Generate greedy reference outputs for Qwen3.5-4B using HuggingFace transformers."""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEST_PROMPTS = [
    "Hello",
    "The capital of France is",
    "What is 2+2?",
    "Write a Python function to check if a number is prime.",
    "Explain quantum computing in simple terms.",
    "The quick brown fox",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--name", type=str, required=True, help="Model name for output file")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt!r}")
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,  # greedy
                temperature=None,
                top_p=None,
            )

        output_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Output: {output_text!r}")

        results.append({
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "output": output_text,
            "full_text": full_text,
            "prompt_tokens": input_ids.tolist()[0],
            "output_tokens": output[0][prompt_len:].tolist(),
        })

    output_path = f"test_data/{args.name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} test cases to {output_path}")


if __name__ == "__main__":
    main()
