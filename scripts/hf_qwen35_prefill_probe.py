#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokens(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [int(x) for x in data]
    return [int(x) for x in data["token_ids"]]


def decode_one(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"<decode_error:{token_id}>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--lengths", required=True, help="comma-separated prompt lengths")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out")
    args = parser.parse_args()

    token_ids = load_tokens(Path(args.tokens_file))
    lengths = [int(x) for x in args.lengths.split(",") if x]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    with torch.no_grad():
        for prompt_len in lengths:
            assert 0 < prompt_len <= len(token_ids), prompt_len
            input_ids = torch.tensor([token_ids[:prompt_len]], device="cuda", dtype=torch.long)
            logits = model(input_ids).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
            top_logprobs, top_ids = torch.topk(logprobs, k=args.top_k)
            generated_token = int(top_ids[0].item())
            results.append(
                {
                    "prompt_len": prompt_len,
                    "last_prompt_token": int(token_ids[prompt_len - 1]),
                    "generated_token": generated_token,
                    "generated_text": decode_one(tokenizer, generated_token),
                    "generated_logprob": float(top_logprobs[0].item()),
                    "top_logprobs": [
                        {
                            "id": int(tok.item()),
                            "text": decode_one(tokenizer, int(tok.item())),
                            "logprob": float(lp.item()),
                        }
                        for tok, lp in zip(top_ids, top_logprobs, strict=True)
                    ],
                }
            )

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
