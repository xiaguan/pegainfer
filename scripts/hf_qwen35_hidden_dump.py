#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def load_tokens(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [int(x) for x in data]
    return [int(x) for x in data["token_ids"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    token_ids = load_tokens(Path(args.tokens_file))
    assert 0 < args.prompt_len <= len(token_ids), args.prompt_len

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    input_ids = torch.tensor([token_ids[: args.prompt_len]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    payload = {
        "prompt_len": args.prompt_len,
        "embedding_last": hidden_states[0][0, -1].float().cpu().tolist(),
        "layers_last": [h[0, -1].float().cpu().tolist() for h in hidden_states[1:]],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
