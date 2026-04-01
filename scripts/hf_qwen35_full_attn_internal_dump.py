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


def flatten_last_token(x: torch.Tensor) -> list[float]:
    return x[0, -1].reshape(-1).float().cpu().tolist()


def split_q_and_gate(q_full_last: list[float], head_dim: int) -> tuple[list[float], list[float]]:
    assert len(q_full_last) % (2 * head_dim) == 0, len(q_full_last)
    q_only = []
    gate_sigmoid = []
    for head_start in range(0, len(q_full_last), 2 * head_dim):
        q_chunk = q_full_last[head_start : head_start + head_dim]
        gate_chunk = q_full_last[head_start + head_dim : head_start + 2 * head_dim]
        q_only.extend(q_chunk)
        gate_sigmoid.extend([1.0 / (1.0 + torch.exp(torch.tensor(-x)).item()) for x in gate_chunk])
    return q_only, gate_sigmoid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--layer-idx", type=int, required=True)
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

    attn = model.model.layers[args.layer_idx].self_attn
    captured: dict[str, list[float]] = {}

    hooks = [
        attn.q_proj.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("q_full_last", flatten_last_token(out))
        ),
        attn.k_proj.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("k_proj_last", flatten_last_token(out))
        ),
        attn.v_proj.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("v_proj_last", flatten_last_token(out))
        ),
        attn.q_norm.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("q_norm_last", flatten_last_token(out))
        ),
        attn.o_proj.register_forward_pre_hook(
            lambda _m, inp: captured.__setitem__("gated_attn_last", flatten_last_token(inp[0]))
        ),
        attn.o_proj.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("o_proj_last", flatten_last_token(out))
        ),
    ]

    input_ids = torch.tensor([token_ids[: args.prompt_len]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)

    for hook in hooks:
        hook.remove()

    q_only_last, gate_sigmoid_last = split_q_and_gate(captured["q_full_last"], attn.head_dim)
    attn_pre_gate_last = [
        0.0 if gate == 0.0 else val / gate
        for val, gate in zip(captured["gated_attn_last"], gate_sigmoid_last)
    ]

    payload = {
        "prompt_len": args.prompt_len,
        "layer_idx": args.layer_idx,
        "q_full_last": captured["q_full_last"],
        "q_only_last": q_only_last,
        "gate_sigmoid_last": gate_sigmoid_last,
        "k_proj_last": captured["k_proj_last"],
        "v_proj_last": captured["v_proj_last"],
        "q_norm_last": captured["q_norm_last"],
        "attn_pre_gate_last": attn_pre_gate_last,
        "gated_attn_last": captured["gated_attn_last"],
        "o_proj_last": captured["o_proj_last"],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
