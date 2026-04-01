#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def load_tokens(path: Path) -> list[int]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [int(x) for x in data]
    return [int(x) for x in data["token_ids"]]


def last_token(x: torch.Tensor) -> list[float]:
    return x[0, -1].reshape(-1).float().cpu().tolist()


def last_rows_flat(x: torch.Tensor, rows: int) -> list[float]:
    return x[-rows:].reshape(-1).float().cpu().tolist()


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

    layer = model.model.layers[args.layer_idx]
    attn = layer.linear_attn
    captured: dict[str, list[float]] = {}

    hooks = [
        layer.input_layernorm.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("input_layernorm_last", last_token(out))
        ),
        attn.in_proj_qkv.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("qkv_proj_last", last_token(out))
        ),
        attn.in_proj_z.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("z_proj_last", last_token(out))
        ),
        attn.in_proj_b.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("b_proj_last", last_token(out))
        ),
        attn.in_proj_a.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("a_proj_last", last_token(out))
        ),
        attn.conv1d.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__(
                "conv1d_last",
                last_token(F.silu(out[:, :, : args.prompt_len]).transpose(1, 2)),
            )
        ),
        attn.norm.register_forward_pre_hook(
            lambda _m, inp: captured.__setitem__(
                "gdr_out_last", last_rows_flat(inp[0], attn.num_v_heads)
            )
        ),
        attn.norm.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__(
                "normed_out_last", last_rows_flat(out, attn.num_v_heads)
            )
        ),
        attn.out_proj.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("out_proj_last", last_token(out))
        ),
    ]

    input_ids = torch.tensor([token_ids[: args.prompt_len]], device="cuda", dtype=torch.long)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)

    for hook in hooks:
        hook.remove()

    payload = {
        "prompt_len": args.prompt_len,
        "layer_idx": args.layer_idx,
        "layer_type": "linear_attention",
        "input_layernorm_last": captured["input_layernorm_last"],
        "qkv_proj_last": captured["qkv_proj_last"],
        "z_proj_last": captured["z_proj_last"],
        "b_proj_last": captured["b_proj_last"],
        "a_proj_last": captured["a_proj_last"],
        "conv1d_last": captured["conv1d_last"],
        "gdr_out_last": captured["gdr_out_last"],
        "normed_out_last": captured["normed_out_last"],
        "out_proj_last": captured["out_proj_last"],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
