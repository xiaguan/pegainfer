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


def last_token(x):
    return x[0, -1].float().cpu().tolist()


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
    layer_type = "full_attention" if hasattr(layer, "self_attn") else "linear_attention"
    attn_module = layer.self_attn if layer_type == "full_attention" else layer.linear_attn
    captured = {}

    hooks = [
        layer.input_layernorm.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("input_layernorm_last", last_token(out))
        ),
        attn_module.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("attn_out_last", last_token(out[0] if isinstance(out, tuple) else out))
        ),
        layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("post_attention_layernorm_last", last_token(out))
        ),
        layer.mlp.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("mlp_out_last", last_token(out))
        ),
        layer.register_forward_hook(
            lambda _m, _inp, out: captured.__setitem__("layer_out_last", last_token(out[0] if isinstance(out, tuple) else out))
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
        "layer_type": layer_type,
        "input_layernorm_last": captured["input_layernorm_last"],
        "attn_out_last": captured["attn_out_last"],
        "hidden_plus_attn_last": None,
        "post_attention_layernorm_last": captured["post_attention_layernorm_last"],
        "mlp_out_last": captured["mlp_out_last"],
        "layer_out_last": captured["layer_out_last"],
    }
    # hidden_plus_attn is the input to post_attention_layernorm.
    # That input is the module input captured by a pre-hook; easiest is reconstruct from layer_out = residual+mlp_out? skip here.
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
