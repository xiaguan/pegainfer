"""
DSV3 FP8 forward reference: embedding → RMSNorm → q_a_proj.

Loads raw safetensors weights (no HF model class), does manual FP8 dequant
and matmul, dumps reference outputs to .npy for Rust-side comparison.

Usage:
    uv run main.py --model-path /data/models/DeepSeek-V3.2 --output-dir ../../test_data/dsv3_ref
"""

import argparse
import json
import os

import numpy as np
import torch
from safetensors import safe_open


def load_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f)


def load_bf16_tensor(st, name: str) -> torch.Tensor:
    """Load a bf16 tensor from safetensors."""
    return st.get_tensor(name).to(torch.bfloat16)


def load_fp8_weight(st, weight_name: str, scale_name: str) -> torch.Tensor:
    """Load FP8 weight and dequantize to bf16.

    Weight: [N, K] fp8_e4m3fn
    Scale:  [ceil(N/128), ceil(K/128)] float32 (2D block-scale)
    Output: [N, K] bf16
    """
    weight_fp8 = st.get_tensor(weight_name)  # float8_e4m3fn
    scale_inv = st.get_tensor(scale_name)  # float32 [ceil(N/128), ceil(K/128)]

    N, K = weight_fp8.shape
    scale_n, scale_k = scale_inv.shape
    assert scale_n == (N + 127) // 128, f"scale_n={scale_n}, expected {(N + 127) // 128}"
    assert scale_k == (K + 127) // 128, f"scale_k={scale_k}, expected {(K + 127) // 128}"

    # Dequantize: for each 128x128 block, multiply fp8 by corresponding scale
    weight_f32 = weight_fp8.to(torch.float32)
    result = torch.zeros(N, K, dtype=torch.float32)

    for ni in range(scale_n):
        for ki in range(scale_k):
            n_start = ni * 128
            n_end = min(n_start + 128, N)
            k_start = ki * 128
            k_end = min(k_start + 128, K)
            result[n_start:n_end, k_start:k_end] = (
                weight_f32[n_start:n_end, k_start:k_end] * scale_inv[ni, ki]
            )

    return result.to(torch.bfloat16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    x_f32 = x.to(torch.float32)
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms).to(torch.bfloat16) * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 9707, 2991, 553, 374, 264, 1296],  # "Hello! What is a test"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.model_path)

    hidden_size = config["hidden_size"]
    rms_norm_eps = config["rms_norm_eps"]
    print(f"hidden_size={hidden_size}, rms_norm_eps={rms_norm_eps}")

    # Open shard containing embedding + layer0
    shard_path = os.path.join(args.model_path, "model-00001-of-000163.safetensors")
    print(f"Loading shard: {shard_path}")
    st = safe_open(shard_path, framework="pt", device="cpu")

    # --- Embedding ---
    token_ids = torch.tensor(args.tokens, dtype=torch.long)
    embed_weight = load_bf16_tensor(st, "model.embed_tokens.weight")
    print(f"embed_tokens: {embed_weight.shape}")  # [vocab_size, hidden_size]

    embedded = embed_weight[token_ids]  # [seq_len, hidden_size]
    print(f"embedded: {embedded.shape}, first 8: {embedded[0, :8].to(torch.float32).tolist()}")

    def save(name: str, tensor: torch.Tensor):
        """Save as raw f32 little-endian binary (easy to read from Rust)."""
        arr = tensor.to(torch.float32).contiguous().numpy()
        path = os.path.join(args.output_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"  saved {name}.bin  shape={list(arr.shape)}  {os.path.getsize(path)} bytes")

    save("embedded", embedded)

    # --- Input LayerNorm ---
    ln_weight = load_bf16_tensor(st, "model.layers.0.input_layernorm.weight")
    normed = rms_norm(embedded, ln_weight, rms_norm_eps)
    print(f"normed: {normed.shape}, first 8: {normed[0, :8].to(torch.float32).tolist()}")
    save("normed", normed)

    # --- q_a_proj (FP8 dequant → matmul) ---
    print("Loading q_a_proj FP8 weight...")
    q_a_proj_bf16 = load_fp8_weight(
        st,
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
    )
    print(f"q_a_proj dequantized: {q_a_proj_bf16.shape}")  # [1536, 7168]

    # Forward: output = normed @ q_a_proj^T  (i.e., [seq, hidden] @ [hidden, 1536])
    # Use float32 for reference accuracy
    q_a_output = (normed.to(torch.float32) @ q_a_proj_bf16.to(torch.float32).T).to(
        torch.bfloat16
    )
    print(
        f"q_a_output: {q_a_output.shape}, first 8: {q_a_output[0, :8].to(torch.float32).tolist()}"
    )
    save("q_a_output", q_a_output)

    # Save metadata
    meta = {
        "token_ids": args.tokens,
        "hidden_size": hidden_size,
        "q_lora_rank": config["q_lora_rank"],
        "rms_norm_eps": rms_norm_eps,
        "seq_len": len(args.tokens),
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nReference outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
