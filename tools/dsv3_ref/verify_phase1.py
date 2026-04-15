"""
DSV3 Phase 1 reference: full 3-layer forward (MLA + Dense FFN).

Loads raw safetensors (no HF model class), does manual FP8 dequant,
runs the **unabsorbed** MLA forward + Dense FFN for 1 token at position 0.

Mathematically equivalent to our absorbed-path Rust implementation.
For position 0 (single KV entry), attention is trivial: softmax=1.0, output=V.

Usage:
    uv run verify_phase1.py --model-path /data/models/DeepSeek-V3.2 \
        --output-dir ../../test_data/dsv3_phase1
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------

class ShardLoader:
    """Lazy safetensors shard loader using the index file."""

    def __init__(self, model_path: str):
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        self.weight_map = index["weight_map"]
        self.model_path = model_path
        self._cache: dict[str, safe_open] = {}

    def _get_shard(self, shard_name: str):
        if shard_name not in self._cache:
            path = os.path.join(self.model_path, shard_name)
            self._cache[shard_name] = safe_open(path, framework="pt", device="cpu")
        return self._cache[shard_name]

    def load(self, name: str) -> torch.Tensor:
        shard_name = self.weight_map[name]
        st = self._get_shard(shard_name)
        return st.get_tensor(name)

    def close(self):
        self._cache.clear()


def dequant_fp8(weight_fp8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequantize FP8 block-scale weight → bf16.

    weight_fp8: [N, K] float8_e4m3fn
    scale_inv:  [ceil(N/128), ceil(K/128)] float32
    """
    N, K = weight_fp8.shape
    weight_f32 = weight_fp8.to(torch.float32)
    result = torch.zeros(N, K, dtype=torch.float32)
    scale_n, scale_k = scale_inv.shape

    for ni in range(scale_n):
        for ki in range(scale_k):
            n_s, n_e = ni * 128, min((ni + 1) * 128, N)
            k_s, k_e = ki * 128, min((ki + 1) * 128, K)
            result[n_s:n_e, k_s:k_e] = weight_f32[n_s:n_e, k_s:k_e] * scale_inv[ni, ki]

    return result.to(torch.bfloat16)


def load_fp8(loader: ShardLoader, prefix: str) -> torch.Tensor:
    """Load and dequantize an FP8 weight matrix."""
    w = loader.load(f"{prefix}.weight")
    s = loader.load(f"{prefix}.weight_scale_inv")
    return dequant_fp8(w, s)


# ---------------------------------------------------------------------------
# Ops
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.to(torch.float32)).to(torch.bfloat16)


def rms_norm_partial(x: torch.Tensor, weight: torch.Tensor, norm_dim: int, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm only the first `norm_dim` elements, leave the rest unchanged."""
    x_f32 = x.to(torch.float32)
    prefix = x_f32[..., :norm_dim]
    rms = torch.sqrt(prefix.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = (prefix / rms * weight.to(torch.float32)).to(torch.bfloat16)
    return torch.cat([normed, x[..., norm_dim:]], dim=-1)


def linear_f32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Y = X @ W^T in f32 precision, result cast to bf16."""
    return (x.to(torch.float32) @ weight.to(torch.float32).T).to(torch.bfloat16)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE using rotate_half (half-split) convention.

    x:   [..., head_dim]
    cos: [head_dim]  (already duplicated: cos[i] == cos[i + half])
    sin: [head_dim]
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # rotate_half: [-x2, x1]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    return (x.to(torch.float32) * cos.to(torch.float32) + x_rotated.to(torch.float32) * sin.to(torch.float32)).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# YaRN RoPE
# ---------------------------------------------------------------------------

def yarn_get_mscale(scale: float, mscale: float) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_yarn_rope(
    head_dim: int,
    max_pos: int,
    theta: float,
    beta_fast: float,
    beta_slow: float,
    factor: float,
    original_max_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute YaRN RoPE cos/sin cache. Returns [max_pos, head_dim] tensors."""
    half_dim = head_dim // 2

    pos_freqs = [theta ** ((2 * i) / head_dim) for i in range(half_dim)]
    inv_freq_extra = [1.0 / f for f in pos_freqs]
    inv_freq_inter = [1.0 / (factor * f) for f in pos_freqs]

    def find_correction_dim(num_rot: float) -> float:
        num = head_dim * math.log(original_max_pos / (num_rot * 2.0 * math.pi))
        den = 2.0 * math.log(theta)
        return num / den

    low = max(0, int(math.floor(find_correction_dim(beta_fast))))
    high = min(head_dim - 1, int(math.ceil(find_correction_dim(beta_slow))))

    inv_freq = []
    for i in range(half_dim):
        if low == high:
            ramp = 0.0 if i < low else 1.0
        else:
            ramp = max(0.0, min(1.0, (i - low) / (high - low)))
        inv_freq.append(inv_freq_inter[i] * ramp + inv_freq_extra[i] * (1.0 - ramp))

    cos_cache = torch.zeros(max_pos, head_dim, dtype=torch.bfloat16)
    sin_cache = torch.zeros(max_pos, head_dim, dtype=torch.bfloat16)
    for pos in range(max_pos):
        for i in range(half_dim):
            freq = pos * inv_freq[i]
            c = math.cos(freq)
            s = math.sin(freq)
            cos_cache[pos, i] = c
            cos_cache[pos, i + half_dim] = c
            sin_cache[pos, i] = s
            sin_cache[pos, i + half_dim] = s
    return cos_cache, sin_cache


# ---------------------------------------------------------------------------
# MLA forward (unabsorbed path, single token)
# ---------------------------------------------------------------------------

def mla_forward_unabsorbed(
    hidden: torch.Tensor,  # [1, hidden_size] bf16
    layer_weights: dict,
    config: dict,
    cos: torch.Tensor,  # [head_dim] for this position
    sin: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """One layer of MLA + Dense FFN. Returns (new_hidden, intermediates)."""
    eps = config["rms_norm_eps"]
    num_heads = config["num_attention_heads"]
    nope_dim = config["qk_nope_head_dim"]
    rope_dim = config["qk_rope_head_dim"]
    kv_lora = config["kv_lora_rank"]
    v_dim = config["v_head_dim"]
    q_head_dim = nope_dim + rope_dim  # 192

    intermediates = {}

    # --- Input LayerNorm ---
    normed = rms_norm(hidden, layer_weights["input_layernorm"], eps)
    intermediates["normed"] = normed

    # --- Q Path ---
    q_compressed = linear_f32(normed, layer_weights["q_a_proj"])  # [1, q_lora_rank]
    q_normed = rms_norm(q_compressed, layer_weights["q_a_layernorm"], eps)
    q_full = linear_f32(q_normed, layer_weights["q_b_proj"])  # [1, num_heads * q_head_dim]
    q_full_reshaped = q_full.reshape(num_heads, q_head_dim)  # [128, 192]

    q_nope = q_full_reshaped[:, :nope_dim]  # [128, 128]
    q_rope = q_full_reshaped[:, nope_dim:]  # [128, 64]

    # RoPE on q_rope — cos/sin are for rope_dim=64
    q_rope = apply_rope(q_rope, cos, sin)

    intermediates["q_full"] = q_full

    # --- KV Path ---
    kv_a = linear_f32(normed, layer_weights["kv_a_proj"])  # [1, kv_lora + rope_dim]
    c_kv = kv_a[:, :kv_lora]  # [1, 512]
    k_rope_raw = kv_a[:, kv_lora:]  # [1, 64]

    c_kv_normed = rms_norm(c_kv, layer_weights["kv_a_layernorm"], eps)  # [1, 512]
    k_rope = apply_rope(k_rope_raw, cos, sin)  # [1, 64]

    # Expand through kv_b_proj to get per-head K_nope and V
    kv_b = linear_f32(c_kv_normed, layer_weights["kv_b_proj"])  # [1, num_heads * (nope + v)]
    kv_b_reshaped = kv_b.reshape(num_heads, nope_dim + v_dim)  # [128, 256]

    k_nope = kv_b_reshaped[:, :nope_dim]  # [128, 128]
    v = kv_b_reshaped[:, nope_dim:]  # [128, 128]

    # Full K: [k_nope, k_rope_broadcast]
    k_rope_broadcast = k_rope.expand(num_heads, -1)  # [128, 64]
    k_full = torch.cat([k_nope, k_rope_broadcast], dim=-1)  # [128, 192]

    # Q full: [q_nope, q_rope]
    q = torch.cat([q_nope, q_rope], dim=-1)  # [128, 192]

    # --- Attention (single token → trivial) ---
    # score_h = q_h @ k_h^T → scalar per head
    # softmax of single element = 1.0
    # attn_out_h = 1.0 * v_h = v_h
    softmax_scale = (q_head_dim ** -0.5) * config["softmax_mscale"]
    scores = (q.to(torch.float32) * k_full.to(torch.float32)).sum(dim=-1) * softmax_scale  # [128]
    # softmax of [scalar] = 1.0 for each head
    attn_out = v  # [128, v_dim=128]

    intermediates["attn_out_per_head"] = attn_out  # [128, 128]

    # --- O Projection + Residual ---
    attn_out_flat = attn_out.reshape(1, num_heads * v_dim)  # [1, 16384]
    o_out = linear_f32(attn_out_flat, layer_weights["o_proj"])  # [1, hidden_size]
    hidden = (hidden.to(torch.float32) + o_out.to(torch.float32)).to(torch.bfloat16)

    intermediates["hidden_after_attn"] = hidden

    # --- Post-attention LayerNorm ---
    normed2 = rms_norm(hidden, layer_weights["post_attention_layernorm"], eps)

    # --- Dense FFN ---
    gate = linear_f32(normed2, layer_weights["gate_proj"])
    up = linear_f32(normed2, layer_weights["up_proj"])
    act = torch.nn.functional.silu(gate.to(torch.float32)).to(torch.bfloat16) * up
    ffn_out = linear_f32(act, layer_weights["down_proj"])
    hidden = (hidden.to(torch.float32) + ffn_out.to(torch.float32)).to(torch.bfloat16)

    intermediates["hidden_after_ffn"] = hidden

    return hidden, intermediates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="../../test_data/dsv3_phase1")
    parser.add_argument("--token-id", type=int, default=1, help="Single token ID (default: BOS=1)")
    parser.add_argument("--position", type=int, default=0, help="Token position for RoPE")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of dense layers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load config ---
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)

    rope_scaling = config.get("rope_scaling", {})
    mscale = yarn_get_mscale(rope_scaling.get("factor", 1.0), rope_scaling.get("mscale", 0.0))
    mscale_all_dim = yarn_get_mscale(rope_scaling.get("factor", 1.0), rope_scaling.get("mscale_all_dim", 0.0))
    softmax_mscale = mscale / mscale_all_dim if mscale_all_dim > 0 else mscale
    config["softmax_mscale"] = softmax_mscale

    rope_dim = config["qk_rope_head_dim"]  # 64
    print(f"Config: hidden={config['hidden_size']}, heads={config['num_attention_heads']}, "
          f"rope_dim={rope_dim}, softmax_mscale={softmax_mscale:.6f}")

    # --- Precompute YaRN RoPE ---
    cos_cache, sin_cache = precompute_yarn_rope(
        head_dim=rope_dim,
        max_pos=args.position + 1,
        theta=config["rope_theta"],
        beta_fast=rope_scaling.get("beta_fast", 32.0),
        beta_slow=rope_scaling.get("beta_slow", 1.0),
        factor=rope_scaling.get("factor", 1.0),
        original_max_pos=rope_scaling.get("original_max_position_embeddings",
                                          config["max_position_embeddings"]),
    )
    cos_pos = cos_cache[args.position]  # [rope_dim]
    sin_pos = sin_cache[args.position]

    # --- Load weights ---
    loader = ShardLoader(args.model_path)
    print("Loading embedding...")
    embed_weight = loader.load("model.embed_tokens.weight").to(torch.bfloat16)

    # Embedding
    token_id = args.token_id
    hidden = embed_weight[token_id].unsqueeze(0)  # [1, hidden_size]
    print(f"Token {token_id} embedded: first 4 = {hidden[0, :4].to(torch.float32).tolist()}")

    def save(name: str, tensor: torch.Tensor):
        arr = tensor.to(torch.float32).contiguous().numpy()
        path = os.path.join(args.output_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"  saved {name}.bin  shape={list(arr.shape)}  {os.path.getsize(path)} bytes")

    save("embedded", hidden)

    # --- Forward through layers ---
    for layer_idx in range(args.num_layers):
        prefix = f"model.layers.{layer_idx}"
        print(f"\n=== Layer {layer_idx} ===")

        print(f"  Loading weights...")
        layer_weights = {
            "input_layernorm": loader.load(f"{prefix}.input_layernorm.weight").to(torch.bfloat16),
            "q_a_proj": load_fp8(loader, f"{prefix}.self_attn.q_a_proj"),
            "q_a_layernorm": loader.load(f"{prefix}.self_attn.q_a_layernorm.weight").to(torch.bfloat16),
            "q_b_proj": load_fp8(loader, f"{prefix}.self_attn.q_b_proj"),
            "kv_a_proj": load_fp8(loader, f"{prefix}.self_attn.kv_a_proj_with_mqa"),
            "kv_a_layernorm": loader.load(f"{prefix}.self_attn.kv_a_layernorm.weight").to(torch.bfloat16),
            "kv_b_proj": load_fp8(loader, f"{prefix}.self_attn.kv_b_proj"),
            "o_proj": load_fp8(loader, f"{prefix}.self_attn.o_proj"),
            "post_attention_layernorm": loader.load(f"{prefix}.post_attention_layernorm.weight").to(torch.bfloat16),
            "gate_proj": load_fp8(loader, f"{prefix}.mlp.gate_proj"),
            "up_proj": load_fp8(loader, f"{prefix}.mlp.up_proj"),
            "down_proj": load_fp8(loader, f"{prefix}.mlp.down_proj"),
        }
        print(f"  Weights loaded. q_a_proj={layer_weights['q_a_proj'].shape}, "
              f"kv_b_proj={layer_weights['kv_b_proj'].shape}")

        hidden, intermediates = mla_forward_unabsorbed(
            hidden, layer_weights, config, cos_pos, sin_pos,
        )

        save(f"layer{layer_idx}_hidden", hidden)
        save(f"layer{layer_idx}_hidden_after_attn", intermediates["hidden_after_attn"])

        print(f"  Hidden after layer {layer_idx}: first 4 = {hidden[0, :4].to(torch.float32).tolist()}")

    # Save metadata
    meta = {
        "token_id": args.token_id,
        "position": args.position,
        "num_layers": args.num_layers,
        "hidden_size": config["hidden_size"],
        "num_attention_heads": config["num_attention_heads"],
        "q_lora_rank": config["q_lora_rank"],
        "kv_lora_rank": config["kv_lora_rank"],
        "qk_nope_head_dim": config["qk_nope_head_dim"],
        "qk_rope_head_dim": config["qk_rope_head_dim"],
        "v_head_dim": config["v_head_dim"],
        "rms_norm_eps": config["rms_norm_eps"],
        "softmax_mscale": softmax_mscale,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPhase 1 reference saved to {args.output_dir}/")
    print("Files: embedded.bin, layer{{0,1,2}}_hidden.bin, layer{{0,1,2}}_hidden_after_attn.bin, meta.json")


if __name__ == "__main__":
    main()
