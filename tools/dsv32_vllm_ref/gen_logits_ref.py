#!/usr/bin/env python3
"""Generate vLLM logits/top-k ground truth for DSV3.2 integration tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_logprob(value: Any) -> float | None:
    if value is None:
        return None
    logprob = _to_float(value)
    if logprob is not None:
        return logprob
    if isinstance(value, dict):
        return _to_float(value.get("logprob"))
    obj_logprob = getattr(value, "logprob", None)
    return _to_float(obj_logprob)


def _extract_top_logprobs(first_step_logprobs: Any) -> list[tuple[int, float]]:
    entries: list[tuple[int, float]] = []
    if first_step_logprobs is None:
        return entries

    if isinstance(first_step_logprobs, list):
        for item in first_step_logprobs:
            if isinstance(item, dict):
                token_id = _to_int(item.get("token_id"))
                if token_id is None:
                    token_id = _to_int(item.get("token"))
                logprob = _extract_logprob(item.get("logprob"))
            else:
                token_id = _to_int(getattr(item, "token_id", None))
                logprob = _extract_logprob(getattr(item, "logprob", None))
            if token_id is not None and logprob is not None:
                entries.append((token_id, logprob))
        entries.sort(key=lambda x: x[1], reverse=True)
        return entries

    if isinstance(first_step_logprobs, dict):
        for key, value in first_step_logprobs.items():
            token_id = _to_int(key)
            if token_id is None:
                token_id = _to_int(getattr(value, "token_id", None))
            logprob = _extract_logprob(value)
            if token_id is not None and logprob is not None:
                entries.append((token_id, logprob))

        if not entries:
            for _, value in first_step_logprobs.items():
                if isinstance(value, dict):
                    token_id = _to_int(value.get("token_id"))
                    if token_id is None:
                        token_id = _to_int(value.get("token"))
                    logprob = _extract_logprob(value.get("logprob"))
                else:
                    token_id = _to_int(getattr(value, "token_id", None))
                    logprob = _extract_logprob(getattr(value, "logprob", None))
                if token_id is not None and logprob is not None:
                    entries.append((token_id, logprob))

    entries.sort(key=lambda x: x[1], reverse=True)
    return entries


def _extract_prompt_token_ids(request_output: Any) -> list[int]:
    raw = getattr(request_output, "prompt_token_ids", None)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    token_ids: list[int] = []
    for item in raw:
        token_id = _to_int(item)
        if token_id is None:
            return []
        token_ids.append(token_id)
    return token_ids


def _load_prompt_cases(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw_cases = raw.get("cases")
    else:
        raw_cases = raw

    if not isinstance(raw_cases, list):
        raise ValueError("prompts file must be a list or an object with a `cases` list")

    cases: list[dict[str, str]] = []
    for idx, item in enumerate(raw_cases):
        if isinstance(item, str):
            cases.append({"name": f"case_{idx:02d}", "prompt": item})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"case #{idx} must be string or object")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"case #{idx} missing non-empty `prompt`")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            name = f"case_{idx:02d}"
        cases.append({"name": name, "prompt": prompt})
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model path (for example /data/models/DeepSeek-V3.2)",
    )
    parser.add_argument(
        "--output-dir",
        default="test_data/dsv32_vllm_logits_ref",
        help="Output directory for manifest.json",
    )
    parser.add_argument(
        "--prompts-file",
        default="tools/dsv32_vllm_ref/prompts.json",
        help="Prompt cases JSON path",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="vLLM tensor_parallel_size",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max_model_len",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="bfloat16",
        help="vLLM dtype",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K candidates to keep from logprobs (must be >= 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Recorded in metadata for reproducibility",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Enable trust_remote_code for tokenizer/model loading",
    )
    parser.add_argument(
        "--disable-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Ensure subprocesses launched by vLLM can resolve tools installed in the
    # same workspace virtualenv (for example `ninja`).
    path_candidates = [
        str((Path(sys.prefix) / "bin").resolve()),
        str(Path(sys.executable).parent),
    ]
    old_path = os.environ.get("PATH", "")
    for path_dir in path_candidates:
        if path_dir and path_dir not in old_path.split(":"):
            old_path = f"{path_dir}:{old_path}"
    os.environ["PATH"] = old_path

    trust_remote_code = args.trust_remote_code and not args.disable_trust_remote_code

    if args.topk < 10:
        raise ValueError("--topk must be >= 10 so `top10_ids` can be produced")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")

    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.json"

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        print(
            "Missing dependency in workspace .venv. Check with:\n"
            "  .venv/bin/python -c \"import vllm, transformers\"",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    cases = _load_prompt_cases(prompts_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=trust_remote_code
    )

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=trust_remote_code,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=args.topk,
    )

    manifest_cases: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        name = case["name"]
        prompt = case["prompt"]
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        if not outputs:
            raise RuntimeError(f"case `{name}` returned empty vLLM outputs")

        req_out = outputs[0]
        prompt_token_ids = _extract_prompt_token_ids(req_out)
        prompt_token_ids_source = "vllm_request_output"
        if not prompt_token_ids:
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_token_ids_source = "hf_encode_no_special_fallback"
        if not prompt_token_ids:
            raise ValueError(f"case `{name}` produced empty token_ids")
        if not req_out.outputs:
            raise RuntimeError(f"case `{name}` has no completion outputs")

        first_out = req_out.outputs[0]
        token_ids = [int(tid) for tid in list(first_out.token_ids)]
        if not token_ids:
            raise RuntimeError(f"case `{name}` generated no tokens")
        generated_token_id = token_ids[0]

        first_step_logprobs = None
        out_logprobs = getattr(first_out, "logprobs", None)
        if isinstance(out_logprobs, list) and out_logprobs:
            first_step_logprobs = out_logprobs[0]

        top_entries = _extract_top_logprobs(first_step_logprobs)
        if not top_entries:
            top_entries = [(generated_token_id, 0.0)]

        top10 = top_entries[:10]
        top10_ids = [token_id for token_id, _ in top10]
        top10_logprobs = [logprob for _, logprob in top10]
        if generated_token_id not in top10_ids:
            top10_ids = [generated_token_id] + top10_ids[:9]

        manifest_cases.append(
            {
                "name": name,
                "prompt": prompt,
                "token_ids": prompt_token_ids,
                "positions": list(range(len(prompt_token_ids))),
                "prompt_token_ids_source": prompt_token_ids_source,
                "generated_token_id": generated_token_id,
                "generated_text": getattr(first_out, "text", ""),
                "top10_ids": top10_ids,
                "top10_logprobs": top10_logprobs,
            }
        )
        print(
            f"[{idx + 1}/{len(cases)}] {name}: prompt_tokens={len(prompt_token_ids)} "
            f"top1={generated_token_id}"
        )

    manifest = {
        "schema_version": "dsv32_vllm_logits_ref.v1",
        "engine": "vllm",
        "model_name": model_path.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "model_path": str(model_path),
            "prompts_file": str(prompts_path),
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "logprobs_topk": args.topk,
            "trust_remote_code": trust_remote_code,
        },
        "cases": manifest_cases,
    }
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path} ({len(manifest_cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
