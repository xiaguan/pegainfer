#!/usr/bin/env python3
"""Generate SGLang greedy top-K logprob ground truth for DSV3.2 integration tests.

Output schema: `dsv32_sglang_ref.v1`. For each prompt case we record
`prompt_token_ids`, `generated_token_ids`, per-position top-K
`(token_id, logprob)` pairs for the output positions, and the logprob of the
actually-sampled token at each position.

Note on greedy invariant: `output_top_logprobs[i][0][0]` is the logit-space
argmax; `generated_token_ids[i]` is what sglang's FlashInfer sampler chose.
Under `temperature=0, top_k=1` these almost always match, but the sampler
resolves near-ties in softmax/FP32 while the top-K reporting sorts the raw
logits, so a small number of positions can legitimately disagree. The
generator counts these and records the count in the manifest meta rather
than failing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ensure_venv_bin_on_path() -> None:
    """sglang spawns subprocesses that shell out to `ninja` (torch extension
    builds). In a bare shell those subprocesses don't inherit the venv's bin
    dir on PATH, and the load fails with `FileNotFoundError: 'ninja'`. Same
    fix the vllm generator applied."""
    candidates = [
        str((Path(sys.prefix) / "bin").resolve()),
        str(Path(sys.executable).parent),
    ]
    current = os.environ.get("PATH", "")
    parts = current.split(":") if current else []
    changed = False
    for cand in candidates:
        if cand and cand not in parts:
            parts.insert(0, cand)
            changed = True
    if changed:
        os.environ["PATH"] = ":".join(parts)


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


def _load_prompt_cases(path: Path, default_max_new_tokens: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(raw_cases, list):
        raise ValueError("prompts file must be a list or an object with a `cases` list")

    cases: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case #{idx} must be an object")

        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"case #{idx} missing non-empty `prompt`")

        name = item.get("name")
        if not isinstance(name, str) or not name:
            name = f"case_{idx:02d}"

        max_new_tokens = _to_int(item.get("max_new_tokens", default_max_new_tokens))
        if max_new_tokens is None or max_new_tokens <= 0:
            raise ValueError(
                f"case #{idx} has invalid `max_new_tokens`: {item.get('max_new_tokens')}"
            )

        cases.append(
            {"name": name, "prompt": prompt, "max_new_tokens": max_new_tokens}
        )
    return cases


def _normalize_top_logprobs(
    raw_top_positions: list[Any] | None, expected_len: int, top_k: int, case_name: str
) -> list[list[list[int | float]]]:
    """Convert sglang's per-position top-K into `[[token_id, logprob], ...]`.

    sglang emits each position as `[(logprob, token_id, text_or_None), ...]`
    already sorted descending by logprob. We drop the text and swap to
    `(token_id, logprob)` pairs so the manifest stays readable.
    """
    if raw_top_positions is None:
        raise RuntimeError(f"case `{case_name}` missing output_top_logprobs")
    if len(raw_top_positions) != expected_len:
        raise RuntimeError(
            f"case `{case_name}` has output_top_logprobs length "
            f"{len(raw_top_positions)}, expected {expected_len}"
        )

    out: list[list[list[int | float]]] = []
    for pos_idx, pos in enumerate(raw_top_positions):
        if pos is None or not isinstance(pos, list) or not pos:
            raise RuntimeError(
                f"case `{case_name}` position {pos_idx}: top logprobs missing"
            )
        if len(pos) < top_k:
            raise RuntimeError(
                f"case `{case_name}` position {pos_idx}: got {len(pos)} top logprobs, "
                f"expected at least {top_k}"
            )
        row: list[list[int | float]] = []
        for entry in pos[:top_k]:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise RuntimeError(
                    f"case `{case_name}` position {pos_idx}: malformed entry {entry!r}"
                )
            logprob = float(entry[0])
            token_id = int(entry[1])
            row.append([token_id, logprob])
        out.append(row)
    return out


def _extract_sampled_logprobs(
    raw_positions: list[Any] | None,
    top_logprobs: list[list[list[int | float]]],
    generated_token_ids: list[int],
    case_name: str,
) -> list[float]:
    """Logprob of the sampled token at each output position.

    sglang's `output_token_logprobs` entries are `(logprob, token_id, text)`
    triples, one per position. If the server didn't provide them (older
    builds), fall back to looking the token up inside the top-K list.
    """
    if raw_positions is not None and len(raw_positions) == len(generated_token_ids):
        out: list[float] = []
        for pos, entry in enumerate(raw_positions):
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                raise RuntimeError(
                    f"case `{case_name}` position {pos}: malformed "
                    f"output_token_logprobs entry {entry!r}"
                )
            out.append(float(entry[0]))
        return out

    # Fallback: search top-K.
    out = []
    for pos, (tok, row) in enumerate(zip(generated_token_ids, top_logprobs)):
        lp: float | None = None
        for t, v in row:
            if int(t) == int(tok):
                lp = float(v)
                break
        if lp is None:
            raise RuntimeError(
                f"case `{case_name}` position {pos}: sampled token {tok} "
                f"missing from top-K and output_token_logprobs not available"
            )
        out.append(lp)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output-dir", default="test_data/dsv32_sglang_ref",
        help="Output directory for manifest.json",
    )
    parser.add_argument(
        "--prompts-file", default="tools/dsv32_sglang_ref/prompts_generation.json",
        help="Prompt cases JSON path (reuses the generation prompt set)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--dtype", choices=["auto", "bfloat16", "float16"], default="bfloat16",
    )
    parser.add_argument("--default-max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Only process the first N cases (for smoke testing)",
    )
    parser.add_argument(
        "--mem-fraction-static", type=float, default=None,
        help="Pass through to sgl.Engine; leave unset for sglang default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _ensure_venv_bin_on_path()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model path not found: {model_path}")
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        raise FileNotFoundError(f"prompts file not found: {prompts_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.default_max_new_tokens <= 0:
        raise ValueError("--default-max-new-tokens must be > 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.json"

    try:
        import sglang as sgl
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    except ImportError as exc:
        print(
            "Missing sglang dependency in workspace .venv. Install with:\n"
            "  uv pip install -p .venv/bin/python -e ../sglang/python",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        import sglang
        engine_version = getattr(sglang, "__version__", "unknown")
    except Exception:
        engine_version = "unknown"

    cases = _load_prompt_cases(prompts_path, args.default_max_new_tokens)
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("--max-cases must be > 0")
        cases = cases[: args.max_cases]
    tokenizer = get_tokenizer(str(model_path))

    prompt_token_ids_per_case: list[list[int]] = []
    for case in cases:
        ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
        if not ids:
            raise ValueError(f"case `{case['name']}` tokenized to empty ids")
        prompt_token_ids_per_case.append([int(x) for x in ids])

    engine_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "tp_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": True,
        "skip_tokenizer_init": True,
        "random_seed": args.seed,
        "context_length": args.max_model_len,
    }
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static

    engine = sgl.Engine(**engine_kwargs)

    manifest_cases: list[dict[str, Any]] = []
    try:
        for idx, (case, prompt_ids) in enumerate(zip(cases, prompt_token_ids_per_case)):
            name = case["name"]
            prompt = case["prompt"]
            max_new_tokens = case["max_new_tokens"]

            sampling_params = {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "top_k": 1,
            }

            output = engine.generate(
                input_ids=prompt_ids,
                sampling_params=sampling_params,
                return_logprob=True,
                top_logprobs_num=args.top_k,
                logprob_start_len=len(prompt_ids),
            )
            if isinstance(output, list):
                if not output:
                    raise RuntimeError(f"case `{name}` returned empty sglang output")
                output = output[0]
            if not isinstance(output, dict):
                raise RuntimeError(
                    f"case `{name}` got unexpected sglang output type: {type(output)}"
                )

            output_ids = output.get("output_ids")
            if not output_ids:
                # Some sglang builds return "token_ids" instead of "output_ids"
                output_ids = output.get("token_ids")
            if not output_ids:
                raise RuntimeError(f"case `{name}` produced no output tokens")
            generated_token_ids = [int(tid) for tid in output_ids]

            meta_info = output.get("meta_info") or {}
            top_logprobs = _normalize_top_logprobs(
                meta_info.get("output_top_logprobs"),
                expected_len=len(generated_token_ids),
                top_k=args.top_k,
                case_name=name,
            )

            # Logprob of the *sampled* token at each position. Falls back to
            # searching the top-K if sglang didn't report it separately (older
            # sglang builds emitted only the top list).
            generated_token_logprobs = _extract_sampled_logprobs(
                meta_info.get("output_token_logprobs"),
                top_logprobs,
                generated_token_ids,
                case_name=name,
            )

            # Count (but don't fail on) positions where the greedy-sampled
            # token disagrees with the logit-space top-1. Expected to be low
            # single digits across a 44-case run.
            argmax_mismatches = sum(
                1
                for tok, row in zip(generated_token_ids, top_logprobs)
                if int(row[0][0]) != tok
            )
            case_argmax_mismatches = argmax_mismatches

            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
            finish_reason = meta_info.get("finish_reason")
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("type")

            manifest_cases.append(
                {
                    "name": name,
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "prompt_token_ids": prompt_ids,
                    "generated_token_ids": generated_token_ids,
                    "generated_token_logprobs": generated_token_logprobs,
                    "generated_text": generated_text,
                    "finish_reason": finish_reason,
                    "argmax_mismatches": case_argmax_mismatches,
                    "output_top_logprobs": top_logprobs,
                }
            )
            print(
                f"[{idx + 1}/{len(cases)}] {name}: prompt_tokens={len(prompt_ids)} "
                f"generated={len(generated_token_ids)} finish={finish_reason} "
                f"argmax_mismatches={case_argmax_mismatches}"
            )
    finally:
        try:
            engine.shutdown()
        except Exception as exc:
            print(f"engine.shutdown() failed: {exc}", file=sys.stderr)

    total_argmax_mismatches = sum(c.get("argmax_mismatches", 0) for c in manifest_cases)
    total_positions = sum(len(c["generated_token_ids"]) for c in manifest_cases)
    manifest = {
        "schema_version": "dsv32_sglang_ref.v1",
        "engine": "sglang",
        "engine_version": engine_version,
        "model_name": model_path.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "model_path": str(model_path),
            "prompts_file": str(prompts_path),
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "seed": args.seed,
            "default_max_new_tokens": args.default_max_new_tokens,
            "top_k": args.top_k,
            "argmax_mismatches_total": total_argmax_mismatches,
            "total_output_positions": total_positions,
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
