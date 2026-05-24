#!/usr/bin/env python3
"""Build local, non-official S1/S8/Smax speed proxy splits.

The official SOAR speed split is not shipped with the public toolkit. This
helper derives deterministic self-test requests from the public accuracy set so
candidate runs can be compared with the same local workload. Outputs are not
official benchmark data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_WORK_ROOT = Path("/home/dataset-local/work/SOAR")
DEFAULT_DATA_PATH = (
    DEFAULT_WORK_ROOT / "repos/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_ROOT / "artifacts/results/local_proxy_speed"
DEFAULT_MODEL_PATH = Path("/home/dataset-local/models/MiniCPM-SALA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local non-official speed proxy splits for SOAR.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--s1-count", type=int, default=8)
    parser.add_argument("--s8-count", type=int, default=16)
    parser.add_argument("--smax-count", type=int, default=24)
    parser.add_argument("--min-output-tokens", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("question") or item.get("input") or item.get("prompt")
            if not prompt:
                raise KeyError(f"Missing prompt field at {path}:{line_num}")
            rows.append({"source_index": line_num - 1, "prompt": prompt, "raw": item})
    return rows


def load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def enrich_rows(rows: list[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        raw = row["raw"]
        prompt = row["prompt"]
        prompt_tokens = raw.get("prompt_tokens")
        response_tokens = raw.get("completion_tokens")
        if prompt_tokens is None:
            prompt_tokens = token_count(tokenizer, prompt)
        if response_tokens is None:
            response = raw.get("model_response") or raw.get("answer") or raw.get("output") or ""
            response_tokens = token_count(tokenizer, str(response))
        enriched.append(
            {
                "source_index": row["source_index"],
                "prompt": prompt,
                "prompt_tokens": int(prompt_tokens),
                "source_response_tokens": int(response_tokens),
            }
        )
    return enriched


def choose_evenly(rows: list[dict[str, Any]], count: int, key: str) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    ordered = sorted(rows, key=lambda item: (item[key], item["source_index"]))
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]

    selected = []
    used: set[int] = set()
    for pos in range(count):
        idx = round(pos * (len(ordered) - 1) / (count - 1))
        while idx in used and idx + 1 < len(ordered):
            idx += 1
        while idx in used and idx > 0:
            idx -= 1
        used.add(idx)
        selected.append(ordered[idx])
    return selected


def choose_smax(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    selected_by_index: dict[int, dict[str, Any]] = {}
    long_prompt_quota = max(1, math.ceil(count * 0.4))
    long_response_quota = max(1, math.ceil(count * 0.3))

    for row in sorted(rows, key=lambda item: item["prompt_tokens"], reverse=True)[
        :long_prompt_quota
    ]:
        selected_by_index[row["source_index"]] = row
    for row in sorted(rows, key=lambda item: item["source_response_tokens"], reverse=True)[
        :long_response_quota
    ]:
        selected_by_index[row["source_index"]] = row
    for row in choose_evenly(rows, count, "prompt_tokens"):
        if len(selected_by_index) >= count:
            break
        selected_by_index[row["source_index"]] = row

    selected = list(selected_by_index.values())
    selected.sort(key=lambda item: (item["prompt_tokens"], item["source_index"]))
    return selected[:count]


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def build_response(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    pieces = [
        " The answer follows from the given context.",
        " Therefore, the relevant evidence is consistent.",
        " This proxy completion is intentionally deterministic.",
        " It is used only for local serving throughput measurement.",
    ]
    text = ""
    piece_idx = 0
    while token_count(tokenizer, text) < target_tokens:
        text += pieces[piece_idx % len(pieces)]
        piece_idx += 1

    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    response = tokenizer.decode(ids, skip_special_tokens=True)
    return response, token_count(tokenizer, response)


def write_split(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    min_output_tokens: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    stats_rows = []
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            target_tokens = clamp(
                row["source_response_tokens"],
                min_output_tokens,
                max_output_tokens,
            )
            response, actual_tokens = build_response(tokenizer, target_tokens)
            out = {
                "question": row["prompt"],
                "model_response": response,
                "source": "local_proxy_from_perf_public_set_length_distribution",
                "source_index": row["source_index"],
                "prompt_tokens": row["prompt_tokens"],
                "source_response_tokens": row["source_response_tokens"],
                "proxy_target_output_tokens": target_tokens,
                "proxy_output_tokens": actual_tokens,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats_rows.append(out)

    return summarize_rows(stats_rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "prompt_tokens": {},
            "proxy_output_tokens": {},
            "source_indices": [],
        }
    prompt_tokens = [int(row["prompt_tokens"]) for row in rows]
    output_tokens = [int(row["proxy_output_tokens"]) for row in rows]
    return {
        "count": len(rows),
        "source_indices": [int(row["source_index"]) for row in rows],
        "prompt_tokens": summarize_values(prompt_tokens),
        "proxy_output_tokens": summarize_values(output_tokens),
    }


def summarize_values(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(mean(ordered), 2),
        "p50": ordered[len(ordered) // 2],
        "sum": sum(ordered),
    }


def main() -> None:
    args = parse_args()
    if not args.data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if args.min_output_tokens <= 0 or args.max_output_tokens <= 0:
        raise ValueError("Output token bounds must be positive")
    if args.min_output_tokens > args.max_output_tokens:
        raise ValueError("--min-output-tokens must be <= --max-output-tokens")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.model_path)
    rows = enrich_rows(load_jsonl(args.data_path), tokenizer)

    splits = {
        "s1": choose_evenly(rows, args.s1_count, "prompt_tokens"),
        "s8": choose_evenly(rows, args.s8_count, "prompt_tokens"),
        "smax": choose_smax(rows, args.smax_count),
    }

    manifest: dict[str, Any] = {
        "kind": "local_proxy_speed_split",
        "official": False,
        "note": (
            "Derived from public accuracy prompts for local self-test only; "
            "do not report as official SOAR speed metrics."
        ),
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
        "min_output_tokens": args.min_output_tokens,
        "max_output_tokens": args.max_output_tokens,
        "splits": {},
    }

    for split_name, split_rows in splits.items():
        split_path = args.output_dir / f"local_proxy_speed_{split_name}.jsonl"
        manifest["splits"][split_name] = {
            "path": str(split_path),
            **write_split(
                path=split_path,
                rows=split_rows,
                tokenizer=tokenizer,
                min_output_tokens=args.min_output_tokens,
                max_output_tokens=args.max_output_tokens,
            ),
        }

    manifest_path = args.output_dir / "local_proxy_speed_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fout:
        json.dump(manifest, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    print(f"wrote={args.output_dir}")
    print(f"manifest={manifest_path}")
    for name, split in manifest["splits"].items():
        print(
            f"{name}: count={split['count']} "
            f"prompt_sum={split['prompt_tokens'].get('sum', 0)} "
            f"output_sum={split['proxy_output_tokens'].get('sum', 0)}"
        )


if __name__ == "__main__":
    main()
