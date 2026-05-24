#!/usr/bin/env python3
"""Generate MiniCPM-SALA EAGLE3 draft-training assets for small probes."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SYNTHETIC_CONVERSATIONS = [
    {
        "conversations": [
            {
                "role": "user",
                "content": "Summarize this deployment rule in one sentence: keep the base model, tokenizer, and evaluation data unchanged.",
            },
            {
                "role": "assistant",
                "content": "The optimization must preserve the original MiniCPM-SALA model, tokenizer, and evaluation inputs while changing only the serving strategy.",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Why should a speculative draft head be validated with accuracy before speed is reported?",
            },
            {
                "role": "assistant",
                "content": "Speculation is useful only if the target model still verifies outputs correctly, so public accuracy is the guardrail before any speed claim.",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Compute 18 plus 24, then explain the arithmetic briefly.",
            },
            {
                "role": "assistant",
                "content": "18 plus 24 equals 42 because 10 plus 20 is 30 and 8 plus 4 is 12.",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Write a short Python function named clamp that bounds x between lo and hi.",
            },
            {
                "role": "assistant",
                "content": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "State the next experiment after an FP16 baseline is trusted.",
            },
            {
                "role": "assistant",
                "content": "After the FP16 baseline is trusted, run narrow optimization probes such as quantization or EAGLE3 and compare them against the same accuracy guardrail.",
            },
        ]
    },
    {
        "conversations": [
            {
                "role": "user",
                "content": "Give a concise checklist for a serving smoke test.",
            },
            {
                "role": "assistant",
                "content": "Start the server, wait for the model endpoint, send one chat request, record the response, and stop the server cleanly.",
            },
        ]
    },
]


def build_probe_conversations(num_records: int, seed: int) -> list[dict]:
    """Build public-eval-independent instruction data for EAGLE3 smoke probes."""
    rng = random.Random(seed)
    topics = [
        "serving latency",
        "CUDA memory",
        "unit testing",
        "vector search",
        "data cleaning",
        "checkpoint resume",
        "HTTP retries",
        "model quantization",
        "scheduler fairness",
        "log analysis",
        "matrix multiplication",
        "cache eviction",
    ]
    records: list[dict] = []

    templates = [
        lambda i: (
            f"Write two concise bullet points about {topics[i % len(topics)]}.",
            (
                f"- {topics[i % len(topics)].capitalize()} should be measured with a clear baseline.\n"
                "- Changes should be verified with a small, reproducible check before a long run."
            ),
        ),
        lambda i: (
            f"Compute {i + 17} + {2 * i + 5} and explain in one sentence.",
            f"{i + 17} + {2 * i + 5} = {3 * i + 22}, because the two addends combine directly.",
        ),
        lambda i: (
            f"Convert this sentence to a shorter status update: experiment {i} finished without errors and wrote its summary file.",
            f"Experiment {i} completed cleanly and wrote its summary.",
        ),
        lambda i: (
            f"Return a Python function named add_{i} that adds {i} to x.",
            f"def add_{i}(x):\n    return x + {i}",
        ),
        lambda i: (
            f"Classify the tone as positive, neutral, or negative: the benchmark completed but the result was mixed.",
            "neutral",
        ),
        lambda i: (
            f"List three checks before launching a long GPU job about {topics[(i + 3) % len(topics)]}.",
            "Check disk space, confirm no conflicting processes, and verify the expected model path.",
        ),
        lambda i: (
            f"Rewrite as JSON with keys id and status: id={i}, status=ready.",
            json.dumps({"id": i, "status": "ready"}),
        ),
        lambda i: (
            f"Explain why a local proxy benchmark is not an official result.",
            "A local proxy benchmark uses a derived workload, so it is useful for comparison but cannot replace the official benchmark split.",
        ),
    ]

    for i in range(num_records):
        template = templates[i % len(templates)]
        user, assistant = template(i)
        if rng.random() < 0.25:
            user += " Keep the answer brief."
        records.append(
            {
                "id": f"synthetic_probe_{i:04d}",
                "conversations": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
            }
        )
    return records


def load_target_config(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    with config_path.open() as f:
        return json.load(f)


def build_draft_config(target_config: dict, draft_vocab_size: int) -> dict:
    hidden_size = int(target_config["hidden_size"])
    num_attention_heads = int(target_config["num_attention_heads"])
    head_dim = hidden_size // num_attention_heads
    eos_token_id = target_config.get("eos_token_id", 2)
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    max_position_embeddings = int(target_config.get("max_position_embeddings", 4096))

    return {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "vocab_size": int(target_config["vocab_size"]),
        "draft_vocab_size": draft_vocab_size,
        "hidden_size": hidden_size,
        "target_hidden_size": hidden_size,
        "intermediate_size": int(target_config["intermediate_size"]),
        "num_hidden_layers": 1,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": int(target_config.get("num_key_value_heads", num_attention_heads)),
        "head_dim": head_dim,
        "hidden_act": target_config.get("hidden_act", "silu"),
        "rms_norm_eps": float(target_config.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(target_config.get("rope_theta", 10000.0)),
        "max_position_embeddings": max_position_embeddings,
        "attention_bias": bool(target_config.get("attention_bias", False)),
        "bos_token_id": int(target_config.get("bos_token_id", 1)),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(target_config.get("pad_token_id", eos_token_id)),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, int(target_config["num_hidden_layers"]) // 2 - 1, int(target_config["num_hidden_layers"]) - 4]
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/home/dataset-local/models/MiniCPM-SALA",
        help="MiniCPM-SALA target model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/dataset-local/work/SOAR/artifacts/draft_heads/eagle3_smoke_assets",
        help="Directory for generated draft config and synthetic training JSONL.",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=4096,
        help="Hot-token vocabulary size for the smoke draft head.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["smoke", "synthetic_probe"],
        default="smoke",
        help="Training data mode. synthetic_probe is larger but still not an eval dataset.",
    )
    parser.add_argument(
        "--num-records",
        type=int,
        default=128,
        help="Number of records for --dataset-mode synthetic_probe.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic synthetic probe generation.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    target_config = load_target_config(model_path)
    draft_config = build_draft_config(target_config, args.draft_vocab_size)

    config_path = output_dir / "minicpm_sala_eagle3_smoke_config.json"
    train_path = output_dir / f"minicpm_sala_eagle3_{args.dataset_mode}_train.jsonl"
    if args.dataset_mode == "smoke":
        rows = SYNTHETIC_CONVERSATIONS
    else:
        if args.num_records <= 0:
            raise ValueError("--num-records must be positive")
        rows = build_probe_conversations(args.num_records, args.seed)
    write_json(config_path, draft_config)
    write_jsonl(train_path, rows)

    summary = {
        "model_path": str(model_path),
        "draft_config_path": str(config_path),
        "train_data_path": str(train_path),
        "draft_vocab_size": args.draft_vocab_size,
        "dataset_mode": args.dataset_mode,
        "num_train_records": len(rows),
        "seed": args.seed,
        "aux_hidden_state_layer_ids": draft_config["eagle_config"]["eagle_aux_hidden_state_layer_ids"],
    }
    write_json(output_dir / "asset_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
