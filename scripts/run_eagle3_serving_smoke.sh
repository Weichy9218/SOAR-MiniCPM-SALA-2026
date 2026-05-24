#!/usr/bin/env bash
set -euo pipefail

# Run a small serving smoke check against a live EAGLE3 SGLang server.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
API_BASE="${API_BASE:-http://127.0.0.1:30000}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/artifacts/results}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/artifacts/logs}"
RUN_NAME="${RUN_NAME:-eagle3_serving_smoke}"
MODE="${MODE:-${RUN_NAME}}"
OFFICIAL="${OFFICIAL:-false}"
EXPECT_PASS="${EXPECT_PASS:-1}"
TMPDIR="${TMPDIR:-/home/dataset-local/tmp}"
HOME="${HOME:-/home/dataset-local}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/dataset-local/.cache}"
export TMPDIR HOME XDG_CACHE_HOME

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [ -z "${DRAFT_MODEL_PATH}" ] || [ ! -d "${DRAFT_MODEL_PATH}" ]; then
  echo "DRAFT_MODEL_PATH must point to a prepared EAGLE3 draft checkpoint." >&2
  exit 1
fi

API_BASE="${API_BASE}" \
MODEL_PATH="${MODEL_PATH}" \
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH}" \
RESULT_DIR="${RESULT_DIR}" \
RUN_NAME="${RUN_NAME}" \
MODE="${MODE}" \
OFFICIAL="${OFFICIAL}" \
EXPECT_PASS="${EXPECT_PASS}" \
python - <<'PY' 2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
import json
import os
import time
from pathlib import Path

import requests

api_base = os.environ["API_BASE"].rstrip("/")
model_path = os.environ["MODEL_PATH"]
draft_model_path = os.environ["DRAFT_MODEL_PATH"]
result_dir = Path(os.environ["RESULT_DIR"])
run_name = os.environ["RUN_NAME"]
mode = os.environ["MODE"]
official = os.environ["OFFICIAL"].lower() == "true"
expect_pass = os.environ["EXPECT_PASS"] != "0"

summary_path = result_dir / f"{run_name}.json"
prediction_path = result_dir / f"{run_name}_predictions.jsonl"
all_runs_path = result_dir / "eagle3_serving_smoke.jsonl"

summary = {
    "status": "failed",
    "scope": "smoke",
    "official": official,
    "expect_pass": expect_pass,
    "mode": mode,
    "base_url": api_base,
    "model_path": model_path,
    "draft_model_path": draft_model_path,
    "models_ok": False,
    "chat_ok": False,
    "models_status": None,
    "models_elapsed_s": None,
    "request_elapsed_s": [],
    "num_requests": 0,
    "prediction_path": str(prediction_path),
    "error": None,
}

try:
    start = time.monotonic()
    models_resp = requests.get(f"{api_base}/v1/models", timeout=30)
    summary["models_elapsed_s"] = round(time.monotonic() - start, 4)
    summary["models_status"] = models_resp.status_code
    models_resp.raise_for_status()
    summary["models_ok"] = True

    prompts = [
        "请用一句话解释 SOAR 推理优化的核心目标。",
        "Return exactly three comma-separated colors.",
    ]
    rows = []
    for prompt in prompts:
        payload = {
            "model": model_path,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
        start = time.monotonic()
        resp = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        elapsed = round(time.monotonic() - start, 4)
        summary["request_elapsed_s"].append(elapsed)
        resp.raise_for_status()
        data = resp.json()
        rows.append(
            {
                "prompt": prompt,
                "status": resp.status_code,
                "elapsed_s": elapsed,
                "finish_reason": data["choices"][0].get("finish_reason"),
                "content": data["choices"][0]["message"]["content"],
                "usage": data.get("usage", {}),
                "id": data.get("id"),
            }
        )

    with prediction_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["chat_ok"] = True
    summary["status"] = "passed"
    summary["num_requests"] = len(rows)
except Exception as exc:
    summary["error"] = repr(exc)
finally:
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with all_runs_path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if expect_pass and summary["status"] != "passed":
        raise SystemExit(1)
    if not expect_pass and summary["status"] == "passed":
        raise SystemExit(1)
PY
