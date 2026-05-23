#!/usr/bin/env bash
set -euo pipefail

# Run the smallest correctness and serving smoke checks against a live SGLang server.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
SOAR_ROOT="${SOAR_ROOT:-${WORK_ROOT}/repos/SOAR-Toolkit}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
API_BASE="${API_BASE:-http://127.0.0.1:30000}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/artifacts/results}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/artifacts/logs}"
TMPDIR="${TMPDIR:-/home/dataset-local/tmp}"
HOME="${HOME:-/home/dataset-local}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/dataset-local/.cache}"
export TMPDIR HOME XDG_CACHE_HOME
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${SOAR_ROOT}/eval_model.py" ]; then
  echo "SOAR eval script not found: ${SOAR_ROOT}/eval_model.py" >&2
  exit 1
fi

API_BASE="${API_BASE}" MODEL_PATH="${MODEL_PATH}" RESULT_DIR="${RESULT_DIR}" python - <<'PY' \
  2>&1 | tee "${LOG_DIR}/baseline_accuracy_smoke.log"
import json
import os
import time

import requests

api_base = os.environ["API_BASE"].rstrip("/")
model_path = os.environ["MODEL_PATH"]
result_dir = os.environ["RESULT_DIR"]
out_path = os.path.join(result_dir, "baseline_smoke_predictions.jsonl")

models_resp = requests.get(f"{api_base}/v1/models", timeout=30)
models_resp.raise_for_status()

prompts = [
    "Answer with one word: ok",
    "Give only the final answer: 2+2=",
]
rows = []
for idx, prompt in enumerate(prompts):
    payload = {
        "model": model_path,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    start = time.time()
    resp = requests.post(
        f"{api_base}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    duration = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    rows.append(
        {
            "index": idx,
            "prompt": prompt,
            "prediction": content,
            "duration_sec": round(duration, 3),
            "usage": data.get("usage", {}),
        }
    )

with open(out_path, "w", encoding="utf-8") as fout:
    for row in rows:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"models_endpoint=ok")
print(f"chat_completions=ok requests={len(rows)}")
print(f"predictions={out_path}")
PY

cat > "${RESULT_DIR}/speed_smoke.jsonl" <<'JSONL'
{"question":"用一句话回答：2+2 等于几？","model_response":"4"}
{"question":"Summarize in one sentence: MiniCPM-SALA is a long-context language model.","model_response":"MiniCPM-SALA is a long-context language model."}
JSONL

SPEED_DATA_S1="${RESULT_DIR}/speed_smoke.jsonl" \
SPEED_DATA_S8="${RESULT_DIR}/speed_smoke.jsonl" \
SPEED_DATA_SMAX="${RESULT_DIR}/speed_smoke.jsonl" \
bash "${SOAR_ROOT}/bench_serving.sh" "${API_BASE}" \
  2>&1 | tee "${LOG_DIR}/baseline_speed_smoke.log"
