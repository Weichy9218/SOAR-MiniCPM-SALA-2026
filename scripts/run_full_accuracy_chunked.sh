#!/usr/bin/env bash
set -euo pipefail

# Run SOAR public accuracy through the official evaluator in restartable chunks.

DATASET_LOCAL_ROOT="${DATASET_LOCAL_ROOT:-/home/dataset-local}"
WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
SOAR_ROOT="${SOAR_ROOT:-${WORK_ROOT}/repos/SOAR-Toolkit}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
API_BASE="${API_BASE:-http://127.0.0.1:30000}"
DATA_PATH="${DATA_PATH:-${SOAR_ROOT}/eval_dataset/perf_public_set.jsonl}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/artifacts/results}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/artifacts/logs}"
RUN_DIR="${RUN_DIR:-${RESULT_DIR}/baseline_accuracy_chunks}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
ACCURACY_CONCURRENCY="${ACCURACY_CONCURRENCY:-4}"
KEEP_RAW_OUTPUTS="${KEEP_RAW_OUTPUTS:-0}"

TMPDIR="${TMPDIR:-${DATASET_LOCAL_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATASET_LOCAL_ROOT}/.cache}"
HF_HOME="${HF_HOME:-${DATASET_LOCAL_ROOT}/.cache/huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${DATASET_LOCAL_ROOT}/.cache/uv}"
if [ "${HOME:-}" = "/home/batchcom" ] && [ -d "${DATASET_LOCAL_ROOT}" ]; then
  HOME="${DATASET_LOCAL_ROOT}"
fi
export TMPDIR XDG_CACHE_HOME HF_HOME UV_CACHE_DIR HOME

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${RUN_DIR}/data"

if [ ! -f "${SOAR_ROOT}/eval_model.py" ]; then
  echo "SOAR eval script not found: ${SOAR_ROOT}/eval_model.py" >&2
  exit 1
fi
if [ ! -f "${DATA_PATH}" ]; then
  echo "Data file not found: ${DATA_PATH}" >&2
  exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

API_BASE="${API_BASE}" MODEL_PATH="${MODEL_PATH}" python - <<'PY'
import os
import requests
import sys

api_base = os.environ["API_BASE"].rstrip("/")
model_path = os.environ["MODEL_PATH"]
try:
    resp = requests.get(f"{api_base}/v1/models", timeout=15)
    resp.raise_for_status()
    model_ids = [item["id"] for item in resp.json().get("data", [])]
except Exception as exc:
    print(f"Server readiness check failed: {exc}", file=sys.stderr)
    sys.exit(1)

if model_ids and model_path not in model_ids:
    print(f"warning: MODEL_PATH={model_path} not in /v1/models ids={model_ids}", file=sys.stderr)
print(f"server_ready=ok api_base={api_base}")
PY

DATA_PATH="${DATA_PATH}" RUN_DIR="${RUN_DIR}" CHUNK_SIZE="${CHUNK_SIZE}" python - <<'PY'
import json
import os
from pathlib import Path

data_path = Path(os.environ["DATA_PATH"])
run_dir = Path(os.environ["RUN_DIR"])
chunk_size = int(os.environ["CHUNK_SIZE"])
if chunk_size <= 0:
    raise SystemExit("CHUNK_SIZE must be positive")

rows = []
with data_path.open("r", encoding="utf-8") as fin:
    for line in fin:
        if line.strip():
            rows.append(json.loads(line))

data_dir = run_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)
manifest = {
    "data_path": str(data_path),
    "num_samples": len(rows),
    "chunk_size": chunk_size,
    "chunks": [],
}
for chunk_id, start in enumerate(range(0, len(rows), chunk_size)):
    end = min(start + chunk_size, len(rows))
    chunk_path = data_dir / f"chunk_{chunk_id:04d}_{start:04d}_{end - 1:04d}.jsonl"
    with chunk_path.open("w", encoding="utf-8") as fout:
        for item in rows[start:end]:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest["chunks"].append(
        {
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "path": str(chunk_path),
        }
    )

with (run_dir / "manifest.json").open("w", encoding="utf-8") as fout:
    json.dump(manifest, fout, ensure_ascii=False, indent=2)

print(f"prepared_chunks={len(manifest['chunks'])} num_samples={len(rows)}")
PY

aggregate_chunks() {
  DATA_PATH="${DATA_PATH}" RUN_DIR="${RUN_DIR}" RESULT_DIR="${RESULT_DIR}" python - <<'PY'
import glob
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
result_dir = Path(os.environ["RESULT_DIR"])
manifest_path = run_dir / "manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing manifest: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
total_samples = int(manifest["num_samples"])
completed_samples = 0
score_sum = 0.0
duration = 0.0
total_tokens = 0
chunks = []

for chunk in manifest["chunks"]:
    chunk_id = int(chunk["chunk_id"])
    chunk_dir = run_dir / f"chunk_{chunk_id:04d}_{int(chunk['start']):04d}_{int(chunk['end']) - 1:04d}"
    pred_path = chunk_dir / "predictions.jsonl"
    summary_path = chunk_dir / "summary.json"
    if not pred_path.exists() or not summary_path.exists():
        chunks.append({**chunk, "status": "pending"})
        continue

    chunk_count = 0
    chunk_score = 0.0
    with pred_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_count += 1
            chunk_score += float(row.get("score", 0.0))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completed_samples += chunk_count
    score_sum += chunk_score
    duration += float(summary.get("duration", 0.0))
    total_tokens += int(summary.get("total_tokens", 0) or 0)
    chunks.append(
        {
            **chunk,
            "status": "complete",
            "num_samples": chunk_count,
            "score_sum": round(chunk_score, 6),
            "ori_accuracy": round((chunk_score / chunk_count) * 100, 2) if chunk_count else 0.0,
            "duration": summary.get("duration"),
            "total_tokens": summary.get("total_tokens"),
        }
    )

ori_accuracy = round((score_sum / completed_samples) * 100, 2) if completed_samples else None
overall_accuracy = min(round(ori_accuracy / 80 * 100, 2), 100) if ori_accuracy is not None else None
status = "complete" if completed_samples == total_samples and total_samples > 0 else "partial"
aggregate = {
    "status": status,
    "data_path": manifest["data_path"],
    "num_samples": total_samples,
    "completed_samples": completed_samples,
    "ori_accuracy": ori_accuracy,
    "overall_accuracy": overall_accuracy,
    "duration": duration,
    "total_tokens": total_tokens,
    "run_dir": str(run_dir),
    "chunks": chunks,
}

run_dir.mkdir(parents=True, exist_ok=True)
for out_path in [
    run_dir / "summary.json",
    result_dir / "baseline_accuracy_full_chunked_summary.json",
]:
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(aggregate, fout, ensure_ascii=False, indent=2)

summary_txt = result_dir / "baseline_accuracy_full_chunked_summary.txt"
summary_txt.write_text(
    "\n".join(
        [
            f"status={status}",
            f"completed_samples={completed_samples}/{total_samples}",
            f"ori_accuracy={ori_accuracy}",
            f"overall_accuracy={overall_accuracy}",
            f"duration={duration:.2f}",
            f"total_tokens={total_tokens}",
            f"run_dir={run_dir}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(f"aggregate_status={status} completed_samples={completed_samples}/{total_samples} ori_accuracy={ori_accuracy}")
PY
}

while IFS=$'\t' read -r chunk_id start end chunk_path; do
  chunk_label="$(printf 'chunk_%04d_%04d_%04d' "${chunk_id}" "${start}" "$((end - 1))")"
  chunk_dir="${RUN_DIR}/${chunk_label}"
  chunk_log="${LOG_DIR}/baseline_accuracy_${chunk_label}.log"
  mkdir -p "${chunk_dir}"

  if [ -f "${chunk_dir}/summary.json" ] && [ -f "${chunk_dir}/predictions.jsonl" ]; then
    echo "skip_completed=${chunk_label}"
    aggregate_chunks
    continue
  fi

  echo "running=${chunk_label} data=${chunk_path}"
  python "${SOAR_ROOT}/eval_model.py" \
    --api_base "${API_BASE}" \
    --model_path "${MODEL_PATH}" \
    --data_path "${chunk_path}" \
    --concurrency "${ACCURACY_CONCURRENCY}" \
    2>&1 | tee "${chunk_log}"

  eval_output_dir="$(grep -a 'Saving results to outputs/' "${chunk_log}" | tail -n 1 | sed 's/^.*Saving results to //')"
  if [ -z "${eval_output_dir}" ] || [ ! -f "${eval_output_dir}/summary.json" ]; then
    echo "Could not locate eval output for ${chunk_label}; see ${chunk_log}" >&2
    exit 1
  fi

  cp "${eval_output_dir}/summary.json" "${chunk_dir}/summary.json"
  cp "${eval_output_dir}/summary.txt" "${chunk_dir}/summary.txt"
  cp "${eval_output_dir}/predictions.jsonl" "${chunk_dir}/predictions.jsonl"
  cat > "${chunk_dir}/metadata.json" <<EOF
{
  "chunk_id": ${chunk_id},
  "start": ${start},
  "end": ${end},
  "chunk_path": "${chunk_path}",
  "log_path": "${chunk_log}",
  "eval_output_dir": "${eval_output_dir}",
  "api_base": "${API_BASE}",
  "model_path": "${MODEL_PATH}",
  "accuracy_concurrency": ${ACCURACY_CONCURRENCY}
}
EOF

  if [ "${KEEP_RAW_OUTPUTS}" != "1" ]; then
    rm -rf "${eval_output_dir}"
  fi

  aggregate_chunks
done < <(
  RUN_DIR="${RUN_DIR}" python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads((Path(os.environ["RUN_DIR"]) / "manifest.json").read_text(encoding="utf-8"))
for chunk in manifest["chunks"]:
    print(f"{chunk['chunk_id']}\t{chunk['start']}\t{chunk['end']}\t{chunk['path']}")
PY
)

aggregate_chunks
