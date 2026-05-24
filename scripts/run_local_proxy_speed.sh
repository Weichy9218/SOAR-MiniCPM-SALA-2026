#!/usr/bin/env bash
set -euo pipefail

# Run non-official local S1/S8/Smax proxy speed splits against a live server.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
SOAR_ROOT="${SOAR_ROOT:-${WORK_ROOT}/repos/SOAR-Toolkit}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
API_BASE="${API_BASE:-http://127.0.0.1:30000}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/artifacts/results}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/artifacts/logs}"
SPEED_PROXY_DIR="${SPEED_PROXY_DIR:-${RESULT_DIR}/local_proxy_speed}"
RUN_NAME="${RUN_NAME:-baseline_fp16}"
TMPDIR="${TMPDIR:-/home/dataset-local/tmp}"
HOME="${HOME:-/home/dataset-local}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/home/dataset-local/.cache}"
export TMPDIR HOME XDG_CACHE_HOME

mkdir -p "${RESULT_DIR}" "${LOG_DIR}" "${SPEED_PROXY_DIR}"

if [ ! -f "${SOAR_ROOT}/bench_serving.sh" ]; then
  echo "SOAR bench script not found: ${SOAR_ROOT}/bench_serving.sh" >&2
  exit 1
fi

if [ ! -f "${SPEED_PROXY_DIR}/local_proxy_speed_manifest.json" ]; then
  python "${WORK_ROOT}/scripts/prepare_local_speed_proxy.py" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${SPEED_PROXY_DIR}"
fi

SPEED_DATA_S1="${SPEED_DATA_S1:-${SPEED_PROXY_DIR}/local_proxy_speed_s1.jsonl}"
SPEED_DATA_S8="${SPEED_DATA_S8:-${SPEED_PROXY_DIR}/local_proxy_speed_s8.jsonl}"
SPEED_DATA_SMAX="${SPEED_DATA_SMAX:-${SPEED_PROXY_DIR}/local_proxy_speed_smax.jsonl}"
export SPEED_DATA_S1 SPEED_DATA_S8 SPEED_DATA_SMAX

for dataset_path in "${SPEED_DATA_S1}" "${SPEED_DATA_S8}" "${SPEED_DATA_SMAX}"; do
  if [ ! -f "${dataset_path}" ]; then
    echo "Speed proxy dataset not found: ${dataset_path}" >&2
    exit 1
  fi
done

python - "${API_BASE}" <<'PY'
import sys
import requests

api_base = sys.argv[1].rstrip("/")
resp = requests.get(f"{api_base}/v1/models", timeout=30)
resp.raise_for_status()
print(f"server_ready={api_base}")
PY

LOG_PATH="${LOG_DIR}/${RUN_NAME}_local_proxy_speed.log"
RESULT_PATH="${RESULT_DIR}/${RUN_NAME}_local_proxy_speed.json"

bash "${SOAR_ROOT}/bench_serving.sh" "${API_BASE}" 2>&1 | tee "${LOG_PATH}"

python - "${LOG_PATH}" "${RESULT_PATH}" "${RUN_NAME}" "${SPEED_PROXY_DIR}" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
run_name = sys.argv[3]
speed_proxy_dir = Path(sys.argv[4])

result = None
for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        result = json.loads(line)
    except json.JSONDecodeError:
        continue
    break

if result is None:
    raise SystemExit(f"No result JSON found in {log_path}")

payload = {
    "run_name": run_name,
    "official": False,
    "note": "Local proxy speed split only; not official SOAR speed data.",
    "speed_proxy_dir": str(speed_proxy_dir),
    "bench_result": result,
}
result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"result={result_path}")
print(json.dumps(payload, ensure_ascii=False))
PY
