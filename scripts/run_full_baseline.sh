#!/usr/bin/env bash
set -euo pipefail

# Run public correctness and all available speed benchmark splits.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
SOAR_ROOT="${SOAR_ROOT:-${WORK_ROOT}/repos/SOAR-Toolkit}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
API_BASE="${API_BASE:-http://127.0.0.1:30000}"
RESULT_DIR="${RESULT_DIR:-${WORK_ROOT}/artifacts/results}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/artifacts/logs}"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

python "${SOAR_ROOT}/eval_model.py" \
  --api_base "${API_BASE}" \
  --model_path "${MODEL_PATH}" \
  --data_path "${SOAR_ROOT}/eval_dataset/perf_public_set.jsonl" \
  --concurrency "${ACCURACY_CONCURRENCY:-32}" \
  2>&1 | tee "${LOG_DIR}/baseline_accuracy_full.log"

if [ -z "${SPEED_DATA_S1:-}" ] && [ -z "${SPEED_DATA_S8:-}" ] && [ -z "${SPEED_DATA_SMAX:-}" ]; then
  echo "No SPEED_DATA_S1/SPEED_DATA_S8/SPEED_DATA_SMAX set; speed benchmark will be skipped by SOAR script." >&2
fi

bash "${SOAR_ROOT}/bench_serving.sh" "${API_BASE}" \
  2>&1 | tee "${LOG_DIR}/baseline_speed_full.log"
