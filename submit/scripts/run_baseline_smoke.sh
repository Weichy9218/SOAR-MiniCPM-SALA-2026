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
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${SOAR_ROOT}/eval_model.py" ]; then
  echo "SOAR eval script not found: ${SOAR_ROOT}/eval_model.py" >&2
  exit 1
fi

python "${SOAR_ROOT}/eval_model.py" \
  --api_base "${API_BASE}" \
  --model_path "${MODEL_PATH}" \
  --data_path "${SOAR_ROOT}/eval_dataset/perf_public_set.jsonl" \
  --concurrency 4 \
  --num_samples 10 \
  --verbose \
  2>&1 | tee "${LOG_DIR}/baseline_accuracy_smoke.log"

cat > "${RESULT_DIR}/speed_smoke.jsonl" <<'JSONL'
{"question":"用一句话回答：2+2 等于几？","model_response":"4"}
{"question":"Summarize in one sentence: MiniCPM-SALA is a long-context language model.","model_response":"MiniCPM-SALA is a long-context language model."}
JSONL

SPEED_DATA_S1="${RESULT_DIR}/speed_smoke.jsonl" \
SPEED_DATA_S8="${RESULT_DIR}/speed_smoke.jsonl" \
SPEED_DATA_SMAX="${RESULT_DIR}/speed_smoke.jsonl" \
bash "${SOAR_ROOT}/bench_serving.sh" "${API_BASE}" \
  2>&1 | tee "${LOG_DIR}/baseline_speed_smoke.log"
