#!/usr/bin/env bash
set -euo pipefail

# Start the official SOAR MiniCPM-SALA baseline server.

MODEL_PATH="${MODEL_PATH:-/models/MiniCPM-SALA}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-/home/dataset-local/work/SOAR/artifacts/logs}"
mkdir -p "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse \
  2>&1 | tee "${LOG_DIR}/baseline_server.log"
