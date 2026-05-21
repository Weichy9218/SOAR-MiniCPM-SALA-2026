#!/usr/bin/env bash
set -euo pipefail

# Start MiniCPM-SALA with an existing GPTQ/GPTQ-Marlin quantized checkpoint.

MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA-GPTQ}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
QUANTIZATION="${QUANTIZATION:-gptq_marlin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
LOG_DIR="${LOG_DIR:-/home/dataset-local/work/SOAR/artifacts/logs}"
mkdir -p "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Quantized model path not found: ${MODEL_PATH}" >&2
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
  --quantization "${QUANTIZATION}" \
  --dtype float16 \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  2>&1 | tee "${LOG_DIR}/quant_${QUANTIZATION}_${KV_CACHE_DTYPE}.log"
