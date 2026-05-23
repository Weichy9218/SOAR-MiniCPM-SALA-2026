#!/usr/bin/env bash
set -euo pipefail

# Start the official SOAR MiniCPM-SALA baseline server.

DATASET_LOCAL_ROOT="${DATASET_LOCAL_ROOT:-/home/dataset-local}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-/home/dataset-local/work/SOAR/artifacts/logs}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"
if [ "${DISABLE_CUDA_GRAPH}" = "1" ]; then
  EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---disable-cuda-graph}"
else
  EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
fi
TMPDIR="${TMPDIR:-${DATASET_LOCAL_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATASET_LOCAL_ROOT}/.cache}"
HF_HOME="${HF_HOME:-${DATASET_LOCAL_ROOT}/.cache/huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${DATASET_LOCAL_ROOT}/.cache/uv}"
if [ "${HOME:-}" = "/home/batchcom" ] && [ -d "${DATASET_LOCAL_ROOT}" ]; then
  HOME="${DATASET_LOCAL_ROOT}"
fi
export TMPDIR XDG_CACHE_HOME HF_HOME UV_CACHE_DIR HOME
mkdir -p "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

read -r -a EXTRA_SERVER_ARGV <<< "${EXTRA_SERVER_ARGS}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse \
  "${EXTRA_SERVER_ARGV[@]}" \
  2>&1 | tee "${LOG_DIR}/baseline_server.log"
