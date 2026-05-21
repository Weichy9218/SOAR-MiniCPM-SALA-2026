#!/usr/bin/env bash
set -euo pipefail

# Prepare the SOAR/SGLang runtime from the local submit package.
# This script intentionally uses uv pip and avoids interactive prompts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${SCRIPT_DIR}"
DEV_WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/dataset-local/.cache/uv}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/dataset-local/.cache/huggingface}"
export MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
export MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
LOCAL_CUDA_HOME="${LOCAL_CUDA_HOME:-/home/dataset-local/cuda-13.1}"
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:---disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse}"

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${MODEL_ROOT}"
if [ -z "${CUDA_HOME:-}" ] && [ -d "${LOCAL_CUDA_HOME}" ]; then
  export CUDA_HOME="${LOCAL_CUDA_HOME}"
fi
if [ -n "${CUDA_HOME:-}" ] && [ -d "${CUDA_HOME}" ]; then
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN=uv
else
  echo "uv is required but was not found in PATH" >&2
  exit 1
fi

if [ -d "${PACKAGE_ROOT}/sglang/python" ]; then
  "${UV_BIN}" pip install --no-deps -e "${PACKAGE_ROOT}/sglang/python"
elif [ -d "${DEV_WORK_ROOT}/repos/sglang/python" ]; then
  "${UV_BIN}" pip install --no-deps -e "${DEV_WORK_ROOT}/repos/sglang/python"
else
  echo "No local sglang/python override found; install the official SOAR/SGLang runtime before running benchmarks." >&2
fi

echo "SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CUDA_HOME=${CUDA_HOME:-}"
