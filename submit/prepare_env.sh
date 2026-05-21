#!/usr/bin/env bash
set -euo pipefail

# Prepare the SOAR/SGLang runtime from the local submit package.
# This script intentionally uses uv pip and avoids interactive prompts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${SCRIPT_DIR}"
DEV_WORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:---disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --skip-server-warmup --dense-as-sparse}"

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
