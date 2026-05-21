#!/usr/bin/env bash
set -euo pipefail

# Download or link MiniCPM-SALA for local smoke tests.
# This is a local helper, not part of the submit package.

MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_DIR="${MODEL_DIR:-${MODEL_ROOT}/MiniCPM-SALA}"
SOURCE="${SOURCE:-huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/dataset-local/.cache/uv}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/home/dataset-local/.cache/huggingface}"

export UV_CACHE_DIR HF_ENDPOINT HF_HOME

mkdir -p "$(dirname "${MODEL_DIR}")" "${UV_CACHE_DIR}" "${HF_HOME}"

if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
  echo "Model already exists: ${MODEL_DIR}"
  exit 0
fi

AVAILABLE_GB="$(df -BG "$(dirname "${MODEL_DIR}")" | awk 'NR == 2 { gsub("G", "", $4); print $4 }')"
if [ -n "${AVAILABLE_GB}" ] && [ "${AVAILABLE_GB}" -lt "${MIN_MODEL_FREE_GB:-25}" ]; then
  echo "Warning: only ${AVAILABLE_GB}GB is available under $(dirname "${MODEL_DIR}"). Full MiniCPM-SALA may not fit." >&2
fi

case "${SOURCE}" in
  modelscope)
    python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("modelscope") else 1)
PY
    modelscope download --model OpenBMB/MiniCPM-SALA --local_dir "${MODEL_DIR}"
    ;;
  huggingface)
    python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
    huggingface-cli download OpenBMB/MiniCPM-SALA --local-dir "${MODEL_DIR}"
    ;;
  *)
    echo "Unknown SOURCE=${SOURCE}; use modelscope or huggingface." >&2
    exit 2
    ;;
esac

test -f "${MODEL_DIR}/config.json"
echo "Prepared model at ${MODEL_DIR}"
