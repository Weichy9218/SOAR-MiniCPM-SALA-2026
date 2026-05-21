#!/usr/bin/env bash
set -euo pipefail

# Download or link MiniCPM-SALA for local smoke tests.
# This is a local helper, not part of the submit package.

MODEL_DIR="${MODEL_DIR:-/models/MiniCPM-SALA}"
SOURCE="${SOURCE:-modelscope}"

mkdir -p "$(dirname "${MODEL_DIR}")"

if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
  echo "Model already exists: ${MODEL_DIR}"
  exit 0
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
