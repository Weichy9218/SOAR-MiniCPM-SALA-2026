#!/usr/bin/env bash
set -euo pipefail

# Prepare a model directory without copying the full original model into the submit package.
# Default mode is a symlink. Set SOAR_MODEL_PREP=quantize_rtn_gptq to run the
# fallback RTN-to-GPTQ helper when it is included in the submit package.

INPUT=""
OUTPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input)
      INPUT="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "${INPUT}" ] || [ -z "${OUTPUT}" ]; then
  echo "Usage: $0 --input <model_path> --output <prepared_model_path>" >&2
  exit 2
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/dataset-local/.cache/uv}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/dataset-local/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/home/dataset-local/tmp}"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"

if [ ! -d "${INPUT}" ]; then
  echo "Input model path does not exist: ${INPUT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

case "${SOAR_MODEL_PREP:-symlink}" in
  symlink)
    ln -sfn "${INPUT}" "${OUTPUT}"
    echo "Prepared model symlink: ${OUTPUT} -> ${INPUT}"
    ;;
  quantize_rtn_gptq)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 "${SCRIPT_DIR}/scripts/quantize_gptq_rtn.py" \
      --input "${INPUT}" \
      --output "${OUTPUT}" \
      --group-size "${SOAR_GPTQ_GROUP_SIZE:-128}" \
      --bits "${SOAR_GPTQ_BITS:-4}"
    ;;
  *)
    echo "Unknown SOAR_MODEL_PREP=${SOAR_MODEL_PREP}; use symlink or quantize_rtn_gptq." >&2
    exit 2
    ;;
esac
