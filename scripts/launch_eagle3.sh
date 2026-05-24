#!/usr/bin/env bash
set -euo pipefail

# Start MiniCPM-SALA with a prepared EAGLE3 draft model.
#
# The default remains the smoke-proven dense fallback path. The native
# minicpm_flashinfer path now has local target-verify/sparse-cache patches and
# passes short serving smoke, but the current synthetic draft is not a candidate
# for accuracy or official speed.

DATASET_LOCAL_ROOT="${DATASET_LOCAL_ROOT:-/home/dataset-local}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-/home/dataset-local/work/SOAR/artifacts/logs}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
FORCE_DENSE_MINICPM="${FORCE_DENSE_MINICPM:-1}"
DENSE_AS_SPARSE="${DENSE_AS_SPARSE:-1}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
SPECULATIVE_ACCEPT_THRESHOLD_SINGLE="${SPECULATIVE_ACCEPT_THRESHOLD_SINGLE:-1.0}"
SPECULATIVE_ACCEPT_THRESHOLD_ACC="${SPECULATIVE_ACCEPT_THRESHOLD_ACC:-1.0}"
SPECULATIVE_DRAFT_ATTENTION_BACKEND="${SPECULATIVE_DRAFT_ATTENTION_BACKEND:-flashinfer}"
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

if [ -z "${DRAFT_MODEL_PATH}" ]; then
  echo "DRAFT_MODEL_PATH is required for EAGLE3 serving." >&2
  echo "Expected a MiniCPM-SALA-compatible draft model/head, for example under artifacts/draft_heads/." >&2
  exit 1
fi

if [ ! -d "${DRAFT_MODEL_PATH}" ]; then
  echo "Draft model path not found: ${DRAFT_MODEL_PATH}" >&2
  exit 1
fi

read -r -a EXTRA_SERVER_ARGV <<< "${EXTRA_SERVER_ARGS}"
DENSE_ARGS=()
if [ "${DENSE_AS_SPARSE}" = "1" ]; then
  DENSE_ARGS+=(--dense-as-sparse)
fi
FORCE_DENSE_ARGS=()
if [ "${FORCE_DENSE_MINICPM}" = "1" ]; then
  FORCE_DENSE_ARGS+=(--force-dense-minicpm)
fi

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-radix-cache \
  --attention-backend "${ATTENTION_BACKEND}" \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  "${DENSE_ARGS[@]}" \
  "${FORCE_DENSE_ARGS[@]}" \
  --dtype float16 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "${DRAFT_MODEL_PATH}" \
  --speculative-num-steps "${SPECULATIVE_NUM_STEPS}" \
  --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}" \
  --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}" \
  --speculative-draft-attention-backend "${SPECULATIVE_DRAFT_ATTENTION_BACKEND}" \
  --speculative-accept-threshold-single "${SPECULATIVE_ACCEPT_THRESHOLD_SINGLE}" \
  --speculative-accept-threshold-acc "${SPECULATIVE_ACCEPT_THRESHOLD_ACC}" \
  "${EXTRA_SERVER_ARGV[@]}" \
  2>&1 | tee "${LOG_DIR}/eagle3_server.log"
