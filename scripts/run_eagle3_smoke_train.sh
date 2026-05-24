#!/usr/bin/env bash
set -euo pipefail

# Train a tiny MiniCPM-SALA-compatible EAGLE3 draft head for serving smoke tests.

DATASET_LOCAL_ROOT="${DATASET_LOCAL_ROOT:-/home/dataset-local}"
WORKDIR="${WORKDIR:-/home/dataset-local/work/SOAR}"
MODEL_ROOT="${MODEL_ROOT:-${DATASET_LOCAL_ROOT}/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
SPECFORGE_ROOT="${SPECFORGE_ROOT:-${WORKDIR}/repos/SpecForge}"
ASSET_DIR="${ASSET_DIR:-${WORKDIR}/artifacts/draft_heads/eagle3_smoke_assets}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKDIR}/artifacts/draft_heads/minicpm_sala_eagle3_smoke}"
CACHE_DIR="${CACHE_DIR:-${WORKDIR}/artifacts/checkpoints/eagle3_smoke_cache}"
RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/artifacts/results}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/artifacts/logs}"
RUN_NAME="${RUN_NAME:-eagle3_smoke}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-4096}"
DATASET_MODE="${DATASET_MODE:-smoke}"
NUM_RECORDS="${NUM_RECORDS:-128}"
SEED="${SEED:-0}"
MAX_LENGTH="${MAX_LENGTH:-256}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TTT_LENGTH="${TTT_LENGTH:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EAGLE3_TARGET_BACKEND="${EAGLE3_TARGET_BACKEND:-sglang}"

TMPDIR="${TMPDIR:-${DATASET_LOCAL_ROOT}/tmp}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${DATASET_LOCAL_ROOT}/.cache}"
HF_HOME="${HF_HOME:-${DATASET_LOCAL_ROOT}/.cache/huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${DATASET_LOCAL_ROOT}/.cache/uv}"
if [ "${HOME:-}" = "/home/batchcom" ] && [ -d "${DATASET_LOCAL_ROOT}" ]; then
  HOME="${DATASET_LOCAL_ROOT}"
fi
export TMPDIR XDG_CACHE_HOME HF_HOME UV_CACHE_DIR HOME TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [ ! -d "${SPECFORGE_ROOT}" ]; then
  echo "SpecForge checkout not found: ${SPECFORGE_ROOT}" >&2
  exit 1
fi

export PYTHONPATH="${SPECFORGE_ROOT}:${PYTHONPATH:-}"

python "${WORKDIR}/scripts/prepare_eagle3_smoke_assets.py" \
  --model-path "${MODEL_PATH}" \
  --output-dir "${ASSET_DIR}" \
  --draft-vocab-size "${DRAFT_VOCAB_SIZE}" \
  --dataset-mode "${DATASET_MODE}" \
  --num-records "${NUM_RECORDS}" \
  --seed "${SEED}"

DRAFT_CONFIG="${ASSET_DIR}/minicpm_sala_eagle3_smoke_config.json"
TRAIN_DATA="${ASSET_DIR}/minicpm_sala_eagle3_${DATASET_MODE}_train.jsonl"

PYTHONPATH="${WORKDIR}/scripts:${SPECFORGE_ROOT}/scripts:${SPECFORGE_ROOT}:${PYTHONPATH:-}" \
torchrun --standalone --nproc_per_node 1 "${WORKDIR}/scripts/train_eagle3_minicpm_hf.py" \
  --target-model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --draft-model-config "${DRAFT_CONFIG}" \
  --train-data-path "${TRAIN_DATA}" \
  --target-model-backend "${EAGLE3_TARGET_BACKEND}" \
  --chat-template qwen \
  --num-epochs 1 \
  --max-num-steps "${MAX_NUM_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --max-length "${MAX_LENGTH}" \
  --ttt-length "${TTT_LENGTH}" \
  --attention-backend sdpa \
  --save-interval "${MAX_NUM_STEPS}" \
  --log-interval 1 \
  --build-dataset-num-proc 1 \
  --dataloader-num-workers 0 \
  --report-to none \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}_train.log"

DRAFT_MODEL_PATH="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'epoch_*_step_*' | sort | tail -1)"
if [ -z "${DRAFT_MODEL_PATH}" ]; then
  echo "No EAGLE3 checkpoint was produced under ${OUTPUT_DIR}" >&2
  exit 1
fi

SUMMARY_TXT="${RESULTS_DIR}/${RUN_NAME}_summary.txt"
SUMMARY_JSON="${RESULTS_DIR}/${RUN_NAME}_summary.json"

cat > "${SUMMARY_TXT}" <<EOF
status=trained
draft_model_path=${DRAFT_MODEL_PATH}
draft_config=${DRAFT_CONFIG}
train_data=${TRAIN_DATA}
target_model=${MODEL_PATH}
backend=${EAGLE3_TARGET_BACKEND}
max_num_steps=${MAX_NUM_STEPS}
max_length=${MAX_LENGTH}
batch_size=${BATCH_SIZE}
ttt_length=${TTT_LENGTH}
draft_vocab_size=${DRAFT_VOCAB_SIZE}
dataset_mode=${DATASET_MODE}
num_records=${NUM_RECORDS}
seed=${SEED}
log=${LOG_DIR}/${RUN_NAME}_train.log
EOF

python - "${SUMMARY_TXT}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

summary = {}
summary_txt = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
for line in summary_txt.read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        summary[key] = value
summary_json.write_text(json.dumps(summary, indent=2) + "\n")
print("DRAFT_MODEL_PATH=" + summary["draft_model_path"])
PY
