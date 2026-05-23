#!/usr/bin/env bash
set -euo pipefail

# Local micro goal: verify the GPU/runtime/model path before launching SGLang.
# It is intentionally small and should finish quickly on the competition box.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/artifacts/logs}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/dataset-local/.cache/uv}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/home/dataset-local/.cache/huggingface}"
TMPDIR="${TMPDIR:-/home/dataset-local/tmp}"

export MODEL_PATH UV_CACHE_DIR HF_ENDPOINT HF_HOME TMPDIR

mkdir -p "${LOG_DIR}" "${MODEL_ROOT}" "${UV_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}"

{
  echo "== SOAR local micro goal =="
  date -Is
  bash "${WORK_ROOT}/scripts/check_soar_readiness.sh"

  python - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["MODEL_PATH"])
print(f"micro_model_path={model_path}")

try:
    import torch
except Exception as exc:
    raise SystemExit(f"micro_goal=blocked: torch import failed: {exc}") from exc

if not torch.cuda.is_available():
    raise SystemExit("micro_goal=blocked: torch.cuda.is_available() is false")

device = torch.cuda.current_device()
free, total = torch.cuda.mem_get_info(device)
print(f"micro_cuda_device={torch.cuda.get_device_name(device)}")
print(f"micro_cuda_mem_free_gb={free / 1024 ** 3:.2f}")
print(f"micro_cuda_mem_total_gb={total / 1024 ** 3:.2f}")

if not model_path.joinpath("config.json").is_file():
    print("micro_model=missing")
    print("micro_goal=runtime_ready_model_pending")
    raise SystemExit(0)

config = json.loads(model_path.joinpath("config.json").read_text())
print(f"micro_model_type={config.get('model_type', 'unknown')}")
print(f"micro_vocab_size={config.get('vocab_size', 'unknown')}")
print(f"micro_hidden_size={config.get('hidden_size', 'unknown')}")
print(f"micro_num_hidden_layers={config.get('num_hidden_layers', 'unknown')}")

try:
    from transformers import AutoTokenizer
except Exception as exc:
    raise SystemExit(f"micro_goal=blocked: transformers import failed: {exc}") from exc

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sample = tokenizer("SOAR micro goal", return_tensors="pt")
print(f"micro_tokenizer_class={tokenizer.__class__.__name__}")
print(f"micro_input_ids_shape={tuple(sample['input_ids'].shape)}")
print("micro_goal=passed")
PY
} 2>&1 | tee "${LOG_DIR}/micro_goal.log"
