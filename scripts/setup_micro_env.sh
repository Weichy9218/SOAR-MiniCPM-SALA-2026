#!/usr/bin/env bash
set -euo pipefail

# Build a small local environment for GPU/model-path smoke tests.
# It reuses the dataset-local uv cache and does not install into base conda.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
VENV_DIR="${VENV_DIR:-${WORK_ROOT}/.venv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/dataset-local/.cache/uv}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/home/dataset-local/.cache/huggingface}"
UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
TMPDIR="${TMPDIR:-/home/dataset-local/tmp}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu130}"
LOCAL_CUDA_HOME="${LOCAL_CUDA_HOME:-/home/dataset-local/cuda-13.1}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers==5.8.1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export UV_CACHE_DIR HF_ENDPOINT HF_HOME UV_LINK_MODE TMPDIR UV_TORCH_BACKEND

if [ -d "${LOCAL_CUDA_HOME}" ]; then
  export CUDA_HOME="${CUDA_HOME:-${LOCAL_CUDA_HOME}}"
  export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TMPDIR}" "$(dirname "${VENV_DIR}")"

if [ -x "${VENV_DIR}/bin/python" ]; then
  echo "Reusing existing virtual environment: ${VENV_DIR}"
else
  uv venv --python "${PYTHON_BIN}" --allow-existing "${VENV_DIR}"
fi

uv pip install --python "${VENV_DIR}/bin/python" \
  --torch-backend "${UV_TORCH_BACKEND}" \
  --index-strategy unsafe-best-match \
  "${TORCH_SPEC}" \
  "${TRANSFORMERS_SPEC}" \
  huggingface_hub \
  safetensors

"${VENV_DIR}/bin/python" - <<'PY'
import os
import torch
import transformers

print(f"cuda_home={os.environ.get('CUDA_HOME', '')}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    print(f"cuda_device={torch.cuda.get_device_name(index)}")
    print(f"cuda_mem_free_gb={free / 1024 ** 3:.2f}")
    print(f"cuda_mem_total_gb={total / 1024 ** 3:.2f}")
PY
