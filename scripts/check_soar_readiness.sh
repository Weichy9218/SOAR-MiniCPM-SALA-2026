#!/usr/bin/env bash
set -euo pipefail

# Fast readiness check before attempting GPU runs.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
MODEL_ROOT="${MODEL_ROOT:-/home/dataset-local/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/MiniCPM-SALA}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/dataset-local/.cache/uv}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/home/dataset-local/.cache/huggingface}"
LOCAL_CUDA_HOME="${LOCAL_CUDA_HOME:-/home/dataset-local/cuda-13.1}"

echo "work_root=${WORK_ROOT}"
echo "model_root=${MODEL_ROOT}"
echo "model_path=${MODEL_PATH}"
echo "uv_cache_dir=${UV_CACHE_DIR}"
echo "hf_endpoint=${HF_ENDPOINT}"
echo "hf_home=${HF_HOME}"
echo "local_cuda_home=${LOCAL_CUDA_HOME}"
echo "cuda_home=${CUDA_HOME:-}"
test -d "${WORK_ROOT}/repos/SOAR-Toolkit" && echo "SOAR-Toolkit=present" || echo "SOAR-Toolkit=missing"
test -d "${WORK_ROOT}/repos/sglang/python/sglang" && echo "sglang_source=present" || echo "sglang_source=missing"
test -d "${WORK_ROOT}/repos/SpecForge/specforge" && echo "SpecForge=present" || echo "SpecForge=missing"
test -f "${MODEL_PATH}/config.json" && echo "model=present" || echo "model=missing"
test -x "${LOCAL_CUDA_HOME}/bin/nvcc" && "${LOCAL_CUDA_HOME}/bin/nvcc" --version | tail -n 1 || true

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${MODEL_ROOT}"
df -h "${WORK_ROOT}" "${MODEL_ROOT}" "${UV_CACHE_DIR}" 2>/dev/null || true

python --version
python - <<'PY'
import importlib.metadata as md
for pkg in ["torch", "sglang", "transformers", "flashinfer-python", "sglang-kernel", "sgl-kernel"]:
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception:
        print(f"{pkg}=missing")
PY
python - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"torch_cuda=unavailable ({exc})")
else:
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        print(f"torch_cuda_device={torch.cuda.get_device_name(index)}")
        print(f"torch_cuda_mem_free_gb={free / 1024 ** 3:.2f}")
        print(f"torch_cuda_mem_total_gb={total / 1024 ** 3:.2f}")
PY
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader || true
