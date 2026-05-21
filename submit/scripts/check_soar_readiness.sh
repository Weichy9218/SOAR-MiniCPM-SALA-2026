#!/usr/bin/env bash
set -euo pipefail

# Fast readiness check before attempting GPU runs.

WORK_ROOT="${WORK_ROOT:-/home/dataset-local/work/SOAR}"
MODEL_PATH="${MODEL_PATH:-/models/MiniCPM-SALA}"

echo "work_root=${WORK_ROOT}"
echo "model_path=${MODEL_PATH}"
test -d "${WORK_ROOT}/repos/SOAR-Toolkit" && echo "SOAR-Toolkit=present" || echo "SOAR-Toolkit=missing"
test -d "${WORK_ROOT}/repos/sglang/python/sglang" && echo "sglang_source=present" || echo "sglang_source=missing"
test -d "${WORK_ROOT}/repos/SpecForge/specforge" && echo "SpecForge=present" || echo "SpecForge=missing"
test -d "${MODEL_PATH}" && echo "model=present" || echo "model=missing"

python --version
python - <<'PY'
import importlib.metadata as md
for pkg in ["torch", "sglang", "transformers", "flashinfer-python", "sglang-kernel", "sgl-kernel"]:
    try:
        print(f"{pkg}={md.version(pkg)}")
    except Exception:
        print(f"{pkg}=missing")
PY
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader || true
