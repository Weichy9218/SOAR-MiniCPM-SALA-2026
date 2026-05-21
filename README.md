# SOAR MiniCPM-SALA 2026

This repository is the organized working tree for rebuilding a submit-ready MiniCPM-SALA inference optimization entry for SOAR.

## Current Status

- Official baseline is blocked before server start in this container because `/models/MiniCPM-SALA` and runtime dependencies are missing.
- SOAR/SGLang/SpecForge source checkouts are kept locally under `repos/` but are intentionally ignored by git.
- Key reproducibility scripts, result placeholders, and submit package files are tracked from this repository root.

## Layout

```text
SOAR/
  README.md
  scripts/                 # local launch, smoke, readiness, quantization helpers
  submit/                  # lightweight submit package scaffold
  artifacts/results/       # tracked result summaries/placeholders
  artifacts/logs/          # local logs, ignored by git
  artifacts/checkpoints/   # local checkpoints, ignored by git
  artifacts/draft_heads/   # local draft heads, ignored by git
  repos/                   # local third-party source checkouts, ignored by git
```

## Local Source Revisions

- SOAR-Toolkit: `2ed4ade`
- SGLang: `791a2f0`
- SpecForge main: `d5fb617`
- SpecForge LK PR #492: `4385d78`

## First Checks

```bash
bash scripts/check_soar_readiness.sh
bash scripts/run_baseline_smoke.sh
```

The smoke script exits early if `/models/MiniCPM-SALA` is missing.

## Baseline Launch

```bash
bash scripts/launch_baseline.sh
```

Equivalent server args:

```bash
python -m sglang.launch_server \
  --model-path /models/MiniCPM-SALA \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

## Submit Package

The current lightweight submit package is under `submit/`. It contains:

- `prepare_env.sh`
- `prepare_model.sh`
- `README_SOAR.md`
- `scripts/`

It intentionally does not include model weights, checkpoints, logs, or full third-party repositories.

