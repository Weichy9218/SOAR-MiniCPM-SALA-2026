# Baseline Summary

Status: **blocked before server start**

## Facts

- Timestamp: 2026-05-21T11:30:50+08:00
- Work root: `/home/dataset-local/work/SOAR`
- Environment log: `/home/dataset-local/work/SOAR/artifacts/logs/env_info.log`
- uv cache: `/home/dataset-local/.cache/uv`
- model root: `/home/dataset-local/models`
- Hugging Face mirror: `HF_ENDPOINT=https://hf-mirror.com`
- Local CUDA toolkit: `/home/dataset-local/cuda-13.1`
- GPU: NVIDIA A100-SXM4-80GB, 0 MiB used at snapshot time
- Python: 3.12.4
- `torch`: not installed
- `sglang`: not installed
- `/home/dataset-local/models/MiniCPM-SALA`: not found
- Upstream model repository size: about 19GB, larger than the current free space on `/home/dataset-local`.
- SOAR-Toolkit: cloned and available at `/home/dataset-local/work/SOAR/repos/SOAR-Toolkit`
- SGLang source: cloned and available at `/home/dataset-local/work/SOAR/repos/sglang`
- SpecForge source: cloned and available at `/home/dataset-local/work/SOAR/repos/SpecForge`
- Repository revisions: SOAR `2ed4ade`, SGLang `791a2f0`, SpecForge `d5fb617`, SpecForge PR #492 `4385d78`

## Official Baseline Server Args

```bash
python -m sglang.launch_server \
  --model-path /home/dataset-local/models/MiniCPM-SALA \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

## Metrics

| metric | value |
|---|---:|
| ori_accuracy | 0.XXX |
| overall_accuracy | 0.XXX |
| S1 Benchmark Duration | 0.XXX |
| S8 Benchmark Duration | 0.XXX |
| Smax Benchmark Duration | 0.XXX |
| TTFT | 0.XXX |
| ITL | 0.XXX |
| output tokens/s | 0.XXX |
| GPU memory peak | 0.XXX |
| OOM | not evaluated |
| server crash | not evaluated |

## Blockers

1. `/home/dataset-local/models/MiniCPM-SALA` does not exist.
2. Base Python environment lacks `torch`, `sglang`, `transformers`, `flashinfer`, and `sgl-kernel`.
3. No Docker/Podman runtime is available to use the official SOAR image directly.
4. Current `/home/dataset-local` free space is about 12GB, below the upstream model size.
5. Hugging Face direct access timed out earlier; new default is `HF_ENDPOINT=https://hf-mirror.com`, with ModelScope as fallback.

## CUDA Note

SOAR Toolkit does not pin a CUDA toolkit version in its submission README. The local SGLang source currently depends on a CUDA 13 stack (`cuda-python>=13.0`, `flashinfer_python[cu13]`, `torch==2.11.0`), so local setup should prefer `/home/dataset-local/cuda-13.1` and avoid mixing with the older `torch==2.6.0+cu124` path.

## Attempted Smoke Test

Command:

```bash
bash /home/dataset-local/work/SOAR/scripts/run_baseline_smoke.sh
```

Result:

```text
Model path not found: /home/dataset-local/models/MiniCPM-SALA
```

Log:

- `/home/dataset-local/work/SOAR/artifacts/logs/baseline_smoke_blocked.log`

## Next Action

Reuse `/home/dataset-local/.cache/uv`, run the local GPU micro goal, make `/home/dataset-local/models/MiniCPM-SALA` available, then install the full SGLang/SOAR runtime and run the 10-sample accuracy smoke test before any quantization or speculative decoding work.

Local model helper:

```bash
SOURCE=huggingface HF_ENDPOINT=https://hf-mirror.com \
MODEL_DIR=/home/dataset-local/models/MiniCPM-SALA \
bash /home/dataset-local/work/SOAR/scripts/prepare_local_model.sh
```
