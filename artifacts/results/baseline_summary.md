# Baseline Summary

Status: **blocked before server start**

## Facts

- Timestamp: 2026-05-21T11:30:50+08:00
- Work root: `/home/dataset-local/work/SOAR`
- Environment log: `/home/dataset-local/work/SOAR/artifacts/logs/env_info.log`
- GPU: NVIDIA A100-SXM4-80GB, 0 MiB used at snapshot time
- Python: 3.12.4
- `torch`: not installed
- `sglang`: not installed
- `/models/MiniCPM-SALA`: not found
- Upstream model repository size: about 19GB, larger than the current free space on `/home/dataset-local`.
- SOAR-Toolkit: cloned and available at `/home/dataset-local/work/SOAR/repos/SOAR-Toolkit`
- SGLang source: cloned and available at `/home/dataset-local/work/SOAR/repos/sglang`
- SpecForge source: cloned and available at `/home/dataset-local/work/SOAR/repos/SpecForge`
- Repository revisions: SOAR `2ed4ade`, SGLang `791a2f0`, SpecForge `d5fb617`, SpecForge PR #492 `4385d78`

## Official Baseline Server Args

```bash
python -m sglang.launch_server \
  --model-path /models/MiniCPM-SALA \
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

1. `/models/MiniCPM-SALA` does not exist.
2. Base Python environment lacks `torch`, `sglang`, `transformers`, `flashinfer`, and `sgl-kernel`.
3. No Docker/Podman runtime is available to use the official SOAR image directly.
4. Current `/home/dataset-local` free space is about 12GB, below the upstream model size.
5. ModelScope is reachable, Hugging Face direct access timed out in this environment.

## Attempted Smoke Test

Command:

```bash
bash /home/dataset-local/work/SOAR/scripts/run_baseline_smoke.sh
```

Result:

```text
Model path not found: /models/MiniCPM-SALA
```

Log:

- `/home/dataset-local/work/SOAR/artifacts/logs/baseline_smoke_blocked.log`

## Next Action

Install the SGLang/SOAR runtime, make `/models/MiniCPM-SALA` available, then run the 10-sample accuracy smoke test before any quantization or speculative decoding work.

Local model helper:

```bash
SOURCE=modelscope MODEL_DIR=/models/MiniCPM-SALA \
bash /home/dataset-local/work/SOAR/scripts/prepare_local_model.sh
```
