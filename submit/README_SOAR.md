# SOAR MiniCPM-SALA Rebuild Notes

## Environment Snapshot

- Work root: `/home/dataset-local/work/SOAR`
- Model path: `/home/dataset-local/models/MiniCPM-SALA`
- Model status: present, 4 safetensors shards, about 18GB.
- uv cache: `/home/dataset-local/.cache/uv`
- Hugging Face mirror: `HF_ENDPOINT=https://hf-mirror.com`
- Local CUDA toolkit: `/home/dataset-local/cuda-13.1`
- Current Python: `Python 3.12.4`
- Current GPU: NVIDIA A100-SXM4-80GB.
- Current SOAR `.venv`: `torch==2.11.0+cu130`, `transformers==5.8.1`, `flashinfer-python==0.6.11.post1`, `sglang-kernel==0.4.2.post2`, `sgl-kernel==0.4.2.post2`, `xgrammar==0.2.1`, `datasets==4.8.5`.
- Local temp/cache guardrails: `TMPDIR=/home/dataset-local/tmp`, `UV_CACHE_DIR=/home/dataset-local/.cache/uv`, `HF_HOME=/home/dataset-local/.cache/huggingface`, `XDG_CACHE_HOME=/home/dataset-local/.cache`, `HOME=/home/dataset-local`.

## Repository Layout

```text
/home/dataset-local/work/SOAR/
  repos/
    SOAR-Toolkit/
    soar_demo_sala/
    SpecForge/
  scripts/
  artifacts/
    logs/
    results/
  submit/
    prepare_env.sh
    prepare_model.sh
    README_SOAR.md
```

## Baseline Launch Command

Required official SGLang server arguments remain present:

```bash
python -m sglang.launch_server \
  --model-path /home/dataset-local/models/MiniCPM-SALA \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

Current stable local launch adds `--disable-cuda-graph` by default through `scripts/launch_baseline.sh`.

Reason: with CUDA graph enabled, the Smax smoke path crashes in `minicpm_flashinfer` CUDA graph replay metadata with `kv_indptr[...] should be non-negative`. Use `DISABLE_CUDA_GRAPH=0` only to reproduce/debug that failure.

## Baseline Status

FP16 baseline smoke is runnable.

Verified smoke:

- `/v1/models`: passed
- 2 short `/v1/chat/completions` requests: passed
- `bench_serving.sh` on 2-request smoke data:
  - S1: 0.78s
  - S8: 1.75s
  - Smax: 0.97s

Logs:

- `/home/dataset-local/work/SOAR/artifacts/logs/baseline_server.log`
- `/home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_smoke.log`
- `/home/dataset-local/work/SOAR/artifacts/logs/baseline_speed_smoke.log`
- `/home/dataset-local/work/SOAR/artifacts/results/baseline_smoke_predictions.jsonl`

Full public accuracy has not been completed yet. Do not report `ori_accuracy` or `overall_accuracy` until the full 150-sample `perf_public_set.jsonl` run finishes.

## Benchmark Commands

Smoke:

```bash
export UV_CACHE_DIR=/home/dataset-local/.cache/uv
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/dataset-local/.cache/huggingface
export MODEL_ROOT=/home/dataset-local/models
export MODEL_PATH=/home/dataset-local/models/MiniCPM-SALA
export LOCAL_CUDA_HOME=/home/dataset-local/cuda-13.1
export CUDA_HOME=/home/dataset-local/cuda-13.1
export TMPDIR=/home/dataset-local/tmp
export HOME=/home/dataset-local
export XDG_CACHE_HOME=/home/dataset-local/.cache

source /home/dataset-local/work/SOAR/.venv/bin/activate
bash /home/dataset-local/work/SOAR/scripts/check_soar_readiness.sh
bash /home/dataset-local/work/SOAR/scripts/run_micro_goal.sh
bash /home/dataset-local/work/SOAR/scripts/launch_baseline.sh
bash /home/dataset-local/work/SOAR/scripts/run_baseline_smoke.sh
```

Full public accuracy and available speed splits:

```bash
bash /home/dataset-local/work/SOAR/scripts/launch_baseline.sh
SPEED_DATA_S1=/path/to/s1.jsonl \
SPEED_DATA_S8=/path/to/s8.jsonl \
SPEED_DATA_SMAX=/path/to/smax.jsonl \
bash /home/dataset-local/work/SOAR/scripts/run_full_baseline.sh
```

## Modified Runtime Notes

- Local SGLang source is loaded from `/home/dataset-local/work/SOAR/repos/soar_demo_sala/sglang/python`.
- A Transformers 5 compatibility patch normalizes chat template `BatchEncoding.input_ids` to `List[int]`; without it `/v1/chat/completions` returns HTTP 400.
- MiniCPM-SALA RoPE/JIT compatibility patches are local to the ignored third-party SGLang checkout and are required for this environment.
- `prepare_model.sh` still defaults to symlinking/preparing the official model and does not copy model weights into the submit package.

## Experiment Results

| run_id | target_precision | draft_loss | quantization | kv_cache_dtype | accuracy_ori | accuracy_overall | s1_duration | s8_duration | smax_duration | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_fp16_smoke | FP16 | none | none | default | 0.XXX | 0.XXX | 0.78 | 1.75 | 0.97 | Smoke only, `--disable-cuda-graph`; full accuracy pending |

## Known Failed / Skipped Items

- Full public accuracy: not completed.
- Full official speed split: not completed because local official S1/S8/Smax split files are not present.
- CUDA graph enabled baseline: Smax smoke crashes in `minicpm_flashinfer` replay metadata.
- EAGLE3 / LK / quantization: intentionally not started before full FP16 baseline accuracy.
