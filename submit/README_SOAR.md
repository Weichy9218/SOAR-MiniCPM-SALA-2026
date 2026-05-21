# SOAR MiniCPM-SALA Rebuild Notes

## Environment Snapshot

- Work root: `/home/dataset-local/work/SOAR`
- Environment log: `/home/dataset-local/work/SOAR/artifacts/logs/env_info.log`
- Current blocker: `/home/dataset-local/models/MiniCPM-SALA` is not present on this container.
- Model root: `/home/dataset-local/models`
- uv cache: `/home/dataset-local/.cache/uv`
- Hugging Face mirror: `HF_ENDPOINT=https://hf-mirror.com`
- Local CUDA toolkit: `/home/dataset-local/cuda-13.1`
- The upstream Hugging Face repository is about 19GB, larger than the current free space on `/home/dataset-local`.
- Current Python: `Python 3.12.4`
- Current CUDA driver runtime from `nvidia-smi`: CUDA 13.0, A100-SXM4-80GB.
- Current Python packages: `torch`, `sglang`, `transformers`, `flashinfer`, and `sgl-kernel` are not installed in the base environment.

## Repository Layout

```text
/home/dataset-local/work/SOAR/
  repos/
    SOAR-Toolkit/
    sglang/
    SpecForge/
  scripts/
  artifacts/
    logs/
    results/
    checkpoints/
    draft_heads/
  submit/
    prepare_env.sh
    prepare_model.sh
    README_SOAR.md
```

Repository revisions:

- SOAR-Toolkit: `2ed4ade`
- SGLang: `791a2f0`
- SpecForge main: `d5fb617`
- SpecForge LK PR #492 ref: `4385d78`

## Baseline Launch Command

Target official SGLang server arguments:

```bash
python -m sglang.launch_server \
  --model-path /home/dataset-local/models/MiniCPM-SALA \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

## Baseline Status

Baseline smoke was attempted and blocked before server start because the model path and runtime dependencies are unavailable in the current base environment.

Readiness log:

- `/home/dataset-local/work/SOAR/artifacts/logs/readiness.log`
- `/home/dataset-local/work/SOAR/artifacts/logs/baseline_smoke_blocked.log`

Required before baseline smoke:

1. Reuse `/home/dataset-local/.cache/uv` and run the local micro goal.
2. Ensure `/home/dataset-local/models/MiniCPM-SALA` exists or prepare a valid symlink to the official model.
3. Install SGLang/SOAR runtime dependencies.
4. Start server with the command above.
5. Run `eval_model.py --num_samples 10` smoke test.
6. Run small serving benchmark smoke test.

## Benchmark Commands

Smoke:

```bash
export UV_CACHE_DIR=/home/dataset-local/.cache/uv
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/dataset-local/.cache/huggingface
export MODEL_ROOT=/home/dataset-local/models
export MODEL_PATH=/home/dataset-local/models/MiniCPM-SALA
export LOCAL_CUDA_HOME=/home/dataset-local/cuda-13.1
bash /home/dataset-local/work/SOAR/scripts/setup_micro_env.sh
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

Expected output files:

- `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy.json`
- `/home/dataset-local/work/SOAR/artifacts/results/baseline_speed_s1.json`
- `/home/dataset-local/work/SOAR/artifacts/results/baseline_speed_s8.json`
- `/home/dataset-local/work/SOAR/artifacts/results/baseline_speed_smax.json`
- `/home/dataset-local/work/SOAR/artifacts/results/baseline_summary.md`

## Modified Files

No runtime source files modified yet. Added reproducibility scripts under `/home/dataset-local/work/SOAR/scripts` and submit scaffolding under `/home/dataset-local/work/SOAR/submit`.

## Quantization Path

SGLang source supports `--quantization gptq`, `--quantization gptq_marlin`, and `--kv-cache-dtype {fp8_e5m2,fp8_e4m3,bfloat16}`. SOAR's `demo-quant.tar.gz` provides an RTN-to-GPTQ format example, but the README warns it is a flow demo rather than a final performance configuration.

`prepare_model.sh` defaults to symlinking the official model. It can optionally run a fallback RTN-to-GPTQ conversion when `SOAR_MODEL_PREP=quantize_rtn_gptq`, but that path is not selected for final use until accuracy confirms it is safe.

Planned launch once a quantized model is available:

```bash
MODEL_PATH=/path/to/MiniCPM-SALA-GPTQ \
QUANTIZATION=gptq_marlin \
KV_CACHE_DTYPE=fp8_e5m2 \
bash /home/dataset-local/work/SOAR/scripts/launch_quant.sh
```

## Speculative Path

SGLang source has built-in `EAGLE3` argument parsing and workers. `topk=1` automatically enforces `num_draft_tokens = num_steps + 1`, which matches the conservative chain-speculation route. This work is intentionally paused until the official baseline smoke test runs successfully.

SpecForge PR #492 was fetched as local branch `pr-492-lk-loss`. It contains `specforge/core/lk_loss.py`, `scripts/train_eagle3.py --lk-loss-type {lambda,alpha}`, `--kl-scale`, `--kl-decay`, and per-position `acceptance_rate_i` logging. Its lambda implementation uses `1 - acceptance_rate` as the TV-side term, so it must be checked against the exact LK^lambda formula before being used for final experiments.

## Experiment Results

| run_id | target_precision | draft_loss | quantization | kv_cache_dtype | accuracy_ori | accuracy_overall | s1_duration | s8_duration | smax_duration | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_fp16 | FP16 | none | none | default | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX | Pending: model/runtime unavailable |

## Known Failed / Skipped Items

- Full baseline correctness and speed benchmark: skipped until model and dependencies are available.
- Local model download was not attempted because `/home/dataset-local` only has about 12GB free while the official model repository is about 19GB.
- Hugging Face direct access timed out earlier; local download now defaults to `HF_ENDPOINT=https://hf-mirror.com`, with ModelScope as a fallback. A local helper exists at `/home/dataset-local/work/SOAR/scripts/prepare_local_model.sh` and defaults to `/home/dataset-local/models/MiniCPM-SALA`.
- EAGLE3 / LK work: intentionally not started before baseline smoke.
- Multi-branch tree, Medusa, Lightning Attention parent-state propagation: out of scope for the first runnable submission path.
