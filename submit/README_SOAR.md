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

Full public accuracy completed on the 150-sample `perf_public_set.jsonl` with `--disable-cuda-graph`: `ori_accuracy=82.53`, `overall_accuracy=100.00`, duration 17260.25s, total_tokens 733451.

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

Local proxy speed self-test:

```bash
RUN_NAME=baseline_fp16 \
bash /home/dataset-local/work/SOAR/scripts/run_local_proxy_speed.sh
```

This proxy split is generated from public accuracy prompts and is marked `official=false`; it is not a replacement for the official SOAR speed split.

Restartable public accuracy:

```bash
CHUNK_SIZE=10 ACCURACY_CONCURRENCY=4 \
bash /home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh
```

## Modified Runtime Notes

- Local SGLang source is loaded from `/home/dataset-local/work/SOAR/repos/soar_demo_sala/sglang/python`.
- A Transformers 5 compatibility patch normalizes chat template `BatchEncoding.input_ids` to `List[int]`; without it `/v1/chat/completions` returns HTTP 400.
- MiniCPM-SALA RoPE/JIT compatibility patches are local to the ignored third-party SGLang checkout and are required for this environment.
- MiniCPM sparse attention compressed-key dtype is patched locally to follow the model runtime dtype. This keeps FP16 baseline behavior intact and lets GPTQ `--dtype float16` smoke avoid query/key dtype mismatch.
- EAGLE3 local patches add MiniCPM helper methods, smoke-training compatibility, native sparse target-verify metadata, and sparse K1/K2 cache accounting. Dense fallback and native sparse serving smokes pass, but the current draft is a synthetic 1-step smoke draft, so EAGLE3 is not part of the current candidate submission.
- `prepare_model.sh` still defaults to symlinking/preparing the official model and does not copy model weights into the submit package.

## Experiment Results

| run_id | target_precision | draft_loss | quantization | kv_cache_dtype | accuracy_ori | accuracy_overall | s1_duration | s8_duration | smax_duration | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_fp16_public | FP16 | none | none | default | 82.53 | 100.00 | 0.78 | 1.75 | 0.97 | Full public accuracy 150/150 complete with `--disable-cuda-graph`; S1/S8/Smax are 2-request smoke durations, official speed split pending |
| baseline_fp16_local_proxy | FP16 | none | none | default | 82.53 | 100.00 | 136.91 | 162.14 | 224.11 | Local proxy only, `official=false`; do not report as official speed |
| gptq_rtn_gs1024_local_proxy | W4A16 | none | gptq | auto | 0.XXX | 0.XXX | 141.52 | 172.86 | 238.65 | Local proxy only; slower than FP16 proxy; full public accuracy not evaluated |
| gptq_rtn_sym_gs128_local_proxy | W4A16 | none | gptq | auto | 0.XXX | 0.XXX | 137.64 | 160.99 | 222.38 | Local proxy only; mixed result vs FP16 proxy; 10-sample accuracy probe stopped after 20:47 with 0/10 completed |
| eagle3_synthetic_dense_smoke | FP16 | EAGLE3 synthetic 1-step | none | default | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX | Dense fallback plumbing smoke only; two short chat requests passed, no official speed or accuracy |
| eagle3_sparse_synthetic_local_proxy | FP16 | EAGLE3 synthetic 1-step | none | default | 0.XXX | 0.XXX | 124.42 | 171.20 | 242.06 | Native sparse smoke/proxy only, `official=false`; S8/Smax slower than FP16 proxy and smoke output quality is poor |
| eagle3_sparse_synthetic_probe_s16 | FP16 | EAGLE3 synthetic_probe 16-step | none | default | 0.XXX | 0.XXX | 134.69 | 170.99 | 240.81 | Native sparse smoke/proxy only, `official=false`; S8/Smax slower than FP16 proxy and smoke output is degenerate |

## Known Failed / Skipped Items

- Full official speed split: not completed because local official S1/S8/Smax split files are not present.
- Local proxy speed split: generated for self-test only from public prompts; not official.
- CUDA graph enabled baseline: Smax smoke crashes in `minicpm_flashinfer` replay metadata.
- FP8 KV cache: `fp8_e5m2` fails first smoke request because FlashAttention on this path only supports fp16/bf16; `fp8_e4m3` fails with Triton `fp8e4nv` unsupported.
- GPTQ RTN gs1024: checkpoint generated and probe run, but local proxy is slower than FP16 and full public accuracy is pending.
- GPTQ RTN sym gs128: ordinary GPTQ smoke and proxy run complete, but improvement is small/mixed; a 10-sample public accuracy probe stalled at 0/10 after 20:47, so full public accuracy is skipped until long generation is fixed.
- GPTQ Marlin: sym gs128 checkpoint is metadata-compatible, but current `sgl_kernel==0.4.2.post2` lacks `gptq_marlin_repack`; gs1024 is not Marlin-compatible because current Marlin group sizes are `-1/32/64/128`.
- EAGLE3 / LK: synthetic 1-step and synthetic-probe 16-step EAGLE3 drafts, dense fallback smoke, and native sparse `minicpm_flashinfer` smoke/proxy are present. Both synthetic draft proxies are slower than FP16 on S8/Smax and have visibly poor output, so no meaningful speculative candidate has been recorded.
