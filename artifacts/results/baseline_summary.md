# Baseline Summary

Status: **FP16 baseline smoke passed with CUDA graph disabled**

## Facts

- Timestamp: 2026-05-22T03:36:15+08:00
- Work root: `/home/dataset-local/work/SOAR`
- Model path: `/home/dataset-local/models/MiniCPM-SALA`
- Model files: `config.json` plus 4 safetensors shards, about 18GB total.
- uv cache: `/home/dataset-local/.cache/uv`
- Hugging Face mirror: `HF_ENDPOINT=https://hf-mirror.com`
- Local CUDA toolkit: `/home/dataset-local/cuda-13.1`
- Python: `3.12.4`
- GPU: NVIDIA A100-SXM4-80GB
- Runtime: `torch==2.11.0+cu130`, `transformers==5.8.1`, `flashinfer-python==0.6.11.post1`, `sglang-kernel==0.4.2.post2`, `sgl-kernel==0.4.2.post2`, `xgrammar==0.2.1`, `datasets==4.8.5`.
- Local patched SGLang source: `/home/dataset-local/work/SOAR/repos/soar_demo_sala/sglang/python`
- The default baseline launch script now adds `--disable-cuda-graph` via `DISABLE_CUDA_GRAPH=1`.
- Reason for the default: with CUDA graph enabled, the Smax smoke path crashes in `minicpm_flashinfer` CUDA graph replay metadata with `kv_indptr[...] should be non-negative`.
- `DISABLE_CUDA_GRAPH=0` can reproduce the original CUDA graph path for debugging.
- Local temp/cache controls are explicit: `TMPDIR=/home/dataset-local/tmp`, `UV_CACHE_DIR=/home/dataset-local/.cache/uv`, `HF_HOME=/home/dataset-local/.cache/huggingface`, `XDG_CACHE_HOME=/home/dataset-local/.cache`, and `HOME=/home/dataset-local` for TVM/JIT cache placement.

## Baseline Server Args

Required official args are still present:

```bash
python -m sglang.launch_server \
  --model-path /home/dataset-local/models/MiniCPM-SALA \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

Current stable local launch adds:

```bash
--disable-cuda-graph
```

## Smoke Metrics

These are **smoke** numbers from the 2-request `artifacts/results/speed_smoke.jsonl`, not full public benchmark results.

| metric | value |
|---|---:|
| chat completions smoke | passed |
| S1 Benchmark Duration | 0.78 s |
| S8 Benchmark Duration | 1.75 s |
| Smax Benchmark Duration | 0.97 s |
| server crash with `--disable-cuda-graph` | no |
| server crash with CUDA graph enabled | yes, Smax smoke |

Full public accuracy is running in restartable chunks because `SOAR-Toolkit/eval_model.py` uses `max_out_len=65536` and some samples produce multi-minute long generations. The current completed-chunks-only partial result is 100/150 samples with `ori_accuracy=76.00` and `overall_accuracy=95.00`; do not report it as final 150-sample accuracy.

Completed-chunk summaries:

- JSON: `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_completed_now.json`
- Markdown: `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_completed_now.md`
- Checkpoint directory: `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_chunks/`

## Logs

- Model download: `/home/dataset-local/work/SOAR/artifacts/logs/prepare_model_hf.log`
- Runtime installs: `/home/dataset-local/work/SOAR/artifacts/logs/install_xgrammar.log`, `/home/dataset-local/work/SOAR/artifacts/logs/install_datasets.log`
- Readiness: `/home/dataset-local/work/SOAR/artifacts/logs/check_soar_readiness_after_extensions.log`, `/home/dataset-local/work/SOAR/artifacts/logs/check_soar_readiness_after_torchao.log`
- Baseline server: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_server.log`
- Short chat smoke: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_smoke.log`
- Speed smoke: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_speed_smoke.log`
- Smoke predictions: `/home/dataset-local/work/SOAR/artifacts/results/baseline_smoke_predictions.jsonl`
- Chunked accuracy log: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_chunked.log`

## Remaining Baseline Work

1. Finish the running chunked public accuracy on all 150 samples and record final `ori_accuracy` / `overall_accuracy`.
2. Obtain or set official speed split paths via `SPEED_DATA_S1`, `SPEED_DATA_S8`, and `SPEED_DATA_SMAX`; only `perf_public_set.jsonl` is present locally.
3. Investigate the CUDA graph Smax crash separately; keep `--disable-cuda-graph` for correctness-stable baseline until fixed.
4. Do not start EAGLE3, LK loss, or quantization experiments before full FP16 baseline accuracy is recorded.
