# Baseline Summary

Status: **FP16 baseline public accuracy completed with CUDA graph disabled**

## Facts

- Timestamp: 2026-05-24T01:57:10+08:00
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
- Local MiniCPM sparse attention patch: compressed-attention key dtype now follows the model runtime dtype (`float16` or `bfloat16`) instead of forcing `bfloat16`; this is needed for GPTQ `--dtype float16` smoke and is in the ignored third-party checkout.
- The default baseline launch script now adds `--disable-cuda-graph` via `DISABLE_CUDA_GRAPH=1`.
- Reason for the default: with CUDA graph enabled, the Smax smoke path crashes in `minicpm_flashinfer` CUDA graph replay metadata with `kv_indptr[...] should be non-negative`.
- `DISABLE_CUDA_GRAPH=0` can reproduce the original CUDA graph path for debugging.
- The official SOAR speed split files are not present locally. `SOAR-Toolkit/README.md` indicates the public toolkit does not provide the competition speed testing dataset for the time being, so local code cannot recreate official S1/S8/Smax files.
- A deterministic local proxy speed split was generated from public accuracy prompts for self-test only. It is marked `official=false` and must not be reported as official SOAR speed.
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

## Full Public Accuracy

Full public accuracy completed through the restartable chunked runner because `SOAR-Toolkit/eval_model.py` uses `max_out_len=65536` and several samples produce long generations.

| metric | value |
|---|---:|
| status | complete |
| completed samples | 150/150 |
| ori_accuracy | 82.53 |
| overall_accuracy | 100.00 |
| duration | 17260.25 s |
| total tokens | 733451 |

The final aggregate is recorded in:

- JSON: `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_full_chunked_summary.json`
- Text: `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_full_chunked_summary.txt`
- Checkpoint directory: `/home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_chunks/`

Chunked resume evidence:

- 15/15 chunk `summary.json` files exist.
- `chunk_0010` through `chunk_0014` were resumed and completed in `artifacts/logs/baseline_accuracy_chunked_resume.log`.

## Local Proxy Speed

The local proxy split exists only to compare local serving candidates while the official speed split is unavailable.

| run | official | S1 Benchmark Duration | S8 Benchmark Duration | Smax Benchmark Duration |
|---|---|---:|---:|---:|
| baseline_fp16 | false | 136.91 s | 162.14 s | 224.11 s |
| gptq_rtn_gs1024 | false | 141.52 s | 172.86 s | 238.65 s |
| gptq_rtn_sym_gs128 | false | 137.64 s | 160.99 s | 222.38 s |
| eagle3_sparse_synthetic | false | 124.42 s | 171.20 s | 242.06 s |
| eagle3_sparse_synthetic_probe_s16 | false | 134.69 s | 170.99 s | 240.81 s |

Proxy split files:

- Manifest: `/home/dataset-local/work/SOAR/artifacts/results/local_proxy_speed/local_proxy_speed_manifest.json`
- S1: `/home/dataset-local/work/SOAR/artifacts/results/local_proxy_speed/local_proxy_speed_s1.jsonl`
- S8: `/home/dataset-local/work/SOAR/artifacts/results/local_proxy_speed/local_proxy_speed_s8.jsonl`
- Smax: `/home/dataset-local/work/SOAR/artifacts/results/local_proxy_speed/local_proxy_speed_smax.jsonl`

Proxy run outputs:

- FP16: `/home/dataset-local/work/SOAR/artifacts/results/baseline_fp16_local_proxy_speed.json`
- GPTQ RTN gs1024: `/home/dataset-local/work/SOAR/artifacts/results/gptq_rtn_gs1024_local_proxy_speed.json`
- GPTQ RTN sym gs128: `/home/dataset-local/work/SOAR/artifacts/results/gptq_rtn_sym_gs128_local_proxy_speed.json`

## Optimization Probes

- FP8 KV cache is not currently compatible with the `minicpm_flashinfer` dense-as-sparse path. `fp8_e5m2` fails the first smoke request with `FlashAttention only support fp16 and bf16 data type`; `fp8_e4m3` fails with Triton `fp8e4nv` unsupported.
- RTN-to-GPTQ W4A16 gs1024 checkpoint was generated at `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-gs1024`. It loads under `--quantization gptq`, but is slower than FP16 on local proxy and public accuracy has not been evaluated.
- RTN-to-GPTQ W4A16 symmetric gs128 checkpoint was generated at `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-sym-gs128`. It passes ordinary GPTQ smoke and has mixed local proxy results: S1 is slightly slower, S8/Smax are slightly faster. A 10-sample public accuracy probe on `chunk_0003` was stopped after `20:47` with `0/10` samples completed while the same FP16 chunk took `111.1713s`; treat this as a negative correctness-runtime signal and do not run full public accuracy before fixing long generation.
- `gptq_marlin` currently blocks on the installed kernel wheel: the sym gs128 checkpoint is metadata-compatible, but server load fails because `sgl_kernel==0.4.2.post2` does not export `gptq_marlin_repack`. The gs1024 variant is also not Marlin-compatible because current Marlin supports group sizes `-1/32/64/128`, not `1024`.
- EAGLE3 synthetic smoke has progressed beyond the draft-head and native sparse target-verify blockers: a 1-step draft exists at `/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`; dense fallback serving smoke passed two short chat requests; native sparse `minicpm_flashinfer` smoke also passed after local target-verify metadata and sparse K1/K2 cache-accounting patches.
- Native sparse EAGLE3 local proxy speed with the synthetic draft is `S1=124.42`, `S8=171.20`, `Smax=242.06`, `official=false`. It is faster than FP16 proxy on S1 but slower on S8/Smax, and the smoke predictions are visibly low quality. Treat this as backend-wiring evidence only, not a candidate result.
- A second EAGLE3 synthetic probe draft was trained for 16 steps on 128 public-eval-independent synthetic records at `/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_synthetic_probe_s16/epoch_0_step_16`. Native sparse smoke passed, but both short predictions were degenerate and hit `finish_reason=length`; local proxy speed is `S1=134.69`, `S8=170.99`, `Smax=240.81`, `official=false`. This is also not a candidate because S8/Smax are slower than FP16 proxy.

## Logs

- Model download: `/home/dataset-local/work/SOAR/artifacts/logs/prepare_model_hf.log`
- Runtime installs: `/home/dataset-local/work/SOAR/artifacts/logs/install_xgrammar.log`, `/home/dataset-local/work/SOAR/artifacts/logs/install_datasets.log`
- Readiness: `/home/dataset-local/work/SOAR/artifacts/logs/check_soar_readiness_after_extensions.log`, `/home/dataset-local/work/SOAR/artifacts/logs/check_soar_readiness_after_torchao.log`
- Baseline server: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_server.log`
- Short chat smoke: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_smoke.log`
- Speed smoke: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_speed_smoke.log`
- Local proxy speed: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_fp16_local_proxy_speed.log`, `/home/dataset-local/work/SOAR/artifacts/logs/gptq_rtn_gs1024_local_proxy_speed.log`, `/home/dataset-local/work/SOAR/artifacts/logs/gptq_rtn_sym_gs128_local_proxy_speed.log`
- FP8 KV failure logs: `/home/dataset-local/work/SOAR/artifacts/logs/fp8kv_fp8_e5m2_server.log`, `/home/dataset-local/work/SOAR/artifacts/logs/fp8kv_fp8_e4m3_server.log`
- GPTQ RTN logs: `/home/dataset-local/work/SOAR/artifacts/logs/quantize_gptq_rtn_gs1024.log`, `/home/dataset-local/work/SOAR/artifacts/logs/quantize_gptq_rtn_sym_gs128.log`, `/home/dataset-local/work/SOAR/artifacts/logs/gptq_rtn_sym_gs128_smoke.log`, `/home/dataset-local/work/SOAR/artifacts/logs/gptq_rtn_sym_gs128_accuracy_probe_chunk_0003.log`, `/home/dataset-local/work/SOAR/artifacts/logs/gptq_rtn_sym_gs128_accuracy_probe_server/quant_gptq_auto.log`, `/home/dataset-local/work/SOAR/artifacts/logs/quant_gptq_marlin_auto.log`
- EAGLE3 smoke logs/results: `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_smoke_train.log`, `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_default_serving_smoke.log`, `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_sparse_synthetic_local_proxy_speed.log`, `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_sparse_synthetic_probe_s16_local_proxy_speed.log`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_smoke_summary.txt`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_synthetic_probe_s16_summary.json`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_serving_smoke_summary.md`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_default_serving_smoke.json`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_after_target_verify_len_patch.json`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_local_proxy_speed.json`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_probe_s16.json`, `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_probe_s16_local_proxy_speed.json`
- Smoke predictions: `/home/dataset-local/work/SOAR/artifacts/results/baseline_smoke_predictions.jsonl`
- Chunked accuracy log: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_chunked.log`
- Chunked accuracy resume log: `/home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_chunked_resume.log`

## Remaining Baseline Work

1. Obtain official speed split paths via `SPEED_DATA_S1`, `SPEED_DATA_S8`, and `SPEED_DATA_SMAX`; only `perf_public_set.jsonl` is present locally, and the local proxy split is not a substitute for official speed.
2. Investigate the CUDA graph Smax crash separately; keep `--disable-cuda-graph` for correctness-stable baseline until fixed.
3. For EAGLE3, keep the synthetic draft and native sparse smoke/proxy as plumbing evidence only. Train a non-synthetic draft before any real EAGLE3 accuracy or speed claim.
4. Do not select GPTQ RTN, Marlin, or FP8 KV as final candidates without public accuracy and official speed evidence.
