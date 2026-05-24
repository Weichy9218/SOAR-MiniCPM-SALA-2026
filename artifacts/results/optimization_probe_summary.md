# Optimization Probe Summary

Status: **FP16 baseline remains the recommended correctness guardrail**

## Speed Split Boundary

The official SOAR S1/S8/Smax speed split files are not present locally. The local proxy split in `artifacts/results/local_proxy_speed/` is deterministic and useful for candidate comparison, but it is marked `official=false` and must not be reported as official SOAR speed.

## Local Proxy Results

| run | official | S1 | S8 | Smax | note |
|---|---|---:|---:|---:|---|
| baseline_fp16 | false | 136.91 | 162.14 | 224.11 | stable with `--disable-cuda-graph` |
| gptq_rtn_gs1024 | false | 141.52 | 172.86 | 238.65 | slower than FP16 proxy |
| gptq_rtn_sym_gs128 | false | 137.64 | 160.99 | 222.38 | mixed: S1 slower, S8/Smax slightly faster; accuracy probe stalled |
| eagle3_sparse_synthetic | false | 124.42 | 171.20 | 242.06 | synthetic 1-step draft; S1 faster but S8/Smax slower |
| eagle3_sparse_synthetic_probe_s16 | false | 134.69 | 170.99 | 240.81 | synthetic 16-step probe draft; S1 slightly faster than FP16 but S8/Smax slower |

## Quantization Probes

- `fp8_e5m2` KV cache fails on the first smoke request in the current `minicpm_flashinfer` dense-as-sparse path: FlashAttention only supports fp16/bf16 there.
- `fp8_e4m3` KV cache fails on the first smoke request with Triton `fp8e4nv` unsupported.
- RTN-to-GPTQ W4A16 group-size 1024 was generated at `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-gs1024`. It loads after the local compressed-attention dtype patch, but is slower than FP16 proxy and has no full public accuracy yet.
- RTN-to-GPTQ W4A16 symmetric group-size 128 was generated at `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-sym-gs128`. It passes ordinary `--quantization gptq` smoke (`0.78/1.18/0.72`) and local proxy (`137.64/160.99/222.38`), but the proxy gain is small and mixed.
- A 10-sample public accuracy probe for the symmetric group-size 128 checkpoint was stopped after `20:47` with `0/10` samples completed. The same chunk completes in `111.1713s` for FP16, while the GPTQ server kept four requests decoding and `#full token` grew from `133232` to `267612`. Treat this as a negative correctness-runtime signal, not an accuracy score.
- `gptq_marlin` on the symmetric group-size 128 checkpoint reaches Marlin compatibility detection, then fails during weight postprocess because the installed `sgl_kernel==0.4.2.post2` does not export `gptq_marlin_repack`. This is a kernel-stack blocker, not a checkpoint-format blocker.

## EAGLE3 Readiness

- A synthetic 1-step MiniCPM-SALA-compatible EAGLE3 draft checkpoint was trained with SpecForge and the local SGLang target backend:
  `/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`.
- Dense fallback serving smoke passed with `ATTENTION_BACKEND=flashinfer`, `--force-dense-minicpm`, `--disable-cuda-graph`, EAGLE3 top-k=1, 3 steps, and 4 draft tokens. The reproducible smoke runner returned HTTP 200 for two short `/v1/chat/completions` requests with request times `2.9984s` and `1.1137s`.
- This dense fallback smoke is `official=false` and validates plumbing only. The draft was trained on synthetic 6-record smoke data for 1 step, and the dense fallback does not preserve the native sparse `minicpm_flashinfer` target path.
- Native sparse EAGLE3 serving with `minicpm_flashinfer` now passes the short serving smoke after local target-verify metadata and sparse K1/K2 cache-accounting patches. The latest smoke returned HTTP 200 for two short `/v1/chat/completions` requests with request times `3.0652s` and `1.4538s`.
- The previous native sparse blockers were:
  - target verification raised `MiniCPM backend does not support speculative decoding (target verify)`;
  - after a first metadata patch, idle memory check reported a +4 sparse-cache allocator over-free. The +4 matched rejected K1 compressed slots from two short smoke requests and was fixed by allocating/freeing MiniCPM sparse K1/K2 slots for EAGLE target verify.
- Native sparse EAGLE3 local proxy speed with the synthetic 1-step draft is `S1=124.42`, `S8=171.20`, `Smax=242.06`, `official=false`. This is mixed and slower than FP16 on S8/Smax, so it is not a candidate promotion.
- A larger public-eval-independent synthetic probe draft was trained for 16 steps on 128 synthetic instruction records at `/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_synthetic_probe_s16/epoch_0_step_16`. Native sparse smoke passed, but predictions were visibly degenerate and both short requests ended by length. Local proxy speed is `S1=134.69`, `S8=170.99`, `Smax=240.81`, `official=false`; relative to FP16 proxy this is about `-1.62%/+5.46%/+7.45%` duration change, so it is also a negative probe.
- Smoke output quality is visibly poor because the draft was trained on synthetic 6-record smoke data for 1 step. Do not run full public accuracy or report speculative speed until a real draft is trained.
- Reproducer: start `scripts/launch_eagle3.sh` with `ATTENTION_BACKEND=minicpm_flashinfer FORCE_DENSE_MINICPM=0 DENSE_AS_SPARSE=1`, then run `EXPECT_PASS=1 RUN_NAME=eagle3_sparse_after_target_verify_len_patch bash scripts/run_eagle3_serving_smoke.sh`.
- Summary artifacts: `artifacts/results/eagle3_serving_smoke_summary.md`, `artifacts/results/eagle3_default_serving_smoke.json`, and `artifacts/results/eagle3_serving_smoke_default_minicpm_flashinfer.json`.
- New sparse artifacts: `artifacts/results/eagle3_sparse_after_target_verify_len_patch.json`, `artifacts/results/eagle3_sparse_synthetic_local_proxy_speed.json`, `artifacts/results/eagle3_sparse_synthetic_probe_s16.json`, `artifacts/results/eagle3_sparse_synthetic_probe_s16_local_proxy_speed.json`, and the corresponding logs under `artifacts/logs/`.
- Next real EAGLE3 step: train a non-synthetic draft, rerun sparse smoke, then run a small accuracy chunk plus local/official speed. Do not promote current EAGLE3 artifacts as candidate metrics.
