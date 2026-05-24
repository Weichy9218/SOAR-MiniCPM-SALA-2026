# SOAR Decision Log

本文件记录已经形成证据的工程和实验决策。格式固定为：decision、evidence、risk、next action。

## D001: Treat FP16 Baseline As The Correctness Guardrail

Decision:

- Use the completed FP16 MiniCPM-SALA baseline as the correctness reference.

Evidence:

- `completed_samples=150/150`
- `ori_accuracy=82.53`
- `overall_accuracy=100`
- `duration=17260.25`
- Artifact: `artifacts/results/baseline_accuracy_full_chunked_summary.txt`

Risk:

- `overall_accuracy=100` can hide task-level weaknesses because it is normalized from `ori_accuracy`.
- `mcq` and `qa` are only `60.0` ori accuracy in the task breakdown.

Next action:

- Update result-facing summaries and use this run as the guardrail for all future candidates.

## D002: Keep CUDA Graph Disabled In Stable Baseline

Decision:

- Keep `DISABLE_CUDA_GRAPH=1` and `--disable-cuda-graph` for the stable baseline.

Evidence:

- CUDA graph enabled path previously crashed in Smax smoke.
- The failing path was `minicpm_flashinfer` CUDA graph replay metadata.
- Observed error included `kv_indptr[...] should be non-negative`.

Risk:

- Disabling CUDA graph may leave launch-overhead speed on the table.
- Enabling it prematurely risks invalidating or crashing long benchmark runs.

Next action:

- Reproduce CUDA graph failure in an isolated smoke.
- Fix or bypass only that path.
- Run smoke -> small accuracy chunk -> full comparison as a separate config.

## D003: Use Chunked Accuracy Runner

Decision:

- Use `scripts/run_full_accuracy_chunked.sh` instead of one monolithic public accuracy run.

Evidence:

- `eval_model.py` uses `max_out_len=65536`.
- Several chunks took around 41-44 minutes.
- A chunk writes checkpoint artifacts only after all 10 requests finish.
- Chunking protected the first 100+ samples while long final requests were still decoding.

Risk:

- Chunk-level checkpointing is still coarse.
- A chunk stuck at `9/10` can lose that chunk if interrupted before `summary.json` and `predictions.jsonl` land.

Next action:

- Consider per-sample checkpointing or smaller chunk size if evaluator semantics remain unchanged.

## D004: Do Not Report Local Proxy Speed As Official SOAR Speed

Decision:

- Keep local proxy speed as a development-only self-test.

Evidence:

- `artifacts/results/local_proxy_speed/local_proxy_speed_manifest.json` has `official=false`.
- The proxy split is derived from public accuracy prompts.
- Official S1/S8/Smax split files are not present locally.

Risk:

- Reporting proxy speed as official would be a benchmark claim error.
- Proxy prompts may not match official concurrency, prompt distribution, output limits, or scoring harness.

Next action:

- Obtain official speed split files or official harness before making speed claims.
- Continue using proxy only to compare local candidates under the same synthetic split.

## D005: Do Not Promote Current FP8 KV Cache Probes

Decision:

- Do not make FP8 KV cache part of the mainline for the current `minicpm_flashinfer` dense-as-sparse path.

Evidence:

- `fp8_e5m2` failed first smoke because FlashAttention supports only fp16/bf16 in this path.
- `fp8_e4m3` failed first smoke with Triton `fp8e4nv` unsupported.

Risk:

- Continuing here could consume time in backend compatibility work before there is a clear benchmark win.

Next action:

- Revisit only if a backend path explicitly supports the needed FP8 KV dtype for MiniCPM-SALA.

## D006: Treat RTN-GPTQ gs1024 As A Probe, Not A Win

Decision:

- Keep `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-gs1024` as a probe artifact only.

Evidence:

- It loads and runs local proxy speed.
- Proxy speed is slower than FP16:
- `baseline_fp16`: S1 `136.91`, S8 `162.14`, Smax `224.11`
- `gptq_rtn_gs1024`: S1 `141.52`, S8 `172.86`, Smax `238.65`
- It has no full public accuracy result yet.

Risk:

- Quantization can reduce memory but still hurt latency or accuracy.
- Proxy speed is not official speed.

Next action:

- Do not invest further unless a stronger quantization path has a credible speed or memory rationale.

## D007: EAGLE3 Draft Smoke Exists, Native Sparse Target Verify Reaches Smoke

Decision:

- Do not start EAGLE3 benchmark claims until a real draft has accuracy and speed evidence.

Evidence:

- A synthetic 1-step draft checkpoint exists at `artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`.
- SpecForge smoke training completed with the local SGLang target backend after narrow compatibility patches.
- SGLang serving can load target and draft on the native `minicpm_flashinfer` path.
- Native sparse EAGLE3 now passes the short serving smoke after local MiniCPM target-verify metadata and sparse K1/K2 cache-accounting patches.
- Latest native sparse smoke: `artifacts/results/eagle3_sparse_after_target_verify_len_patch.json`, two requests, elapsed `3.0652s` and `1.4538s`, `official=false`.
- Native sparse local proxy speed with the synthetic draft: S1 `124.42`, S8 `171.20`, Smax `242.06`, `official=false`.
- FP16 local proxy speed: S1 `136.91`, S8 `162.14`, Smax `224.11`.
- The synthetic EAGLE3 proxy is faster on S1 but slower on S8/Smax, and smoke predictions are visibly low quality.
- Dense fallback serving with `ATTENTION_BACKEND=flashinfer --force-dense-minicpm` returns HTTP 200 for two short chat requests, but this changes the target attention path and uses a synthetic 1-step draft.

Risk:

- Dense fallback smoke can be mistaken for a valid speculative speed result.
- A synthetic 1-step draft has no expected quality or acceptance behavior.
- Implementing sparse target verify incorrectly can silently damage correctness on long-context linear-attention states.
- Relaxing acceptance thresholds for speed would change correctness semantics.
- Local proxy speed is not official SOAR speed.

Next action:

- Keep current sparse target-verify patches as smoke-level evidence.
- Train a non-synthetic draft before any EAGLE3 accuracy claim.
- Run sparse smoke with conservative top-k=1 and thresholds at `1.0`, then a small accuracy chunk, then local/official speed.
- Continue monitoring `eagle3_server.log` for sparse allocator leaks because the current validation is smoke/proxy only.

## D008: Treat RTN-GPTQ sym-gs128 As A Negative Probe, Not A Final Candidate

Decision:

- Keep `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-sym-gs128` as a negative quantization probe unless long generation is fixed.

Evidence:

- Ordinary `--quantization gptq` server loads with about `5.82GB` model memory.
- Smoke passes with S1/S8/Smax `0.78/1.18/0.72`.
- Local proxy speed is `137.64/160.99/222.38`.
- FP16 local proxy speed is `136.91/162.14/224.11`.
- A 10-sample public accuracy probe on `chunk_0003` was stopped after `20:47` with `0/10` samples completed.
- The same `chunk_0003` took `111.1713s` under FP16 and had `ori_accuracy=100.0`.
- During the GPTQ probe, the server kept four requests decoding and `#full token` grew from `133232` to `267612`.

Risk:

- The proxy gain is small and mixed across speed axes.
- RTN quantization can hurt correctness; no final accuracy evidence exists.
- The probe suggests pathological long generation or stop-token behavior on correctness data.
- Local proxy speed is not official SOAR speed.

Next action:

- Do not run full accuracy unless long generation is fixed and a stronger proxy result makes the cost worthwhile.

## D009: GPTQ Marlin Is Blocked By Kernel Stack

Decision:

- Do not keep retrying `gptq_marlin` on the current wheel stack.

Evidence:

- Symmetric group-size 128 checkpoint is detected as convertible to `gptq_marlin`.
- Server load fails during weight postprocess with missing `sgl_kernel.gptq_marlin_repack`.
- Current installed `sgl_kernel==0.4.2.post2` exposes no Marlin ops.
- Group-size 1024 checkpoints are not Marlin-compatible because current Marlin supports `-1/32/64/128`.

Risk:

- Rebuilding or replacing `sgl_kernel` may perturb the validated FP16 baseline environment.

Next action:

- Treat Marlin as a separate environment-change experiment: rebuild/install the SOAR demo kernel stack with rollback notes, then rerun smoke before any accuracy/speed claims.

## D010: GitHub Sync Is Blocked By Credentials, Not By Code State

Decision:

- Do not keep retrying `git push` until auth is configured.

Evidence:

- `git push origin main` failed with no interactive username available.
- `gh auth status` showed not logged in.
- No `GH_TOKEN` or `GITHUB_TOKEN` was available.

Risk:

- Repeated push attempts do not fix auth and create noise.

Next action:

- Run `gh auth login` or provide a valid token.
- Then push `main` to `Weichy9218/SOAR-MiniCPM-SALA-2026`.

## D011: Treat Dense EAGLE3 Fallback As Plumbing Smoke Only

Decision:

- Keep dense fallback EAGLE3 results out of candidate comparisons.

Evidence:

- `artifacts/results/eagle3_dense_serving_smoke.json` records `official=false`.
- The smoke used `ATTENTION_BACKEND=flashinfer` and `--force-dense-minicpm`, not the baseline `minicpm_flashinfer` sparse target path.
- The two chat requests completed, but no public accuracy, private accuracy, S1/S8/Smax, or acceptance-quality run exists.

Risk:

- Dense fallback could be mistaken for native sparse EAGLE3 evidence.
- It may not match MiniCPM-SALA's intended sparse linear-attention runtime behavior.

Next action:

- Use dense fallback only to debug EAGLE3 wiring.
- Promote no EAGLE3 result until native sparse serving passes correctness and speed gates.

## D012: Treat Synthetic-Probe EAGLE3 16-Step As A Negative Probe

Decision:

- Do not run full public accuracy for the current 16-step synthetic EAGLE3 draft.

Evidence:

- Draft checkpoint: `artifacts/draft_heads/minicpm_sala_eagle3_synthetic_probe_s16/epoch_0_step_16`.
- Training data was 128 public-eval-independent synthetic instruction records, not public/private eval data.
- Native sparse `minicpm_flashinfer` smoke passed with two short chat requests.
- Smoke predictions were visibly degenerate and both requests ended with `finish_reason=length`.
- Local proxy speed was `S1=134.69`, `S8=170.99`, `Smax=240.81`, `official=false`.
- FP16 proxy speed was `S1=136.91`, `S8=162.14`, `Smax=224.11`.
- Relative to FP16 proxy, this draft is slightly faster on S1 but slower on S8/Smax.

Risk:

- Running full public accuracy would spend hours on a candidate with poor output and negative concurrent-speed evidence.
- Synthetic draft data does not represent the official workload.
- Local proxy speed is not official SOAR speed.

Next action:

- Keep the artifact as serving evidence only.
- Train a non-synthetic draft before further EAGLE3 accuracy or speed claims.

## D012: Do Not Promote Synthetic Native Sparse EAGLE3

Decision:

- Keep `eagle3_sparse_synthetic` as a backend-wiring result, not an optimization candidate.

Evidence:

- Native sparse EAGLE3 smoke now passes: `artifacts/results/eagle3_sparse_after_target_verify_len_patch.json`.
- Local proxy speed is `S1=124.42`, `S8=171.20`, `Smax=242.06`, `official=false`.
- Compared with FP16 local proxy (`136.91/162.14/224.11`), synthetic EAGLE3 only improves S1 and regresses S8/Smax.
- Smoke predictions show degenerate output, consistent with a 1-step synthetic draft.
- No public accuracy has been run for EAGLE3.

Risk:

- Running full public accuracy now would likely waste several hours and produce a non-candidate result.
- Reporting proxy speed or smoke output as EAGLE3 progress would overstate the evidence.

Next action:

- Train or obtain a real MiniCPM-SALA-compatible EAGLE3 draft.
- After training, rerun native sparse smoke, a small public accuracy chunk, and local proxy speed before considering full public accuracy.
