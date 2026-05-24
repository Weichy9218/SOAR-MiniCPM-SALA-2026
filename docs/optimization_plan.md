# Optimization Plan

本文件负责给出 SOAR MiniCPM-SALA 的 correctness-first 优化路线和每一步的进入条件。

## Principle

Correctness first, speed second.

Before optimizing, establish a complete FP16 baseline:

- `completed_samples=150/150`
- final `ori_accuracy`
- final `overall_accuracy`
- full speed metrics for S1, S8, and Smax when official split files are available

Do not treat smoke or partial accuracy as final benchmark evidence.

## Phase 1: Finish Stable FP16 Baseline

Status: complete.

Goal:

- Complete public accuracy on all 150 samples.
- Preserve stable runtime semantics.
- Record final correctness and speed baseline.

Current stable server configuration:

```bash
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 8192
--skip-server-warmup
--dense-as-sparse
--disable-cuda-graph
```

Done when:

- `artifacts/results/baseline_accuracy_full_chunked_summary.txt` says `status=complete`.
- `artifacts/results/baseline_accuracy_full_chunked_summary.json` includes all 15 chunks as complete.
- `artifacts/results/baseline_summary.md` is updated with final values.
- The command and environment variables are recorded.

Current result:

```text
completed_samples=150/150
ori_accuracy=82.53
overall_accuracy=100
duration=17260.25
total_tokens=733451
```

Remaining gap:

- Official S1/S8/Smax speed inputs are not available locally.
- `artifacts/results/local_proxy_speed/` is deterministic and useful for local candidate comparison, but it is marked `official=false`.
- Do not report local proxy values as official SOAR speed.

## Phase 2: Observability Improvements

Goal:

- Make long runs easier to audit and safer to resume.

Recommended changes:

- Add a small status script that prints:
  - current chunk
  - aggregate completed samples
  - latest server decode line
  - GPU utilization and VRAM
  - newest checkpoint directory
- Add per-sample checkpointing wrapper if evaluator semantics allow it.
- Keep chunk metadata explicit:
  - server flags
  - git commit
  - model path
  - evaluator commit
  - `max_out_len`

These are engineering improvements and should not change scoring semantics.

## Phase 3: Full Speed Benchmark

Goal:

- Measure S1, S8, and Smax under the same stable FP16 baseline.

Inputs needed:

```bash
SPEED_DATA_S1=/path/to/s1.jsonl
SPEED_DATA_S8=/path/to/s8.jsonl
SPEED_DATA_SMAX=/path/to/smax.jsonl
```

Rules:

- Do not report only tokens/s.
- Record benchmark duration, TTFT, ITL, output tokens/s, peak memory, crash/OOM status.
- Keep accuracy and speed configs aligned.
- If official splits are still unavailable, use local proxy speed only as a development smoke and label it `official=false`.

Current local proxy values:

| run | official | S1 | S8 | Smax | note |
|---|---|---:|---:|---:|---|
| `baseline_fp16` | false | 136.91 | 162.14 | 224.11 | stable with `--disable-cuda-graph` |
| `gptq_rtn_gs1024` | false | 141.52 | 172.86 | 238.65 | slower than FP16 proxy |
| `gptq_rtn_sym_gs128` | false | 137.64 | 160.99 | 222.38 | mixed proxy; 10-sample accuracy probe stalled |
| `eagle3_sparse_synthetic_probe_s16` | false | 134.69 | 170.99 | 240.81 | synthetic 16-step draft; S8/Smax slower than FP16 proxy |

## Phase 4: Quantization Mainline

Goal:

- Reduce memory and/or walltime without material accuracy loss.

Candidates:

- FP16 target baseline
- GPTQ / GPTQ-Marlin target
- GPTQ / GPTQ-Marlin + FP8 KV cache

Current probes:

- `fp8_e5m2` KV cache fails on first smoke in the current `minicpm_flashinfer` dense-as-sparse path because FlashAttention only supports fp16/bf16 there.
- `fp8_e4m3` KV cache fails on first smoke with Triton `fp8e4nv` unsupported.
- RTN-to-GPTQ W4A16 group-size 1024 loads, but is slower than FP16 on local proxy speed and has no full public accuracy yet.
- RTN-to-GPTQ W4A16 symmetric group-size 128 loads and passes smoke. Local proxy is mixed: S1 `137.64` is slower than FP16 `136.91`, while S8 `160.99` and Smax `222.38` are slightly faster than FP16 `162.14` and `224.11`.
- `gptq_marlin` is currently blocked by kernel stack, not by checkpoint metadata: the sym-gs128 checkpoint is detected as Marlin-compatible, but installed `sgl_kernel==0.4.2.post2` lacks `gptq_marlin_repack`.

Current judgement:

- Do not promote FP8 KV cache in the current backend path.
- Do not treat RTN-GPTQ gs1024 or sym-gs128 as winning candidates without official speed and full public accuracy.
- Do not rebuild or replace `sgl_kernel` casually while FP16 baseline is the guardrail; Marlin kernel work should be a separate environment-change experiment with rollback notes.
- If quantization is revisited, use the completed FP16 baseline as the guardrail and run a bounded smoke -> proxy -> accuracy sequence.

Decision gates:

- Candidate loads cleanly.
- Public accuracy does not materially regress.
- S1/S8/Smax all have complete metrics.
- If speed improves only on one axis but hurts another, document it instead of calling it a win.

Artifacts:

- `artifacts/results/quant_matrix.csv`
- `artifacts/results/baseline_summary.md`
- candidate-specific logs

## Phase 5: CUDA Graph Investigation

Goal:

- Determine whether CUDA graph can be safely re-enabled.

Known issue:

- CUDA graph enabled Smax smoke crashes in `minicpm_flashinfer` replay metadata.

Safe order:

1. Reproduce the crash with `DISABLE_CUDA_GRAPH=0`.
2. Minimize the failing smoke case.
3. Fix or bypass only the CUDA graph path.
4. Run S1/S8/Smax smoke.
5. Run a small isolated accuracy chunk.
6. If all pass, run full baseline as a separate configuration.

Do not mix CUDA graph enabled chunks with disabled chunks in one aggregate.

## Phase 6: EAGLE3 Top-k=1

Goal:

- Try conservative chain speculation after baseline and quantization evidence exists.

Default conservative serving flags:

```bash
--speculative-algorithm EAGLE3
--speculative-draft-model-path <draft_head_path>
--speculative-eagle-topk 1
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-accept-threshold-single 1.0
--speculative-accept-threshold-acc 1.0
```

Entry conditions:

- Stable FP16 baseline complete.
- Draft head exists or training path is clear.
- Forward smoke passes:
  - hidden state shape
  - draft logits shape
  - vocab dimension
  - dtype/device placement

Decision gates:

- Top-k=1 verify completes.
- Accuracy does not regress.
- S1 improves or acceptance metrics show clear future value.
- If not working within a bounded time budget, stop and return to quantization/mainline serving.

Current EAGLE3 status:

- `scripts/launch_eagle3.sh` records the expected serving path.
- A synthetic 1-step MiniCPM-SALA-compatible draft now exists at `artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`.
- Dense fallback serving smoke passes with `ATTENTION_BACKEND=flashinfer --force-dense-minicpm`, but this is plumbing-only and not a candidate result.
- Native sparse `minicpm_flashinfer` EAGLE3 serving now passes a short smoke after local target-verify metadata and sparse K1/K2 cache-accounting patches.
- Native sparse local proxy speed with the synthetic draft is `S1=124.42`, `S8=171.20`, `Smax=242.06`, `official=false`; this is worse than FP16 proxy on S8/Smax and is not a candidate result.
- A 128-record synthetic-probe draft trained for 16 steps also passes native sparse smoke, but its outputs are degenerate and local proxy speed is `S1=134.69`, `S8=170.99`, `Smax=240.81`, `official=false`; this remains a negative probe and should not move to full public accuracy.
- The next real EAGLE step is non-synthetic draft training, then sparse smoke, a small accuracy chunk, and speed gates.

## Phase 7: LK Loss

Goal:

- Compare KL draft and LK-style draft only after serving path is functional.

Known caution:

- SpecForge PR #492 has LK-related code, but the lambda loss formulation must be checked before use.
- Do not cherry-pick loss code just because it runs.

Evidence required:

- Training-side metrics.
- Serving-side acceptance metrics.
- Public accuracy.
- S1/S8/Smax speed.

## Current Next Action

As of `2026-05-24 CST`, the FP16 public accuracy baseline is complete. The next actions are:

1. Keep all result-facing summaries aligned with `completed_samples=150/150`, `ori_accuracy=82.53`, and `overall_accuracy=100`.
2. Obtain or reconstruct the official SOAR speed benchmark inputs before claiming S1/S8/Smax.
3. Keep local proxy speed as self-test only and label it `official=false`.
4. If optimizing immediately, first improve observability and per-sample checkpointing because they preserve benchmark semantics.
5. Run CUDA graph, quantization, multi-GPU, and EAGLE as separate configurations with separate artifacts.
6. For EAGLE, do not run full public accuracy until a real draft replaces the synthetic 1-step smoke draft.
