# EAGLE3 Serving Smoke Summary

Status: **draft head trained; dense fallback and native MiniCPM sparse serving smoke passed**

## Draft Checkpoint

- `DRAFT_MODEL_PATH`: `/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`
- Training script: `scripts/run_eagle3_smoke_train.sh`
- Target backend used for smoke training: `sglang`
- Training scope: 1 step on synthetic smoke data, not public/private eval data.
- Draft config: `/home/dataset-local/work/SOAR/artifacts/draft_heads/eagle3_smoke_assets/minicpm_sala_eagle3_smoke_config.json`
- Train data: `/home/dataset-local/work/SOAR/artifacts/draft_heads/eagle3_smoke_assets/minicpm_sala_eagle3_smoke_train.jsonl`

## Serving Smoke Results

| mode | models endpoint | chat smoke | official | result |
|---|---|---|---|---|
| `ATTENTION_BACKEND=flashinfer FORCE_DENSE_MINICPM=1 DENSE_AS_SPARSE=1` | passed | passed, 2 requests | false | serving plumbing smoke only |
| `ATTENTION_BACKEND=minicpm_flashinfer FORCE_DENSE_MINICPM=0 DENSE_AS_SPARSE=1` | passed | passed, 2 requests | false | native sparse smoke only after local target-verify/cache patches |

Dense fallback smoke artifacts:

- Summary: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_default_serving_smoke.json`
- Predictions: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_default_serving_smoke_predictions.jsonl`
- Server log snapshot: `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_force_dense_flashinfer_dense_as_sparse/eagle3_server.log`

Native sparse failure artifacts:

- Summary: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_serving_smoke_default_minicpm_flashinfer.json`
- Server log snapshot: `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_default_minicpm_flashinfer/eagle3_server.log`
- Root error: `MiniCPM backend does not support speculative decoding (target verify)`.

Native sparse pass artifacts:

- Summary: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_after_target_verify_len_patch.json`
- Predictions: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_after_target_verify_len_patch_predictions.jsonl`
- Local proxy speed: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_local_proxy_speed.json`
- Local proxy log: `/home/dataset-local/work/SOAR/artifacts/logs/eagle3_sparse_synthetic_local_proxy_speed.log`
- Proxy result: `S1=124.42`, `S8=171.20`, `Smax=242.06`, `official=false`.

Synthetic probe 16-step artifacts:

- Draft: `/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_synthetic_probe_s16/epoch_0_step_16`
- Training summary: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_synthetic_probe_s16_summary.json`
- Smoke summary: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_probe_s16.json`
- Predictions: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_probe_s16_predictions.jsonl`
- Local proxy speed: `/home/dataset-local/work/SOAR/artifacts/results/eagle3_sparse_synthetic_probe_s16_local_proxy_speed.json`
- Proxy result: `S1=134.69`, `S8=170.99`, `Smax=240.81`, `official=false`.
- Boundary: smoke passed, but predictions are degenerate and both short smoke requests ended by `finish_reason=length`; S8/Smax remain slower than FP16 proxy.

## Boundary

This result validates that MiniCPM-SALA-compatible EAGLE3 draft checkpoints can be trained and loaded by SGLang, and that the native sparse target path can answer short chat requests after local patches. It is **not** an accuracy result, not an official speed result, and not yet a final optimization candidate. Both current synthetic drafts regress S8/Smax versus FP16 proxy, and their smoke predictions are visibly low quality. Full public accuracy and official S1/S8/Smax should wait for a non-synthetic draft.
