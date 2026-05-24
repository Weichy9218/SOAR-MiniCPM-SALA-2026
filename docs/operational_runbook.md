# Operational Runbook

本文件负责给出 SOAR baseline 长运行期间的检查、监控、resume 和事故判断步骤。

## Quick Status

Check aggregate:

```bash
cat /home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_full_chunked_summary.txt
```

Check runner progress:

```bash
tail -80 /home/dataset-local/work/SOAR/artifacts/logs/baseline_accuracy_chunked_resume.log
```

Check server decode:

```bash
tail -120 /home/dataset-local/work/SOAR/artifacts/logs/baseline_server.log
```

Check processes:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args --sort=pid \
  | grep -Ei 'SOAR|baseline|chunk|eval_model|sglang|launch_server' \
  | grep -v grep
```

Check GPU:

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,pstate,temperature.gpu --format=csv,noheader,nounits
```

## Normal Long-Decode State

This is normal:

```text
Generating: 90%|...| 9/10
Decode batch, #running-req: 1, #full token: increasing, gen throughput: non-zero
```

Interpretation:

- The evaluator is waiting for the final request in the chunk.
- The server is still decoding.
- No chunk checkpoint is written until the last request finishes.

Expected behavior after completion:

```text
Generating: 100%|...| 10/10
Generation completed ...
Detailed results saved to outputs/<timestamp>/predictions.jsonl
aggregate_status=<partial-or-complete> completed_samples=<N+10>/150 ...
running=chunk_XXXX_...
```

## Healthy Vs Unhealthy

Healthy:

- Server process exists.
- Evaluator process exists.
- `#full token` keeps increasing.
- GPU power/utilization are non-zero.
- No traceback or CUDA error.

Unhealthy:

- Server process disappeared.
- Evaluator process disappeared.
- `baseline_server.log` stops updating for a long time while evaluator still waits.
- New traceback, OOM, CUDA assertion, or HTTP API error appears.
- Chunk output directory is missing `summary.json` after evaluator exits.

## Resume Rules

Use the chunk runner for resume:

```bash
source /home/dataset-local/work/SOAR/.venv/bin/activate
export MODEL_PATH=/home/dataset-local/models/MiniCPM-SALA
export API_BASE=http://127.0.0.1:30000
export CHUNK_SIZE=10
export ACCURACY_CONCURRENCY=4
bash /home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh
```

The runner skips chunks with both:

```text
summary.json
predictions.jsonl
```

Do not manually mark a chunk complete.

## What Not To Do Mid-Run

Do not:

- Kill the evaluator just because progress is stuck at `9/10`.
- Change `max_out_len`.
- Change `CHUNK_SIZE` and mix old/new chunk manifests without understanding the checkpoint layout.
- Change model path, tokenizer, or chat template.
- Enable CUDA graph for the remaining chunks.
- Start EAGLE or quantization on the same server.

If you must stop, prefer stopping between chunks after a checkpoint lands.

## After Full Accuracy Completes

Run these checks:

```bash
cat /home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_full_chunked_summary.txt
find /home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_chunks -maxdepth 2 -name summary.json | sort | wc -l
find /home/dataset-local/work/SOAR/artifacts/results/baseline_accuracy_chunks -maxdepth 2 -name predictions.jsonl | sort | wc -l
```

Expected:

```text
status=complete
completed_samples=150/150
15 chunk summary files
15 chunk predictions files
```

Then update:

- `artifacts/results/baseline_summary.md`
- `artifacts/results/quant_matrix.csv` if needed
- `submit/README_SOAR.md`

Current completed baseline:

```text
completed_samples=150/150
ori_accuracy=82.53
overall_accuracy=100
duration=17260.25
total_tokens=733451
```

## Local Proxy Speed Checks

Use local proxy speed only when official SOAR speed splits are unavailable:

```bash
source /home/dataset-local/work/SOAR/.venv/bin/activate
bash /home/dataset-local/work/SOAR/scripts/run_local_proxy_speed.sh
```

Interpretation rules:

- `artifacts/results/local_proxy_speed/local_proxy_speed_manifest.json` says `official=false`.
- Proxy speed is useful for comparing local candidates under the same synthetic split.
- Proxy speed is not reportable as official SOAR S1/S8/Smax.

Current proxy evidence:

```text
baseline_fp16: S1=136.91 S8=162.14 Smax=224.11
gptq_rtn_gs1024: S1=141.52 S8=172.86 Smax=238.65
gptq_rtn_sym_gs128: S1=137.64 S8=160.99 Smax=222.38
eagle3_sparse_synthetic: S1=124.42 S8=171.20 Smax=242.06
eagle3_sparse_synthetic_probe_s16: S1=134.69 S8=170.99 Smax=240.81
```

The GPTQ RTN probes should not be promoted without stronger evidence: gs1024 is slower on all proxy axes, and sym-gs128 is mixed with only small S8/Smax proxy gains. In addition, sym-gs128 stalled on a 10-sample public accuracy probe for `20:47` with `0/10` completed while FP16 completed the same chunk in `111.1713s`. The synthetic EAGLE3 proxies should also not be promoted: they regress S8/Smax, use synthetic draft data, have visibly poor smoke output, and have no public accuracy.

## EAGLE3 Smoke Checks

Current EAGLE3 evidence is smoke/proxy only:

```text
draft=/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1
dense_smoke=artifacts/results/eagle3_dense_serving_smoke.json
native_sparse_smoke=artifacts/results/eagle3_sparse_after_target_verify_len_patch.json
native_sparse_proxy=artifacts/results/eagle3_sparse_synthetic_local_proxy_speed.json
synthetic_probe_s16_smoke=artifacts/results/eagle3_sparse_synthetic_probe_s16.json
synthetic_probe_s16_proxy=artifacts/results/eagle3_sparse_synthetic_probe_s16_local_proxy_speed.json
```

Interpretation rules:

- Synthetic 1-step draft training passing means the adapter and checkpoint format are loadable; it does not imply useful acceptance or speed.
- Dense fallback serving with `ATTENTION_BACKEND=flashinfer --force-dense-minicpm` may be used for wiring debug only.
- Native sparse `minicpm_flashinfer` EAGLE3 now passes the short smoke after local target-verify metadata and sparse K1/K2 cache-accounting patches.
- The previous sparse-cache over-free showed up as `token_to_kv_pool_allocator memory leak detected` with available slots larger than max by 4; monitor `eagle3_server.log` for recurrence.
- The 16-step synthetic-probe draft also passes native sparse smoke, but predictions are degenerate and proxy speed is `S1=134.69`, `S8=170.99`, `Smax=240.81`, so it should not enter full public accuracy.
- Do not run full public accuracy for EAGLE3 until a non-synthetic draft is trained and a small chunk shows acceptable correctness.

Native sparse smoke reproducer:

```bash
DRAFT_MODEL_PATH=/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1 \
ATTENTION_BACKEND=minicpm_flashinfer \
FORCE_DENSE_MINICPM=0 \
DENSE_AS_SPARSE=1 \
bash /home/dataset-local/work/SOAR/scripts/launch_eagle3.sh

EXPECT_PASS=1 \
RUN_NAME=eagle3_sparse_after_target_verify_len_patch \
DRAFT_MODEL_PATH=/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1 \
bash /home/dataset-local/work/SOAR/scripts/run_eagle3_serving_smoke.sh
```

Local proxy command after the server is up:

```bash
RUN_NAME=eagle3_sparse_synthetic \
bash /home/dataset-local/work/SOAR/scripts/run_local_proxy_speed.sh
```

Current result is `S1=124.42`, `S8=171.20`, `Smax=242.06`, `official=false`; this is not a candidate promotion.

## Git Hygiene

Commit:

- `README.md`
- `AGENTS.md`
- `docs/`
- `scripts/`
- `submit/`
- `artifacts/results/*.md`
- `artifacts/results/*.csv`

Do not commit:

- `.venv/`
- `repos/`
- model weights
- logs
- chunk checkpoints
- raw `outputs/`
- large generated files

Current known GitHub push blocker:

```text
GitHub remote is configured, but this machine currently lacks push credentials.
```

Use one of:

```bash
gh auth login
```

or set a valid `GH_TOKEN` / `GITHUB_TOKEN`, then:

```bash
git -C /home/dataset-local/work/SOAR push origin main
```
