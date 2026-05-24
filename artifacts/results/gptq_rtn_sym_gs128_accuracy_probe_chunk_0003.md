# GPTQ RTN Sym GS128 Accuracy Probe

Status: **stopped; no completed samples after 20:47**

## Setup

- Model: `/home/dataset-local/models/MiniCPM-SALA-GPTQ-RTN-sym-gs128`
- Serving: `--quantization gptq`, `--dtype float16`, `--disable-cuda-graph`, `--attention-backend minicpm_flashinfer`, `--dense-as-sparse`
- Data: `artifacts/results/baseline_accuracy_chunks/data/chunk_0003_0030_0039.jsonl`
- Samples: 10
- Concurrency: 4

## Evidence

- FP16 baseline on the same chunk: `ori_accuracy=100.0`, duration `111.1713s`, total tokens `3423`.
- GPTQ probe evaluator log stayed at `Generating: 0/10` after `20:47`.
- Server decode stayed healthy but long-running: `#running-req: 4`, `#full token` increased from `133232` at `19:37:08` to `267612` at `19:57:35`.
- Logs:
  - `artifacts/logs/gptq_rtn_sym_gs128_accuracy_probe_chunk_0003.log`
  - `artifacts/logs/gptq_rtn_sym_gs128_accuracy_probe_server/quant_gptq_auto.log`

## Decision

Do not run full public accuracy for `gptq_rtn_sym_gs128` now. Its local proxy result was already mixed, and this chunk probe suggests pathological long generation or stop-token behavior on correctness data.
