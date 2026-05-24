# SOAR System Map

本文件负责解释 `/home/dataset-local/work/SOAR` 当前系统的组成、职责边界和数据流。

## Project Goal

这个 workspace 的目标不是训练一个新 base model，而是把 MiniCPM-SALA 做成可运行、可评测、可优化、可提交的 SOAR inference project。

当前阶段的主线是：

```text
MiniCPM-SALA FP16 baseline
  -> stable SGLang serving
  -> SOAR public accuracy evaluation
  -> speed/quantization/speculative optimization candidates
  -> reproducible submission notes
```

## Main Directories

`/home/dataset-local/work/SOAR`

- 根项目目录，保存本地脚本、docs、submit skeleton 和可 git 的结果摘要。

`/home/dataset-local/work/SOAR/repos/`

- 第三方源码 checkout，例如 `SOAR-Toolkit`。
- 视为 external dependency，不把完整第三方仓库 vendoring 到本项目提交里。

`/home/dataset-local/work/SOAR/scripts/`

- 本项目自己的入口脚本。
- 负责 server launch、chunked evaluation、smoke tests、quant/proxy experiments。

`/home/dataset-local/work/SOAR/artifacts/results/`

- 可保留的结果摘要和小型结构化结果。
- 包括 baseline summaries、proxy speed JSON、probe summaries、CSV matrices。

`/home/dataset-local/work/SOAR/artifacts/logs/`

- 本地运行日志。
- 用于 audit，不应进 git。

`/home/dataset-local/work/SOAR/artifacts/checkpoints/`

- 本地 checkpoints。
- 不应进 git。

`/home/dataset-local/work/SOAR/artifacts/draft_heads/`

- EAGLE3 draft head 的默认本地位置。
- 当前没有 MiniCPM-SALA-compatible draft head。

`/home/dataset-local/models/`

- 本地模型目录。
- FP16 baseline model path 是 `/home/dataset-local/models/MiniCPM-SALA`。

## Runtime Components

Server:

- Entrypoint: `scripts/launch_baseline.sh`
- Backend: SGLang server
- Model: `/home/dataset-local/models/MiniCPM-SALA`
- Stable attention path: `minicpm_flashinfer`
- Current stability choice: `--disable-cuda-graph`

Chunk runner:

- Entrypoint: `scripts/run_full_accuracy_chunked.sh`
- Splits 150 public samples into 15 chunks of 10.
- Skips chunks that already have both `summary.json` and `predictions.jsonl`.
- Aggregates chunk outputs into final summary files.

Evaluator:

- Entrypoint: `repos/SOAR-Toolkit/eval_model.py`
- Sends requests to `http://127.0.0.1:30000`.
- Uses `max_out_len=65536`.
- Scores task-specific outputs after responses return.

## Data Flow

```text
repos/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl
  -> scripts/run_full_accuracy_chunked.sh
  -> artifacts/results/baseline_accuracy_chunks/data/chunk_*.jsonl
  -> repos/SOAR-Toolkit/eval_model.py
  -> SGLang server at :30000
  -> outputs/<timestamp>/predictions.jsonl
  -> outputs/<timestamp>/summary.json
  -> artifacts/results/baseline_accuracy_chunks/chunk_*/
  -> artifacts/results/baseline_accuracy_full_chunked_summary.{txt,json}
```

## Source Of Truth

Source inputs:

- Model path: `/home/dataset-local/models/MiniCPM-SALA`
- Evaluation data: `repos/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`
- Evaluator code: `repos/SOAR-Toolkit/eval_model.py`
- Server scripts: `scripts/launch_baseline.sh`
- Runner scripts: `scripts/run_full_accuracy_chunked.sh`

Primary final correctness result:

- `artifacts/results/baseline_accuracy_full_chunked_summary.txt`
- `artifacts/results/baseline_accuracy_full_chunked_summary.json`

Derived or local-only artifacts:

- `outputs/<timestamp>/`
- `artifacts/logs/`
- `artifacts/results/baseline_accuracy_chunks/`
- `artifacts/results/local_proxy_speed/`

Reportable boundary:

- Final public accuracy is reportable for the stable FP16 config.
- Local proxy speed is not official SOAR speed because its manifest says `official=false`.

## What SOAR Is Doing Here

In this workspace, SOAR is being used as a benchmark/evaluation and optimization target.

The completed baseline asks:

```text
Given MiniCPM-SALA and the SOAR public set,
how accurate is the stable FP16 serving configuration?
```

It does not ask:

```text
Can we train MiniCPM-SALA from scratch?
Can we fine-tune the target model?
Can we change the benchmark data or scoring rule?
```

After baseline, optimization experiments ask:

```text
Can we reduce latency or memory while preserving the correctness baseline?
```

That is why quantization, CUDA graph, multi-GPU, and EAGLE must be treated as separate configurations rather than mixed into one baseline aggregate.
