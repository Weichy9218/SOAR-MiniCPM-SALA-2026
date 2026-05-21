# SOAR Project Agent Instructions

本文件记录 `/home/dataset-local/work/SOAR` 项目的长期开发偏好。进入本仓库工作时，优先遵守这里的规则，再结合用户当轮请求执行。

## 目标与语气

- 默认用中文回复；技术名词、CLI flag、API、benchmark 指标保留英文。
- 目标是把 MiniCPM-SALA 的 SOAR 推理优化做成可运行、可评测、可回滚的工程项目。
- correctness 第一，speed 第二；没有 baseline 数据前，不进入复杂 speculative / LK 实验。
- 不编造 metrics、结果、论文结论或实现状态。缺失值用 `0.XXX` 或明确写 `pending/blocked`。

## 机器与系统

- 服务器系统：Ubuntu 22.04.5 LTS。
- CPU：12 核。
- 内存：约 120 GB。
- GPU：NVIDIA A100-SXM4-80GB * 1。
- 当前 `nvidia-smi` 驱动 runtime 显示 CUDA 13.0。
- 本地 CUDA toolkit 优先使用 `/home/dataset-local/cuda-13.1`，但提交脚本不能强依赖这个路径。
- `/home/dataset-local` 当前是主要可用盘，总量约 50 GB；空间紧张时先运行 `df -h /home/dataset-local`。

## 路径约定

- 项目根目录固定为 `/home/dataset-local/work/SOAR`。
- 所有 SOAR 项目文件尽量保存在 `/home/dataset-local` 下，避免散落到 `/home/dataset-local/work` 根目录或系统盘。
- 第三方源码放在 `/home/dataset-local/work/SOAR/repos/`。
- 本地脚本放在 `/home/dataset-local/work/SOAR/scripts/`。
- 提交包骨架放在 `/home/dataset-local/work/SOAR/submit/`。
- 可 git 的实验结果摘要放在 `/home/dataset-local/work/SOAR/artifacts/results/`。
- 本地日志放在 `/home/dataset-local/work/SOAR/artifacts/logs/`，不进 git。
- checkpoints 放在 `/home/dataset-local/work/SOAR/artifacts/checkpoints/`，不进 git。
- EAGLE3 draft head 放在 `/home/dataset-local/work/SOAR/artifacts/draft_heads/`，不进 git，除非最终提交明确需要小权重。
- 模型下载与处理输出默认放在 `/home/dataset-local/models/`。
- MiniCPM-SALA 默认路径为 `/home/dataset-local/models/MiniCPM-SALA`。

## 环境约定

- 使用 `uv`，不要直接用普通 `pip install`。
- 优先复用 uv cache：

```bash
export UV_CACHE_DIR=/home/dataset-local/.cache/uv
```

- Hugging Face 下载优先走 mirror：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/dataset-local/.cache/huggingface
```

- 本地开发可设置：

```bash
export MODEL_ROOT=/home/dataset-local/models
export MODEL_PATH=/home/dataset-local/models/MiniCPM-SALA
export LOCAL_CUDA_HOME=/home/dataset-local/cuda-13.1
```

- 如果 `/home/dataset-local/cuda-13.1` 存在，可以在本地脚本中导出为 `CUDA_HOME`；如果不存在，沿用平台环境，不要让脚本因为 `CUDA_HOME` 未定义而退出。
- 当前 SGLang 源码偏向 CUDA 13 栈：`cuda-python>=13.0`、`flashinfer_python[cu13]`、`torch==2.11.0`、`sglang-kernel==0.4.2.post2`。
- 不要无意切回 `torch==2.6.0+cu124`，避免和 SGLang cu13 依赖混用。

## SOAR 工作流

- 先跑官方 FP16 baseline，再谈量化和 speculative。
- baseline 默认 server args：

```bash
--disable-radix-cache
--attention-backend minicpm_flashinfer
--chunked-prefill-size 8192
--skip-server-warmup
--dense-as-sparse
```

- 参数必须用连字符形式，例如 `--dense-as-sparse`。
- 推荐检查顺序：

```bash
bash scripts/check_soar_readiness.sh
bash scripts/setup_micro_env.sh
source .venv/bin/activate
bash scripts/run_micro_goal.sh
bash scripts/launch_baseline.sh
bash scripts/run_baseline_smoke.sh
```

- 模型不存在或依赖缺失时，只修 baseline 环境，不进入 EAGLE3 / LK。
- 完整 benchmark 需要同时记录 correctness、S1、S8、Smax，不要只报告 tokens/s。

## 开发边界

- 不替换 base model、tokenizer 或 eval dataset。
- 不放宽 speculative acceptance threshold 换速度。
- 不手写新的 CUDA kernel，优先复用 SOAR/SGLang 现有 GPTQ、Marlin、KV cache dtype、speculative kernel。
- 不把大模型、完整第三方仓库、logs、checkpoints、cache、临时下载物提交到 git。
- `repos/` 是本地第三方 checkout，根仓库只记录 revision 和 patch/脚本，不 vendoring 大仓库。
- 修改 `submit/` 时同时考虑本地开发环境和官方评测环境：本地路径可以作为默认值，但不能成为不可替代的硬依赖。

## Git 与提交

- 根仓库：`/home/dataset-local/work/SOAR`。
- GitHub remote：`Weichy9218/SOAR-MiniCPM-SALA-2026`。
- 小步提交，提交内容要能回滚。
- 未经用户明确要求，不要运行 destructive git 命令。
- 可提交内容：README、AGENTS.md、scripts、submit、`artifacts/results/*.md`、`artifacts/results/*.csv`。
- 不提交内容：`.venv/`、`repos/`、模型权重、logs、checkpoints、draft head 大权重、cache。

## 验证纪律

- 修改 shell 脚本后运行 `bash -n`。
- 修改 Python 脚本后至少运行 `python -m py_compile`。
- 环境相关改动后运行 `bash scripts/check_soar_readiness.sh`。
- GPU 相关判断优先用实际 `torch.cuda` smoke，而不是只看 `nvidia-smi`。
- 每次实验或失败都更新对应记录：
  - `artifacts/results/baseline_summary.md`
  - `artifacts/results/quant_matrix.csv`
  - `artifacts/results/ablation_matrix.csv`
  - `submit/README_SOAR.md`

