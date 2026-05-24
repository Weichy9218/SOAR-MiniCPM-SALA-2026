# SOAR MiniCPM-SALA 2026

这个仓库是 SOAR 比赛中 MiniCPM-SALA 推理优化项目的整理版工作树。当前目标不是宣称已经完成优化，而是把从零重建过程中已经做好的 baseline 复现脚手架、量化候选路径、提交包骨架、实验记录格式和下一步执行路线集中到一个可 git、可回滚、可继续推进的项目根目录中。

项目根目录固定为：

```bash
/home/dataset-local/work/SOAR
```

项目级 agent 工作偏好记录在 `AGENTS.md`。后续让 Codex 继续开发时，先读取这个文件，它记录了本服务器的路径、uv cache、模型目录、CUDA/GPU、git 和验证纪律。

环境和模型目录也固定到 `/home/dataset-local`，避免污染系统盘或散落到 `/home/dataset-local/work`：

```bash
export UV_CACHE_DIR=/home/dataset-local/.cache/uv
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/dataset-local/.cache/huggingface
export MODEL_ROOT=/home/dataset-local/models
export MODEL_PATH=/home/dataset-local/models/MiniCPM-SALA
export LOCAL_CUDA_HOME=/home/dataset-local/cuda-13.1
```

当前执行策略是先做本地显卡 **micro goal**：验证 `torch + CUDA + tokenizer/model config` 这条最小链路，再启动 SGLang server。这样可以把问题分成两类：环境/GPU 问题和 serving/runtime 问题。

### CUDA 版本判断

SOAR Toolkit 的提交说明没有单独写死 CUDA toolkit 版本，官方要求的核心是通过 `prepare_env.sh` 自定义环境、用 `uv pip install` 安装依赖，并最终能在评测环境中启动 MiniCPM-SALA server。

但当前本地 SGLang 源码已经明确偏向 CUDA 13 栈：

- `cuda-python>=13.0`
- `flashinfer_python[cu13]==0.6.11.post1`
- `flashinfer_cubin==0.6.11.post1`
- `nvidia-cutlass-dsl[cu13]==4.5.1`
- `torch==2.11.0`
- `sglang-kernel==0.4.2.post2`

因此，本项目本地优先使用 `/home/dataset-local/cuda-13.1` 作为 `LOCAL_CUDA_HOME`，路径存在时再导出为 `CUDA_HOME`。`nvidia-smi` 显示驱动 runtime CUDA 13.0，`/home/dataset-local/cuda-13.1/bin/nvcc` 可用；这对 Python wheel 运行通常不是问题，编译自定义 kernel 时才更敏感。原则上不再走 `torch==2.6.0+cu124`，避免和 SGLang cu13 依赖栈混用。提交脚本不会强制要求评测环境也存在这个本地路径，路径不存在时会沿用平台已有 CUDA。

## 当前状态

当前代码库已经从 **reproducibility scaffold** 推进到 **FP16 baseline public accuracy 已完成**。官方必需的 baseline args 已保留，当前为了稳定通过 Smax smoke，默认额外加 `--disable-cuda-graph`。

事实：

- SOAR-Toolkit、SGLang、SpecForge 已经 clone 到本地 `repos/`，并记录了 revision。
- 官方 baseline 启动参数已经固化到脚本中；`scripts/launch_baseline.sh` 默认使用 `DISABLE_CUDA_GRAPH=1`。
- MiniCPM-SALA 已下载到 `/home/dataset-local/models/MiniCPM-SALA`，4 个 safetensors shard 约 18GB。
- SOAR `.venv` 已具备 `torch==2.11.0+cu130`、`transformers==5.8.1`、`flashinfer-python==0.6.11.post1`、`sglang-kernel==0.4.2.post2`、`sgl-kernel==0.4.2.post2`、`xgrammar==0.2.1` 和 `datasets==4.8.5`。
- readiness check 已通过，`torch.cuda` 可见本地 A100。
- correctness smoke 使用短 `/v1/chat/completions` 请求；serving smoke 使用 `SOAR-Toolkit/bench_serving.sh` 跑 2 条样本的 S1/S8/Smax。
- 当前 smoke 结果：S1 0.78s，S8 1.75s，Smax 0.97s，均在 `--disable-cuda-graph` 下通过。
- Full public accuracy 已在 `SOAR-Toolkit/eval_dataset/perf_public_set.jsonl` 的 150 条样本上完成：`ori_accuracy=82.53`、`overall_accuracy=100.00`、duration 17260.25s、total_tokens 733451。
- Full public accuracy 使用 `scripts/run_full_accuracy_chunked.sh` 的可恢复 chunked runner 完成，15/15 个 chunk summary 均已落盘。
- 官方 speed split 本地不可得；`SOAR-Toolkit/README.md` 明确 public toolkit 暂不提供 speed testing dataset。当前新增的是 `local_proxy_speed` 自测 split，不是 official metric。
- Local proxy speed：FP16 baseline 为 S1 136.91s / S8 162.14s / Smax 224.11s；RTN-GPTQ gs1024 为 S1 141.52s / S8 172.86s / Smax 238.65s；RTN-GPTQ sym gs128 为 S1 137.64s / S8 160.99s / Smax 222.38s。全部标记为 `official=false`。
- 量化主线已经准备了 `gptq_marlin` / `fp8 kv cache` 的启动脚本和 RTN-to-GPTQ fallback helper。
- FP8 KV cache 已做 smoke probe：`fp8_e5m2` 和 `fp8_e4m3` 都会在当前 `minicpm_flashinfer` dense-as-sparse path 上失败，暂不作为候选。
- RTN-to-GPTQ W4A16 checkpoint 已生成 gs1024 和 sym gs128 两个 probe。gs1024 local proxy 比 FP16 慢；sym gs128 在 S8/Smax proxy 上略快但 S1 略慢。sym gs128 的 10-sample public accuracy probe 在 `chunk_0003` 跑了 20:47 仍 `0/10` 完成，而 FP16 同 chunk 只用 111.17s，说明有长生成/停止行为风险，不能作为 final。
- `gptq_marlin` 当前不是数据格式问题，而是 kernel-stack blocker：sym gs128 checkpoint metadata 兼容，但已安装 `sgl_kernel==0.4.2.post2` 不导出 `gptq_marlin_repack`。
- EAGLE3 synthetic smoke 已推进：1-step draft checkpoint 已生成，dense fallback serving smoke 可回包；native sparse `minicpm_flashinfer` serving smoke 在本地 target-verify metadata 和 sparse K1/K2 cache-accounting patches 后也可回包。1-step synthetic draft 的 local proxy speed 为 S1 124.42s / S8 171.20s / Smax 242.06s，`official=false`；16-step synthetic_probe draft 的 local proxy speed 为 S1 134.69s / S8 170.99s / Smax 240.81s，`official=false`。两者 S8/Smax 都慢于 FP16 proxy 且输出质量明显不好，所以只是 backend-wiring 证据，不是候选。
- submit package 骨架已经生成在 `submit/`。
- `artifacts/results/` 中已经有 baseline、quantization、A/B/C/D ablation 的结果表占位。

当前仍未完成：

- 本地只找到 `SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`；官方 speed split 需要通过 `SPEED_DATA_S1`、`SPEED_DATA_S8`、`SPEED_DATA_SMAX` 显式提供。
- 开启 CUDA graph 的原始路线会在 Smax smoke 崩溃，栈落在 `minicpm_flashinfer` CUDA graph replay metadata，报错 `kv_indptr[...] should be non-negative`。当前稳定 baseline 默认禁用 CUDA graph，`DISABLE_CUDA_GRAPH=0` 可复现该问题。
- 当前容器没有 Docker/Podman，因此不能直接用官方 SOAR 镜像。
- Hugging Face 下载默认使用 mirror：`HF_ENDPOINT=https://hf-mirror.com`；ModelScope 保留为 fallback。
- 本地 CUDA toolkit 优先使用 `/home/dataset-local/cuda-13.1`，对齐当前 SGLang 的 cu13 依赖。
- 本地 package/temp/cache 已显式固定到 `/home/dataset-local`：`TMPDIR=/home/dataset-local/tmp`、`UV_CACHE_DIR=/home/dataset-local/.cache/uv`、`HF_HOME=/home/dataset-local/.cache/huggingface`、`XDG_CACHE_HOME=/home/dataset-local/.cache`、`HOME=/home/dataset-local`。
- 已清理 `/home/batchcom` 中可再生的大 cache；TinyStories checkpoints 已删除以释放空间，模型和 runtime cache 都留在 `/home/dataset-local`。

因此，当前 FP16 baseline correctness 已建立；official speed 仍必须等待真实 S1/S8/Smax split。Local proxy 只能用于本地比较，不可替代官方 speed。

## 代码库实现了什么

这个仓库目前实现的是一套“先跑通官方 baseline，再逐步打开优化”的工程控制面。

### 1. 项目结构收拢

原来散落在 `/home/dataset-local/work` 下的 SOAR 相关目录已经统一收进本仓库：

```text
SOAR/
  README.md
  scripts/
  submit/
  artifacts/
  repos/
```

其中：

- `scripts/`：本地执行脚本，负责 readiness check、baseline launch、smoke eval、full eval、quant launch、模型下载辅助。
- `submit/`：轻量提交包骨架，包含平台会调用的 `prepare_env.sh` 和 `prepare_model.sh`。
- `artifacts/results/`：可纳入 git 的结果摘要和实验矩阵占位。
- `artifacts/logs/`：本地日志，不纳入 git。
- `artifacts/checkpoints/`：本地 checkpoint，不纳入 git。
- `artifacts/draft_heads/`：本地 EAGLE3 draft head，不纳入 git。
- `repos/`：第三方源码 checkout，不纳入 git。

`.gitignore` 已经排除了 `repos/`、logs、checkpoints、draft heads、模型权重和常见缓存，避免把大文件或第三方仓库历史推到 GitHub。

### 2. 官方 baseline 复现入口

官方 baseline 的 server args 被固定在：

- `scripts/launch_baseline.sh`
- `submit/scripts/launch_baseline.sh`

核心启动参数：

```bash
python -m sglang.launch_server \
  --model-path /home/dataset-local/models/MiniCPM-SALA \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse
```

这里特别保留了 SOAR README 强调的连字符参数名，例如 `--dense-as-sparse`，没有使用下划线版本。

### 3. readiness check

`scripts/check_soar_readiness.sh` 做运行前体检：

- 检查项目根目录。
- 检查 `SOAR-Toolkit`、`SGLang`、`SpecForge` 是否存在。
- 检查 `/home/dataset-local/models/MiniCPM-SALA` 是否存在。
- 打印 `UV_CACHE_DIR`、`HF_ENDPOINT`、`HF_HOME`、`MODEL_ROOT`。
- 打印 Python 版本。
- 检查关键 Python package 是否安装。
- 打印 GPU 型号和显存。

当前输出显示：

```text
model=present
torch=2.11.0+cu130
sglang=0.0.0.dev0
transformers=5.8.1
flashinfer-python=0.6.11.post1
sglang-kernel=0.4.2.post2
sgl-kernel=0.4.2.post2
torch_cuda_available=True
```

这说明模型、cu13 runtime 和 GPU micro runtime 已解锁。

### 4. baseline smoke 和 full benchmark 入口

`scripts/run_baseline_smoke.sh` 做最小可行检查：

- 调用 `/v1/models`。
- 发送 2 条短 `/v1/chat/completions` 请求，写入 `artifacts/results/baseline_smoke_predictions.jsonl`。
- 构造一个极小的 speed smoke JSONL。
- 调用 SOAR `bench_serving.sh` 跑 S1/S8/Smax smoke。

如果模型路径不存在，它会 fail fast，避免把后续错误误判成 SGLang 问题。

`scripts/run_full_baseline.sh` 是完整 public correctness 和 speed benchmark 的入口：

- public correctness 使用 `SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`。
- speed benchmark 读取 `SPEED_DATA_S1`、`SPEED_DATA_S8`、`SPEED_DATA_SMAX`。
- 日志写到 `artifacts/logs/`。

`scripts/run_full_accuracy_chunked.sh` 是当前已验证的 full public accuracy 可恢复入口：

- 将 150 条 public samples 切成 15 个 chunk，每个 chunk 独立调用官方 `eval_model.py`。
- 已完成的 chunk 通过 `summary.json` 和 `predictions.jsonl` 跳过。
- 最终 aggregate 写入 `artifacts/results/baseline_accuracy_full_chunked_summary.{json,txt}`。

`scripts/prepare_local_speed_proxy.py` 和 `scripts/run_local_proxy_speed.sh` 是本地自测入口：

- 从 public accuracy prompts 生成 deterministic S1/S8/Smax proxy split。
- 输出目录为 `artifacts/results/local_proxy_speed/`。
- manifest 中明确 `official=false`。
- 只能用于本地候选间相对比较，不能写成官方 SOAR speed。

### 5. 模型准备辅助

`scripts/prepare_local_model.sh` 是本地 helper，不是最终提交逻辑。它用于把 MiniCPM-SALA 准备到：

```bash
/home/dataset-local/models/MiniCPM-SALA
```

默认优先走 Hugging Face mirror：

```bash
SOURCE=huggingface \
HF_ENDPOINT=https://hf-mirror.com \
MODEL_DIR=/home/dataset-local/models/MiniCPM-SALA \
bash scripts/prepare_local_model.sh
```

如果 Hugging Face mirror 不稳定，可以切换到 ModelScope：

```bash
SOURCE=modelscope \
MODEL_DIR=/home/dataset-local/models/MiniCPM-SALA \
bash scripts/prepare_local_model.sh
```

当前模型已通过 Hugging Face mirror 下载完成，日志在 `artifacts/logs/prepare_model_hf.log`。再次下载前仍应先看 `df -h /home/dataset-local`，避免覆盖或重复拉取大文件。

### 6. 量化候选路径

当前量化主线没有手写 CUDA kernel，而是对齐 SOAR/SGLang 已有路径：

- `--quantization gptq`
- `--quantization gptq_marlin`
- `--kv-cache-dtype fp8_e5m2`
- `--kv-cache-dtype fp8_e4m3`

对应启动脚本：

```bash
MODEL_PATH=/path/to/MiniCPM-SALA-GPTQ \
QUANTIZATION=gptq_marlin \
KV_CACHE_DTYPE=fp8_e5m2 \
bash scripts/launch_quant.sh
```

`scripts/quantize_gptq_rtn.py` 是一个 RTN-to-GPTQ fallback helper，来源思路对齐 SOAR demo-quant。它的作用是生成 SGLang 能识别的 GPTQ 格式：

- `qweight`
- `scales`
- `qzeros`
- `g_idx`
- `quantize_config.json`

当前 probe 结果：

- `fp8_e5m2`：server 可以启动，但首个 smoke request 在 sparse attention 路径报 `FlashAttention only support fp16 and bf16 data type`。
- `fp8_e4m3`：server 可以启动，但首个 smoke request 在 Triton 编译时报 `fp8e4nv` unsupported。
- `MiniCPM-SALA-GPTQ-RTN-gs1024`：RTN-to-GPTQ asymmetric checkpoint 已生成，local proxy 比 FP16 慢约 S1 3.37%、S8 6.61%、Smax 6.49%；full public accuracy 未评估。
- `MiniCPM-SALA-GPTQ-RTN-sym-gs128`：RTN-to-GPTQ symmetric checkpoint 已生成，普通 `--quantization gptq` smoke 通过，local proxy 为 S1 137.64s / S8 160.99s / Smax 222.38s；相对 FP16 是 S1 略慢、S8/Smax 略快。10-sample `chunk_0003` accuracy probe 在 20:47 后仍 `0/10` 完成，server 四个请求持续 decode，暂不跑 full public accuracy。
- `gptq_marlin`：gs1024 group size 不受当前 Marlin 支持；sym gs128 会通过 Marlin compatibility detection，但加载时失败于 `gptq_marlin_repack` missing，需要匹配 SOAR demo kernel stack 或 rebuild `sgl_kernel`。

注意：这个 fallback 是格式跑通工具，不是最终优化结论。是否能作为 final candidate，必须先跑 public accuracy 和 official speed，不能只看能否加载或 local proxy。

### 7. submit package 骨架

`submit/` 是当前轻量提交包：

```text
submit/
  prepare_env.sh
  prepare_model.sh
  README_SOAR.md
  scripts/
```

`prepare_env.sh`：

- 使用 `uv pip install`。
- 默认复用 `/home/dataset-local/.cache/uv`。
- 默认导出 `HF_ENDPOINT=https://hf-mirror.com`。
- 默认模型路径为 `/home/dataset-local/models/MiniCPM-SALA`。
- 如果提交包内存在 `sglang/python`，会用 editable install 覆盖 SGLang。
- 在当前开发树中，也能 fallback 到 `repos/sglang/python`。
- 导出默认 `SGLANG_SERVER_ARGS`。

`prepare_model.sh`：

- 默认模式是 symlink，不复制完整原模型。
- 可选 `SOAR_MODEL_PREP=quantize_rtn_gptq` 调用 RTN-to-GPTQ helper。
- 不会把原始模型塞进提交包。

### 8. 实验记录格式

当前已经建立三个结果文件：

- `artifacts/results/baseline_summary.md`
- `artifacts/results/quant_matrix.csv`
- `artifacts/results/ablation_matrix.csv`

这些文件只记录已验证事实和占位值 `0.XXX`，没有编造指标。当前 FP16 baseline public accuracy 已完成；S1/S8/Smax official split 仍缺失。2-request smoke 和 local proxy 都不能当 official speed split。

`quant_matrix.csv` 已预留：

- FP16 baseline
- FP16 local proxy
- FP8 KV failure probes
- RTN-GPTQ gs1024 local proxy
- RTN-GPTQ sym gs128 local proxy
- GPTQ-Marlin kernel/blocker probes
- W4A16/GPTQ/Marlin candidate
- W4A16/GPTQ/Marlin + FP8 KV candidate

`ablation_matrix.csv` 已预留 speculative / draft loss / quantization / accepted length / walltime 等字段，后续 A/B/C/D 实验直接补结果。

### 9. EAGLE3 / LK loss 的准备状态

当前已经有 MiniCPM-SALA EAGLE3 synthetic 1-step 和 synthetic_probe 16-step draft checkpoints。`scripts/launch_eagle3.sh` 已经固化 EAGLE3 serving 参数和 `DRAFT_MODEL_PATH` 前置检查；dense fallback serving smoke 能回包，native sparse `minicpm_flashinfer` serving smoke 也已在本地 target-verify metadata 和 sparse K1/K2 cache-accounting patches 后跑通。

已经完成的准备：

- SGLang 源码中已确认存在 EAGLE3 参数解析和 worker。
- `scripts/launch_eagle3.sh` 已加入，默认 `--speculative-algorithm EAGLE3`、`--speculative-eagle-topk 1`、`--speculative-num-steps 3`、`--speculative-num-draft-tokens 4`。
- `scripts/run_eagle3_smoke_train.sh` 已能通过 SpecForge 训练 synthetic 1-step draft，输出到 `artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`。
- Dense fallback serving smoke 使用 `ATTENTION_BACKEND=flashinfer --force-dense-minicpm`，两条短 chat 请求 HTTP 200；结果在 `artifacts/results/eagle3_default_serving_smoke.json`。
- Native sparse `minicpm_flashinfer` serving smoke 使用 `ATTENTION_BACKEND=minicpm_flashinfer FORCE_DENSE_MINICPM=0 DENSE_AS_SPARSE=1`，两条短 chat 请求 HTTP 200；结果在 `artifacts/results/eagle3_sparse_after_target_verify_len_patch.json`。
- Native sparse EAGLE3 1-step local proxy speed 为 S1 124.42s / S8 171.20s / Smax 242.06s，`official=false`。相比 FP16 proxy 的 136.91s / 162.14s / 224.11s，只在 S1 更快，S8/Smax 更慢。
- Native sparse EAGLE3 synthetic_probe 16-step local proxy speed 为 S1 134.69s / S8 170.99s / Smax 240.81s，`official=false`。相比 FP16 proxy，S1 只小幅更快，S8/Smax 更慢；short smoke 两条预测都退化并以 `finish_reason=length` 结束。
- 当前 draft 只用 synthetic 6-record smoke data 训练 1 step，不能冒充有效 draft 或 speed 结果。
- 当前 native sparse smoke predictions 明显退化，说明现在的 EAGLE3 结果只能证明 serving path 接通，不能证明 correctness 或可用加速。
- `topk=1` 时，SGLang 会自动约束 `num_draft_tokens = num_steps + 1`，符合单路径 chain speculation。
- SpecForge PR #492 已 fetch 到本地分支 `pr-492-lk-loss`。
- PR #492 包含：
  - `specforge/core/lk_loss.py`
  - `scripts/train_eagle3.py --lk-loss-type {lambda,alpha}`
  - `--kl-scale`
  - `--kl-decay`
  - per-position `acceptance_rate_i` logging

注意：PR #492 的 lambda loss 实现使用 `1 - acceptance_rate` 作为 TV-side term，和目标 LK^lambda 公式需要再核对，不能无脑 cherry-pick。

## 本地源码 revision

第三方源码存在本地 `repos/`，但不纳入 git。

- SOAR-Toolkit: `2ed4ade`
- SGLang: `791a2f0`
- SpecForge main: `d5fb617`
- SpecForge LK PR #492: `4385d78`

## 常用命令

进入项目：

```bash
cd /home/dataset-local/work/SOAR
```

检查环境：

```bash
bash scripts/check_soar_readiness.sh
```

建立本地 micro env：

```bash
bash scripts/setup_micro_env.sh
source .venv/bin/activate
```

运行本地显卡 micro goal：

```bash
bash scripts/run_micro_goal.sh
```

准备模型：

```bash
SOURCE=huggingface \
HF_ENDPOINT=https://hf-mirror.com \
MODEL_DIR=/home/dataset-local/models/MiniCPM-SALA \
bash scripts/prepare_local_model.sh
```

启动 baseline server：

```bash
bash scripts/launch_baseline.sh
```

跑 baseline smoke：

```bash
bash scripts/run_baseline_smoke.sh
```

跑 full baseline：

```bash
SPEED_DATA_S1=/path/to/s1.jsonl \
SPEED_DATA_S8=/path/to/s8.jsonl \
SPEED_DATA_SMAX=/path/to/smax.jsonl \
bash scripts/run_full_baseline.sh
```

跑 local proxy speed 自测：

```bash
RUN_NAME=baseline_fp16 bash scripts/run_local_proxy_speed.sh
```

注意：这会使用 `artifacts/results/local_proxy_speed/`，不是 official SOAR speed split。

恢复或复核 full public accuracy：

```bash
CHUNK_SIZE=10 ACCURACY_CONCURRENCY=4 \
bash scripts/run_full_accuracy_chunked.sh
```

启动量化 candidate：

```bash
MODEL_PATH=/path/to/MiniCPM-SALA-GPTQ \
QUANTIZATION=gptq_marlin \
KV_CACHE_DTYPE=fp8_e5m2 \
bash scripts/launch_quant.sh
```

启动 EAGLE3 candidate：

```bash
DRAFT_MODEL_PATH=/path/to/minicpm-sala-eagle3-draft \
bash scripts/launch_eagle3.sh
```

如果 `DRAFT_MODEL_PATH` 为空或不存在，脚本会 fail fast。当前可用于 plumbing smoke 的 draft 在 `artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1`，但它不是性能候选。

EAGLE3 serving smoke：

```bash
DRAFT_MODEL_PATH=/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1 \
RUN_NAME=eagle3_default_serving_smoke \
MODE=force_dense_minicpm_flashinfer_dense_as_sparse \
bash scripts/run_eagle3_serving_smoke.sh
```

Native sparse EAGLE3 smoke：

```bash
DRAFT_MODEL_PATH=/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1 \
ATTENTION_BACKEND=minicpm_flashinfer \
FORCE_DENSE_MINICPM=0 \
DENSE_AS_SPARSE=1 \
bash scripts/launch_eagle3.sh

EXPECT_PASS=1 \
RUN_NAME=eagle3_sparse_after_target_verify_len_patch \
DRAFT_MODEL_PATH=/home/dataset-local/work/SOAR/artifacts/draft_heads/minicpm_sala_eagle3_smoke/epoch_0_step_1 \
bash scripts/run_eagle3_serving_smoke.sh
```

## 下一步计划

### Phase 1: 解锁 baseline

目标：官方 FP16 baseline 可以完整跑通 correctness + S1/S8/Smax。

步骤：

1. 运行 `bash scripts/setup_micro_env.sh`，优先复用 `/home/dataset-local/.cache/uv`，并保持 `UV_TORCH_BACKEND=cu130`。
2. `source .venv/bin/activate` 后运行 `bash scripts/run_micro_goal.sh`，确认本地 A100 可被 `torch` 使用；当前 micro goal 已通过。
3. 运行 `bash scripts/check_soar_readiness.sh`，确认 `model=present` 且关键包不再 missing。
4. 启动 `bash scripts/launch_baseline.sh`。默认会加 `--disable-cuda-graph`；如需复现原始 CUDA graph crash，使用 `DISABLE_CUDA_GRAPH=0 bash scripts/launch_baseline.sh`。
5. 运行 `bash scripts/run_baseline_smoke.sh`；当前 smoke 已通过。
6. Full public accuracy 已完成并记录：`ori_accuracy=82.53`、`overall_accuracy=100.00`。
7. 准备官方 speed split 后设置 `SPEED_DATA_S1`、`SPEED_DATA_S8`、`SPEED_DATA_SMAX` 跑完整 S1/S8/Smax。
8. 在 official split 不可得时，只能用 `scripts/run_local_proxy_speed.sh` 做本地候选比较，并保持 `official=false` 标注。
9. 更新 `artifacts/results/baseline_summary.md`。

done when：

- `baseline_accuracy_full_chunked_summary.json` 存在且 `status=complete`。
- `baseline_speed_s1.json`、`baseline_speed_s8.json`、`baseline_speed_smax.json` 存在。
- `baseline_summary.md` 写入 `ori_accuracy`、`overall_accuracy`、Benchmark Duration、TTFT/ITL/output tokens/s、GPU memory peak、OOM/server crash 状态。

### Phase 2: 量化主线

目标：优先得到 correctness 不掉且 walltime 有收益的 target model 配置。

步骤：

1. 用现有 SGLang GPTQ/GPTQ-Marlin 路径，不写新 kernel。
2. 如果有官方/成熟 GPTQ 权重，优先直接加载。
3. 如果没有，先用 `quantize_gptq_rtn.py` 做格式 smoke，但不要默认当 final；当前 gs1024 RTN-GPTQ local proxy 已慢于 FP16，sym gs128 只有混合小幅 proxy 结果且 10-sample accuracy probe 出现长生成。
4. 分别测试：
   - FP16 target
   - GPTQ/Marlin target
   - GPTQ/Marlin + FP8 KV cache
5. 每个配置必须先跑 smoke；有明显 local proxy 收益后再跑 full public accuracy 和 official speed。
6. 更新 `artifacts/results/quant_matrix.csv`。

done when：

- 至少一个量化 candidate 能加载。
- accuracy 不明显下降。
- S1/S8/Smax Benchmark Duration 有完整记录。
- 若量化失败，记录失败日志和原因，回退 FP16。

### Phase 3: EAGLE3 topk=1 单路径

目标：只做保守 chain speculation，不做多分支 tree。

默认配置：

```bash
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path <draft_head_path> \
--speculative-eagle-topk 1 \
--speculative-num-steps 3 \
--speculative-num-draft-tokens 4 \
--speculative-accept-threshold-single 1.0 \
--speculative-accept-threshold-acc 1.0
```

步骤：

1. 保留当前 native sparse MiniCPM `TARGET_VERIFY` smoke-level patch，并继续监控 sparse cache allocator leak；dense fallback 只能用于 wiring debug。
2. 用非 synthetic 数据训练或提供 MiniCPM-SALA-compatible EAGLE3 draft model/head，放到 `artifacts/draft_heads/` 或 `/home/dataset-local/models/`。
3. 基于 `llama_eagle3.py` 和 `minicpm.py` 判断是否需要 `minicpm_sala_eagle3.py`。
4. 使用 `scripts/launch_eagle3.sh` 做 serving launch smoke。
5. 先只做 forward smoke：
   - target hidden states shape
   - draft logits shape
   - vocab dim
   - dtype/device
6. 再跑 topk=1 chain verify。
7. 如果 sparse target verify 报错，先定位 metadata/cache state，不要放宽 acceptance threshold。
8. 一天内修不通就停止 speculative，回到量化主线。

done when：

- draft head 可加载。
- topk=1 verify 能完成。
- accuracy 不掉。
- S1 有正收益，或 acceptance 指标显示有继续价值。

### Phase 4: LK^lambda draft training

目标：只在 baseline 和最小 EAGLE3 serving 路径可用后，比较 KL draft 和 LK^lambda draft。

步骤：

1. 基于 SpecForge PR #492 移植最小 LK loss。
2. 核对目标公式：

```text
p = softmax(target_logits).detach()
q = softmax(draft_logits)
alpha = sum_x min(p_x, q_x)
tv = 1 - alpha
kl = sum_x p_x * (log p_x - log q_x)
lambda = exp(-eta * stop_gradient(alpha))
loss = lambda * kl + (1 - lambda) * tv
```

3. 先跑 100-1000 samples smoke train。
4. 记录 `acceptance_rate_i`，因为 accepted length 是链式事件。
5. 跑 A/B/C/D：
   - A: FP16 target + KL draft
   - B: FP16 target + LK draft
   - C: quantized target + KL draft
   - D: quantized target + LK draft

done when：

- 每个可运行 baseline 有 training-side metrics。
- 每个可运行 baseline 有 serving-side accuracy + S1/S8/Smax。
- `artifacts/results/ablation_matrix.csv` 完整更新。

### Phase 5: 最终提交选择

选择规则：

1. accuracy 明显下降的配置淘汰。
2. server crash、OOM、long-tail hang 的配置淘汰。
3. 只提升 S1 但拖慢 S8/Smax 的 speculative 配置不作为 final。
4. walltime 优先于单独的 accepted length。
5. 提交包越小越好，工程风险越低越好。

最低可接受 final：

- FP16 baseline 可跑。
- correctness eval 通过。
- S1/S8/Smax 有完整结果。
- 量化或 speculative 至少一个方向有正收益。
- README 和结果表完整记录。

## 下一轮给 Codex 的 prompt

下面这段可以直接作为下一轮 prompt 使用：

```text
你是 Codex，继续在 /home/dataset-local/work/SOAR 项目中推进 SOAR MiniCPM-SALA 2026。

请先读取 README.md、submit/README_SOAR.md、artifacts/results/baseline_summary.md，并运行：

  bash scripts/check_soar_readiness.sh

当前原则：
- correctness 第一，speed 第二。
- FP16 baseline public accuracy 已完成；官方 speed split 仍缺失。local proxy speed 已生成但只能自测，不能当 official。
- 不替换 base model/tokenizer。
- 不放宽 speculative acceptance threshold。
- 不写新的 CUDA kernel。
- 不把模型权重、logs、checkpoints、第三方完整仓库推入 git。

下一步目标：
1. 补齐真实官方 FP16 baseline speed split，拿到 S1/S8/Smax full Benchmark Duration；不要把 `local_proxy_speed` 当官方 split。
2. 固定环境变量：

   export UV_CACHE_DIR=/home/dataset-local/.cache/uv
   export HF_ENDPOINT=https://hf-mirror.com
   export HF_HOME=/home/dataset-local/.cache/huggingface
   export MODEL_ROOT=/home/dataset-local/models
   export MODEL_PATH=/home/dataset-local/models/MiniCPM-SALA

3. 先跑本地显卡 micro goal：

   bash scripts/setup_micro_env.sh
   source .venv/bin/activate
   bash scripts/run_micro_goal.sh

4. 确认 `/home/dataset-local/models/MiniCPM-SALA/config.json` 存在；当前模型已下载完成。
5. 安装或修复完整 SGLang/SOAR runtime dependencies，优先使用 uv pip，并继续复用 `/home/dataset-local/.cache/uv`；当前 smoke 依赖已满足。
6. 启动稳定 baseline：

   bash scripts/launch_baseline.sh

7. 跑 smoke：

   bash scripts/run_baseline_smoke.sh

8. Full public accuracy 已完成：`ori_accuracy=82.53`、`overall_accuracy=100.00`；准备官方 speed split 后再跑 full S1/S8/Smax。
9. Quant/EAGLE probes 已记录：FP8 KV 当前不兼容；RTN-GPTQ gs1024 local proxy 慢于 FP16；RTN-GPTQ sym gs128 只有 mixed proxy 且 10-sample accuracy probe 长生成；`gptq_marlin` 卡在 kernel symbol；EAGLE3 synthetic 1-step 和 synthetic_probe 16-step drafts、dense fallback smoke 和 native sparse smoke/proxy 已跑通，但 synthetic drafts 输出质量差且 S8/Smax proxy 慢于 FP16。
10. 更新：
   - artifacts/results/baseline_summary.md
   - artifacts/results/quant_matrix.csv
   - submit/README_SOAR.md

如果 speed split 或 baseline server 失败，请只修 baseline，并基于 logs 给出最小修复。EAGLE3 当前只能继续训练/接入真实 draft，不要把 synthetic draft 的 dense fallback 或 native sparse smoke/proxy 当候选结果。
```
