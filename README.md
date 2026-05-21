# SOAR MiniCPM-SALA 2026

这个仓库是 SOAR 比赛中 MiniCPM-SALA 推理优化项目的整理版工作树。当前目标不是宣称已经完成优化，而是把从零重建过程中已经做好的 baseline 复现脚手架、量化候选路径、提交包骨架、实验记录格式和下一步执行路线集中到一个可 git、可回滚、可继续推进的项目根目录中。

项目根目录固定为：

```bash
/home/dataset-local/work/SOAR
```

## 当前状态

当前代码库已经完成的是 **reproducibility scaffold**，还没有完成真正的 speed benchmark 或 EAGLE3/LK 实验。

事实：

- SOAR-Toolkit、SGLang、SpecForge 已经 clone 到本地 `repos/`，并记录了 revision。
- 官方 baseline 启动参数已经固化到脚本中。
- correctness smoke、serving benchmark smoke、full baseline benchmark 的入口脚本已经写好。
- 量化主线已经准备了 `gptq_marlin` / `fp8 kv cache` 的启动脚本和 RTN-to-GPTQ fallback helper。
- submit package 骨架已经生成在 `submit/`。
- `artifacts/results/` 中已经有 baseline、quantization、A/B/C/D ablation 的结果表占位。

当前阻塞：

- `/models/MiniCPM-SALA` 不存在。
- 当前 Python 环境缺少 `torch`、`sglang`、`transformers`、`flashinfer-python`、`sglang-kernel` 等 runtime dependencies。
- 当前容器没有 Docker/Podman，因此不能直接用官方 SOAR 镜像。
- `/home/dataset-local` 剩余空间约 12GB，而 MiniCPM-SALA 模型仓库约 19GB，不适合放在本地盘；如果下载，优先放 `/models/MiniCPM-SALA`。
- Hugging Face 直连超时；ModelScope 页面可达。

因此，当前阶段严格停止在 baseline smoke 前，未进入 speculative decoding 或 LK loss 训练。

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
  --model-path /models/MiniCPM-SALA \
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
- 检查 `/models/MiniCPM-SALA` 是否存在。
- 打印 Python 版本。
- 检查关键 Python package 是否安装。
- 打印 GPU 型号和显存。

当前输出显示：

```text
model=missing
torch=missing
sglang=missing
transformers=missing
flashinfer-python=missing
sglang-kernel=missing
```

这就是 baseline 不能启动的直接原因。

### 4. baseline smoke 和 full benchmark 入口

`scripts/run_baseline_smoke.sh` 做最小可行检查：

- 调用 SOAR `eval_model.py`。
- 使用 `--num_samples 10`。
- 构造一个极小的 speed smoke JSONL。
- 调用 SOAR `bench_serving.sh` 跑 S1/S8/Smax smoke。

如果模型路径不存在，它会 fail fast，避免把后续错误误判成 SGLang 问题。

`scripts/run_full_baseline.sh` 是完整 public correctness 和 speed benchmark 的入口：

- public correctness 使用 `SOAR-Toolkit/eval_dataset/perf_public_set.jsonl`。
- speed benchmark 读取 `SPEED_DATA_S1`、`SPEED_DATA_S8`、`SPEED_DATA_SMAX`。
- 日志写到 `artifacts/logs/`。

### 5. 模型准备辅助

`scripts/prepare_local_model.sh` 是本地 helper，不是最终提交逻辑。它用于把 MiniCPM-SALA 准备到：

```bash
/models/MiniCPM-SALA
```

默认优先走 ModelScope：

```bash
SOURCE=modelscope MODEL_DIR=/models/MiniCPM-SALA \
bash scripts/prepare_local_model.sh
```

这个脚本没有被自动执行，因为模型约 19GB，而本地盘空间不足；是否下载需要根据系统盘空间和网络状态决定。

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

注意：这个 fallback 是格式跑通工具，不是最终优化结论。是否能作为 final candidate，必须先跑 public accuracy，不能只看能否加载。

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

这些文件目前只记录事实和占位值 `0.XXX`，没有编造指标。

`quant_matrix.csv` 已预留：

- FP16 baseline
- W4A16/GPTQ/Marlin candidate
- W4A16/GPTQ/Marlin + FP8 KV candidate

`ablation_matrix.csv` 已预留 speculative / draft loss / quantization / accepted length / walltime 等字段，后续 A/B/C/D 实验直接补结果。

### 9. EAGLE3 / LK loss 的准备状态

当前没有实现 MiniCPM-SALA EAGLE3 adapter，也没有训练 draft head。原因是 baseline 尚未跑通，按 correctness-first 路线必须先暂停 speculative。

已经完成的准备：

- SGLang 源码中已确认存在 EAGLE3 参数解析和 worker。
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

准备模型：

```bash
SOURCE=modelscope MODEL_DIR=/models/MiniCPM-SALA \
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

启动量化 candidate：

```bash
MODEL_PATH=/path/to/MiniCPM-SALA-GPTQ \
QUANTIZATION=gptq_marlin \
KV_CACHE_DTYPE=fp8_e5m2 \
bash scripts/launch_quant.sh
```

## 下一步计划

### Phase 1: 解锁 baseline

目标：官方 FP16 baseline 可以完整跑通 correctness + S1/S8/Smax。

步骤：

1. 安装 runtime dependencies，优先复用官方 SOAR/SGLang 版本。
2. 把模型准备到 `/models/MiniCPM-SALA`。
3. 运行 `bash scripts/check_soar_readiness.sh`，确认 `model=present`，关键包不再 missing。
4. 启动 `bash scripts/launch_baseline.sh`。
5. 运行 `bash scripts/run_baseline_smoke.sh`。
6. smoke 通过后运行 full public accuracy 和 S1/S8/Smax。
7. 更新 `artifacts/results/baseline_summary.md`。

done when：

- `baseline_accuracy.json` 存在。
- `baseline_speed_s1.json`、`baseline_speed_s8.json`、`baseline_speed_smax.json` 存在。
- `baseline_summary.md` 写入 `ori_accuracy`、`overall_accuracy`、Benchmark Duration、TTFT/ITL/output tokens/s、GPU memory peak、OOM/server crash 状态。

### Phase 2: 量化主线

目标：优先得到 correctness 不掉且 walltime 有收益的 target model 配置。

步骤：

1. 用现有 SGLang GPTQ/GPTQ-Marlin 路径，不写新 kernel。
2. 如果有官方/成熟 GPTQ 权重，优先直接加载。
3. 如果没有，先用 `quantize_gptq_rtn.py` 做格式 smoke，但不要默认当 final。
4. 分别测试：
   - FP16 target
   - GPTQ/Marlin target
   - GPTQ/Marlin + FP8 KV cache
5. 每个配置必须先跑 accuracy，再跑 speed。
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

1. 基于 `llama_eagle3.py` 和 `minicpm.py` 判断是否需要 `minicpm_sala_eagle3.py`。
2. 先只做 forward smoke：
   - target hidden states shape
   - draft logits shape
   - vocab dim
   - dtype/device
3. 再跑 topk=1 chain verify。
4. 如果 Lightning Attention target verify 报错，先判断 topk=1 是否能退化为普通连续 verify。
5. 一天内修不通就停止 speculative，回到量化主线。

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
- baseline 没跑通前，不做 EAGLE3/LK。
- 不替换 base model/tokenizer。
- 不放宽 speculative acceptance threshold。
- 不写新的 CUDA kernel。
- 不把模型权重、logs、checkpoints、第三方完整仓库推入 git。

下一步目标：
1. 解锁官方 FP16 baseline。
2. 确认 /models/MiniCPM-SALA 是否存在；如果不存在，优先尝试 ModelScope 下载到 /models/MiniCPM-SALA，不要下载到 /home/dataset-local。
3. 安装或修复 SGLang/SOAR runtime dependencies，优先使用 uv pip。
4. 启动：

   bash scripts/launch_baseline.sh

5. 跑：

   bash scripts/run_baseline_smoke.sh

6. smoke 通过后，跑 full public accuracy 和 S1/S8/Smax。
7. 更新：
   - artifacts/results/baseline_summary.md
   - artifacts/results/quant_matrix.csv
   - submit/README_SOAR.md

如果 baseline 仍失败，请只修 baseline，并基于 logs 给出最小修复；不要进入 speculative。
```

