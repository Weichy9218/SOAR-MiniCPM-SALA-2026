# SOAR Inference Optimization Primer

本文件是 SOAR 项目的学习主线：用本项目真实代码解释 MiniCPM-SALA baseline 怎样做 inference，SOAR 怎样评价 correctness/speed，以及推理优化应该从哪些层下手。

## 0. How To Read This Note

先抓住一个总图：

```text
JSONL sample
  -> evaluator sends HTTP request
  -> SGLang server runs MiniCPM-SALA prefill/decode
  -> evaluator scores model text
  -> chunk runner aggregates metrics
  -> optimizer compares candidate configs
```

本项目里有三种角色，不要混淆：

| role | code | responsibility | runs model? |
|---|---|---|---|
| server | `scripts/launch_baseline.sh` | 启动 SGLang，加载 MiniCPM-SALA，接收请求 | yes |
| evaluator | `repos/SOAR-Toolkit/eval_model.py` | 读数据、发请求、收文本、打分 | no |
| runner | `scripts/run_full_accuracy_chunked.sh` | 切 chunk、resume、aggregate | no |

所以 baseline 不是训练。它没有 optimizer、backward、gradient update 或新 checkpoint。它是 serving + evaluation。

## 1. Inference Basics

一次 LLM inference 可以粗分成：

```text
prompt -> tokenize -> prefill -> decode -> detokenize -> score
```

`prefill`

- 输入 prompt 有 `N` 个 tokens，模型一次性处理这些 tokens。
- 产物是每一层的 KV cache。
- 长上下文任务的 prefill 很重，因为 attention 要看大量历史 tokens。

`decode`

- 自回归生成阶段，一步生成一个 token。
- 每一步都要读取历史 KV cache。
- 如果生成 `T` 个 tokens，就要做 `T` 次 decode step。

`KV cache`

- Transformer 每层 attention 都会产生 key/value。
- decode 时复用历史 KV，避免每步重新计算整个 prompt。
- KV cache 让 decode 成为可行的，但会吃显存和显存带宽。

一个简化成本模型：

```text
prefill cost ~= O(prompt_tokens^2) attention + O(prompt_tokens) MLP
decode cost per token ~= O(current_context_tokens) attention + O(1) MLP per generated token
KV memory ~= layers * context_tokens * hidden_kv_size * dtype_bytes
```

这不是精确公式，但足够指导优化：长 prompt 容易卡 prefill，长输出容易卡 decode，长上下文 decode 会不断读越来越大的 KV cache。

## 2. Baseline Serving Path

Server 由 `scripts/launch_baseline.sh` 启动，核心命令在 [launch_baseline.sh](/home/dataset-local/work/SOAR/scripts/launch_baseline.sh:35)：

```bash
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 8192 \
  --skip-server-warmup \
  --dense-as-sparse \
  --disable-cuda-graph
```

这些 flag 的学习含义：

| flag | meaning | optimization angle |
|---|---|---|
| `--model-path` | 加载 MiniCPM-SALA 权重 | 换模型会改变 benchmark，不是安全优化 |
| `--attention-backend minicpm_flashinfer` | 使用当前 MiniCPM-SALA 适配的 attention backend | attention backend 是高价值优化点，但兼容风险高 |
| `--chunked-prefill-size 8192` | 长 prompt prefill 分块 | 降低长 prompt 峰值压力，可能影响调度/吞吐 |
| `--disable-radix-cache` | 关闭 prefix cache | 保持当前 stable baseline 语义 |
| `--dense-as-sparse` | 当前 SOAR/MiniCPM serving path 需要的兼容配置 | 不应随便去掉 |
| `--disable-cuda-graph` | 禁用 CUDA graph | 当前稳定选择，因为 enabled path 有 Smax crash 证据 |

Server 是 GPU 工作发生的地方。Evaluator 只是 client。

## 3. Evaluator Request Path

Evaluator 的 API 调用在 [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:30)：

```python
url = f"{api_base}/v1/chat/completions"
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
}
payload.update(sampling_kwargs)
resp = requests.post(url, json=payload, timeout=timeout)
content = result["choices"][0]["message"]["content"]
```

这段代码说明：

- evaluator 不直接调用 PyTorch model。
- evaluator 调 SGLang 的 OpenAI-compatible API。
- prompt 以 chat message 形式发给 server。
- server 负责 tokenize、prefill、decode、detokenize。
- evaluator 拿到文本后再打分。

Generation 参数在 [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:131)：

```python
sampling_kwargs = {
    "temperature": 0,
    "max_tokens": max_out_len,
    "stop": list(set(self.stop_words + stopping_criteria)),
}
```

full public accuracy 的关键行在 [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:339)：

```python
outputs = model.generate(inputs, max_out_len=65536)
```

这解释了为什么部分样本非常慢：每条 request 最多允许生成 65536 tokens。如果模型没有提前遇到 stop token，单条 decode 可以持续很久。

Client-side concurrency 在 [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:173)：

```python
with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
    futures = {executor.submit(_infer, i, raw_inputs[i]): i for i in range(len(raw_inputs))}
```

这不是 data parallel training，只是 evaluator 同时向 server 发多个 HTTP requests。Server 会把这些请求调度成 batch。

## 4. Chunk Runner Path

`scripts/run_full_accuracy_chunked.sh` 是工程层保护，不是模型算法。它做四件事。

第一，检查 server 是否活着，在 [run_full_accuracy_chunked.sh](/home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh:44)：

```python
resp = requests.get(f"{api_base}/v1/models", timeout=15)
```

第二，把 public set 切成 chunks，在 [run_full_accuracy_chunked.sh](/home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh:64)：

```python
for chunk_id, start in enumerate(range(0, len(rows), chunk_size)):
    end = min(start + chunk_size, len(rows))
    chunk_path = data_dir / f"chunk_{chunk_id:04d}_{start:04d}_{end - 1:04d}.jsonl"
```

第三，跳过已完成 chunk，在 [run_full_accuracy_chunked.sh](/home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh:217)：

```bash
if [ -f "${chunk_dir}/summary.json" ] && [ -f "${chunk_dir}/predictions.jsonl" ]; then
  echo "skip_completed=${chunk_label}"
  aggregate_chunks
  continue
fi
```

第四，聚合 correctness，在 [run_full_accuracy_chunked.sh](/home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh:167)：

```python
ori_accuracy = round((score_sum / completed_samples) * 100, 2)
overall_accuracy = min(round(ori_accuracy / 80 * 100, 2), 100)
```

为什么需要 chunking：

- official evaluator 只在一个 chunk 全部 request 完成后写 `predictions.jsonl`。
- 一个 chunk 卡在 `9/10` 时，最后一条可能还在长 decode。
- chunk 完成后 aggregate 才从例如 `110/150` 跳到 `120/150`。
- chunking 让 crash/resume 的损失从 150 条缩小到一个 chunk。

## 5. SOAR Accuracy Metrics

SOAR public accuracy 数据：

```text
repos/SOAR-Toolkit/eval_dataset/perf_public_set.jsonl
```

每条样本至少包含：

```json
{"task": "mcq", "question": "...", "gold": "B"}
```

或者：

```json
{"task": "cwe", "question": "...", "gold": ["word1", "word2"]}
```

Scoring 在 [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:369)：

```python
if task in mcq_tasks:
    score, extracted = score_mcq(pred, gold)
elif task in long_context_tasks:
    score = score_exact_match(pred, gold, task)
```

任务含义：

| task | scoring idea | optimization implication |
|---|---|---|
| `mcq` | 抽取 A/B/C/D，完全匹配得分 | 输出格式很重要，长 reasoning 可能干扰抽取 |
| `niah` | needle-in-a-haystack，命中答案得分 | 主要测长上下文检索 |
| `qa` | 长文档问答，命中候选答案得分 | prompt 长，答案可能短 |
| `cwe` | common word extraction，按关键词覆盖率 | 可能要求输出多个词 |
| `fwe` | frequent word extraction，按关键词覆盖率 | 长列表统计，可能引发长输出 |

最终指标在 [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:396)：

```python
avg_score = (correct_count / len(dataset)) * 100
overall_accuracy = min(round(avg_score / 80 * 100, 2), 100)
```

解释：

- `ori_accuracy` 是原始平均分。
- `overall_accuracy` 是相对 80 分 baseline 的归一化分，上限 100。
- `overall_accuracy=100` 不代表每个 task 都完美。

本项目 FP16 baseline 结果：

```text
completed_samples=150/150
ori_accuracy=82.53
overall_accuracy=100
duration=17260.25s
total_tokens=733451
```

Task-level 结果：

| task | samples | ori accuracy | input tokens | output tokens |
|---|---:|---:|---:|---:|
| `mcq` | 30 | 60.0 | 8128 | 284894 |
| `niah` | 30 | 100.0 | 2219512 | 10552 |
| `qa` | 30 | 60.0 | 2147102 | 3056 |
| `fwe` | 30 | 100.0 | 2044642 | 155922 |
| `cwe` | 30 | 92.67 | 2224932 | 279027 |

学习重点：

- `niah/qa/fwe/cwe` 的 input tokens 很大，说明长上下文 prefill 压力是真实存在的。
- `mcq/cwe/fwe` 的 output tokens 可以很大，说明长 decode 压力也是真实存在的。
- 优化必须同时守住 correctness 和 speed，不能只看 tokens/s。

## 6. SOAR Speed Metrics: S1, S8, Smax

S1/S8/Smax 是 speed benchmark 档位，不是 task 类型。

SOAR Toolkit 在 [bench_serving.sh](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/bench_serving.sh:11) 写明：

```bash
SPEED_DATA_S1   - S1(并发度1) 数据集路径
SPEED_DATA_S8   - S8(并发度8) 数据集路径
SPEED_DATA_SMAX - Smax(不设并发上限) 数据集路径
```

bench command 在 [bench_serving.sh](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/bench_serving.sh:143)：

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --host ${HOST} \
  --port ${PORT} \
  --dataset-name custom \
  --dataset-path ${CONVERTED_DATA} \
  --num-prompts ${NUM_PROMPTS} \
  --flush-cache
```

S1/S8 额外加并发上限，在 [bench_serving.sh](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/bench_serving.sh:152)：

```bash
--max-concurrency 1
--max-concurrency 8
```

Smax 不加 `--max-concurrency`，即尽可能把所有 requests 压上去。

直觉：

| split | concurrency | what it stresses |
|---|---:|---|
| S1 | 1 | 单请求 latency、prefill/decode kernel efficiency |
| S8 | 8 | continuous batching、scheduler、KV memory pressure |
| Smax | unlimited | server saturation、queueing、OOM/crash stability |

SOAR speed 输出是 `Benchmark duration (s)`，越低越好。官方 speed split 当前不在本地，所以本项目的 `local_proxy_speed/` 只能学习和自测，不能作为 official result。

## 7. Why This Run Was Slow

观察到的慢不是 silent hang，而是长请求 decode：

```text
Generating: 90%|...| 9/10
Decode batch, #running-req: 1, #full token: increasing, gen throughput: ~25-28 token/s
```

含义：

- evaluator 已完成这个 chunk 的 9 条 request。
- 最后一条 request 还在 server 端 decode。
- `#full token` 持续增加表示 server 活着。
- 进度条只在 request 返回后更新，不会按 token 更新。
- `predictions.jsonl` 只在整个 chunk 完成后写。

完成的 FP16 public accuracy run：

```text
duration=17260.25 seconds ~= 4.79 hours
completed_samples=150/150
total_tokens=733451
```

几个长 chunk：

| chunk | duration seconds | approximate minutes | reason |
|---:|---:|---:|---|
| 10 | 2523.94 | 42.07 | long final request |
| 11 | 2633.71 | 43.90 | long final request |
| 12 | 2455.56 | 40.93 | long final request |
| 13 | 2583.15 | 43.05 | highest token count |
| 14 | 2580.47 | 43.01 | long final request |

判断 slow vs stuck：

| state | healthy slow | likely problem |
|---|---|---|
| server log | decode lines continue | no new lines for a long time |
| token counter | `#full token` increases | token counter frozen |
| throughput | non-zero | zero or absent |
| process | server/evaluator alive | process disappeared |
| error | no traceback | CUDA error/OOM/API error |

ETA 粗估：

```text
remaining_minutes ~= remaining_decode_tokens / tokens_per_second / 60
```

但只能粗估，因为模型可能提前 stop，也可能接近 `max_tokens=65536` 才停。

## 8. Optimization Map

推理优化不要从“我想打开某个开关”开始，而要从 bottleneck 开始：

```text
Is it prefill-bound?
Is it decode-bound?
Is it memory-bound?
Is it scheduler/batching-bound?
Is it correctness-constrained?
```

### 8.1 Workflow Optimization

目标：更可观测、更可恢复，不改变模型输出。

例子：

- chunking
- per-sample checkpointing
- status script
- readiness check
- structured metadata

本项目已有 chunking，但 checkpoint 粒度还是 chunk-level。下一步高价值工程改进是 per-sample checkpointing，但要确保不改变 evaluator scoring。

### 8.2 Concurrency And Batching

目标：让 GPU 不被单请求低利用率拖住。

相关概念：

- `client concurrency`：evaluator 同时发多少 requests。
- `server batching`：SGLang 把多少 active requests 合成 batch。
- `continuous batching`：request 动态进入/退出 batch，减少空等。
- `tail latency`：最慢 request 的完成时间。

风险：

- concurrency 太低，GPU 吃不满。
- concurrency 太高，KV cache 爆显存或 queueing 变差。
- long requests 会拖住 chunk completion。

本项目 accuracy runner 默认 `ACCURACY_CONCURRENCY=4`，speed benchmark 的 S1/S8/Smax 则分别测试不同并发压力。

### 8.3 Attention And KV Cache

目标：降低长上下文成本。

可调方向：

- attention backend
- chunked prefill size
- prefix/radix cache
- KV cache dtype
- max context and memory scheduling

理论直觉：

- prompt 越长，prefill 越贵。
- decode 越长，读 KV cache 越多。
- KV dtype 越小，显存和带宽压力越低，但 kernel/backend 必须支持。

当前项目事实：

- stable path 使用 `minicpm_flashinfer`。
- `--chunked-prefill-size 8192` 已启用。
- FP8 KV 在当前 backend path 已有 smoke failure，不应作为 mainline。

### 8.4 CUDA Graph

目标：减少 CPU launch overhead，把重复形状的 GPU kernel replay 化。

适合：

- shape 稳定。
- repeated decode。
- launch overhead 占比高。

不适合乱开，因为：

- serving shape 动态变化。
- long-context metadata 更复杂。
- 当前 enabled path 在 Smax smoke 崩过。

本项目决策：

- stable baseline 保持 `--disable-cuda-graph`。
- CUDA graph 是 separate config。
- 必须先 reproduce crash，再 small smoke，再 small accuracy chunk，再 full comparison。

### 8.5 Quantization

目标：降低权重显存、显存带宽，可能降低 latency。

常见类型：

| type | changes | likely benefit | risk |
|---|---|---|---|
| weight-only INT4/INT8 | only weights | lower model memory/bandwidth | accuracy drop, kernel overhead |
| GPTQ/AWQ | calibrated weight quant | better accuracy than naive RTN | preprocessing cost |
| GPTQ-Marlin | quant format + optimized kernel | speed on supported GPUs/shapes | kernel stack compatibility |
| KV cache quant | KV cache dtype | lower KV memory/bandwidth | backend support, accuracy/stability |

为什么 quantization 不一定更快：

- 如果 bottleneck 是 attention over KV cache，weight-only quant 只优化一部分。
- 如果 kernel 不够优化，dequant overhead 会吃掉收益。
- 如果 batch/shape 不匹配 optimized kernel，理论收益落不到实际。

本项目事实：

- RTN-GPTQ gs1024 能跑 local proxy，但比 FP16 慢。
- RTN-GPTQ sym-gs128 能跑，proxy 结果 mixed：S1 略慢，S8/Smax 略快。
- `gptq_marlin` 当前卡在 `sgl_kernel.gptq_marlin_repack` 缺失。
- 这些 quant probes 都还没有 full public accuracy，不是 final candidate。

### 8.6 Speculative Decoding / EAGLE

目标：用小 draft model 猜 tokens，用大 target model 验证，减少 target decode steps。

简化流程：

```text
draft proposes k tokens
target verifies them in one pass
accepted tokens are emitted
rejected token falls back to target
```

收益条件：

```text
speedup exists if draft_cost + verify_cost < target_cost_for_same_tokens
and acceptance_rate is high enough
```

风险：

- draft model/head 必须和 target hidden states/vocab/tokenizer 对齐。
- acceptance 阈值不能为了速度乱放松。
- draft 太慢或 acceptance 太低会变慢。

本项目事实：

- `scripts/launch_eagle3.sh` 是 wrapper。
- 本地没有 MiniCPM-SALA-compatible draft head。
- SpecForge 当前 auto draft mapping 偏 Llama-style。
- 先要解决 draft head/adapter，再谈 EAGLE benchmark。

### 8.7 Multi-GPU

两种多卡不要混：

| mode | helps | does not help |
|---|---|---|
| data parallel | 多 chunks 总 walltime | 单条 long request latency |
| tensor parallel | 单请求模型计算 | 通信开销和 backend 风险 |

当前机器是单 A100-SXM4-80GB。学习顺序上，先把 single-GPU bottleneck、correctness、speed split 讲清楚，再考虑多卡更稳。

## 9. Experiment Design

推理优化实验要固定不变量：

- base model
- tokenizer
- eval dataset
- scoring rule
- `max_out_len`
- chat template
- server semantic flags

每个 candidate 都要记录：

```text
Experiment:
Candidate:
Baseline config:
Changed variables:
Unchanged variables:
Dataset/split:
Command:
Artifacts:
Status:
Metrics:
Failure evidence:
Decision:
Next action:
```

状态标签：

| label | meaning |
|---|---|
| `final` | 完整 benchmark/evaluation 完成，输入/命令/结果可追溯 |
| `partial` | 只用于监控，不可当 final |
| `proxy` | 本地自测，不是 official benchmark |
| `smoke` | 能启动/跑小样本，不证明性能或正确性 |
| `blocked` | 缺 artifact、dependency、credential 或 official input |
| `rejected` | 已有失败证据，不应在相同假设下反复尝试 |

Candidate promotion gate：

- correctness 不显著低于 FP16 baseline。
- speed 在 official S1/S8/Smax 或明确标注的 proxy split 上改善。
- artifact 和 config 单独保存。
- 没有隐藏改变 benchmark semantics。
- failure mode 和 rollback path 清楚。

## 10. Practical Learning Path

建议按这个顺序读代码。

1. [launch_baseline.sh](/home/dataset-local/work/SOAR/scripts/launch_baseline.sh:35)

看 server 怎样加载模型、使用 attention backend、控制 CUDA graph。

2. [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:30)

看 evaluator 怎样调用 `/v1/chat/completions`。

3. [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:117)

看 `generate()` 如何设置 sampling、并发发送 requests。

4. [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:339)

理解 `max_out_len=65536` 为什么会制造长 decode。

5. [eval_model.py](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/eval_model.py:369)

看不同 task 如何 scoring。

6. [run_full_accuracy_chunked.sh](/home/dataset-local/work/SOAR/scripts/run_full_accuracy_chunked.sh:64)

看 chunk split/resume/aggregate 机制。

7. [bench_serving.sh](/home/dataset-local/work/SOAR/repos/SOAR-Toolkit/bench_serving.sh:91)

看 S1/S8/Smax 如何映射到 `--max-concurrency`。

8. [prepare_local_speed_proxy.py](/home/dataset-local/work/SOAR/scripts/prepare_local_speed_proxy.py:1)

看 local proxy 为什么只是学习和自测工具。

## 11. Current Project State

Correctness:

```text
FP16 public accuracy complete
completed_samples=150/150
ori_accuracy=82.53
overall_accuracy=100
```

Speed:

```text
official S1/S8/Smax split: not available locally
local proxy: available, official=false
```

Stable serving:

```text
--disable-cuda-graph
--attention-backend minicpm_flashinfer
--chunked-prefill-size 8192
--dense-as-sparse
```

Optimization status:

| candidate | status | reason |
|---|---|---|
| CUDA graph | blocked/risky | Smax smoke crash |
| FP8 KV | rejected for current path | backend dtype support failure |
| RTN-GPTQ gs1024 | probe only | slower proxy, no full accuracy |
| RTN-GPTQ sym-gs128 | mixed probe | tiny proxy gains only on S8/Smax, no full accuracy |
| GPTQ-Marlin | blocked | missing `sgl_kernel.gptq_marlin_repack` |
| EAGLE3 | smoke/proxy only | synthetic draft, dense fallback smoke, and native sparse smoke/proxy pass; synthetic draft regresses S8/Smax proxy and has no accuracy |
| multi-GPU | future separate config | current machine single-GPU, topology change |

## 12. Glossary

`prefill`

- 对输入 prompt 做前向，生成 KV cache。

`decode`

- 自回归生成阶段，一步一个 token。

`KV cache`

- 保存历史 attention key/value，减少重复计算，但占显存。

`client concurrency`

- evaluator 同时发出的 HTTP requests 数。

`continuous batching`

- server 动态合并多个不同进度的 requests，提高 GPU 利用率。

`TTFT`

- Time To First Token，首 token 延迟。

`ITL`

- Inter-Token Latency，相邻输出 tokens 的间隔。

`Benchmark Duration`

- speed harness 跑完整个 split 的总耗时。

`ori_accuracy`

- evaluator 直接算出的平均分。

`overall_accuracy`

- `ori_accuracy` 相对 80 分 baseline 的归一化分，上限 100。

`proxy speed`

- 本地构造的速度自测，不是 official benchmark。
