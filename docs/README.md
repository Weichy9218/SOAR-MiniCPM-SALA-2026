# SOAR Knowledge Base

本目录负责沉淀 SOAR MiniCPM-SALA 项目的长期知识、运行经验和决策依据，避免关键判断只留在聊天记录或临时日志里。

## Reading Order

1. `inference_optimization_primer.md`
   - 主学习笔记：从代码和理论理解 SOAR baseline inference。
   - 覆盖 prefill/decode、KV cache、client concurrency、S1/S8/Smax、SOAR metrics、quantization、EAGLE、多卡。
   - 如果只读一篇，读这篇。

2. `system_map.md`
   - 项目地图：目录、脚本、模型、评测器、artifact 的职责边界。
   - 用来快速定位文件，不重复解释推理理论。

3. `operational_runbook.md`
   - 操作手册：检查 server、runner、GPU、resume、summary、Git hygiene。
   - 用来处理“现在跑到哪里了、慢还是卡住了、怎么恢复”。

4. `optimization_plan.md`
   - 给出 correctness-first 的优化路线。
   - 明确什么时候可以进入 quantization、EAGLE3、LK loss。
   - 区分安全工程优化和会改变 benchmark 语义的优化。

5. `decision_log.md`
   - 记录已经做过的关键工程/实验决策。
   - 每条决策都写明 evidence、risk 和 next action。
   - 用于避免重复踩 CUDA graph、FP8 KV、proxy speed、EAGLE draft head 这些坑。

## Current Durable Rules

- 本地实验环境不是官方 speed 资源或官方容器；除完整 public accuracy 外，所有 speed/smoke/proxy 数字都必须标注证据等级，不能直接写成官方比赛成绩。
- FP16 baseline correctness 已完成：`completed_samples=150/150`，`ori_accuracy=82.53`，`overall_accuracy=100`。
- full public accuracy 已有 final result；以后 partial status 只能用于运行监控，不能覆盖 final baseline。
- 当前稳定 baseline 使用 `--disable-cuda-graph`；CUDA graph enabled path 已知会在 Smax smoke 崩溃，需要单独修复和重跑完整对照。
- 不中途修改 `max_out_len`、dataset、tokenizer、base model 或 scoring rule，否则 baseline 不可比。
- official SOAR speed split files 当前不在本地；`local_proxy_speed/` 只能作为 self-test，不能报告成 official S1/S8/Smax。
- EAGLE3、LK loss、quantization 都必须以完整 FP16 baseline 为参照。
- 当前 EAGLE3 只有 synthetic 1-step draft、dense fallback smoke、native sparse smoke/proxy；native sparse target verify 已到 smoke 级别，但 synthetic draft 输出质量差且 S8/Smax proxy 慢于 FP16，不能报告 speculative speed。
