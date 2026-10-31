# 原生循环 A/B/C 对照实验

对应 `spec/drafts/two-loop-evolution.md` §8B（P3）。用同一套 OpenRSI 运行时跑三组：

| 组 | 做法 | 回答什么 |
| --- | --- | --- |
| A | 冻结初始 agent，不做改进 | 能力和成本的基线 |
| B | 等搜索预算，每一代都从根版本出发（`IndependentAlgorithm`） | 排除"只是多试了几次" |
| C | 等搜索预算，下一代可以继承接受过的改进（`GreedyAlgorithm`） | 改进能不能累积 |

脚本保证的规则：

- 搜索预算按 agent 调用次数算，覆盖练习题、开发期考试打分、反事实重跑和失败重试；到上限立刻停，正在评估的候选记为未完成，不删记录。
- 每组锁定最终版本之后，才各做一次封存考试（`final` 划分）；封存考试的调用另计，各组相同。搜索期间有任何组件请求 `final` 题，实验直接判为无效。
- 报告逐题结果、通过率、两两配对比较（修好、改坏、平均差值）、接受和拒绝的原因、调用次数和耗时。

```bash
python benchmarks/native_loop/native_loop.py --budget 400 --seeds 0 1 2 --out out/native_loop
```

默认用 `examples/secret_ops` 做合成冒烟实验，报告会标明"synthetic smoke，非 benchmark"：它只说明机制在工作，不能当成能力结论。换成真实任务时，替换 `write_config` 里的 env、agent、proposer。

## 一次合成冒烟结果

`--budget 400 --seeds 0 1 2`，封存考试 40 题：

| 组 | 封存考试通过率（种子 0 / 1 / 2） |
| --- | --- |
| A | 0.000 / 0.125 / 0.125 |
| B | 0.375 / 0.500 / 0.500 |
| C | 1.000 / 1.000 / 1.000 |

三组都没有"改坏"的题；C 相对 B 每个种子修好 20–25 道。这是合成任务上的机制检查，不是 benchmark。
