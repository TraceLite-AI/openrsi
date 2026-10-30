# 和 RSIHub 拉齐的对照实验（做法 A：用 RSIHub 的评测器）

在 RSIHub 里跑同一个方法的两个版本，只换算子、其余完全相同，看 OpenRSI 的组件有没有带来差别。
同时用 RSIHub 公布的数字检查我们的基线有没有对上。

## 设置

和 RSIHub README 结果表里 "Terminal-Bench 2 / MiniSWE / Hyperagents" 那一行对齐：

| 项 | 取值 |
|---|---|
| 基准 | Terminal-Bench 2.0 全部 89 题，划分 train 50 / gate 19 / sealed 20（seed 0） |
| 目标 agent | mini-swe-agent（commit 388da74），GPT-5.4，reasoning high，每题最多 100 步、3 美元 |
| 变异 | Codex，GPT-5.4，reasoning xhigh，可改 `target/` 和 `operators/` |
| 方法 | Hyperagents：按分数和子代数选父代，每代 1 个子代，共 10 代 |
| 评分 | 每题 1 次，pass@1；训练分 = train 50 题；全量分 = 89 题 |

两组的唯一区别：

| 算子 | baseline（RSIHub 原样） | openrsi |
|---|---|---|
| validate | `hyperagents`：Python 编译检查 | `openrsi_guard`：同样的编译检查，加上护栏（不许改 `evaluator/`、不许写入密钥） |
| gate | `parent_eligible`：评估完整就入档 | `openrsi_no_regression`：评估完整，且总分不降、逐题变差不超过 2 道 |
| select | `score_child_prop` | 不变 |

可选第三组 `shinka`：在 openrsi 基础上把 select 换成 `openrsi_shinka`。

RSIHub 公布的这一行：训练分 58.0% → 68.0%，全量分 56.2% → 69.7%。我们的 baseline 应该接近这个数。
不会完全一样：他们论文实验用的具体划分没有公开，而且单次运行本身有噪声。同一个原始 agent，在他们表里不同方法那几行的训练分是 58% 到 70% 不等，说明单次运行的差别可能有 ±6 个百分点。

## 花费估计

每组约 760 次做题（第 0 代 50 题，加上 10 代 × 每代 50 题训练和 19 题 gate，再加 sealed 20 题），外加 10 次 Codex 变异。
全量评估另需 3 × 89 = 267 次（原始 agent、两组各自的最佳版本）。
RSIHub 的 hill_climb 配方给约 140 次做题的预算是 150 美元，按每次约 1 美元算，两组加全量评估大约 **1800 美元**，主要看你的 API 价格。
建议先跑第 5 步的冒烟测试（1 代），看实际单价再决定。

## 步骤

需要：Linux、Docker、uv、Git 2.25+、Python 3.12、OpenAI API key。

```bash
# 0. RSIHub（我们按 commit d6554299 读的源码；用更新的版本也行，出问题时把 commit 告诉我）
git clone https://github.com/simple-agent-lab/RSIHub && cd RSIHub
echo "OPENAI_API_KEY=..." > .env
export RSIHUB=$PWD

# 1. 下载数据集、构建 Codex 变异镜像。这一步会执行 uv sync，所以要在装 OpenRSI 之前做
./scripts/setup_terminal_bench.sh hyperagents

# 2. OpenRSI 环境（必须是 Python 3.12，算子会在 RSIHub 的进程里加载这里的包）
cd /path/to/openrsi
uv venv -p 3.12 .venv-parity && . .venv-parity/bin/activate
uv pip install -e sdk/python -e components/gate -e components/guard -e components/lineage -e adapters/rsihub pyyaml

# 3. 装算子并检查（必须全部 PASS）
bash field_tests/rsihub/r1_operators.sh $RSIHUB

# 4. 生成配方（会打印确认 50 / 19 / 20 划分）
python benchmarks/rsihub_parity/make_recipes.py $RSIHUB

# 5. 冒烟测试：每组 1 代，确认能跑通、看单价
WORKSPACE=$RSIHUB/runs/smoke-baseline bash benchmarks/rsihub_parity/run_arm.sh $RSIHUB baseline 1
WORKSPACE=$RSIHUB/runs/smoke-openrsi  bash benchmarks/rsihub_parity/run_arm.sh $RSIHUB openrsi 1

# 6. 正式实验（两组可以同时跑，前提是机器扛得住 2 × 10 个并发容器）
bash benchmarks/rsihub_parity/run_arm.sh $RSIHUB baseline
bash benchmarks/rsihub_parity/run_arm.sh $RSIHUB openrsi

# 7. 全量 89 题评估：原始 agent 一次，两组最佳版本各一次
python benchmarks/rsihub_parity/full_eval.py $RSIHUB $RSIHUB/runs/openrsi-parity-baseline 0
python benchmarks/rsihub_parity/full_eval.py $RSIHUB $RSIHUB/runs/openrsi-parity-baseline best
python benchmarks/rsihub_parity/full_eval.py $RSIHUB $RSIHUB/runs/openrsi-parity-openrsi best

# 8. 汇总
python benchmarks/rsihub_parity/compare.py $RSIHUB
```

## 发回什么

`benchmarks/rsihub_parity/out/` 整个目录：`compare.md`、`compare.json`、每组的 `*.log` 和 `*.report.txt`。
另外附上 RSIHub 的 commit 和各 workspace 下的 `archive.jsonl`。

哪一步失败就把那一步的终端输出发来。最可能出问题的两处：

- 第 7 步：`evolve init` 不一定接受本地仓库（`file://`）作为 target，`run --max-generations 0` 也不一定被接受（脚本失败时会打印替代命令）。
- gate 拿不到逐题分数：这时 `openrsi_no_regression` 会退回只比总分，原因里写着"只比总分"，compare 的逐代列表里能看到。

## 之后

拿到这组结果后做两件事：

1. 如果差别明显、预算允许，扩到其他方法（AHE、GEPA、A-Evolve）、Codex 目标 agent 和 τ³ Banking（Harbor 里的 `sierra-research/tau3-bench` banking 部分，97 题，划分 50 / 20 / 27）。
2. 做法 B：OpenRSI 自己的运行时通过 Harbor 跑同样的基准和划分。
