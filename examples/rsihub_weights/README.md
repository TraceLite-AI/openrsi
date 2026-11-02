# RSIHub 题库：同一个循环里改代码 + 训权重

RSIHub 本身只会改 agent 的代码，它的流程里没有"训练权重"这一步。这个例子反过来：由 AnyRSI 当总指挥，把
RSIHub 的题库（Harbor 格式）搬进来，同一个循环里两样都改：

- **代码**：`anyrsi_llm:CodeRewriter` 读答错的 train 题（过程尾部、判错原因、归因）和 agent 的可改源码，
  生成 `HarnessEvolvable` 的文件写入补丁
- **权重**：`anyrsi_weights` 的提案器把 train 轨迹变成训练数据，交给 verl 或自带训练器训出新 checkpoint

每一代由 ShinkaEvolve 在两个提案器里选一个，**一代只改一样**，所以分数涨了知道是谁的功劳。候选要在 gate 题上真的变好才被接受。

```
  RSIHub 题库（Harbor 数据集）── RSIHubTaskEnv：按 RSIHub 的规则分成 train / gate / sealed
        │                                       （映射成 AnyRSI 的 train / holdout / final）
        ▼
  HarborAgent：每道题调一次 harbor run ──► 容器里跑 agent（当前代码 + vLLM 上的当前权重）
        │                                         │
        ▼                                         ▼
  HarborRewardJudge：读验证脚本写的 result.json（自己核对哈希，不信 agent 转述的分数）
        │
        ├── 答错的题 ──► code：改 agent 代码        ┐
        └── train 轨迹 ──► weights：训新 checkpoint ┴─► 候选版本 ──► gate 题准入 ──► 谱系
```

## 三个新组件（在 `adapters/rsihub/anyrsi_rsihub/harbor.py`）

| 组件 | 做什么 |
|---|---|
| `RSIHubTaskEnv` | 读 Harbor 数据集（带 `task.toml` 的子目录），按 RSIHub `splits._assign` 的规则划分：按 `sha256(种子\0题名)` 排序、按比例取整、余数给小数部分最大的划分。同样的比例和种子，分出来的题和 RSIHub 一样。`anyrsi-rsihub split <数据集>` 可以打印划分结果，和 RSIHub 的 `task-split.json` 对照 |
| `HarborAgent` | 对一道题调用一次 `harbor run`，把 `result.json` 和 `agent/trajectory.json`（ATIF 格式）转成 AnyRSI 轨迹。被进化的代码目录加到 Harbor 进程的 `PYTHONPATH` 最前面，也可以用 `{repo}` 占位符传给 agent。上下文里有权重条目时，先用新 checkpoint 启动或重启推理服务 |
| `HarborRewardJudge` | 按验证脚本的奖励打分。自己重新读 `result.json`：文件不在、哈希对不上、题名对不上、没有奖励，都判失败 |

## 两种跑法

| 配置 | Harbor | 推理 | 训练 | 用途 |
|---|---|---|---|---|
| `anyrsi.mock.yaml` | 假的 `fake_harbor.py` | 假服务 `mock_llm.py` | 假训练 `fake_train.py` | 检查流程，CI 里跑，约 25 秒 |
| `anyrsi.yaml` | 真 Harbor（在 RSIHub 环境里） | vLLM | verl SFT | 正式跑（模板，还没在真机上跑过） |

### 本地调通（不需要 GPU、Docker、API key）

```bash
cd examples/rsihub_weights
python make_toy_dataset.py toy_tasks
anyrsi run anyrsi.mock.yaml
```

玩具题库有 60 道：1/3 是乘法题，假模型永远算错，只有 agent 代码里加了乘法工具才能做对（要改代码）；
2/3 是加法题，答对率取决于权重（要训权重）。一次运行的结果大致是：

```
 gen  result
  · root holdout=0.200
   1  接受  harness：加乘法工具                       → 0.400
   2  接受  weights：KTO 训练（9 道错题也用上了）       → 0.733
   ...
   6  接受  weights                                   → 1.000
```

假模型的答案是固定的，所以权重用 KTO（答对标"好"、答错标"坏"，不需要同一道题一对一错）。

### 正式跑：Terminal-Bench 2 + mini-swe-agent + verl

需要：Linux、Docker、多卡 GPU（配置里是 8 卡训练 Qwen3-32B）、uv、OpenAI API key（给改代码的强模型用）。

```bash
# 1. RSIHub 和题库（和 benchmarks/rsihub_parity 的第 0、1 步一样）
git clone https://github.com/simple-agent-lab/RSIHub && cd RSIHub && ./scripts/setup_terminal_bench.sh hyperagents

# 2. 被进化的 agent 代码
git clone https://github.com/SWE-agent/mini-swe-agent /path/to/mini-swe-agent && git -C /path/to/mini-swe-agent checkout 388da74

# 3. AnyRSI（加 verl、vLLM）
pip install -e sdk/python -e runtime -e components/judge -e components/proposer -e components/algorithm \
            -e components/gate -e components/lineage -e components/guard -e components/harness \
            -e components/llm -e components/weights -e adapters/rsihub
pip install vllm verl

# 4. 先看划分对不对（应当是 50 / 19 / 20）
anyrsi-rsihub split /path/to/RSIHub/.evolve-assets/terminal-bench-2.0/raw/terminal-bench

# 5. 把 anyrsi.yaml 里的 /path/to/... 换掉，然后
anyrsi run examples/rsihub_weights/anyrsi.yaml
```

## 已知的不确定之处（正式跑之前要确认）

正式配置是按 RSIHub（commit d655429）和 Harbor 的源码写的，**没有在真机上跑过**，下面几处最可能要调：

1. **候选代码怎么进容器。** RSIHub 的 `CandidateMiniSweAgent` 从 `EVOLVE_CANDIDATE_SOURCE` 安装候选代码，
   RSIHub 自己的评测脚本还额外传了 `--mounts`，把代码目录挂进容器。我们只传了环境变量；如果容器里找不到代码，
   要在 `extra_args` 里加上同样的 `--mounts`。
2. **容器里访问 vLLM。** 配置里用 `host.docker.internal:8100`。Linux 上的 Docker 默认没有这个名字，
   要在 Harbor 的 Docker 设置里加 `host-gateway`，或者把 vLLM 放到容器能访问的地址。
3. **模型能不能做 Terminal-Bench。** 终端任务很难，能本地训练的开源模型在 TB2 上起点可能很低。
   训练分太低时，`DistillProposer` 攒不够答对的轨迹，权重这一路会一直没有提案。
4. **花费。** 每代 10 道 train 题加 19 道 gate 题，每道题一次 agent 运行；改代码还要调一次强模型。
   先把 `max_generations` 设成 1 试一次，看单价。

## 和 RSIHub 原样跑的区别

| | RSIHub | 这个例子 |
|---|---|---|
| 谁决定改什么 | RSIHub 的 driver + 变异算子（Codex） | AnyRSI 的提案器 + 算法（Shinka 在 code / weights 间选） |
| 能改什么 | agent 代码（和算子） | agent 代码 + 模型权重 |
| 怎么评 | 一次评一整批题 | 每道题单独调一次 Harbor（同样的题、同样的验证脚本） |
| 准入 | gate 算子 | `HoldoutGate`（总分提升，且逐题变差不超过 2 道） |
| 划分 | `splits._assign` | 同一个规则（`assign_splits`） |
