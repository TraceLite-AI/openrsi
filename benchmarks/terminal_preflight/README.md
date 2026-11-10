# 在 Terminal-Bench 上跑进化之前：先确认能测准

上一次五代实验（`benchmarks/terminal_multigen_20261103`）1061 次评测里有 643 次判分器没跑到测试（装不上测试要的
Python 3.13，安装目录是只读的），却被记成普通的 0 分。19 道考试题里 13 道判分器经常没起来、4 道起点 agent 每次都过、
2 道时过时不过——没有一道题能证明一个改法变好了，所以任何方法都只能 0 次接受。

下面五步都不进化（第 4 步只临时试一个手写修改），按顺序做，每一步不过关就停下来先解决。

## 1. 修判分环境

判分器不要继承 agent 的 uv 设置（`UV_PYTHON=3.12`、只读的 `UV_PYTHON_INSTALL_DIR=/opt/evolve/uv/python`）：
去掉它们，或者把 `UV_PYTHON_INSTALL_DIR` 指向容器里可写的目录，让题目的测试脚本自己装要的 Python；
网络受限时预先把要用的版本装进共享目录。不要把整个共享目录改成可写。

## 2. 跑前体检（`preflight.py`）

```bash
python benchmarks/terminal_preflight/preflight.py examples/rsihub_weights/anyrsi.step1.yaml --per-split 5
```

每道抽到的题跑两次：标准答案（Harbor 的 oracle agent）应该通过，什么都不做（nop）应该不通过，而且两次测试都要真的跑起来。
有一项不对就退出码 1，列出是哪道题、哪一项。全部 OK 再往下走。你的 Harbor 版本 oracle / nop 参数不同时用
`--oracle-args`、`--nop-args` 改。

## 3. 测噪声、挑考试题（`benchmarks/noise/noise.py` + `headroom.py`）

```bash
python benchmarks/noise/noise.py examples/rsihub_weights/anyrsi.step1.yaml --repeats 3 --split all --out noise_all.json
python benchmarks/terminal_preflight/headroom.py noise_all.json --size 19
```

起点 agent 把练习题和考试题各做 3 遍（封存题不碰，免得泄漏）。`headroom.py` 把每道题分成：判分有问题、每次都过、
时过时不过、每次都失败；从后两类里挑考试题，输出一行 `gate: [...]`，放进配置的 `RSIHubTaskEnv` 里：

```yaml
env:
  python: "anyrsi_rsihub:RSIHubTaskEnv"
  with: {dataset: ..., seed: 0, gate: ["task-a", "task-b", ...]}
```

封存的 20 道题仍按 seed 划分、不变；指定的考试题不能和封存题重叠；其余的题都归练习题。
有提升空间的题少于 8 道时，说明这个做题模型在这些题上没多少可提升的（上一次能正常判分的 44 道题里 terra 已经过了 33 道）：
换一个弱一点的做题模型（比如 gpt-5.4-mini），提案器可以继续用强模型。

## 4. 手写补丁测试（`handpatch.py`）

```bash
cd /path/to/mini-swe-agent && git diff > my_change.diff      # 你认为最好的一个修改
python benchmarks/terminal_preflight/handpatch.py examples/rsihub_weights/anyrsi.step1.yaml --diff my_change.diff --repeats 3
```

起点版本和打了补丁的版本交替各跑 3 遍考试题，报告配对差值、95% 区间、变好 / 变差的题、起点自己的噪声，
以及"这批题大约多大的提升才测得出来"。结论三选一：

- **测出提升**：有空间，瓶颈在提案器，值得跑进化。
- **测不出差别**：连人手写的最好修改都看不出来，自动循环也不会成功。看"能测出的最小提升"：太大就加题、加遍数；
  否则换一个弱一点的做题模型，或者选失败原因相近的一类题。
- **测出变差**：这个修改本身不对，换一个再测。

补丁只在测试期间生效，不写谱系。改提示、记忆这类可进化体时用 `--patch`（JSON：`{"target": ..., "ops": [...]}`）。

## 5. 再跑进化

框架现在会这样处理判分问题：

- `HarborRewardJudge` 看 `verifier/` 里的输出：测试没开始（没有 `ctrf.json`、输出里也没有测试开始的迹象）就把这次标成
  **无法评分**（必需的 rubric 项 `verifier_started` 结果是 `error`），不当成 0 分。
- 无法评分的题不算分、不参与准入比较（父版本和候选只在两边都能评分的题上比）、不交给提案器。
- 一批题里无法评分的超过 `loop.max_unscorable`（默认 0.2）就停下报错，不会再空跑出一整轮 0 分。
- 每个测试用例（`ctrf.json`）各是一条 rubric，没过的带上报错开头，提案器能看到具体是哪个测试、为什么没过。
  `score_by: tests` 时分数改成通过的测试比例（细一些的信号，通过与否仍看奖励）。
