# 现场测试

单元测试（`tests/`）只能用替身验证。这里的测试要在真实环境里跑：你的机器、真实的 verl、真实的 RSIHub。
每个脚本都会打印 `PASS` 或 `FAIL`，结果写在 `field_tests/out/` 下。全部跑完后执行 `bash field_tests/collect.sh`，把生成的 `openrsi-field-report.tgz` 发回来；只跑了一部分也可以发。

建议按下表顺序跑，前面的不过就先停下来发结果。

| 编号 | 测什么 | 需要 | 耗时 | 花钱 |
|---|---|---|---|---|
| L0 | 本机基线：安装、单元测试、四个示例、每个组件作为 MCP 子进程的一致性测试 | Python 3.10+、git | 约 3 分钟 | 否 |
| V1 | 我们的奖励和 verl 自带 GSM8K 奖励逐条对比；verl 能加载我们的奖励函数 | verl（CPU 即可） | 1 分钟 | 否 |
| V2 | 导出的 RL 数据和 verl 格式一致，verl 的 RLHFDataset 能读；防泄漏 | verl、transformers | 1 分钟 | 否 |
| V3 | 小模型 GRPO，奖励用我们的判别器；可选对照：verl 自带奖励 | 1 张 GPU、vllm | 20～40 分钟 | 否 |
| V4 | 权重进化：连续两次 verl SFT，第二次从第一次的输出继续；回滚 | 1 张 GPU | 20～40 分钟 | 否 |
| R1 | OpenRSI 算子装进 RSIHub：能发现、描述、校验配置，能 import 到 OpenRSI | RSIHub 仓库、uv | 1 分钟 | 否 |
| R2 | 拿已有的 RSIHub workspace 离线重放：逐题分数、准入对比、护栏、篡改评估器能否被拦 | 一个跑过的 workspace | 1～5 分钟 | 否 |
| R3 | 真实 RSIHub 配方换上我们的三个算子跑 1～2 代 | Docker、Terminal-Bench 数据、OpenAI 等 API | 视配方 | 是 |

## L0 本机基线

```bash
git clone https://github.com/zsicheng0214/openrsi && cd openrsi
bash field_tests/run_local.sh            # 可以指定解释器：bash field_tests/run_local.sh python3.12
```

在 `field_tests/out/local/venv` 里新建环境，不碰你现有的环境。通过标准：最后一行 `合计：N 通过，0 失败`。

后面的测试需要一个装了 OpenRSI 的环境。可以直接用 L0 的：`. field_tests/out/local/venv/bin/activate`，再装 verl；或者把下面这些装进你已有的 verl 环境：

```bash
pip install -e sdk/python -e components/judge -e components/gate -e components/guard -e components/lineage \
            -e adapters/verl -e adapters/rsihub pandas pyarrow
```

## verl

先用 verl 仓库自带的脚本准备 GSM8K：

```bash
python <verl 仓库>/examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

```bash
python field_tests/verl/v1_reward_parity.py --data ~/data/gsm8k/test.parquet
python field_tests/verl/v2_data_export.py --src ~/data/gsm8k --tokenizer Qwen/Qwen2.5-0.5B-Instruct
MODEL=Qwen/Qwen2.5-0.5B-Instruct N_GPUS=1 STEPS=30 CONTROL=1 bash field_tests/verl/v3_grpo_smoke.sh
python field_tests/verl/v4_weights_sft.py --src ~/data/gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --gpus 1
```

- **V1** 通过：verl 的加载函数能加载 `field_tests/verl/gsm8k_reward.py`，每类样例（正确、错误、没有答案标记、两个标记、带逗号）和 verl 的打分完全一致。
- **V2** 通过：导出数据的列和 verl 的一致，题目和答案逐条相同，答案不在 prompt 里；把 holdout 导成训练集、把 train 导成验证集都会报错；RLHFDataset 能读出第一条。
- **V3** 通过：两次运行都跑完；训练前的验证分数两者一致；训练后上升。参数名按较新的 `verl.trainer.main_ppo` 写，你的版本报"未知参数"就把报错发回来；显存不够可以调小 `BATCH`、`RESP_LEN`，或者设 `ROLLOUT=hf`。
- **V4** 通过：两次训练都成功，第二次的起点是第一次的输出，输出目录里有 `config.json` 和权重文件，回滚后回到原始模型。不同 verl 版本的 SFT 入口和输出目录结构不一样：报错时看 `field_tests/out/verl/v4.json` 里的 `layout`，用 `--model-glob` 指定模型所在子路径，或用 `--trainer-json` 传完整的训练命令。

## RSIHub

RSIHub 要求 Python 3.12+。OpenRSI 最好也用 3.12 的环境（算子脚本在 RSIHub 的进程里运行，会用到 OpenRSI 环境里编译过的包）。

```bash
bash field_tests/rsihub/r1_operators.sh /path/to/RSIHub
python field_tests/rsihub/r2_replay.py /path/to/RSIHub/runs/<某个跑过的 workspace>
python field_tests/rsihub/r3_recipe.py /path/to/RSIHub --base hill_climb     # 然后按它打印的命令跑
```

- **R1** 通过：三个算子都能被 `evolve operator list` 发现、`describe` 正常；合法配置通过、非法和未知字段被拒；RSIHub 的 Python 能 import 到 OpenRSI。
- **R2** 通过：能算出逐题分数；篡改评估器的探针被拦下；每个候选的改动都能检查。准入结果和 RSIHub 的不一致是预期的（我们要求逐题不回退，比 hillclimb 严），我需要看具体是哪些。
- **R3** 通过：`preflight` 和 `verify` 通过；每一代都有 `validate/openrsi_guard.json` 和带 `no_regression` 的 `gate.json`。注意不要在装完 OpenRSI 之后执行 `uv sync`，它会删掉 RSIHub venv 里的额外包。

没有跑过的 workspace 也没关系：R3 跑完的 workspace 可以直接用于 R2。

## 发回什么

```bash
bash field_tests/collect.sh
```

打包 `field_tests/out/` 下的日志和 JSON，不含模型权重和数据。另外请附上：GPU 型号、`pip show verl vllm torch | grep -E "Name|Version"`、RSIHub 的 commit（`git -C /path/to/RSIHub rev-parse HEAD`）。
