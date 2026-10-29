# 小模型跑通全流程

用一个开源小模型（默认 Qwen2.5-0.5B-Instruct）做 GSM8K，同一轮进化里同时改两样东西：

- **系统提示**：被进化的小模型自己读答错的 train 题，改写系统提示（`openrsi_llm:PromptRewriter`）
- **模型权重**：把判别器判为通过的 train 轨迹拿去 SFT（`openrsi_verl:DistillProposer` + `train_sft.py`），也就是拒绝采样式的自我蒸馏

每一代由 ShinkaEvolve 的 UCB 在这两个提案器里选一个。候选要在 holdout 上真的变好才被接受。
权重变了，执行体 `ChatAgent` 会用新的 checkpoint 重启推理服务。

```
  GSM8K train 题 ──► ChatAgent（vLLM 上的当前权重 + 当前系统提示）──► 判别器（抽出 "#### 数字" 比对答案）
        ▲                                                                   │
        │                        ┌── prompt：小模型改写系统提示 ◄────────────┤ 答错的题
  准入（holdout）◄── 候选版本 ◄───┤                                           │
        │                        └── weights：通过的轨迹做 SFT ◄─────────────┘ 答对的题
        ▼
      谱系（每个版本 = 系统提示快照 + 权重 checkpoint）
```

起点的系统提示是空的，小模型一般不会按 `#### 数字` 给答案，所以 holdout 接近 0。
要把分数做上去，prompt 进化得先学会"答案写在 #### 后面"，权重进化才有足够的通过样本可以训练。

## 三种跑法

| 配置 | 推理 | 训练 | 用途 | 时间 |
|---|---|---|---|---|
| `openrsi.mock.yaml` | 假服务 `mock_llm.py` | 假训练 `fake_train.py` | 检查流程，CI 里跑 | 几秒 |
| `openrsi.cpu.yaml` | `hf_server.py`（transformers） | `train_sft.py` | 没有 GPU 时用真实小模型走一遍 | 4 核 CPU 约 1 小时 |
| `openrsi.yaml` | vLLM | `train_sft.py` | 正式跑 | 1 张 GPU 约 1～2 小时 |

### 在 GPU 上跑

```bash
pip install -e sdk/python -e runtime -e components/judge -e components/proposer -e components/algorithm \
            -e components/gate -e components/lineage -e components/evolvables -e components/guard \
            -e components/llm -e adapters/verl
pip install vllm torch transformers

# 数据：用 verl 的预处理脚本，或者不装 verl 时用 prepare_gsm8k.py
python <verl 仓库>/examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
#   或：python examples/small_model/prepare_gsm8k.py ~/data/gsm8k 再把 openrsi.yaml 里的 data_dir 改过去

cd examples/small_model
openrsi run openrsi.yaml
openrsi log openrsi.yaml
```

单卡上推理服务和训练同时占显存：vLLM 设了 `--gpu-memory-utilization 0.3`，0.5B 全参训练还需要 8GB 左右。
换更大的模型时，把训练放到另一张卡（训练命令前加 `CUDA_VISIBLE_DEVICES=1`），或者调小这个比例。

可以调的地方：

- `base_model`：换成别的开源模型，比如 Qwen2.5-1.5B-Instruct
- `PromptRewriter` 的 `base_url` / `model`：改写系统提示可以换成更强的模型（任何 OpenAI 兼容接口），被进化的仍是小模型
- `holdout`：题越多准入越可靠，评估越慢；`gate` 的 `significance_z` 要求提升超过噪声
- 训练命令：`train_sft.py` 是单卡最小实现；换成 verl 的 SFT 入口见 `adapters/verl` 和 `field_tests/verl/v4_weights_sft.py`

### 跑完发回什么

`.openrsi/lineage.jsonl`（谱系）、`openrsi log openrsi.yaml` 的输出、`.openrsi/vllm-logs/`（推理服务日志）、
`.openrsi/weights/checkpoints/*.log`（训练日志）、`.openrsi/sft/` 里任意一个文件的前几行。

## 在 CPU 上实际跑过的结果

在开发环境（4 核 CPU，无 GPU）用 `openrsi.cpu.yaml` 跑过 Qwen2.5-0.5B-Instruct，每代 12 道 train、15 道 holdout、3 代，约 1 小时：

| 代 | 提案器 | 结果 |
|---|---|---|
| 根 | — | holdout 0.20（3 / 15） |
| 1 | prompt | 没有可用的改写（被过滤或和原来一样），本代跳过 |
| 2 | weights | 用 4 条答对的 train 轨迹 SFT 3 步（loss 0.11～0.18），换上新 checkpoint 重启推理服务后 holdout 0.133，准入拒绝 |
| 3 | prompt | 同第 1 代 |

整条链路都走通了：推理服务按权重快照启动和重启、判别器抽答案打分、两种提案器轮流、真实训练产出完整的 Hugging Face 模型目录、准入拒绝了没变好的候选、谱系可续跑。
分数没有提升在意料之中：15 道题的 holdout 噪声很大（2 道和 3 道的差别），4 条样本的 SFT 也太少。要看到提升，需要 GPU 上的规模（每代 64 道 train、200 道 holdout）。

跑的过程中发现并修掉的问题：

- GSM8K 题目里有 U+2028 这类字符，`str.splitlines()` 会在那里断行，读 JSONL 出错。谱系、anchor 集等所有 JSONL 读取都改成按 `\n` 切分
- Qwen2.5-0.5B 不听"按 #### 作答"，几乎总是写 `\boxed{}`，长推导还会超过 256 token。判别器现在两种格式都认，输出长度放宽
- 0.5B 改写自己的系统提示时，会把错题原文抄进去还开始解题（被准入拒绝）。现在抄题的改写会被丢掉；配置里建议把改写换成更强的模型
- 起点没有按格式作答的提示时，一道题都答不对，权重进化无从开始。`PromptEvolvable` 现在可以给初始版本

## 文件

- `gsm8k_env.py`：GSM8K 环境，读 verl 的 parquet 或 jsonl；题目用原始问题，不带 verl 加的格式提示
- `train_sft.py`：只依赖 torch 和 transformers 的 SFT 脚本，只在最后一条 assistant 消息上算损失
- `hf_server.py`：transformers 实现的最小 OpenAI 兼容推理服务
- `prepare_gsm8k.py`：从 Hugging Face 下载 GSM8K 转成 jsonl
- `mock_llm.py`、`fake_train.py`、`make_toy_data.py`：不需要模型的假服务、假训练和玩具数据
