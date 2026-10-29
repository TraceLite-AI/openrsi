"""GSM8K 环境：读 verl 预处理好的 parquet（examples/data_preprocess/gsm8k.py 的输出），也接受 jsonl。

- train：从 train 文件里每代抽 train_per_round 道（按代号固定随机种子，可复现）
- holdout：test 文件的前 holdout 道，固定不变，用来准入
- final：test 文件里紧接着的 final 道，只在最后报告用

题目用原始问题（extra_info.question），不带 verl 加在 prompt 里的格式提示，
这样"要求按 #### 输出答案"必须由 prompt 进化自己学会。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from openrsi import Environment, Task

VERL_SUFFIX = 'Let\'s think step by step and output the final answer after "####".'


def _load(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    else:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for r in rows:
        if "question" in r:  # jsonl：{"question", "answer"}，answer 可以是 GSM8K 原始解答（以 #### 结尾）
            q, a = r["question"], str(r["answer"]).split("####")[-1].strip().replace(",", "")
        else:
            extra = r.get("extra_info") or {}
            q = extra.get("question") or r["prompt"][0]["content"].replace(VERL_SUFFIX, "").strip()
            a = str(r["reward_model"]["ground_truth"])
        out.append({"question": q, "answer": a})
    return out


class GSM8KEnv(Environment):
    name, version = "example/gsm8k-env", "0.4.0"

    def __init__(self, data_dir: str, train_per_round: int = 64, holdout: int = 200, final: int = 300,
                 seed: int = 0):
        d = Path(data_dir).expanduser().resolve()
        def pick(stem):
            return (next((p for p in (d / f"{stem}.parquet", d / f"{stem}.jsonl") if p.exists()), None))
        train_file, test_file = pick("train"), pick("test")
        if not train_file or not test_file:
            raise FileNotFoundError(f"{d} 下需要 train / test 的 .parquet 或 .jsonl")
        self.train, self.test = _load(train_file), _load(test_file)
        self.train_per_round, self.holdout, self.final, self.seed = train_per_round, holdout, final, seed

    def _task(self, split: str, idx: int, row: dict) -> Task:
        return Task(id=f"{split}-{idx}", input={"question": row["question"]}, split=split,
                    reference={"expected": row["answer"]})

    def tasks(self, split: str, round: int = 0, limit: int | None = None) -> list[Task]:
        if split == "train":
            idx = random.Random(f"{self.seed}-{round}").sample(range(len(self.train)),
                                                               min(self.train_per_round, len(self.train)))
            out = [self._task("train", i, self.train[i]) for i in idx]
        elif split == "holdout":
            out = [self._task("holdout", i, self.test[i]) for i in range(min(self.holdout, len(self.test)))]
        elif split == "final":
            lo = min(self.holdout, len(self.test))
            out = [self._task("final", i, self.test[i]) for i in range(lo, min(lo + self.final, len(self.test)))]
        else:
            out = []
        return out[:limit] if limit else out
