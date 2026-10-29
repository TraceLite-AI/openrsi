"""示例：判别器 + 可进化 memory 组成一个最小的自我改进循环（不需要 LLM）。

环境里有 8 个符号，每个背后是一个未知的二元运算（比如 ⊕ 其实是 a+b+1）。
agent 从 memory 里找规则来算，找不到就猜 a+b。每轮：

  1. 在 train 题上跑，判别器按证据判对错
  2. 从答错的题里归纳规则，写成 memory 补丁
  3. 在 holdout 题（数字范围和 train 不重叠）上验证
  4. holdout 变好就保留新快照，否则回滚

holdout 的结果只用来做准入，从不用来归纳规则。

    python examples/memory_evolution.py
"""
from __future__ import annotations

import random

from openrsi import Item, Patch, Step, Task, Trajectory
from openrsi_judge import EvidenceJudge, Rubric
from openrsi_memory import MemoryEvolvable

OPS = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "mul+1": lambda a, b: a * b + 1, "add+1": lambda a, b: a + b + 1, "2a+b": lambda a, b: 2 * a + b,
    "a+2b": lambda a, b: a + 2 * b, "max": lambda a, b: max(a, b), "min": lambda a, b: min(a, b),
    "a*a+b": lambda a, b: a * a + b, "a-b-1": lambda a, b: a - b - 1, "absdiff": lambda a, b: abs(a - b),
}
SYMBOLS = list("⊕⊗⊙⊛⊞⊠⊡⊘")


class Env:
    def __init__(self, seed: int = 0):
        self.seed = seed
        rng = random.Random(seed)
        self.secret = dict(zip(SYMBOLS, rng.sample(sorted(OPS), len(SYMBOLS))))

    def _task(self, split: str, i: int, sym: str, a: int, b: int) -> Task:
        return Task(id=f"{split}-{i}-{sym}", split=split, tags=[sym],
                    input={"sym": sym, "a": a, "b": b, "expected": OPS[self.secret[sym]](a, b)})

    def train(self, round_: int, symbols: int = 3, per_symbol: int = 3) -> list[Task]:
        rng = random.Random(f"{self.seed}-train-{round_}")
        return [self._task("train", round_, s, rng.randint(0, 9), rng.randint(0, 9))
                for s in rng.sample(SYMBOLS, symbols) for _ in range(per_symbol)]

    def holdout(self, per_symbol: int = 5) -> list[Task]:
        rng = random.Random(f"{self.seed}-holdout")
        return [self._task("holdout", i, s, rng.randint(20, 99), rng.randint(20, 99))
                for s in SYMBOLS for i in range(per_symbol)]


def agent(task: Task, memory: MemoryEvolvable) -> Trajectory:
    x = task.input
    rules = memory.retrieve(tags=[x["sym"]], k=1)
    op = rules[0].data["op"] if rules else "add"
    value = OPS[op](x["a"], x["b"])
    return Trajectory(
        id=f"traj-{task.id}", task_id=task.id, snapshot_id=memory.snapshot().id,
        used_items=[r.id for r in rules],
        steps=[Step(id="s0", type="tool_call", name="compute", call_id="c",
                    arguments={"op": op, "a": x["a"], "b": x["b"]}),
               Step(id="s1", type="tool_result", name="compute", call_id="c", content=str(value))],
        output=value,
    )


def reflect(failures: list[tuple[Task, int]], memory: MemoryEvolvable) -> list[dict]:
    """从答错的 train 题里归纳规则：在假设空间里找和所有观察一致的第一个运算。"""
    obs: dict[str, list[tuple[int, int, int]]] = {}
    for task, expected in failures:
        x = task.input
        obs.setdefault(x["sym"], []).append((x["a"], x["b"], expected))
    current = memory.items()
    ops = []
    for sym, rows in sorted(obs.items()):
        name = next((n for n, f in OPS.items() if all(f(a, b) == y for a, b, y in rows)), None)
        if name is None:
            continue
        item = Item(id=f"rule:{sym}", kind="rule", text=f"{sym} means {name}", tags=[sym],
                    data={"symbol": sym, "op": name})
        ops.append({"op": "replace" if item.id in current else "add", "item": item.model_dump(mode="json")})
    return ops


def run(rounds: int = 10, seed: int = 0, verbose: bool = True) -> list[dict]:
    env = Env(seed)
    memory = MemoryEvolvable(target="memory")
    judge = EvidenceJudge(Rubric.model_validate({"items": [{"id": "correct", "check": "output_equals", "required": True}]}))
    holdout = env.holdout()

    def score(tasks: list[Task]) -> tuple[float, list[tuple[Task, int]]]:
        failures = []
        for t in tasks:
            traj = agent(t, memory)
            v = judge.evaluate(t, traj, judge.collect(t, traj))
            if not v.passed:
                failures.append((t, t.input["expected"]))
        return 1 - len(failures) / len(tasks), failures

    baseline, _ = score(holdout)
    log = []
    if verbose:
        print(f"{'round':>5} {'train':>6} {'holdout':>8} {'cand':>6}  result")
    for r in range(1, rounds + 1):
        train_score, failures = score(env.train(r))
        ops = reflect(failures, memory)
        row = {"round": r, "train": train_score, "baseline": baseline, "candidate": None, "accepted": False}
        if ops:
            before = memory.snapshot()
            snap = memory.apply(Patch(target="memory", base_snapshot=before.id, ops=ops,
                                      rationale=f"round {r}: {len(ops)} rules from train failures"))
            cand, _ = score(holdout)
            row["candidate"] = cand
            if cand > baseline:
                row["accepted"], row["snapshot"], baseline = True, snap.id, cand
            else:
                memory.rollback(before.id)
        log.append(row)
        if verbose:
            res = f"accepted → {row['snapshot']}" if row["accepted"] else ("rolled back" if ops else "no change")
            cand = "-" if row["candidate"] is None else f"{row['candidate']:.2f}"
            print(f"{r:>5} {train_score:>6.2f} {row['baseline']:>8.2f} {cand:>6}  {res}")
    if verbose:
        print(f"\nfinal holdout = {baseline:.2f}; memory has {len(memory.items())} rules")
        print("lineage:", " ← ".join(s.id for s in memory.history()))
    return log


if __name__ == "__main__":
    run()
