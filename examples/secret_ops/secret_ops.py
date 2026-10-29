"""秘密运算符示例的三个自定义组件：环境、执行体、提案器。其余组件都用官方的。

8 个符号，每个背后是一个未知的二元运算（比如 ⊕ 其实是 a+b+1）。agent 从 memory 里找规则，
找不到就猜 a+b。提案器从答错的 train 题里归纳规则。标准答案放在 task.reference 里，agent 看不到。
"""
from __future__ import annotations

import random

from openrsi import Agent, Environment, Item, Patch, ProposalRequest, Proposer, Step, Task, Trajectory

OPS = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
    "mul+1": lambda a, b: a * b + 1, "add+1": lambda a, b: a + b + 1, "2a+b": lambda a, b: 2 * a + b,
    "a+2b": lambda a, b: a + 2 * b, "max": lambda a, b: max(a, b), "min": lambda a, b: min(a, b),
    "a*a+b": lambda a, b: a * a + b, "a-b-1": lambda a, b: a - b - 1, "absdiff": lambda a, b: abs(a - b),
}
SYMBOLS = list("⊕⊗⊙⊛⊞⊠⊡⊘")


class SecretOpsEnv(Environment):
    name, version = "example/secret-ops-env", "0.2.0"

    def __init__(self, seed: int = 0, symbols_per_round: int = 3, per_symbol: int = 3, holdout_per_symbol: int = 5):
        self.seed, self.symbols_per_round, self.per_symbol, self.holdout_per_symbol = (
            seed, symbols_per_round, per_symbol, holdout_per_symbol)
        rng = random.Random(seed)
        self.secret = dict(zip(SYMBOLS, rng.sample(sorted(OPS), len(SYMBOLS))))

    def _task(self, split: str, key: str, sym: str, a: int, b: int) -> Task:
        return Task(id=f"{split}-{key}-{sym}", split=split, tags=[sym], input={"sym": sym, "a": a, "b": b},
                    reference={"expected": OPS[self.secret[sym]](a, b)})

    def tasks(self, split: str, round: int = 0, limit: int | None = None) -> list[Task]:
        if split == "holdout":  # 固定一套，数字范围和 train 不重叠
            rng = random.Random(f"{self.seed}-holdout")
            out = [self._task("holdout", str(i), s, rng.randint(20, 99), rng.randint(20, 99))
                   for s in SYMBOLS for i in range(self.holdout_per_symbol)]
        elif split == "train":
            rng = random.Random(f"{self.seed}-train-{round}")
            out = [self._task("train", f"{round}.{i}", s, rng.randint(0, 9), rng.randint(0, 9))
                   for s in rng.sample(SYMBOLS, self.symbols_per_round) for i in range(self.per_symbol)]
        else:
            out = []
        return out[:limit] if limit else out


class RuleAgent(Agent):
    name, version = "example/rule-agent", "0.2.0"

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        x = task.input
        rule = next((i for i in context if i.kind == "rule" and i.data.get("symbol") == x["sym"]), None)
        op = rule.data["op"] if rule else "add"
        value = OPS[op](x["a"], x["b"])
        return Trajectory(
            id=f"traj-{task.id}-{snapshot_id}", task_id=task.id, snapshot_id=snapshot_id,
            used_items=[rule.id] if rule else [],
            steps=[Step(id="s0", type="tool_call", name="compute", call_id="c",
                        arguments={"op": op, "a": x["a"], "b": x["b"]}),
                   Step(id="s1", type="tool_result", name="compute", call_id="c", content=str(value))],
            output=value,
        )


class RuleInducer(Proposer):
    """在假设空间里找和答错题目一致的运算，写成规则。只读 train episode（运行时保证）。"""

    name, version = "example/rule-inducer", "0.2.0"

    def propose(self, request: ProposalRequest) -> list[Patch]:
        current = {d["id"] for d in request.state or []}
        obs: dict[str, list[tuple[int, int, int]]] = {}
        for ep in request.episodes:
            if not ep.verdict.passed:
                x = ep.task.input
                obs.setdefault(x["sym"], []).append((x["a"], x["b"], ep.task.reference["expected"]))
        ops = []
        for sym, rows in sorted(obs.items()):
            name = next((n for n, f in OPS.items() if all(f(a, b) == y for a, b, y in rows)), None)
            if name:
                item = Item(id=f"rule:{sym}", kind="rule", text=f"{sym} means {name}", tags=[sym],
                            data={"symbol": sym, "op": name})
                ops.append({"op": "replace" if item.id in current else "add", "item": item.model_dump(mode="json")})
        if not ops:
            return []
        return [Patch(target=request.target.target, base_snapshot=request.parent_snapshot, ops=ops,
                      rationale=f"{len(ops)} rules induced from train failures", proposer_ref=self.ref)]
