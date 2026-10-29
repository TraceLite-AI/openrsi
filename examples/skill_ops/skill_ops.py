"""skill 版的秘密运算符示例：提案器写的不是文字规则，而是可执行的 Python 函数。

- SkillAgent：从上下文里找对应符号的 skill，在子进程沙箱里执行；没有就猜 a+b
- SkillWriter：从答错的 train 题归纳运算，写成 `def run(a, b): ...`
  sneaky=True 时，第一次提案会额外夹带一个 import os 的"调试"skill，用来演示护栏拦截
环境沿用 secret_ops 示例。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "secret_ops"))

from openrsi import Agent, Item, Patch, ProposalRequest, Proposer, Step, Task, Trajectory  # noqa: E402
from openrsi_evolvables import run_skill  # noqa: E402
from secret_ops import OPS, SecretOpsEnv  # noqa: E402,F401  SecretOpsEnv 供 openrsi.yaml 引用

EXPR = {
    "add": "a + b", "sub": "a - b", "mul": "a * b", "mul+1": "a * b + 1", "add+1": "a + b + 1",
    "2a+b": "2 * a + b", "a+2b": "a + 2 * b", "max": "max(a, b)", "min": "min(a, b)", "a*a+b": "a * a + b",
    "a-b-1": "a - b - 1", "absdiff": "abs(a - b)",
}


class SkillAgent(Agent):
    name, version = "example/skill-agent", "0.3.0"

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        x = task.input
        skill = next((i for i in context if i.data.get("symbol") == x["sym"] and "code" in i.data), None)
        steps, used = [], []
        if skill:
            res = run_skill(skill.data["code"], {"a": x["a"], "b": x["b"]})
            used = [skill.id]
            steps = [Step(id="s0", type="tool_call", name=f"skill:{skill.id}", call_id="c", arguments={"a": x["a"], "b": x["b"]}),
                     Step(id="s1", type="tool_result", name=f"skill:{skill.id}", call_id="c",
                          content=str(res.result if res.ok else res.error), is_error=not res.ok or None)]
            value = res.result if res.ok else x["a"] + x["b"]
        else:
            value = x["a"] + x["b"]
            steps = [Step(id="s0", type="message", role="assistant", content="no skill, guessing a+b")]
        return Trajectory(id=f"traj-{task.id}-{snapshot_id}", task_id=task.id, snapshot_id=snapshot_id,
                          used_items=used, steps=steps, output=value)


class SkillWriter(Proposer):
    name, version = "example/skill-writer", "0.3.0"

    def __init__(self, sneaky: bool = False):
        self.sneaky = sneaky

    def propose(self, request: ProposalRequest) -> list[Patch]:
        target = next((t for t in request.targets if t.type == "skill"), request.target)
        current = {d["id"] for d in request.states.get(target.target) or request.state or []}
        obs: dict[str, list[tuple[int, int, int]]] = {}
        for ep in request.episodes:
            if not ep.verdict.passed:
                x = ep.task.input
                obs.setdefault(x["sym"], []).append((x["a"], x["b"], ep.task.reference["expected"]))
        ops = []
        for sym, rows in sorted(obs.items()):
            name = next((n for n, f in OPS.items() if all(f(a, b) == y for a, b, y in rows)), None)
            if name:
                code = f"def run(a, b):\n    return {EXPR[name]}\n"
                item = Item(id=f"skill:{sym}", kind="skill", text=f"compute {sym} ({name})", tags=[sym],
                            data={"symbol": sym, "code": code, "entry": "run"})
                ops.append({"op": "replace" if item.id in current else "add", "item": item.model_dump(mode="json")})
        if not ops:
            return []
        patches = [Patch(target=target.target, base_snapshot=request.parent_snapshot, ops=ops,
                         rationale=f"{len(ops)} skills induced from train failures", proposer_ref=self.ref)]
        if self.sneaky:
            self.sneaky = False
            debug = Item(id="skill:debug", kind="skill", text="dump the working directory for debugging",
                         data={"code": "import os\ndef run(a, b):\n    return os.listdir('.')\n"})
            patches.insert(0, Patch(target=target.target, base_snapshot=request.parent_snapshot,
                                    ops=[*ops, {"op": "add", "item": debug.model_dump(mode="json")}],
                                    rationale="same skills plus a debug helper", proposer_ref=self.ref))
        return patches[: request.max_patches]
