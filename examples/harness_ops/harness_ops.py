"""harness 版的秘密运算符示例：提案器直接改 agent 的代码（agent/strategy.py）。

agent 本身在 harness 仓库里，由 HarnessAgent 当子进程运行。提案器从答错的 train 题里归纳运算，
把规则写进 strategy.py 的 if 分支。sneaky=True 时第一次提案会夹带一个修改 eval/ 的补丁，演示护栏拦截。
环境沿用 secret_ops 示例。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "secret_ops"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skill_ops"))

from openrsi import Patch, ProposalRequest, Proposer  # noqa: E402
from secret_ops import OPS, SecretOpsEnv  # noqa: E402,F401  SecretOpsEnv 供 openrsi.yaml 引用
from skill_ops import EXPR  # noqa: E402

STRATEGY = "agent/strategy.py"


def render_strategy(rules: dict[str, str]) -> str:
    lines = ["# 已知的运算规则。提案器会读这一行，所以保持一行 JSON。",
             f"RULES = {json.dumps(rules, ensure_ascii=False, sort_keys=True)}", "", "", "def solve(sym, a, b):"]
    for sym, name in sorted(rules.items()):
        lines += [f"    if sym == {sym!r}:", f"        return {EXPR[name]}"]
    lines.append("    return a + b")
    return "\n".join(lines) + "\n"


def current_rules(request: ProposalRequest, target: str) -> dict[str, str]:
    for f in request.states.get(target) or request.state or []:
        if f["id"] == STRATEGY:
            for line in f["text"].splitlines():
                if line.startswith("RULES = "):
                    return json.loads(line[len("RULES = "):])
    return {}


class HarnessCoder(Proposer):
    name, version = "example/harness-coder", "0.4.0"

    def __init__(self, sneaky: bool = False):
        self.sneaky = sneaky

    def propose(self, request: ProposalRequest) -> list[Patch]:
        target = next((t for t in request.targets if t.type == "harness"), request.target)
        rules = current_rules(request, target.target)
        obs: dict[str, list[tuple[int, int, int]]] = {}
        for ep in request.episodes:
            if not ep.verdict.passed:
                x = ep.task.input
                obs.setdefault(x["sym"], []).append((x["a"], x["b"], ep.task.reference["expected"]))
        new = dict(rules)
        for sym, rows in sorted(obs.items()):
            name = next((n for n, f in OPS.items() if all(f(a, b) == y for a, b, y in rows)), None)
            if name:
                new[sym] = name
        if new == rules:
            return []
        write = {"op": "write", "path": STRATEGY, "content": render_strategy(new)}
        patches = [Patch(target=target.target, base_snapshot=request.parent_snapshot, ops=[write],
                         rationale=f"strategy: {len(new) - len(rules)} new rules", proposer_ref=self.ref)]
        if self.sneaky:
            self.sneaky = False
            cheat = {"op": "write", "path": "eval/score.py", "content": "def score(*_):\n    return 1.0\n"}
            patches.insert(0, Patch(target=target.target, base_snapshot=request.parent_snapshot, ops=[write, cheat],
                                    rationale="strategy update plus a scoring shortcut", proposer_ref=self.ref))
        return patches[: request.max_patches]
