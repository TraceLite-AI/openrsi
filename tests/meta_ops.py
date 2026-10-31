"""元层进化测试用的提案器和 agent（基于 secret_ops 示例）。"""
from __future__ import annotations

from openrsi import Item, Patch

from secret_ops import RuleAgent, RuleInducer

META = ("improver", "curriculum")


def max_rules(request) -> int:
    for items in (request.hints.get("improver") or {}).values():
        for d in items or []:
            if isinstance(d, dict) and "max_rules" in (d.get("data") or {}):
                return int(d["data"]["max_rules"])
    return 1


class StrategicInducer(RuleInducer):
    """解题侧：每个补丁最多学 max_rules 条规则（从 improver 策略读）。
    元层：被问到 improver 时，把 max_rules 加 step（step=0 就是一个不改变行为的候选）。"""

    name, version = "test/strategic-inducer", "0.1.0"

    def __init__(self, step: int = 1):
        self.step = step

    def propose(self, request):
        if request.target.type == "improver":
            current = next((d for d in request.state or [] if d.get("id") == "max_rules"), None)
            k = int(current["data"]["max_rules"]) if current else 1
            item = Item(id="max_rules", kind="improver", text=f"learn at most {k + self.step} rules per patch",
                        data={"max_rules": k + self.step, "order": 0})
            return [Patch(target=request.target.target, base_snapshot=request.parent_snapshot, proposer_ref=self.ref,
                          ops=[{"op": "replace" if current else "add", "item": item.model_dump(mode="json")}])]
        if request.target.type in META:
            return []
        patches = super().propose(request)
        for p in patches:
            p.ops = p.ops[: max_rules(request)]
        return patches


class SolverOnlyInducer(RuleInducer):
    """普通的规则归纳，元层请求一律不理（让 CurriculumProposer 负责课程）。"""

    name, version = "test/solver-only-inducer", "0.1.0"

    def propose(self, request):
        return [] if request.target.type in META else super().propose(request)


class ContextRecordingAgent(RuleAgent):
    """记下 agent 每次拿到的上下文条目的 kind，检查元层内容没有交给 agent。"""

    seen_kinds: set[str] = set()

    def run(self, task, context, snapshot_id=None):
        ContextRecordingAgent.seen_kinds |= {i.kind for i in context}
        return super().run(task, context, snapshot_id)


class SneakyProposer(RuleInducer):
    """只要请求里出现 improver 目标就去改它；运行时在普通代不应当把元层对象交给提案器。"""

    name, version = "test/sneaky", "0.1.0"

    def propose(self, request):
        imp = next((t for t in request.targets or [] if t.type == "improver"), None)
        if imp is not None:
            item = Item(id="max_rules", kind="improver", text="sneaky", data={"max_rules": 9, "order": 0})
            return [Patch(target=imp.target, ops=[{"op": "replace", "item": item.model_dump(mode="json")}],
                          proposer_ref=self.ref)]
        return [] if request.target.type in META else super().propose(request)
