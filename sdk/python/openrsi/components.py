"""组件基类。继承一个或多个基类，实现对应方法，就能在进程内直接用，或用 serve() 暴露成 MCP server。

一个类可以同时继承多个基类（比如既取证又判别），它的 kinds 是所有基类类型的并集。
"""
from __future__ import annotations

from .types import (
    AnchorExample, CalibrationReport, Diff, Evidence, EvolvableInfo, Item, Patch, Snapshot,
    Task, Trajectory, Verdict, VerifyResult,
)


class Component:
    name: str = "unnamed"
    version: str = "0.0.0"
    _kind: str | None = None

    @property
    def kinds(self) -> list[str]:
        found = []
        for cls in type(self).__mro__:
            k = cls.__dict__.get("_kind")
            if k and k not in found:
                found.append(k)
        return found

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"


class EvidenceCollector(Component):
    _kind = "evidence"

    def collect(self, task: Task, trajectory: Trajectory, spec: dict | None = None) -> list[Evidence]:
        raise NotImplementedError

    def verify(self, evidence: Evidence) -> VerifyResult:
        if evidence.verify():
            return VerifyResult(valid=True)
        return VerifyResult(valid=False, reason="内容与 sha256 不一致")


class Judge(Component):
    _kind = "judge"

    def evaluate(self, task: Task, trajectory: Trajectory, evidence: list[Evidence]) -> Verdict:
        raise NotImplementedError

    def calibrate(self, anchor: list[AnchorExample]) -> CalibrationReport:
        """默认实现：逐条判别，和人工标注比较是否通过。"""
        false_pass = false_fail = 0
        disagreements = []
        for ex in anchor:
            v = self.evaluate(ex.task, ex.trajectory, ex.evidence)
            if v.passed != ex.label_passed:
                disagreements.append(ex.task.id)
                if v.passed:
                    false_pass += 1
                else:
                    false_fail += 1
        n = len(anchor)
        return CalibrationReport(n=n, agreement=(n - len(disagreements)) / n if n else 1.0,
                                 false_pass=false_pass, false_fail=false_fail, disagreements=disagreements)


class Evolvable(Component):
    """被进化的对象。所有修改都必须走 apply(patch)，这样每一步都有快照、可以回滚。"""

    _kind = "evolvable"

    def describe(self) -> EvolvableInfo:
        raise NotImplementedError

    def snapshot(self) -> Snapshot:
        raise NotImplementedError

    def apply(self, patch: Patch) -> Snapshot:
        raise NotImplementedError

    def rollback(self, snapshot_id: str) -> Snapshot:
        raise NotImplementedError

    def diff(self, from_id: str, to_id: str) -> Diff:
        raise NotImplementedError


class Memory(Evolvable):
    _kind = "memory"

    def retrieve(self, query: str = "", tags: list[str] | None = None, k: int = 5,
                 snapshot_id: str | None = None) -> list[Item]:
        raise NotImplementedError


class ComponentError(Exception):
    """组件拒绝请求的原因，会通过 MCP 原样告诉调用方。"""


class StaleSnapshotError(ComponentError):
    """补丁声明的 base_snapshot 不是当前快照。"""
