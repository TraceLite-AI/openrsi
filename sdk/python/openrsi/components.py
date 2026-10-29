"""组件基类。继承一个或多个基类，实现对应方法，就能在进程内直接用，或用 serve() 暴露成 MCP server。

一个类可以同时继承多个基类（比如既取证又判别），它的 kinds 是所有基类类型的并集。
"""
from __future__ import annotations

from typing import Any

from .types import (
    AlgorithmStatus, AnchorExample, AskResult, Attribution, CalibrationReport, Diff, Episode, Evidence,
    EvolvableInfo, GateDecision, GateRequest, Item, LineageEntry, Patch, ProposalRequest, Snapshot,
    Task, TellRequest, Trajectory, Verdict, VerifyResult,
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


class ComponentError(Exception):
    """组件拒绝请求的原因，会通过 MCP 原样告诉调用方。"""


class StaleSnapshotError(ComponentError):
    """补丁声明的 base_snapshot 不是当前快照。"""


# ---------------------------------------------------------------- 执行面

class Environment(Component):
    _kind = "env"

    def tasks(self, split: str, round: int = 0, limit: int | None = None) -> list[Task]:
        raise NotImplementedError


class Agent(Component):
    """执行体本身不持有 memory：运行时检索好相关条目，作为 context 传进来。"""

    _kind = "agent"

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        raise NotImplementedError


# ---------------------------------------------------------------- 评估面

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


class Attributor(Component):
    _kind = "attribution"

    def explain(self, episodes: list[Episode]) -> list[Attribution]:
        raise NotImplementedError


# ---------------------------------------------------------------- 进化面

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

    def export(self, snapshot_id: str | None = None) -> Any:
        """可选：快照的完整内容。不实现时提案器只能看到 describe 的信息。"""
        raise NotImplementedError


class Memory(Evolvable):
    _kind = "memory"

    def retrieve(self, query: str = "", tags: list[str] | None = None, k: int = 5,
                 snapshot_id: str | None = None) -> list[Item]:
        raise NotImplementedError


class Proposer(Component):
    _kind = "proposer"

    def propose(self, request: ProposalRequest) -> list[Patch]:
        raise NotImplementedError


class Algorithm(Component):
    """管理候选快照的种群。ask 选父代，tell 回报结果。第一次 tell 用来登记根快照。"""

    _kind = "algorithm"

    def ask(self) -> AskResult:
        raise NotImplementedError

    def tell(self, result: TellRequest) -> AlgorithmStatus:
        raise NotImplementedError


class Gate(Component):
    _kind = "gate"

    def check(self, request: GateRequest) -> GateDecision:
        raise NotImplementedError


# ---------------------------------------------------------------- 基础设施

class Lineage(Component):
    _kind = "lineage"

    def record(self, entry: LineageEntry) -> LineageEntry:
        raise NotImplementedError

    def log(self, target: str | None = None, status: str | None = None, limit: int | None = None) -> list[LineageEntry]:
        raise NotImplementedError

    def best(self, metric: str = "holdout", target: str | None = None) -> LineageEntry | None:
        raise NotImplementedError
