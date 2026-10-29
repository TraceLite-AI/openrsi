"""OpenRSI 协议的数据格式。

这些模型是协议的唯一来源：spec/schemas/ 下的 JSON Schema 由它们导出，
MCP 工具的输入输出也直接用它们。字段说明见 spec/PROTOCOL.md。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = "openrsi/0.4"


class _Model(BaseModel):
    # 0.x 阶段允许额外字段，方便组件携带自己的扩展信息
    model_config = ConfigDict(extra="allow")


def canonical_json(value: Any) -> str:
    """稳定的 JSON 序列化：键排序、无多余空格。哈希都基于它。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- 任务与轨迹

Split = Literal["train", "holdout", "anchor", "final"]


class Task(_Model):
    """一道题。split 决定它能流向哪里：只有 train 的轨迹可以进入归因和提案。

    reference 放标准答案等只给判别器看的信息；运行时交给 agent 之前会去掉它。
    """

    id: str
    input: Any = None
    reference: Any = Field(default=None, description="只给判别器看，不得交给 agent")
    split: Split = "train"
    tags: list[str] = Field(default_factory=list)
    env_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


StepType = Literal["message", "tool_call", "tool_result", "observation"]


class Step(_Model):
    """轨迹里的一步。tool_call 和 tool_result 通过 call_id 配对。"""

    type: StepType
    id: str | None = None
    role: str | None = None
    content: Any = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
    call_id: str | None = None
    is_error: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def for_agent(task: "Task") -> "Task":
    """agent 能看到的那部分题目。"""
    return task.model_copy(update={"reference": None})


class Trajectory(_Model):
    id: str
    task_id: str
    steps: list[Step] = Field(default_factory=list)
    output: Any = None
    used_items: list[str] = Field(default_factory=list, description="本次用到的 memory 条目 id")
    snapshot_id: str | None = Field(default=None, description="跑在哪个快照上")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- 证据

class Evidence(_Model):
    """一条可独立复核的证据。sha256 是 content 的哈希，id 由类型、来源和哈希决定。"""

    id: str
    type: str
    content: Any = None
    uri: str | None = None
    sha256: str
    source: str
    collected_at: str
    collector: str

    @classmethod
    def make(cls, type: str, content: Any, source: str, collector: str, uri: str | None = None) -> "Evidence":
        digest = content_hash(content)
        eid = "ev_" + hashlib.sha256(f"{type}|{source}|{digest}".encode()).hexdigest()[:16]
        return cls(id=eid, type=type, content=content, uri=uri, sha256=digest,
                   source=source, collected_at=now_iso(), collector=collector)

    def verify(self) -> bool:
        """内容是否和哈希一致。只有 uri 没有内容的证据无法在本地复核，返回 True。"""
        if self.content is None and self.uri:
            return True
        return content_hash(self.content) == self.sha256


class EvidenceBundle(_Model):
    evidence: list[Evidence] = Field(default_factory=list)


class VerifyResult(_Model):
    valid: bool
    reason: str | None = None


# ---------------------------------------------------------------- 判决

RubricOutcome = Literal["pass", "fail", "skip", "error"]


class RubricResult(_Model):
    id: str
    result: RubricOutcome
    evidence_ids: list[str] = Field(default_factory=list, description="这条结论引用的证据")
    weight: float = 1.0
    required: bool = False
    note: str | None = None


class Verdict(_Model):
    """判别结果。每条 rubric 结论必须引用证据：要么是调用方给的，要么在 collected 里。"""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    rubric: list[RubricResult] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    judge_ref: str
    evidence_ids: list[str] = Field(default_factory=list, description="判别时实际采信的全部证据")
    collected: list[Evidence] = Field(default_factory=list, description="判别器自己补充收集的证据")
    notes: list[str] = Field(default_factory=list)


class AnchorExample(_Model):
    """人工标注的样本，用来校准判别器。"""

    task: Task
    trajectory: Trajectory
    evidence: list[Evidence] = Field(default_factory=list)
    label_passed: bool
    label_score: float | None = None


class CalibrationReport(_Model):
    n: int
    agreement: float
    false_pass: int = Field(description="人工判不通过、判别器判通过的数量")
    false_fail: int
    disagreements: list[str] = Field(default_factory=list, description="判断不一致的 task id")


# ---------------------------------------------------------------- 可进化体

class Item(_Model):
    """memory 里的一条记录。"""

    id: str
    text: str = ""
    tags: list[str] = Field(default_factory=list)
    kind: str = "note"
    data: dict[str, Any] = Field(default_factory=dict)


class Patch(_Model):
    """对一个可进化体的修改。ops 的格式由可进化体的类型决定（memory 见 MemoryOp）。"""

    target: str
    ops: list[dict[str, Any]]
    id: str | None = None
    base_snapshot: str | None = Field(default=None, description="期望的当前快照；不一致则拒绝，防止覆盖别人的修改")
    rationale: str | None = None
    proposer_ref: str | None = None


class MemoryOp(_Model):
    op: Literal["add", "remove", "replace"]
    item: Item | None = None
    item_id: str | None = None


class Snapshot(_Model):
    id: str
    target: str
    parent_ids: list[str] = Field(default_factory=list)
    patch_id: str | None = None
    content_hash: str
    created_at: str
    message: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class Diff(_Model):
    from_id: str
    to_id: str
    ops: list[dict[str, Any]] = Field(default_factory=list)


class EvolvableInfo(_Model):
    target: str
    type: str
    head: str | None
    surfaces: list[str] = Field(default_factory=list)
    size: int = 0
    example_ops: list[dict[str, Any]] = Field(
        default_factory=list, description="一组无害的示例 ops，一致性测试用它验证 apply/rollback")


class RetrieveResult(_Model):
    items: list[Item] = Field(default_factory=list)
    snapshot_id: str | None = None


class SkillResult(_Model):
    """一次 skill 调用的结果。代码在隔离的子进程里运行。"""

    ok: bool
    result: Any = None
    error: str | None = None
    stdout: str = ""
    duration_ms: float = 0.0


# ---------------------------------------------------------------- 护栏

class GuardFinding(_Model):
    rule: str
    severity: Literal["block", "warn"]
    message: str
    op_index: int | None = None
    item_id: str | None = None


class GuardReport(_Model):
    allowed: bool = Field(description="有任何 block 级别的发现就是 False")
    findings: list[GuardFinding] = Field(default_factory=list)
    guard_ref: str | None = None


class ExportResult(_Model):
    snapshot_id: str
    content: Any = None


# ---------------------------------------------------------------- 环境与执行体

class TaskBundle(_Model):
    tasks: list[Task] = Field(default_factory=list)


class Episode(_Model):
    """一次完整的执行记录：题目、轨迹、判决，以及判决用到的证据。"""

    task: Task
    trajectory: Trajectory
    verdict: Verdict
    evidence: list[Evidence] = Field(default_factory=list)


# ---------------------------------------------------------------- 归因

class AttributionTarget(_Model):
    kind: Literal["step", "item", "component", "patch", "rubric"]
    ref: str
    weight: float = Field(ge=-1.0, le=1.0, description="负数是责任，正数是功劳")
    note: str | None = None


class Attribution(_Model):
    episode_id: str = Field(description="对应 trajectory.id")
    targets: list[AttributionTarget] = Field(default_factory=list)
    method: str
    rationale: str | None = None


class AttributionBundle(_Model):
    attributions: list[Attribution] = Field(default_factory=list)


# ---------------------------------------------------------------- 提案

class ProposalRequest(_Model):
    """给提案器的输入。episodes 只能来自 train 划分。

    只有一个可进化体时看 target / state 即可。同时进化多个时，targets 和 states 列出全部，
    补丁的 target 字段决定改哪一个；parent_snapshot 是组合快照的 id。
    """

    target: EvolvableInfo
    parent_snapshot: str
    targets: list[EvolvableInfo] = Field(default_factory=list, description="全部可进化体（0.3）")
    states: dict[str, Any] = Field(default_factory=dict, description="每个可进化体在父版本里的完整内容（0.3）")
    episodes: list[Episode] = Field(default_factory=list)
    attributions: list[Attribution] = Field(default_factory=list)
    state: Any = Field(default=None, description="父快照的完整内容（rsi_evolvable_export 的结果）")
    max_patches: int = Field(default=1, ge=1)
    hints: dict[str, Any] = Field(default_factory=dict,
                                  description="算法给提案器的建议，如 {'proposer': 'lesson'}（0.4）")


class ProposalBundle(_Model):
    patches: list[Patch] = Field(default_factory=list)


# ---------------------------------------------------------------- 进化算法

class AskResult(_Model):
    generation: int
    parent_id: str | None = Field(description="下一代从哪个快照出发；None 表示停止")
    done: bool = False
    reason: str | None = None
    proposer: str | None = Field(default=None, description="建议这一代用哪个提案器（0.4，自适应提案器选择）")


class TellRequest(_Model):
    generation: int
    parent_id: str | None = None
    candidate_id: str | None = None
    accepted: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    task_scores: dict[str, float] = Field(default_factory=dict, description="候选在每道 holdout 题上的分数（0.4）")
    proposer: str | None = Field(default=None, description="产生这个候选的提案器（0.4）")


class ScreenResult(_Model):
    """评估前的筛查：和已经评估过的候选太像就不必再花评估成本（0.4）。"""

    accept: bool
    novelty: float = Field(ge=0.0, le=1.0, description="1 表示全新，0 表示完全重复")
    reason: str | None = None


class AlgorithmStatus(_Model):
    generation: int
    best_id: str | None = None
    best_score: float | None = None
    archive_size: int = 0
    done: bool = False
    reason: str | None = None


# ---------------------------------------------------------------- 准入

class GateRequest(_Model):
    candidate_id: str
    baseline: dict[str, float] = Field(description="父快照的分数，如 holdout、anchor、cost、n_holdout")
    candidate: dict[str, float]
    baseline_tasks: dict[str, float] = Field(default_factory=dict, description="父快照每道 holdout 题的分数（0.4）")
    candidate_tasks: dict[str, float] = Field(default_factory=dict, description="候选每道 holdout 题的分数（0.4）")
    primary_metric: str | None = Field(
        default=None, description="本次要求提升的指标；判别器自身的修改用 anchor，因为它会改变 holdout 的打分标准（0.3）")


class GateCheck(_Model):
    name: str
    passed: bool
    detail: str | None = None


class GateDecision(_Model):
    candidate_id: str
    accepted: bool
    checks: list[GateCheck] = Field(default_factory=list)
    reason: str | None = None


# ---------------------------------------------------------------- 谱系

class LineageEntry(_Model):
    id: str = Field(description="快照 id")
    target: str
    parent_ids: list[str] = Field(default_factory=list)
    generation: int = 0
    status: Literal["root", "accepted", "rejected"]
    metrics: dict[str, float] = Field(default_factory=dict)
    decision: GateDecision | None = None
    patch: Patch | None = None
    snapshots: dict[str, str] = Field(default_factory=dict,
                                      description="同时进化多个可进化体时，每个目标对应的快照 id（0.3）")
    task_scores: dict[str, float] = Field(default_factory=dict, description="每道 holdout 题的分数（0.4）")
    created_at: str = Field(default_factory=now_iso)


class LineageLog(_Model):
    entries: list[LineageEntry] = Field(default_factory=list)


# ---------------------------------------------------------------- 组件自述

class ComponentInfo(_Model):
    name: str
    version: str
    protocol: str = PROTOCOL_VERSION
    kinds: list[str]
    tools: list[str]


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    m.__name__: m
    for m in (Task, Step, Trajectory, Evidence, Verdict, RubricResult, AnchorExample, CalibrationReport,
              Item, Patch, MemoryOp, Snapshot, Diff, EvolvableInfo, ExportResult, Episode, Attribution,
              ProposalRequest, AskResult, TellRequest, AlgorithmStatus, GateRequest, GateDecision,
              LineageEntry, SkillResult, GuardReport, ScreenResult, ComponentInfo)
}
