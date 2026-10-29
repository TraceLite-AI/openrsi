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

PROTOCOL_VERSION = "openrsi/0.1"


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
    """一道题。split 决定它能流向哪里：只有 train 的轨迹可以进入归因和提案。"""

    id: str
    input: Any = None
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


# ---------------------------------------------------------------- v0.2 预留

class AttributionTarget(_Model):
    kind: Literal["step", "item", "component", "patch"]
    ref: str
    weight: float


class Attribution(_Model):
    episode_id: str
    targets: list[AttributionTarget]
    method: str
    rationale: str | None = None


class GateDecision(_Model):
    candidate_id: str
    accepted: bool
    checks: list[dict[str, Any]] = Field(default_factory=list)
    reason: str | None = None


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
              Item, Patch, MemoryOp, Snapshot, Diff, EvolvableInfo, Attribution, GateDecision, ComponentInfo)
}
