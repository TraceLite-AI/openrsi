"""协议定义了哪些工具：名字、所属组件类型、是否必选。"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = ("evidence", "judge", "evolvable", "memory")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str  # "component" 表示所有组件都要实现
    required: bool
    description: str


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("rsi_component_info", "component", True, "报告组件名、版本、协议版本和实现的组件类型。"),
    # 取证
    ToolSpec("rsi_evidence_collect", "evidence", True, "从任务和轨迹中收集可核验的证据。"),
    ToolSpec("rsi_evidence_verify", "evidence", True, "复核一条证据的内容和哈希是否一致。"),
    # 判别
    ToolSpec("rsi_judge_evaluate", "judge", True, "基于证据给出判决，每条结论引用证据。"),
    ToolSpec("rsi_judge_calibrate", "judge", False, "在人工标注的 anchor 集上计算与人工判断的一致率。"),
    # 可进化体
    ToolSpec("rsi_evolvable_describe", "evolvable", True, "描述可进化体：类型、当前快照、可进化的部分。"),
    ToolSpec("rsi_evolvable_snapshot", "evolvable", True, "返回当前快照。"),
    ToolSpec("rsi_evolvable_apply", "evolvable", True, "应用一个补丁，生成新快照并设为当前。"),
    ToolSpec("rsi_evolvable_rollback", "evolvable", True, "把当前快照切回指定快照。"),
    ToolSpec("rsi_evolvable_diff", "evolvable", True, "计算两个快照之间的差异。"),
    # memory（同时也是 evolvable）
    ToolSpec("rsi_memory_retrieve", "memory", True, "按查询和标签检索 memory 条目。"),
)

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def required_tools(kind: str) -> set[str]:
    names = {t.name for t in TOOLS if t.required and t.kind in (kind, "component")}
    if kind == "memory":
        names |= required_tools("evolvable")
    return names


def kinds_from_tools(names: set[str] | list[str]) -> list[str]:
    """根据实现了哪些工具，推断组件属于哪些类型。"""
    present = set(names)
    return [k for k in KINDS if required_tools(k) <= present]
