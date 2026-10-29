"""协议定义了哪些工具：名字、所属组件类型、是否必选。"""
from __future__ import annotations

from dataclasses import dataclass

KINDS = ("env", "agent", "evidence", "judge", "attribution", "evolvable", "memory", "skill",
         "proposer", "algorithm", "gate", "guard", "lineage")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: str  # "component" 表示所有组件都要实现
    required: bool
    description: str


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("rsi_component_info", "component", True, "报告组件名、版本、协议版本和实现的组件类型。"),
    # 执行面
    ToolSpec("rsi_env_tasks", "env", True, "按划分（train/holdout/anchor/final）和轮次给出任务。"),
    ToolSpec("rsi_agent_run", "agent", True, "在给定的 memory 上下文中执行一个任务，返回轨迹。"),
    # 评估面
    ToolSpec("rsi_evidence_collect", "evidence", True, "从任务和轨迹中收集可核验的证据。"),
    ToolSpec("rsi_evidence_verify", "evidence", True, "复核一条证据的内容和哈希是否一致。"),
    ToolSpec("rsi_judge_evaluate", "judge", True, "基于证据给出判决，每条结论引用证据。"),
    ToolSpec("rsi_judge_calibrate", "judge", False, "在人工标注的 anchor 集上计算与人工判断的一致率。"),
    ToolSpec("rsi_attribution_explain", "attribution", True, "说明每个 episode 的成败归因于哪些步骤、条目或组件。"),
    # 进化面
    ToolSpec("rsi_evolvable_describe", "evolvable", True, "描述可进化体：类型、当前快照、可进化的部分。"),
    ToolSpec("rsi_evolvable_snapshot", "evolvable", True, "返回当前快照。"),
    ToolSpec("rsi_evolvable_apply", "evolvable", True, "应用一个补丁，生成新快照并设为当前。"),
    ToolSpec("rsi_evolvable_rollback", "evolvable", True, "把当前快照切回指定快照。"),
    ToolSpec("rsi_evolvable_diff", "evolvable", True, "计算两个快照之间的差异。"),
    ToolSpec("rsi_evolvable_export", "evolvable", False, "导出某个快照的完整内容，供提案器参考。"),
    ToolSpec("rsi_memory_retrieve", "memory", True, "按查询和标签检索 memory 条目。"),
    ToolSpec("rsi_skill_invoke", "skill", True, "在隔离环境里运行某个快照中的一个 skill，返回结果或错误。"),
    ToolSpec("rsi_proposer_propose", "proposer", True, "根据 train episode 和归因结果提出候选补丁。"),
    ToolSpec("rsi_algorithm_ask", "algorithm", True, "下一代从哪个快照出发。"),
    ToolSpec("rsi_algorithm_tell", "algorithm", True, "告知一代的结果，返回算法状态。"),
    ToolSpec("rsi_gate_check", "gate", True, "比较候选和父快照的分数，决定是否接受。"),
    ToolSpec("rsi_guard_inspect", "guard", True, "补丁生效前检查：危险代码、受保护条目、泄露的密钥等。"),
    # 基础设施
    ToolSpec("rsi_lineage_record", "lineage", True, "记录一个快照及其分数、准入决定。"),
    ToolSpec("rsi_lineage_log", "lineage", True, "按时间列出谱系记录，可按目标和状态过滤。"),
    ToolSpec("rsi_lineage_best", "lineage", True, "返回某个指标最高的已接受快照。"),
)

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def required_tools(kind: str) -> set[str]:
    names = {t.name for t in TOOLS if t.required and t.kind in (kind, "component")}
    if kind in ("memory", "skill"):
        names |= required_tools("evolvable")
    return names


def kinds_from_tools(names: set[str] | list[str]) -> list[str]:
    """根据实现了哪些工具，推断组件属于哪些类型。"""
    present = set(names)
    return [k for k in KINDS if required_tools(k) <= present]
