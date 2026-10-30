"""把组件暴露成 MCP server。工具名和参数严格按协议，外部框架看到的就是普通 MCP 工具。"""
from __future__ import annotations

import functools
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .components import (
    Agent, Algorithm, Attributor, Component, ComponentError, Environment, EvidenceCollector, Evolvable, Gate,
    Guard, Judge, Lineage, Memory, Proposer, Skills,
)
from .protocol import TOOLS_BY_NAME
from .types import (
    AlgorithmStatus, AnchorExample, AskResult, AttributionBundle, CalibrationReport, ComponentInfo, Diff,
    Episode, Evidence, EvidenceBundle, EvolvableInfo, ExportResult, GateDecision, GateRequest, GuardReport, Item,
    LineageEntry, LineageLog, PROTOCOL_VERSION, Patch, ProposalBundle, ProposalRequest, RetrieveResult,
    ScreenResult, SkillResult, Snapshot, Task, TaskBundle, TellRequest, Trajectory, Verdict, VerifyResult,
)


def _overrides(component: Component, base: type, method: str) -> bool:
    return getattr(type(component), method) is not getattr(base, method)


def build_server(component: Component, *, name: str | None = None) -> MCPServer:
    server = MCPServer(
        name or component.name,
        version=component.version,
        instructions=f"OpenRSI component ({PROTOCOL_VERSION}). Kinds: {', '.join(component.kinds)}.",
    )
    registered: list[str] = []

    def add(fn, tool: str) -> None:
        server.add_tool(_report_errors(fn), name=tool, description=TOOLS_BY_NAME[tool].description)
        registered.append(tool)

    c: Any = component

    if isinstance(component, Environment):
        def env_tasks(split: str, round: int = 0, limit: int | None = None) -> TaskBundle:
            return TaskBundle(tasks=c.tasks(split, round, limit))

        add(env_tasks, "rsi_env_tasks")

    if isinstance(component, Agent):
        def agent_run(task: Task, context: list[Item] | None = None, snapshot_id: str | None = None) -> Trajectory:
            return c.run(task, context or [], snapshot_id)

        add(agent_run, "rsi_agent_run")

    if isinstance(component, EvidenceCollector):
        def collect(task: Task, trajectory: Trajectory, spec: dict | None = None) -> EvidenceBundle:
            return EvidenceBundle(evidence=c.collect(task, trajectory, spec))

        def verify(evidence: Evidence) -> VerifyResult:
            return c.verify(evidence)

        add(collect, "rsi_evidence_collect")
        add(verify, "rsi_evidence_verify")

    if isinstance(component, Judge):
        def evaluate(task: Task, trajectory: Trajectory, evidence: list[Evidence] | None = None) -> Verdict:
            return c.evaluate(task, trajectory, evidence or [])

        def calibrate(anchor: list[AnchorExample]) -> CalibrationReport:
            return c.calibrate(anchor)

        add(evaluate, "rsi_judge_evaluate")
        add(calibrate, "rsi_judge_calibrate")

    if isinstance(component, Attributor):
        def explain(episodes: list[Episode]) -> AttributionBundle:
            return AttributionBundle(attributions=c.explain(episodes))

        add(explain, "rsi_attribution_explain")

    if isinstance(component, Evolvable):
        def describe() -> EvolvableInfo:
            return c.describe()

        def snapshot() -> Snapshot:
            return c.snapshot()

        def apply(patch: Patch) -> Snapshot:
            return c.apply(patch)

        def rollback(snapshot_id: str) -> Snapshot:
            return c.rollback(snapshot_id)

        def diff(from_id: str, to_id: str) -> Diff:
            return c.diff(from_id, to_id)

        add(describe, "rsi_evolvable_describe")
        add(snapshot, "rsi_evolvable_snapshot")
        add(apply, "rsi_evolvable_apply")
        add(rollback, "rsi_evolvable_rollback")
        add(diff, "rsi_evolvable_diff")

        if _overrides(component, Evolvable, "export"):
            def export(snapshot_id: str | None = None) -> ExportResult:
                sid = snapshot_id or c.snapshot().id
                return ExportResult(snapshot_id=sid, content=c.export(sid))

            add(export, "rsi_evolvable_export")

    if isinstance(component, Memory):
        def retrieve(query: str = "", tags: list[str] | None = None, k: int = 5,
                     snapshot_id: str | None = None) -> RetrieveResult:
            items = c.retrieve(query, tags, k, snapshot_id)
            return RetrieveResult(items=items, snapshot_id=snapshot_id or c.snapshot().id)

        add(retrieve, "rsi_memory_retrieve")

    if isinstance(component, Skills):
        def invoke(skill_id: str, arguments: dict | None = None, snapshot_id: str | None = None) -> SkillResult:
            return c.invoke(skill_id, arguments or {}, snapshot_id)

        add(invoke, "rsi_skill_invoke")

    if isinstance(component, Guard):
        def inspect(patch: Patch, target: EvolvableInfo | None = None) -> GuardReport:
            return c.inspect(patch, target)

        add(inspect, "rsi_guard_inspect")

    if isinstance(component, Proposer):
        def propose(request: ProposalRequest) -> ProposalBundle:
            return ProposalBundle(patches=c.propose(request))

        add(propose, "rsi_proposer_propose")

    if isinstance(component, Algorithm):
        def ask() -> AskResult:
            return c.ask()

        def tell(result: TellRequest) -> AlgorithmStatus:
            return c.tell(result)

        add(ask, "rsi_algorithm_ask")
        add(tell, "rsi_algorithm_tell")

        if _overrides(component, Algorithm, "screen"):
            def screen(patch: Patch, parent_id: str) -> ScreenResult:
                return c.screen(patch, parent_id)

            add(screen, "rsi_algorithm_screen")

    if isinstance(component, Gate):
        def gate_check(request: GateRequest) -> GateDecision:
            return c.check(request)

        add(gate_check, "rsi_gate_check")

    if isinstance(component, Lineage):
        def record(entry: LineageEntry) -> LineageEntry:
            return c.record(entry)

        def log(target: str | None = None, status: str | None = None, limit: int | None = None) -> LineageLog:
            return LineageLog(entries=c.log(target, status, limit))

        def best(metric: str = "holdout", target: str | None = None) -> LineageLog:
            found = c.best(metric, target)
            return LineageLog(entries=[found] if found else [])

        add(record, "rsi_lineage_record")
        add(log, "rsi_lineage_log")
        add(best, "rsi_lineage_best")

    tools = ["rsi_component_info", *registered]

    def info() -> ComponentInfo:
        return ComponentInfo(name=component.name, version=component.version,
                             kinds=component.kinds, tools=tools)

    add(info, "rsi_component_info")
    return server


# 调用方能修正的错误：原因要传回去。其他异常属于组件内部故障，细节留在服务端。
_CLIENT_ERRORS = (ComponentError, ValueError, KeyError)


def _report_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _CLIENT_ERRORS as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

    return wrapper


def serve(component: Component, transport: str = "stdio", **kwargs: Any) -> None:
    """以 MCP server 方式运行组件（默认 stdio，供其他进程或框架启动）。

    transport="streamable-http" 时可以传 host、port、streamable_http_path。
    """
    build_server(component).run(transport, **kwargs)  # type: ignore[arg-type]
