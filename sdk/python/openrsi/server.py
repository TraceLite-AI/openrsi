"""把组件暴露成 MCP server。工具名和参数严格按协议，外部框架看到的就是普通 MCP 工具。"""
from __future__ import annotations

import functools

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .components import Component, ComponentError, EvidenceCollector, Evolvable, Judge, Memory
from .protocol import TOOLS_BY_NAME
from .types import (
    AnchorExample, CalibrationReport, ComponentInfo, Diff, Evidence, EvidenceBundle, EvolvableInfo,
    PROTOCOL_VERSION, Patch, RetrieveResult, Snapshot, Task, Trajectory, Verdict, VerifyResult,
)


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

    if isinstance(component, EvidenceCollector):
        c_ev = component

        def collect(task: Task, trajectory: Trajectory, spec: dict | None = None) -> EvidenceBundle:
            return EvidenceBundle(evidence=c_ev.collect(task, trajectory, spec))

        def verify(evidence: Evidence) -> VerifyResult:
            return c_ev.verify(evidence)

        add(collect, "rsi_evidence_collect")
        add(verify, "rsi_evidence_verify")

    if isinstance(component, Judge):
        c_j = component

        def evaluate(task: Task, trajectory: Trajectory, evidence: list[Evidence] | None = None) -> Verdict:
            return c_j.evaluate(task, trajectory, evidence or [])

        def calibrate(anchor: list[AnchorExample]) -> CalibrationReport:
            return c_j.calibrate(anchor)

        add(evaluate, "rsi_judge_evaluate")
        add(calibrate, "rsi_judge_calibrate")

    if isinstance(component, Evolvable):
        c_e = component

        def describe() -> EvolvableInfo:
            return c_e.describe()

        def snapshot() -> Snapshot:
            return c_e.snapshot()

        def apply(patch: Patch) -> Snapshot:
            return c_e.apply(patch)

        def rollback(snapshot_id: str) -> Snapshot:
            return c_e.rollback(snapshot_id)

        def diff(from_id: str, to_id: str) -> Diff:
            return c_e.diff(from_id, to_id)

        add(describe, "rsi_evolvable_describe")
        add(snapshot, "rsi_evolvable_snapshot")
        add(apply, "rsi_evolvable_apply")
        add(rollback, "rsi_evolvable_rollback")
        add(diff, "rsi_evolvable_diff")

    if isinstance(component, Memory):
        c_m = component

        def retrieve(query: str = "", tags: list[str] | None = None, k: int = 5,
                     snapshot_id: str | None = None) -> RetrieveResult:
            items = c_m.retrieve(query, tags, k, snapshot_id)
            return RetrieveResult(items=items, snapshot_id=snapshot_id or c_m.snapshot().id)

        add(retrieve, "rsi_memory_retrieve")

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


def serve(component: Component, transport: str = "stdio") -> None:
    """以 MCP server 方式运行组件（默认 stdio，供其他进程或框架启动）。"""
    build_server(component).run(transport)  # type: ignore[arg-type]
