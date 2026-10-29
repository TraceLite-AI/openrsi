"""通过 MCP 调用任何 OpenRSI 组件，不管它是什么语言写的。

    async with connect(["uvx", "openrsi-judge", "--rubric", "rubric.yaml"]) as c:
        verdict = await c.judge.evaluate(task, trajectory)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, TypeVar

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel

from .protocol import kinds_from_tools
from .types import (
    AnchorExample, CalibrationReport, ComponentInfo, Diff, Evidence, EvidenceBundle, EvolvableInfo,
    Item, Patch, RetrieveResult, Snapshot, Task, Trajectory, Verdict, VerifyResult,
)

M = TypeVar("M", bound=BaseModel)


class ToolCallError(RuntimeError):
    """组件返回了错误（参数不合法、补丁冲突等）。"""


def _dump(x: Any) -> Any:
    if isinstance(x, BaseModel):
        return x.model_dump(mode="json", exclude_none=True)
    if isinstance(x, list):
        return [_dump(i) for i in x]
    return x


class RemoteComponent:
    def __init__(self, client: Client):
        self._client = client
        self.tools: list[str] = []
        self.info: ComponentInfo | None = None

    async def _load(self) -> None:
        listed = await self._client.list_tools()
        self.tools = [t.name for t in listed.tools]
        if "rsi_component_info" in self.tools:
            self.info = await self.call("rsi_component_info", {}, ComponentInfo)

    @property
    def kinds(self) -> list[str]:
        return kinds_from_tools(self.tools)

    async def call(self, tool: str, args: dict[str, Any], model: type[M]) -> M:
        result = await self._client.call_tool(tool, {k: _dump(v) for k, v in args.items() if v is not None})
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise ToolCallError(f"{tool}: {text}")
        return model.model_validate(result.structured_content)

    @property
    def evidence(self) -> "RemoteEvidence":
        return RemoteEvidence(self)

    @property
    def judge(self) -> "RemoteJudge":
        return RemoteJudge(self)

    @property
    def evolvable(self) -> "RemoteEvolvable":
        return RemoteEvolvable(self)

    @property
    def memory(self) -> "RemoteMemory":
        return RemoteMemory(self)


class _Facade:
    def __init__(self, rc: RemoteComponent):
        self._rc = rc


class RemoteEvidence(_Facade):
    async def collect(self, task: Task, trajectory: Trajectory, spec: dict | None = None) -> list[Evidence]:
        bundle = await self._rc.call("rsi_evidence_collect",
                                     {"task": task, "trajectory": trajectory, "spec": spec}, EvidenceBundle)
        return bundle.evidence

    async def verify(self, evidence: Evidence) -> VerifyResult:
        return await self._rc.call("rsi_evidence_verify", {"evidence": evidence}, VerifyResult)


class RemoteJudge(_Facade):
    async def evaluate(self, task: Task, trajectory: Trajectory, evidence: list[Evidence] | None = None) -> Verdict:
        return await self._rc.call("rsi_judge_evaluate",
                                   {"task": task, "trajectory": trajectory, "evidence": evidence or []}, Verdict)

    async def calibrate(self, anchor: list[AnchorExample]) -> CalibrationReport:
        return await self._rc.call("rsi_judge_calibrate", {"anchor": anchor}, CalibrationReport)


class RemoteEvolvable(_Facade):
    async def describe(self) -> EvolvableInfo:
        return await self._rc.call("rsi_evolvable_describe", {}, EvolvableInfo)

    async def snapshot(self) -> Snapshot:
        return await self._rc.call("rsi_evolvable_snapshot", {}, Snapshot)

    async def apply(self, patch: Patch) -> Snapshot:
        return await self._rc.call("rsi_evolvable_apply", {"patch": patch}, Snapshot)

    async def rollback(self, snapshot_id: str) -> Snapshot:
        return await self._rc.call("rsi_evolvable_rollback", {"snapshot_id": snapshot_id}, Snapshot)

    async def diff(self, from_id: str, to_id: str) -> Diff:
        return await self._rc.call("rsi_evolvable_diff", {"from_id": from_id, "to_id": to_id}, Diff)


class RemoteMemory(RemoteEvolvable):
    async def retrieve(self, query: str = "", tags: list[str] | None = None, k: int = 5,
                       snapshot_id: str | None = None) -> list[Item]:
        res = await self._rc.call("rsi_memory_retrieve",
                                  {"query": query, "tags": tags, "k": k, "snapshot_id": snapshot_id},
                                  RetrieveResult)
        return res.items


Target = Any  # MCPServer 实例 / StdioServerParameters / URL / 命令列表


@asynccontextmanager
async def connect(target: Target, *, env: dict[str, str] | None = None) -> AsyncIterator[RemoteComponent]:
    """连接一个组件。target 可以是命令列表、URL、StdioServerParameters，或测试时直接传 MCPServer。

    命令列表启动的子进程默认只拿到 MCP 认为安全的少数环境变量；需要传 PYTHONPATH、API key 等时用 env。
    """
    if isinstance(target, (list, tuple)):
        target = StdioServerParameters(command=target[0], args=list(target[1:]), env=env)
    async with Client(target) as client:
        rc = RemoteComponent(client)
        await rc._load()
        yield rc
