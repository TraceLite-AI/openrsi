"""通过 MCP 调用任何 OpenRSI 组件，不管它是什么语言写的。

    async with connect(["openrsi-judge", "--rubric", "rubric.yaml"]) as c:
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
    AlgorithmStatus, AnchorExample, AskResult, Attribution, AttributionBundle, CalibrationReport, ComponentInfo,
    Diff, Episode, Evidence, EvidenceBundle, EvolvableInfo, ExportResult, GateDecision, GateRequest, GuardReport, Item,
    LineageEntry, LineageLog, Patch, ProposalBundle, ProposalRequest, RetrieveResult, Snapshot, Task, TaskBundle,
    ScreenResult, SkillResult, TellRequest, Trajectory, Verdict, VerifyResult,
)

M = TypeVar("M", bound=BaseModel)


class ToolCallError(RuntimeError):
    """组件返回了错误（参数不合法、补丁冲突等）。"""


class ComponentUnavailableError(RuntimeError):
    """组件进程退出、连接断开或不响应，调用没有完成。"""


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
        try:
            result = await self._client.call_tool(tool, {k: _dump(v) for k, v in args.items() if v is not None})
        except Exception as e:  # noqa: BLE001  传输层故障：说清楚是哪个组件、哪个调用
            who = f"{self.info.name}:{self.info.version}" if self.info else "组件"
            raise ComponentUnavailableError(f"{who} 在 {tool} 时不可用（进程退出、连接断开或超时）："
                                            f"{type(e).__name__}: {e}") from e
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            raise ToolCallError(f"{tool}: {text}")
        return model.model_validate(result.structured_content)

    # 每种组件类型一个门面，方法名和本地基类一致，都是 async
    @property
    def env(self) -> "RemoteEnvironment":
        return RemoteEnvironment(self)

    @property
    def agent(self) -> "RemoteAgent":
        return RemoteAgent(self)

    @property
    def evidence(self) -> "RemoteEvidence":
        return RemoteEvidence(self)

    @property
    def judge(self) -> "RemoteJudge":
        return RemoteJudge(self)

    @property
    def attribution(self) -> "RemoteAttributor":
        return RemoteAttributor(self)

    @property
    def evolvable(self) -> "RemoteEvolvable":
        return RemoteEvolvable(self)

    @property
    def memory(self) -> "RemoteMemory":
        return RemoteMemory(self)

    @property
    def skill(self) -> "RemoteSkills":
        return RemoteSkills(self)

    @property
    def guard(self) -> "RemoteGuard":
        return RemoteGuard(self)

    @property
    def proposer(self) -> "RemoteProposer":
        return RemoteProposer(self)

    @property
    def algorithm(self) -> "RemoteAlgorithm":
        return RemoteAlgorithm(self)

    @property
    def gate(self) -> "RemoteGate":
        return RemoteGate(self)

    @property
    def lineage(self) -> "RemoteLineage":
        return RemoteLineage(self)


class _Facade:
    def __init__(self, rc: RemoteComponent):
        self._rc = rc

    @property
    def kinds(self) -> list[str]:
        return self._rc.kinds


class RemoteEnvironment(_Facade):
    async def tasks(self, split: str, round: int = 0, limit: int | None = None) -> list[Task]:
        res = await self._rc.call("rsi_env_tasks", {"split": split, "round": round, "limit": limit}, TaskBundle)
        return res.tasks


class RemoteAgent(_Facade):
    async def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        return await self._rc.call("rsi_agent_run", {"task": task, "context": context, "snapshot_id": snapshot_id},
                                   Trajectory)


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


class RemoteAttributor(_Facade):
    async def explain(self, episodes: list[Episode]) -> list[Attribution]:
        res = await self._rc.call("rsi_attribution_explain", {"episodes": episodes}, AttributionBundle)
        return res.attributions


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

    async def export(self, snapshot_id: str | None = None) -> Any:
        if "rsi_evolvable_export" not in self._rc.tools:
            return None
        res = await self._rc.call("rsi_evolvable_export", {"snapshot_id": snapshot_id}, ExportResult)
        return res.content


class RemoteMemory(RemoteEvolvable):
    async def retrieve(self, query: str = "", tags: list[str] | None = None, k: int = 5,
                       snapshot_id: str | None = None) -> list[Item]:
        res = await self._rc.call("rsi_memory_retrieve",
                                  {"query": query, "tags": tags, "k": k, "snapshot_id": snapshot_id},
                                  RetrieveResult)
        return res.items


class RemoteSkills(RemoteEvolvable):
    async def invoke(self, skill_id: str, arguments: dict | None = None, snapshot_id: str | None = None) -> SkillResult:
        return await self._rc.call("rsi_skill_invoke",
                                   {"skill_id": skill_id, "arguments": arguments or {}, "snapshot_id": snapshot_id},
                                   SkillResult)


class RemoteGuard(_Facade):
    async def inspect(self, patch: Patch, target: EvolvableInfo | None = None) -> GuardReport:
        return await self._rc.call("rsi_guard_inspect", {"patch": patch, "target": target}, GuardReport)


class RemoteProposer(_Facade):
    async def propose(self, request: ProposalRequest) -> list[Patch]:
        res = await self._rc.call("rsi_proposer_propose", {"request": request}, ProposalBundle)
        return res.patches


class RemoteAlgorithm(_Facade):
    async def ask(self) -> AskResult:
        return await self._rc.call("rsi_algorithm_ask", {}, AskResult)

    async def tell(self, result: TellRequest) -> AlgorithmStatus:
        return await self._rc.call("rsi_algorithm_tell", {"result": result}, AlgorithmStatus)

    async def screen(self, patch: Patch, parent_id: str) -> ScreenResult | None:
        if "rsi_algorithm_screen" not in self._rc.tools:
            return None
        return await self._rc.call("rsi_algorithm_screen", {"patch": patch, "parent_id": parent_id}, ScreenResult)


class RemoteGate(_Facade):
    async def check(self, request: GateRequest) -> GateDecision:
        return await self._rc.call("rsi_gate_check", {"request": request}, GateDecision)


class RemoteLineage(_Facade):
    async def record(self, entry: LineageEntry) -> LineageEntry:
        return await self._rc.call("rsi_lineage_record", {"entry": entry}, LineageEntry)

    async def log(self, target: str | None = None, status: str | None = None,
                  limit: int | None = None) -> list[LineageEntry]:
        res = await self._rc.call("rsi_lineage_log", {"target": target, "status": status, "limit": limit}, LineageLog)
        return res.entries

    async def best(self, metric: str = "holdout", target: str | None = None) -> LineageEntry | None:
        res = await self._rc.call("rsi_lineage_best", {"metric": metric, "target": target}, LineageLog)
        return res.entries[0] if res.entries else None


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
