"""走 MCP 的端到端测试：进程内连接、真实 stdio 子进程、一致性测试、清单、spec 同步。"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters

from openrsi import (Evidence, Judge, Patch, Task, ToolCallError, Trajectory, Verdict, RubricResult, build_server,
                     connect, load_manifest)
from openrsi.conformance import FIXTURE_TASK, FIXTURE_TRAJECTORY, check
from openrsi.export_spec import render
from openrsi_judge import EvidenceJudge, Rubric
from openrsi_memory import MemoryEvolvable

ROOT = Path(__file__).resolve().parents[1]
RUBRIC = Rubric.model_validate({"items": [
    {"id": "tp", "check": "tests_pass", "required": True},
    {"id": "eq", "check": "output_equals"},
]})


def run(coro):
    return asyncio.run(coro)


def test_judge_over_mcp_matches_in_process():
    judge = EvidenceJudge(RUBRIC)
    local = judge.evaluate(FIXTURE_TASK, FIXTURE_TRAJECTORY, judge.collect(FIXTURE_TASK, FIXTURE_TRAJECTORY))

    async def remote():
        async with connect(build_server(judge)) as c:
            assert set(c.kinds) == {"evidence", "judge"}
            ev = await c.evidence.collect(FIXTURE_TASK, FIXTURE_TRAJECTORY)
            return await c.judge.evaluate(FIXTURE_TASK, FIXTURE_TRAJECTORY, ev)

    v = run(remote())
    assert v.passed == local.passed and v.score == local.score
    assert [r.model_dump() for r in v.rubric] == [r.model_dump() for r in local.rubric]


def test_memory_over_mcp_and_errors():
    async def go():
        async with connect(build_server(MemoryEvolvable())) as c:
            m = c.memory
            s0 = await m.snapshot()
            s1 = await m.apply(Patch(target="memory", base_snapshot=s0.id,
                                     ops=[{"op": "add", "item": {"id": "a", "text": "remote item"}}]))
            assert [i.id for i in await m.retrieve("remote")] == ["a"]
            with pytest.raises(ToolCallError, match="当前是"):
                await m.apply(Patch(target="memory", base_snapshot=s0.id, ops=[]))
            with pytest.raises(ToolCallError):
                await c.call("rsi_judge_evaluate", {}, Verdict)  # 不存在的工具
            await m.rollback(s0.id)
            assert (await m.retrieve("remote")) == []
            return s1

    assert run(go()).parent_ids


def test_conformance_passes_for_official_components():
    for comp in (EvidenceJudge(RUBRIC), MemoryEvolvable()):
        report = run(check(build_server(comp)))
        assert report.ok, report.text()


class LyingJudge(Judge):
    """引用不存在的证据：一致性测试必须抓住。"""
    name, version = "test/lying-judge", "0.0.1"

    def evaluate(self, task: Task, trajectory: Trajectory, evidence: list[Evidence]) -> Verdict:
        return Verdict(score=1.0, passed=True, judge_ref=self.ref,
                       rubric=[RubricResult(id="trust_me", result="pass", evidence_ids=["ev_made_up"])])


def test_conformance_catches_bad_judge():
    report = run(check(build_server(LyingJudge())))
    assert not report.ok
    assert any("ev_made_up" in r.detail for r in report.results if not r.ok)


def test_conformance_over_real_stdio_subprocess(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        str(ROOT / p) for p in ("sdk/python", "components/judge", "components/memory"))}
    params = StdioServerParameters(command=sys.executable, args=["-m", "openrsi_memory", "--path", str(tmp_path)],
                                   env=env)
    report = run(check(params))
    assert report.ok, report.text()
    assert (tmp_path / "HEAD").exists()


@pytest.mark.parametrize("component, factory", [
    ("judge", lambda: EvidenceJudge(RUBRIC)),
    ("memory", MemoryEvolvable),
])
def test_manifest_matches_served_tools(component, factory):
    manifest = load_manifest(ROOT / "components" / component / "rsi-component.yaml")

    async def served():
        async with connect(build_server(factory())) as c:
            return c.info

    info = run(served())
    assert manifest.name == info.name and manifest.version == info.version
    assert set(manifest.implements) == set(info.tools)
    assert set(manifest.kinds) == set(info.kinds)


def test_spec_files_are_up_to_date():
    stale = [rel for rel, text in render().items()
             if not (ROOT / "spec" / rel).exists() or (ROOT / "spec" / rel).read_text(encoding="utf-8") != text]
    assert not stale, f"spec 过期，运行 python -m openrsi.export_spec spec 重新生成：{stale}"
