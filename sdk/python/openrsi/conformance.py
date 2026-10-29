"""一致性测试：通过 MCP 检查任意组件是否符合协议，与组件用什么语言实现无关。

    python -m openrsi.conformance -- python -m openrsi_judge --rubric rubric.yaml
    python -m openrsi.conformance --url http://localhost:8000/mcp

注意：evolvable 检查会 apply 一个探针补丁再回滚，请对一个临时实例运行。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .client import RemoteComponent, ToolCallError, connect
from .protocol import required_tools
from .types import PROTOCOL_VERSION, AnchorExample, Evidence, Patch, Step, Task, Trajectory

FIXTURE_TASK = Task(id="conformance-task", input={"question": "fix the failing test", "expected": "done"},
                    split="train", tags=["conformance"])

FIXTURE_TRAJECTORY = Trajectory(
    id="conformance-traj", task_id="conformance-task",
    steps=[
        Step(id="s0", type="message", role="user", content="fix the failing test"),
        Step(id="s1", type="tool_call", role="assistant", name="run_tests", arguments={"cmd": "pytest"}, call_id="c1"),
        Step(id="s2", type="tool_result", role="tool", name="run_tests", call_id="c1", content="2 passed in 0.10s"),
        Step(id="s3", type="message", role="assistant", content="done"),
    ],
    output="done",
)


@dataclass
class CheckResult:
    kind: str
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Report:
    component: str
    kinds: list[str]
    results: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def text(self) -> str:
        lines = [f"{self.component}  kinds={','.join(self.kinds) or '-'}"]
        for r in self.results:
            mark = "PASS" if r.ok else "FAIL"
            lines.append(f"  [{mark}] {r.kind:<9} {r.name}{'  — ' + r.detail if r.detail else ''}")
        passed = sum(r.ok for r in self.results)
        lines.append(f"{passed}/{len(self.results)} checks passed")
        return "\n".join(lines)


async def _run(report: Report, kind: str, name: str, fn: Callable[[], Awaitable[Any]]) -> Any:
    try:
        out = await fn()
        report.results.append(CheckResult(kind, name, True))
        return out
    except AssertionError as e:
        report.results.append(CheckResult(kind, name, False, str(e) or "assertion failed"))
    except Exception as e:  # noqa: BLE001  一致性测试要把任何异常都记成失败
        report.results.append(CheckResult(kind, name, False, f"{type(e).__name__}: {e}"))
    return None


async def check_component(rc: RemoteComponent) -> Report:
    info = rc.info
    report = Report(component=f"{info.name}:{info.version}" if info else "(no rsi_component_info)", kinds=rc.kinds)

    async def _info():
        assert info is not None, "缺少 rsi_component_info"
        assert info.protocol.split("/")[0] == PROTOCOL_VERSION.split("/")[0], f"协议不匹配: {info.protocol}"
        assert info.kinds, "kinds 为空"
        for k in info.kinds:
            missing = required_tools(k) - set(rc.tools)
            assert not missing, f"声明了 {k} 但缺少工具 {sorted(missing)}"

    await _run(report, "component", "info 与声明的工具一致", _info)

    evidence: list[Evidence] = []
    if "evidence" in rc.kinds:
        evidence = await _check_evidence(rc, report) or []
    if "judge" in rc.kinds:
        await _check_judge(rc, report, evidence)
    if "evolvable" in rc.kinds:
        await _check_evolvable(rc, report)
    return report


async def _check_evidence(rc: RemoteComponent, report: Report) -> list[Evidence] | None:
    async def collect():
        ev = await rc.evidence.collect(FIXTURE_TASK, FIXTURE_TRAJECTORY)
        ids = [e.id for e in ev]
        assert len(ids) == len(set(ids)), "证据 id 重复"
        for e in ev:
            assert e.verify(), f"{e.id} 的 sha256 与内容不符"
        return ev

    ev = await _run(report, "evidence", "collect 返回合法且自洽的证据", collect)
    if not ev:
        return ev

    async def deterministic():
        again = await rc.evidence.collect(FIXTURE_TASK, FIXTURE_TRAJECTORY)
        assert [e.id for e in again] == [e.id for e in ev], "同样输入两次收集的证据 id 不同"

    async def verify_ok():
        assert (await rc.evidence.verify(ev[0])).valid, "未篡改的证据被判为无效"

    async def verify_tampered():
        bad = ev[0].model_copy(update={"content": {"tampered": True}})
        assert not (await rc.evidence.verify(bad)).valid, "篡改过的证据被判为有效"

    await _run(report, "evidence", "收集结果可复现（id 稳定）", deterministic)
    await _run(report, "evidence", "verify 接受原始证据", verify_ok)
    await _run(report, "evidence", "verify 拒绝篡改的证据", verify_tampered)
    return ev


async def _check_judge(rc: RemoteComponent, report: Report, evidence: list[Evidence]) -> None:
    async def evaluate():
        v = await rc.judge.evaluate(FIXTURE_TASK, FIXTURE_TRAJECTORY, evidence)
        known = {e.id for e in evidence} | {e.id for e in v.collected}
        for e in v.collected:
            assert e.verify(), f"判别器补充的证据 {e.id} 哈希不符"
        for r in v.rubric:
            stray = set(r.evidence_ids) - known
            assert not stray, f"rubric {r.id} 引用了不存在的证据 {sorted(stray)}"
            if r.result in ("pass", "fail"):
                assert r.evidence_ids or r.note, f"rubric {r.id} 给出结论却既无证据也无说明"
        assert set(v.evidence_ids) <= known, "verdict.evidence_ids 含有未知证据"
        return v

    await _run(report, "judge", "evaluate 的每条结论都引用已知证据", evaluate)

    if "rsi_judge_calibrate" in rc.tools:
        async def calibrate():
            anchor = [AnchorExample(task=FIXTURE_TASK, trajectory=FIXTURE_TRAJECTORY, evidence=evidence, label_passed=True)]
            rep = await rc.judge.calibrate(anchor)
            assert rep.n == 1 and 0.0 <= rep.agreement <= 1.0, f"报告不合理: {rep}"

        await _run(report, "judge", "calibrate 返回合理的报告", calibrate)


async def _check_evolvable(rc: RemoteComponent, report: Report) -> None:
    ev = rc.evolvable
    info = await _run(report, "evolvable", "describe", ev.describe)
    if info is None:
        return
    if not info.example_ops:
        report.results.append(CheckResult("evolvable", "提供 example_ops", False, "describe 未给出 example_ops，无法继续检查"))
        return
    s0 = await _run(report, "evolvable", "snapshot", ev.snapshot)
    if s0 is None:
        return

    async def apply():
        s1 = await ev.apply(Patch(target=info.target, base_snapshot=s0.id, ops=info.example_ops,
                                  rationale="openrsi conformance probe"))
        assert s0.id in s1.parent_ids, "新快照的 parent_ids 不含原快照"
        assert (await ev.snapshot()).id == s1.id, "apply 后当前快照没有更新"
        return s1

    s1 = await _run(report, "evolvable", "apply 生成子快照并成为当前", apply)
    if s1 is None:
        return

    if "memory" in rc.kinds:
        async def retrieve_new():
            probe = next((op.get("item", {}).get("id") for op in info.example_ops if op.get("op") == "add"), None)
            items = await rc.memory.retrieve(tags=None, query="openrsi conformance probe", k=50)
            assert probe is None or probe in [i.id for i in items], "apply 后检索不到新增条目"

        await _run(report, "memory", "retrieve 能看到新快照的内容", retrieve_new)

    async def diff():
        d = await ev.diff(s0.id, s1.id)
        assert d.ops, "两个不同快照的 diff 为空"

    await _run(report, "evolvable", "diff 反映修改", diff)

    async def rollback():
        back = await ev.rollback(s0.id)
        assert back.id == s0.id and back.content_hash == s0.content_hash, "回滚后内容与原快照不一致"

    await _run(report, "evolvable", "rollback 恢复原快照", rollback)

    async def stale():
        try:
            await ev.apply(Patch(target=info.target, base_snapshot=s1.id, ops=info.example_ops))
        except ToolCallError:
            return
        raise AssertionError("base_snapshot 不是当前快照时应拒绝补丁")

    await _run(report, "evolvable", "拒绝基于过期快照的补丁", stale)


async def check(target: Any, *, env: dict[str, str] | None = None) -> Report:
    async with connect(target, env=env) as rc:
        return await check_component(rc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openrsi-conformance")
    ap.add_argument("--url", help="Streamable HTTP 地址")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- 之后是启动组件的命令")
    args = ap.parse_args(argv)
    cmd = [c for c in args.command if c != "--"]
    if not args.url and not cmd:
        ap.error("需要 --url 或 -- <命令>")
    try:
        # 命令行里启动组件，就像在 shell 里启动一样，继承当前环境变量
        report = asyncio.run(check(args.url or cmd, env=None if args.url else dict(os.environ)))
    except Exception as e:  # noqa: BLE001
        print(f"无法连接组件：{type(e).__name__}。检查命令能否单独运行：{' '.join(cmd) or args.url}", file=sys.stderr)
        return 2
    print(report.text())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
