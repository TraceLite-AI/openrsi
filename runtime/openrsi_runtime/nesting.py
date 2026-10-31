"""循环即 agent（同事文档的 P11）：把一整轮内层 OpenRSI 运行包装成外层的一次"做题"。

    LoopAgent   每道外层题都在全新的目录里跑一次内层循环（固定代数，可选 agent 调用预算），
                输出内层的提升：最佳考试分 − 根版本考试分
    GainJudge   按内层提升给外层打分

外层的可进化体里放内层配置的覆盖项（Item 的 kind 是 loop_config，data = {path, value}）。这样外层的
12 个维度原样复用：外层提案器改内层的算法、提案器、归因方法……外层准入看"同样预算下内层提升多少"。

不动点：只有 OVERRIDABLE 里的路径能被外层改。内层的环境、判别器、取证、准入、护栏、谱系、记录、环境格，
以及预算（代数、调用次数）都不能改；改了这道题直接判失败，原因写进轨迹。外层题目可以通过
task.input.env_with 指定内层环境的构造参数（这是外层出题，不是被进化的东西）。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from openrsi import Agent, Evidence, Item, Judge, Step, Task, Trajectory, Verdict, content_hash

OVERRIDABLE = ("components.algorithm", "components.proposer", "components.attribution", "loop.max_patches",
               "loop.retrieve_k", "loop.counterfactual_reruns", "loop.meta_every", "loop.meta_min_gain")


class BudgetExhausted(RuntimeError):
    pass


class _Counting:
    def __init__(self, inner: Any, budget: int | None):
        self.inner, self.budget, self.calls = inner, budget, 0
        self.kinds = getattr(inner, "kinds", ["agent"])

    async def run(self, *a: Any, **k: Any) -> Any:
        if self.budget is not None and self.calls >= self.budget:
            raise BudgetExhausted(f"内层预算 {self.budget} 次 agent 调用已用完")
        self.calls += 1
        return await self.inner.run(*a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _set(data: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def overrides_from(context: list[Item]) -> list[tuple[str, Any]]:
    out = []
    for item in sorted((i for i in context if i.kind == "loop_config"), key=lambda i: i.id):
        path = item.data.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"内层配置覆盖项 {item.id} 缺少 data.path")
        if not any(path == p or path.startswith(p + ".") for p in OVERRIDABLE):
            raise ValueError(f"{path} 是不动点，外层不能改（可改：{', '.join(OVERRIDABLE)}）")
        out.append((path, item.data.get("value")))
    return out


class LoopAgent(Agent):
    """with:
    template      内层的 openrsi.yaml（路径）
    generations   每次内层运行的代数（预算，外层不能改）
    max_agent_calls  可选：内层 agent 调用次数上限
    workdir       内层运行的工作目录（默认 .openrsi/nested）
    """

    name = "openrsi/loop-agent"
    version = "0.4.3"

    def __init__(self, template: str, generations: int = 3, max_agent_calls: int | None = None,
                 workdir: str = ".openrsi/nested"):
        self.template = Path(template).resolve()
        self.generations, self.max_agent_calls = generations, max_agent_calls
        self.workdir = Path(workdir).resolve()

    def _config(self, task: Task, overrides: list[tuple[str, Any]]) -> dict[str, Any]:
        data = yaml.safe_load(self.template.read_text(encoding="utf-8"))
        data = copy.deepcopy(data)
        env_with = (task.input or {}).get("env_with") if isinstance(task.input, dict) else None
        if env_with:
            data["components"]["env"]["with"] = {**data["components"]["env"].get("with", {}), **env_with}
        for path, value in overrides:
            _set(data, path, value)
        data.setdefault("loop", {})["generations"] = self.generations
        return data

    async def _inner(self, cfg_path: Path) -> dict[str, Any]:
        from .config import load_config
        from .loop import Runner
        from .wiring import Wiring

        cfg = load_config(cfg_path)
        async with Wiring(cfg) as w:
            counter = _Counting(w.components["agent"], self.max_agent_calls)
            w.components["agent"] = counter
            runner, stopped = Runner(w, cfg.loop), "完成"
            try:
                await runner.run(self.generations)
            except BudgetExhausted as e:
                stopped = str(e)
            log = await w.lineage.log()
        root = next(e for e in log if e.status == "root")
        best = max((e for e in log if e.status in ("root", "accepted")), key=lambda e: e.metrics.get("holdout", 0.0))
        return {"root": root.metrics.get("holdout", 0.0), "best": best.metrics.get("holdout", 0.0),
                "generations": max(e.generation for e in log), "agent_calls": counter.calls, "stopped": stopped,
                "accepted": sum(1 for e in log if e.status == "accepted")}

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        tid = f"traj-{task.id}-{snapshot_id}"
        try:
            overrides = overrides_from(context)
        except ValueError as e:
            return Trajectory(id=tid, task_id=task.id, snapshot_id=snapshot_id, output=None,
                              steps=[Step(id="s0", type="observation", content=str(e), is_error=True)],
                              metadata={"error": str(e)})
        data = self._config(task, overrides)
        key = hashlib.sha256(json.dumps([task.id, snapshot_id, data], sort_keys=True, default=str).encode()).hexdigest()
        run_dir = self.workdir / key[:16]
        if run_dir.exists():
            shutil.rmtree(run_dir)  # 每次都从全新的状态开始，内层之间不串
        run_dir.mkdir(parents=True)
        (run_dir / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        with concurrent.futures.ThreadPoolExecutor(1) as ex:  # 外层可能已经在事件循环里，内层放到独立线程
            result = ex.submit(asyncio.run, self._inner(run_dir / "openrsi.yaml")).result()
        gain = result["best"] - result["root"]
        return Trajectory(id=tid, task_id=task.id, snapshot_id=snapshot_id, output=round(gain, 6),
                          steps=[Step(id="s0", type="observation",
                                      content=f"内层 {result['generations']} 代，接受 {result['accepted']} 个，"
                                              f"考试分 {result['root']:.3f} → {result['best']:.3f}（{result['stopped']}）")],
                          metadata={"inner": result, "overrides": [p for p, _ in overrides],
                                    "inner_config_hash": content_hash(data)})


class GainJudge(Judge):
    """按内层提升打分：score = 提升 / scale（截到 0–1），提升 ≥ min_gain 算通过。内层出错（没有输出）算 0。"""

    name = "openrsi/gain-judge"
    version = "0.4.3"

    def __init__(self, scale: float = 1.0, min_gain: float = 0.0):
        self.scale, self.min_gain = scale, min_gain

    def evaluate(self, task: Task, trajectory: Trajectory, evidence: list[Evidence] | None = None) -> Verdict:
        gain = trajectory.output if isinstance(trajectory.output, (int, float)) else None
        ev = Evidence.make("inner_result", {"gain": gain, "inner": trajectory.metadata.get("inner"),
                                            "error": trajectory.metadata.get("error")},
                           f"trajectory:{trajectory.id}:output", self.ref)
        score = 0.0 if gain is None else max(0.0, min(1.0, gain / self.scale))
        passed = gain is not None and gain >= self.min_gain and gain > 0
        error = trajectory.metadata.get("error")
        return Verdict(passed=passed, score=score, judge_ref=self.ref, evidence_ids=[ev.id], collected=[ev],
                       notes=[f"内层提升 {gain:+.4f}" if gain is not None else f"内层没有结果：{error}"])


__all__ = ["GainJudge", "LoopAgent", "OVERRIDABLE", "overrides_from"]
