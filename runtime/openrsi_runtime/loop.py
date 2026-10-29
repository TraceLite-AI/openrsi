"""自我改进主循环。

每一代：
  1. 进化算法选父快照，memory 切过去
  2. agent 在父快照上做 train 题；取证、判别
  3. 归因，提案器根据 train 结果提出补丁
  4. 每个补丁生成一个候选快照，在 holdout 上打分
  5. 准入组件比较候选和父快照，结果写进谱系，回报给算法

硬规则（由本模块保证，不依赖组件自觉）：
  - 交给 agent 的题目去掉了 reference（标准答案）
  - 只有 train 的 episode 会进入归因和提案；holdout 只用来打分
  - env 返回的题目划分不对时直接报错
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable

from openrsi import (ComponentError, Episode, GateRequest, LineageEntry, ProposalRequest, Task, TellRequest,
                     ToolCallError)
from openrsi.types import for_agent

from .config import LoopConfig


class IsolationError(RuntimeError):
    """数据划分被破坏，比如 env 把 holdout 题标成了 train。"""


@dataclass
class Candidate:
    id: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""


@dataclass
class GenerationReport:
    generation: int
    parent_id: str
    parent_metrics: dict[str, float]
    train_score: float
    candidates: list[Candidate] = field(default_factory=list)
    note: str = ""


def _query(task: Task) -> str:
    x = task.input
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return " ".join(str(v) for v in x.values())
    return "" if x is None else str(x)


def _check_split(tasks: list[Task], split: str) -> None:
    wrong = [t.id for t in tasks if t.split != split]
    if wrong:
        raise IsolationError(f"env 在 {split} 里返回了其他划分的题目：{wrong[:5]}")


class Runner:
    def __init__(self, w: Any, loop: LoopConfig, on_generation: Callable[[GenerationReport], None] | None = None,
                 log: Callable[[str], None] = lambda s: None):
        self.w = w
        self.cfg = loop
        self.on_generation = on_generation
        self.log = log

    # ------------------------------------------------------------ 执行与打分

    async def episode(self, task: Task, snapshot_id: str) -> Episode:
        visible = for_agent(task)
        context = await self.w.memory.retrieve(query=_query(visible), tags=visible.tags or None,
                                               k=self.cfg.retrieve_k, snapshot_id=snapshot_id)
        traj = await self.w.agent.run(visible, context, snapshot_id)
        traj.snapshot_id = traj.snapshot_id or snapshot_id
        if not traj.used_items:  # agent 没报告就当作检索到的都用上了
            traj.used_items = [i.id for i in context]
        evidence = await self.w.evidence.collect(task, traj) if self.w.evidence else []
        verdict = await self.w.judge.evaluate(task, traj, evidence)
        return Episode(task=task, trajectory=traj, verdict=verdict, evidence=[*evidence, *verdict.collected])

    async def score(self, tasks: list[Task], snapshot_id: str, prefix: str) -> tuple[dict[str, float], list[Episode]]:
        episodes = [await self.episode(t, snapshot_id) for t in tasks]
        if not episodes:
            return {prefix: 0.0, f"{prefix}_pass": 0.0, f"n_{prefix}": 0.0}, []
        return {
            prefix: fmean(e.verdict.score for e in episodes),
            f"{prefix}_pass": fmean(1.0 if e.verdict.passed else 0.0 for e in episodes),
            f"n_{prefix}": float(len(episodes)),
        }, episodes

    # ------------------------------------------------------------ 主循环

    async def run(self, generations: int | None = None) -> list[GenerationReport]:
        w = self.w
        limit = generations if generations is not None else self.cfg.generations
        info = await w.memory.describe()
        target = info.target

        holdout = await w.env.tasks("holdout", 0, self.cfg.holdout_limit)
        _check_split(holdout, "holdout")
        if not holdout:
            raise IsolationError("env 没有给出 holdout 题，无法做准入")
        self.holdout = holdout  # 整个运行用同一套 holdout，候选之间才可比

        entries = await w.lineage.log(target=target)
        if not entries:
            head = await w.memory.snapshot()
            m, _ = await self.score(holdout, head.id, "holdout")
            entries = [await w.lineage.record(LineageEntry(id=head.id, target=target, parent_ids=head.parent_ids,
                                                           generation=0, status="root", metrics=m))]
            self.log(f"root {head.id}: holdout={m['holdout']:.3f}")
        else:
            self.log(f"从谱系续跑：{len(entries)} 条记录")

        metrics_by_id = await self._seed_algorithm(entries)
        last_gen = max(e.generation for e in entries)
        reports: list[GenerationReport] = []

        while limit is None or len(reports) < limit:
            ask = await w.algorithm.ask()
            if ask.parent_id is None:
                self.log(f"停止：{ask.reason}")
                break
            gen = max(ask.generation, last_gen + 1)
            last_gen = gen
            report = await self._generation(gen, ask.parent_id, info, metrics_by_id)
            reports.append(report)
            if self.on_generation:
                self.on_generation(report)

        best = await w.lineage.best(self.cfg.metric, target)
        if best:
            await w.memory.rollback(best.id)
            self.log(f"memory 切到最佳快照 {best.id}（{self.cfg.metric}={best.metrics.get(self.cfg.metric, 0):.3f}）")
        return reports

    async def _seed_algorithm(self, entries: list[LineageEntry]) -> dict[str, dict[str, float]]:
        """把谱系重放给进化算法，这样中断后能接着跑。"""
        root = next((e for e in entries if e.status == "root"), None)
        if root is None:
            raise RuntimeError("谱系里没有根快照记录，无法续跑")
        for e in entries:
            if e.status in ("root", "accepted"):
                try:
                    await self.w.memory.rollback(e.id)
                except Exception as err:  # noqa: BLE001
                    raise RuntimeError(f"谱系记录了快照 {e.id}，但 memory 里没有它（{err}）。"
                                       "memory 和谱系的状态要一起保存、一起删除。") from err
        await self.w.algorithm.tell(TellRequest(generation=root.generation, candidate_id=root.id, accepted=True,
                                                metrics=root.metrics))
        metrics = {root.id: root.metrics}
        for e in entries:
            if e is root:
                continue
            await self.w.algorithm.tell(TellRequest(generation=e.generation, parent_id=e.parent_ids[0] if e.parent_ids else None,
                                                    candidate_id=e.id, accepted=e.status == "accepted",
                                                    metrics=e.metrics))
            if e.status == "accepted":
                metrics[e.id] = e.metrics
        return metrics

    async def _generation(self, gen: int, parent: str, info: Any, metrics_by_id: dict) -> GenerationReport:
        w = self.w
        await w.memory.rollback(parent)
        train = await w.env.tasks("train", gen, self.cfg.train_limit)
        _check_split(train, "train")
        episodes = [await self.episode(t, parent) for t in train]
        report = GenerationReport(generation=gen, parent_id=parent, parent_metrics=metrics_by_id[parent],
                                  train_score=fmean(e.verdict.score for e in episodes) if episodes else 0.0)

        attributions = await w.attribution.explain(episodes) if w.attribution and episodes else []
        state = await w.memory.export(parent)
        patches = await w.proposer.propose(ProposalRequest(
            target=info, parent_snapshot=parent, episodes=episodes, attributions=attributions, state=state,
            max_patches=self.cfg.max_patches))

        if not patches:
            report.note = "提案器没有提出修改"
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
            return report

        for patch in patches[: self.cfg.max_patches]:
            await w.memory.rollback(parent)
            patch = patch.model_copy(update={"target": info.target, "base_snapshot": parent})
            try:
                snap = await w.memory.apply(patch)
            except (ToolCallError, ComponentError, ValueError, KeyError) as e:
                report.candidates.append(Candidate(id=None, reason=f"补丁被拒：{e}"))
                await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
                continue

            m, _ = await self.score(self.holdout, snap.id, "holdout")
            m["train_of_parent"] = report.train_score
            decision = await w.gate.check(GateRequest(candidate_id=snap.id, baseline=metrics_by_id[parent], candidate=m))
            status = "accepted" if decision.accepted else "rejected"
            await w.lineage.record(LineageEntry(id=snap.id, target=info.target, parent_ids=[parent], generation=gen,
                                                status=status, metrics=m, decision=decision, patch=patch))
            if decision.accepted:
                metrics_by_id[snap.id] = m
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=snap.id,
                                               accepted=decision.accepted, metrics=m))
            report.candidates.append(Candidate(id=snap.id, metrics=m, accepted=decision.accepted,
                                               reason=decision.reason or ""))
        await w.memory.rollback(parent)
        return report
