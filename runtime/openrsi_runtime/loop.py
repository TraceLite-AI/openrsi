"""自我改进主循环。

每一代：
  1. 进化算法选一个父版本，所有可进化体切过去
  2. agent 在父版本上做 train 题；取证、判别
  3. 归因，提案器根据 train 结果提出补丁（每个补丁指定改哪个可进化体）
  4. 护栏检查补丁内容；通过的补丁生成候选版本，在 holdout 上打分
  5. 准入组件比较候选和父版本，结果写进谱系，回报给算法

"版本"是每个可进化体各取一个快照的组合（genome）。只有一个可进化体时，版本 id 就是它的快照 id，
和 0.2 完全一致；有多个时是组合的哈希，谱系里的 snapshots 字段记录每个可进化体对应的快照。

硬规则（由本模块保证，不依赖组件自觉）：
  - 交给 agent 的题目去掉了 reference（标准答案）
  - 只有 train 的 episode 会进入归因和提案；holdout 只用来打分
  - env 返回的题目划分不对时直接报错
  - 配置了护栏时，被拦下的补丁不会生效
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable

from openrsi import (ComponentError, Episode, EvolvableInfo, GateCheck, GateDecision, GateRequest, Item, LineageEntry,
                     Patch, ProposalRequest, Task, TellRequest, ToolCallError, content_hash)
from openrsi.types import canonical_json, for_agent

from .config import LoopConfig

Genome = dict[str, str]  # 可进化体 target → 快照 id


class IsolationError(RuntimeError):
    """数据划分被破坏，比如 env 把 holdout 题标成了 train。"""


@dataclass
class Candidate:
    id: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ""
    target: str | None = None


@dataclass
class GenerationReport:
    generation: int
    parent_id: str
    parent_metrics: dict[str, float]
    train_score: float
    candidates: list[Candidate] = field(default_factory=list)
    note: str = ""


@dataclass
class _Target:
    facade: Any
    info: EvolvableInfo
    kinds: list[str]


def genome_id(genome: Genome) -> str:
    if len(genome) == 1:
        return next(iter(genome.values()))
    return "cfg_" + hashlib.sha256(canonical_json(genome).encode()).hexdigest()[:16]


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
        self.targets: dict[str, _Target] = {}
        self.genomes: dict[str, Genome] = {}
        self.metrics: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------ 可进化体

    async def _load_targets(self) -> None:
        for facade in self.w.evolvables.values():
            info = await facade.describe()
            if info.target in self.targets:
                raise ValueError(f"两个可进化体用了同一个 target 名 {info.target}")
            self.targets[info.target] = _Target(facade, info, list(facade.kinds))
        if not self.targets:
            raise ValueError("没有可进化体")
        self.lineage_target = "+".join(sorted(self.targets))

    async def checkout(self, genome: Genome) -> None:
        for t, sid in genome.items():
            await self.targets[t].facade.rollback(sid)

    def _genome_of(self, entry: LineageEntry) -> Genome:
        if entry.snapshots:
            return dict(entry.snapshots)
        if len(self.targets) == 1:  # 0.2 的谱系没有 snapshots 字段
            return {next(iter(self.targets)): entry.id}
        raise RuntimeError(f"谱系记录 {entry.id} 没有 snapshots，无法对应到多个可进化体")

    # ------------------------------------------------------------ 执行与打分

    async def context(self, task: Task, genome: Genome) -> list[Item]:
        """agent 能看到的内容：memory 按题目检索，其他可进化体（prompt、skill）整个给。"""
        out: list[Item] = []
        for t, tgt in self.targets.items():
            if "memory" in tgt.kinds:
                out += await tgt.facade.retrieve(query=_query(task), tags=task.tags or None, k=self.cfg.retrieve_k,
                                                 snapshot_id=genome[t])
            else:
                exported = await tgt.facade.export(genome[t])
                if isinstance(exported, list):
                    out += [Item.model_validate(d) for d in exported if isinstance(d, dict) and "id" in d]
        return out

    async def episode(self, task: Task, genome: Genome) -> Episode:
        visible = for_agent(task)
        context = await self.context(visible, genome)
        gid = genome_id(genome)
        traj = await self.w.agent.run(visible, context, gid)
        traj.snapshot_id = traj.snapshot_id or gid
        if not traj.used_items:  # agent 没报告就当作给它的都用上了
            traj.used_items = [i.id for i in context]
        evidence = await self.w.evidence.collect(task, traj) if self.w.evidence else []
        verdict = await self.w.judge.evaluate(task, traj, evidence)
        return Episode(task=task, trajectory=traj, verdict=verdict, evidence=[*evidence, *verdict.collected])

    async def score(self, tasks: list[Task], genome: Genome, prefix: str) -> tuple[dict[str, float], list[Episode]]:
        episodes = [await self.episode(t, genome) for t in tasks]
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
        await self._load_targets()

        holdout = await w.env.tasks("holdout", 0, self.cfg.holdout_limit)
        _check_split(holdout, "holdout")
        if not holdout:
            raise IsolationError("env 没有给出 holdout 题，无法做准入")
        self.holdout = holdout  # 整个运行用同一套 holdout，候选之间才可比

        entries = await w.lineage.log(target=self.lineage_target)
        if not entries:
            genome = {t: (await tgt.facade.snapshot()).id for t, tgt in self.targets.items()}
            m, _ = await self.score(holdout, genome, "holdout")
            root = LineageEntry(id=genome_id(genome), target=self.lineage_target, generation=0, status="root",
                                metrics=m, snapshots=genome)
            entries = [await w.lineage.record(root)]
            self.log(f"root {root.id}: holdout={m['holdout']:.3f}")
        else:
            self.log(f"从谱系续跑：{len(entries)} 条记录")

        await self._seed_algorithm(entries)
        last_gen = max(e.generation for e in entries)
        reports: list[GenerationReport] = []

        while limit is None or len(reports) < limit:
            ask = await w.algorithm.ask()
            if ask.parent_id is None:
                self.log(f"停止：{ask.reason}")
                break
            gen = max(ask.generation, last_gen + 1)
            last_gen = gen
            report = await self._generation(gen, ask.parent_id)
            reports.append(report)
            if self.on_generation:
                self.on_generation(report)

        best = await w.lineage.best(self.cfg.metric, self.lineage_target)
        if best:
            await self.checkout(self._genome_of(best))
            self.log(f"切到最佳版本 {best.id}（{self.cfg.metric}={best.metrics.get(self.cfg.metric, 0):.3f}）")
        return reports

    async def _seed_algorithm(self, entries: list[LineageEntry]) -> None:
        """把谱系重放给进化算法，这样中断后能接着跑。"""
        root = next((e for e in entries if e.status == "root"), None)
        if root is None:
            raise RuntimeError("谱系里没有根版本记录，无法续跑")
        for e in entries:
            if e.status in ("root", "accepted"):
                genome = self._genome_of(e)
                try:
                    await self.checkout(genome)
                except Exception as err:  # noqa: BLE001
                    raise RuntimeError(f"谱系记录了版本 {e.id}，但可进化体里没有它的快照（{err}）。"
                                       "可进化体和谱系的状态要一起保存、一起删除。") from err
                self.genomes[e.id], self.metrics[e.id] = genome, e.metrics
        await self.w.algorithm.tell(TellRequest(generation=root.generation, candidate_id=root.id, accepted=True,
                                                metrics=root.metrics))
        for e in entries:
            if e is not root:
                await self.w.algorithm.tell(TellRequest(
                    generation=e.generation, parent_id=e.parent_ids[0] if e.parent_ids else None,
                    candidate_id=e.id, accepted=e.status == "accepted", metrics=e.metrics))

    async def _generation(self, gen: int, parent: str) -> GenerationReport:
        w = self.w
        parent_genome = self.genomes[parent]
        await self.checkout(parent_genome)
        train = await w.env.tasks("train", gen, self.cfg.train_limit)
        _check_split(train, "train")
        episodes = [await self.episode(t, parent_genome) for t in train]
        report = GenerationReport(generation=gen, parent_id=parent, parent_metrics=self.metrics[parent],
                                  train_score=fmean(e.verdict.score for e in episodes) if episodes else 0.0)

        attributions = await w.attribution.explain(episodes) if w.attribution and episodes else []
        states = {t: await tgt.facade.export(parent_genome[t]) for t, tgt in self.targets.items()}
        primary = next(iter(self.targets))
        patches = await w.proposer.propose(ProposalRequest(
            target=self.targets[primary].info, targets=[tgt.info for tgt in self.targets.values()],
            parent_snapshot=parent, episodes=episodes, attributions=attributions,
            state=states[primary], states=states, max_patches=self.cfg.max_patches))

        if not patches:
            report.note = "提案器没有提出修改"
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
            return report

        for patch in patches[: self.cfg.max_patches]:
            report.candidates.append(await self._candidate(gen, parent, parent_genome, patch, report.train_score))
        await self.checkout(parent_genome)
        return report

    async def _candidate(self, gen: int, parent: str, parent_genome: Genome, patch: Patch,
                         train_score: float) -> Candidate:
        w = self.w
        if len(self.targets) == 1:
            t = next(iter(self.targets))
        elif patch.target in self.targets:
            t = patch.target
        else:
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
            return Candidate(id=None, reason=f"补丁目标 {patch.target} 不存在，可选：{sorted(self.targets)}")
        tgt = self.targets[t]
        await self.checkout(parent_genome)
        patch = patch.model_copy(update={"target": tgt.info.target, "base_snapshot": parent_genome[t]})

        guard_checks: list[GateCheck] = []
        if w.guard:
            g = await w.guard.inspect(patch, tgt.info)
            guard_checks = [GateCheck(name=f"guard:{f.rule}", passed=f.severity != "block",
                                      detail=f"[{f.item_id or '-'}] {f.message}") for f in g.findings]
            if not g.allowed:
                blocked = "blocked_" + content_hash(patch.model_dump(mode="json"))[:16]
                rules = sorted({f.rule for f in g.findings if f.severity == "block"})
                decision = GateDecision(candidate_id=blocked, accepted=False, checks=guard_checks,
                                        reason=f"护栏拦截：{', '.join(rules)}")
                await w.lineage.record(LineageEntry(id=blocked, target=self.lineage_target, parent_ids=[parent],
                                                    generation=gen, status="rejected", decision=decision, patch=patch))
                await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=blocked,
                                                   accepted=False))
                return Candidate(id=blocked, reason=decision.reason, target=t)

        try:
            snap = await tgt.facade.apply(patch)
        except (ToolCallError, ComponentError, ValueError, KeyError) as e:
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
            return Candidate(id=None, reason=f"补丁被拒：{e}", target=t)

        genome = {**parent_genome, t: snap.id}
        cid = genome_id(genome)
        m, _ = await self.score(self.holdout, genome, "holdout")
        m["train_of_parent"] = train_score
        decision = await w.gate.check(GateRequest(candidate_id=cid, baseline=self.metrics[parent], candidate=m))
        decision.checks += guard_checks  # 护栏的警告也留档
        status = "accepted" if decision.accepted else "rejected"
        await w.lineage.record(LineageEntry(id=cid, target=self.lineage_target, parent_ids=[parent], generation=gen,
                                            status=status, metrics=m, decision=decision, patch=patch,
                                            snapshots=genome))
        if decision.accepted:
            self.genomes[cid], self.metrics[cid] = genome, m
        await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=cid,
                                           accepted=decision.accepted, metrics=m))
        return Candidate(id=cid, metrics=m, accepted=decision.accepted, reason=decision.reason or "", target=t)
