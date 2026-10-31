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
  - 交给 agent 的题目去掉了 reference（标准答案）；判别器的 rubric 不进 agent 的上下文
  - 只有 train 的 episode 会进入归因和提案；holdout 只用来打分
  - env 返回的题目划分不对时直接报错
  - 配置了护栏时，被拦下的补丁不会生效

执行绑定与不可变记录（P2，见 bindings.py、records.py）：cells 里配置了 binding 时，每道题执行前准备并核对
环境，agent 必须回报所用方案；配置了 records 时，原始记录按内容寻址存档，续跑时运行清单必须一致。

环境格模式（编排文件写了 cells，见 cells.py）：取题经 CellRegistry，逐题分数的键是 <task_id>@<cell_id>，
谱系的 env_cells 记下格的指纹，续跑时必须一致。

判别器参与进化时（loop.judge_every）：
  - agent 侧和判别器轮流进化，同一代只改一边，分数变化才能归因
  - 判别器的修改会改变 holdout 的打分标准，所以它的准入看 anchor（与人工标注的一致率），不看 holdout
  - 每接受一次判别器的修改，版本进入新的"纪元"（judge_epoch）。progress = 2 × 纪元 + holdout，
    新纪元总是优先：旧纪元的高分是在更宽松的标准下得到的
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable

from openrsi import (AnchorExample, Attribution, AttributionTarget, ComponentError, Episode, Evidence, EvolvableInfo,
                     GateCheck, GateDecision, GateRequest, Item, LineageEntry, Patch, ProposalRequest, Task,
                     TellRequest, ToolCallError, __version__ as OPENRSI_VERSION, content_hash)
from openrsi.types import canonical_json, for_agent

from .cells import CellError, CellRegistry, task_key
from .config import CellsConfig, LoopConfig
from .records import DerivedRecord, EvidenceRecord, RecordError, RecordStore, RunManifest, sanitized_config

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
                 log: Callable[[str], None] = lambda s: None, anchor: list[AnchorExample] | None = None,
                 cells: CellsConfig | None = None):
        self.w = w
        self.cfg = loop
        self.on_generation = on_generation
        self.log = log
        if anchor is None and loop.anchor and getattr(w, "config", None) is not None:
            from .config import load_anchor
            anchor = load_anchor(w.config)
        self.anchor = anchor or []
        if cells is None and getattr(w, "config", None) is not None:
            cells = getattr(w.config, "cells", None)
        self.cells_cfg = cells
        self.registry: CellRegistry | None = None
        self.binding: Any = None
        self._plans: dict[str, Any] = {}
        config = getattr(w, "config", None)
        self.records = (RecordStore(config.base_dir / config.records.path)
                        if config is not None and getattr(config, "records", None) else None)
        self.manifest: RunManifest | None = None
        self.run_id = ""
        self._episode_records: dict[str, str] = {}
        self.judge_target: str | None = None
        self.targets: dict[str, _Target] = {}
        self.genomes: dict[str, Genome] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.task_scores: dict[str, dict[str, float]] = {}
        self.last_task_scores: dict[str, float] = {}

    # ------------------------------------------------------------ 可进化体

    async def _load_targets(self) -> None:
        for facade in self.w.evolvables.values():
            info = await facade.describe()
            if info.target in self.targets:
                raise ValueError(f"两个可进化体用了同一个 target 名 {info.target}")
            self.targets[info.target] = _Target(facade, info, list(facade.kinds))
            if "judge" in facade.kinds:
                self.judge_target = info.target
        if not self.targets:
            raise ValueError("没有可进化体")
        if self.judge_target and not self.cfg.judge_every:
            self.log(f"判别器 {self.judge_target} 是可进化体，但没有配置 loop.judge_every，本次不进化它")
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

    async def context(self, task: Task, genome: Genome, exclude: set[str] | None = None) -> list[Item]:
        """agent 能看到的内容：memory 按题目检索，其他可进化体（prompt、skill）整个给；判别器的 rubric 不给。"""
        out: list[Item] = []
        for t, tgt in self.targets.items():
            if t == self.judge_target:
                continue
            if "memory" in tgt.kinds:
                out += await tgt.facade.retrieve(query=_query(task), tags=task.tags or None, k=self.cfg.retrieve_k,
                                                 snapshot_id=genome[t])
            else:
                exported = await tgt.facade.export(genome[t])
                if isinstance(exported, list):
                    out += [Item.model_validate(d) for d in exported if isinstance(d, dict) and "id" in d]
        return [i for i in out if not exclude or i.id not in exclude]

    async def tasks(self, split: str, round: int, limit: int | None) -> list[Task]:
        if self.registry:
            return await self.registry.tasks(self.w.env, split, round, limit)
        return await self.w.env.tasks(split, round, limit)

    def _key(self, task: Task) -> str:
        return task_key(task) if self.registry else task.id

    @property
    def env_cells(self) -> dict[str, str]:
        return dict(self.registry.fingerprints) if self.registry else {}

    async def episode(self, task: Task, genome: Genome, exclude: set[str] | None = None) -> Episode:
        visible = for_agent(task)
        coords = self.registry.coords_for_agent(task) if self.registry else None
        if coords is not None:
            visible = visible.model_copy(update={"metadata": {**visible.metadata, "env_coords": coords}})
        receipt = None
        if self.binding is not None and task.env_ref:
            plan, receipt = self._bind(task)
            visible = visible.model_copy(update={"metadata": {
                **visible.metadata, "binding": {"env": plan.env, "plan_hash": plan.plan_hash, "image": plan.image}}})
        context = await self.context(visible, genome, exclude)
        gid = genome_id(genome)
        traj = await self.w.agent.run(visible, context, gid)
        traj.snapshot_id = traj.snapshot_id or gid
        if receipt is not None and traj.metadata.get("binding_plan_hash") != receipt.plan_hash:
            raise CellError(f"agent 没有回报执行方案（metadata.binding_plan_hash），说明它不支持执行绑定；"
                            f"严格模式下不能把题交给它（题 {task.id}@{task.env_ref}）")
        if self.registry and task.env_ref and not traj.id.endswith(f"@{task.env_ref}"):
            traj.id = f"{traj.id}@{task.env_ref}"  # 同一道题在不同格里的轨迹要能区分（归因按轨迹 id 对号）
        if not traj.used_items:  # agent 没报告就当作给它的都用上了
            traj.used_items = [i.id for i in context]
        evidence = await self.w.evidence.collect(task, traj) if self.w.evidence else []
        verdict = await self.w.judge.evaluate(task, traj, evidence)
        env = [self.registry.evidence(task)] if self.registry and task.env_ref else []
        if receipt is not None:
            env.append(Evidence.make("exec_receipt", receipt.model_dump(mode="json"), f"runtime:binding:{task.env_ref}",
                                     self.binding.name))
        ep = Episode(task=task, trajectory=traj, verdict=verdict, evidence=[*evidence, *verdict.collected, *env])
        if self.records is not None:
            self._record_episode(ep, receipt)
        return ep

    def _bind(self, task: Task) -> tuple[Any, Any]:
        """执行前准备并核对这道题所在格的环境；不符就停。"""
        cell = self.registry.cells[task.env_ref]
        plan = self._plans.get(cell.id)
        if plan is None:
            plan = self._plans[cell.id] = self.binding.resolve(cell, self.registry.fingerprints[cell.id])
        receipt = self.binding.prepare(plan)
        check = self.binding.verify(plan, receipt, cell)
        if not check.ok:
            raise CellError(f"环境格 {cell.id} 的实际执行环境和声明不符：{'; '.join(check.mismatches)}")
        return plan, receipt

    def _record_episode(self, ep: Episode, receipt: Any) -> None:
        """原始记录只追加：题目、轨迹、证据、回执各存一份，再写一条把它们串起来的 EvidenceRecord；判决作为派生记录。"""
        rs, rid = self.records, self.run_id
        traj_h = rs.put(ep.trajectory, "trajectory", rid)
        rec = EvidenceRecord(run_id=rid, episode_id=ep.trajectory.id, task_digest=rs.put(ep.task, "task", rid),
                             cell_fingerprint=self.registry.fingerprints.get(ep.task.env_ref) if self.registry else None,
                             execution_receipt_hash=rs.put(receipt, "receipt", rid) if receipt is not None else None,
                             raw_trajectory_hash=traj_h, raw_artifact_hashes=[rs.put(e, "evidence", rid) for e in ep.evidence],
                             producer_ref=ep.trajectory.snapshot_id or "")
        rec_h = rs.put(rec, "evidence_record", rid)
        self._episode_records[ep.trajectory.id] = rec_h
        rs.put(DerivedRecord(run_id=rid, what="verdict", source_hashes=[rec_h], method_ref=ep.verdict.judge_ref,
                             output_hash=rs.put(ep.verdict, "verdict", rid)), "derived_record", rid)

    def _record_attributions(self, attributions: list[Attribution]) -> None:
        for a in attributions:
            src = self._episode_records.get(a.episode_id)
            if src is None:
                raise RecordError(f"归因 {a.method} 指向的 episode {a.episode_id} 没有原始记录")
            self.records.put(DerivedRecord(run_id=self.run_id, what="attribution", source_hashes=[src],
                                           method_ref=a.method,
                                           output_hash=self.records.put(a, "attribution", self.run_id)),
                             "derived_record", self.run_id)

    def _make_manifest(self) -> RunManifest:
        config = self.w.config
        refs = {k: v.rsplit(" (", 1)[0] for k, v in sorted(getattr(self.w, "refs", {}).items())}
        return RunManifest(protocol=config.protocol, config_hash=content_hash(sanitized_config(config)),
                           components=refs, cells=self.env_cells, openrsi=OPENRSI_VERSION)

    async def score(self, tasks: list[Task], genome: Genome, prefix: str) -> tuple[dict[str, float], list[Episode]]:
        await self.checkout(genome)  # 判别器按当前快照打分，所以先把所有可进化体切到这个版本
        episodes = [await self.episode(t, genome) for t in tasks]
        if not episodes:
            m = {prefix: 0.0, f"{prefix}_pass": 0.0, f"n_{prefix}": 0.0}
        else:
            m = {
                prefix: fmean(e.verdict.score for e in episodes),
                f"{prefix}_pass": fmean(1.0 if e.verdict.passed else 0.0 for e in episodes),
                f"n_{prefix}": float(len(episodes)),
            }
            # 每个 rubric 项的通过率，可以用作 MAP-Elites 的行为特征或多目标指标
            by_item: dict[str, list[float]] = {}
            for e in episodes:
                for r in e.verdict.rubric:
                    if r.result in ("pass", "fail"):
                        by_item.setdefault(r.id, []).append(1.0 if r.result == "pass" else 0.0)
            m.update({f"rubric:{k}": fmean(v) for k, v in by_item.items()})
        self.last_task_scores = {self._key(e.task): e.verdict.score for e in episodes}
        if self.registry and len(self.last_task_scores) != len(episodes):
            raise CellError("逐题分数的键重复：同一道题在同一个格里被评了两次")
        if self.anchor:
            rep = await self.w.judge.calibrate(self.anchor)
            m["anchor"], m["n_anchor"] = rep.agreement, float(rep.n)
        return m, episodes

    @staticmethod
    def _with_epoch(m: dict[str, float], epoch: float) -> dict[str, float]:
        return {**m, "judge_epoch": epoch, "progress": 2 * epoch + m.get("holdout", 0.0)}

    def _allowed_targets(self, gen: int) -> list[str]:
        if self.judge_target is None:
            return list(self.targets)
        if self.cfg.judge_every and gen % self.cfg.judge_every == 0:
            return [self.judge_target]
        return [t for t in self.targets if t != self.judge_target]

    async def counterfactual(self, episodes: list[Episode], genome: Genome) -> list[Attribution]:
        """拿掉一条用过的条目，重跑同一道题：分数掉了说明它有用，涨了说明它有害。"""
        budget = self.cfg.counterfactual_reruns
        out = []
        for ep in episodes:
            targets = []
            for item_id in ep.trajectory.used_items:
                if budget <= 0:
                    break
                budget -= 1
                alt = await self.episode(ep.task, genome, exclude={item_id})
                delta = ep.verdict.score - alt.verdict.score
                targets.append(AttributionTarget(kind="item", ref=item_id, weight=max(-1.0, min(1.0, delta)),
                                                 note=f"拿掉后分数 {ep.verdict.score:.2f} → {alt.verdict.score:.2f}"))
            if targets:
                out.append(Attribution(episode_id=ep.trajectory.id, targets=targets, method="counterfactual"))
            if budget <= 0:
                break
        return out

    # ------------------------------------------------------------ 主循环

    async def run(self, generations: int | None = None) -> list[GenerationReport]:
        w = self.w
        limit = generations if generations is not None else self.cfg.generations
        await self._load_targets()
        if self.cells_cfg is not None:
            self.registry = await CellRegistry.open(w.env, self.cells_cfg, self.log)
            if self.cells_cfg.binding is not None:
                from .wiring import _import
                self.binding = _import(self.cells_cfg.binding.python)(**self.cells_cfg.binding.with_)
        if getattr(w, "config", None) is not None:
            self.manifest = self._make_manifest()
            self.run_id = self.manifest.manifest_hash[:16]
            if self.records is not None:
                self.records.put(self.manifest, "manifest", self.run_id)

        holdout = await self.tasks("holdout", 0, self.cfg.holdout_limit)
        _check_split(holdout, "holdout")
        if not holdout:
            raise IsolationError("env 没有给出 holdout 题，无法做准入")
        self.holdout = holdout  # 整个运行用同一套 holdout，候选之间才可比

        entries = await w.lineage.log(target=self.lineage_target)
        if not entries:
            genome = {t: (await tgt.facade.snapshot()).id for t, tgt in self.targets.items()}
            m, _ = await self.score(holdout, genome, "holdout")
            m = self._with_epoch(m, 0.0)
            root = LineageEntry(id=genome_id(genome), target=self.lineage_target, generation=0, status="root",
                                metrics=m, snapshots=genome, task_scores=self.last_task_scores,
                                env_cells=self.env_cells,
                                manifest=self.manifest.manifest_hash if self.manifest else None)
            entries = [await w.lineage.record(root)]
            self.log(f"root {root.id}: holdout={m['holdout']:.3f}")
        else:
            self.log(f"从谱系续跑：{len(entries)} 条记录")
            root = next((e for e in entries if e.status == "root"), None)
            if root is not None and self.registry:
                self.registry.check_lineage(root.env_cells)
            elif root is not None and root.env_cells:
                raise CellError("谱系是在环境格模式下跑的，现在的配置没有写 cells，分数不能接着比")
            if root is not None and self.manifest and root.manifest and root.manifest != self.manifest.manifest_hash:
                msg = (f"运行清单变了（{root.manifest[:12]} → {self.manifest.manifest_hash[:12]}）："
                       "组件版本、配置或环境格和谱系开始时不一样，新旧结果不能混在一起")
                if self.records is not None or self.binding is not None:
                    raise RecordError(msg)
                self.log("警告：" + msg)

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
            report = await self._generation(gen, ask.parent_id, ask.proposer)
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
                self.genomes[e.id], self.metrics[e.id], self.task_scores[e.id] = genome, e.metrics, e.task_scores
        await self.w.algorithm.tell(TellRequest(generation=root.generation, candidate_id=root.id, accepted=True,
                                                metrics=root.metrics, task_scores=root.task_scores))
        for e in entries:
            if e is not root:
                await self.w.algorithm.tell(TellRequest(
                    generation=e.generation, parent_id=e.parent_ids[0] if e.parent_ids else None,
                    candidate_id=e.id, accepted=e.status == "accepted", metrics=e.metrics,
                    task_scores=e.task_scores, proposer=e.patch.proposer_ref if e.patch else None))

    async def _generation(self, gen: int, parent: str, proposer_hint: str | None = None) -> GenerationReport:
        w = self.w
        parent_genome = self.genomes[parent]
        await self.checkout(parent_genome)
        train = await self.tasks("train", gen, self.cfg.train_limit)
        _check_split(train, "train")
        episodes = [await self.episode(t, parent_genome) for t in train]
        report = GenerationReport(generation=gen, parent_id=parent, parent_metrics=self.metrics[parent],
                                  train_score=fmean(e.verdict.score for e in episodes) if episodes else 0.0)

        attributions = await w.attribution.explain(episodes) if w.attribution and episodes else []
        if self.cfg.counterfactual_reruns and episodes:
            attributions += await self.counterfactual(episodes, parent_genome)
            await self.checkout(parent_genome)
        if self.records is not None:
            self._record_attributions(attributions)
        states = {t: await tgt.facade.export(parent_genome[t]) for t, tgt in self.targets.items()}
        allowed = self._allowed_targets(gen)
        if not allowed:
            report.note = "本代没有可以修改的对象"
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
            return report
        if allowed == [self.judge_target]:
            report.note = "判别器进化代"
        patches = await w.proposer.propose(ProposalRequest(
            target=self.targets[allowed[0]].info, targets=[self.targets[t].info for t in allowed],
            parent_snapshot=parent, episodes=episodes, attributions=attributions,
            state=states[allowed[0]], states=states, max_patches=self.cfg.max_patches,
            hints={"proposer": proposer_hint} if proposer_hint else {}))

        if not patches:
            report.note = "提案器没有提出修改"
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False,
                                               proposer=proposer_hint))
            return report

        for patch in patches[: self.cfg.max_patches]:
            report.candidates.append(await self._candidate(gen, parent, parent_genome, patch, report.train_score,
                                                           allowed))
        await self.checkout(parent_genome)
        return report

    async def _candidate(self, gen: int, parent: str, parent_genome: Genome, patch: Patch,
                         train_score: float, allowed: list[str]) -> Candidate:
        w = self.w

        async def reject(reason: str, t: str | None = None) -> Candidate:
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False,
                                               proposer=patch.proposer_ref))
            return Candidate(id=None, reason=reason, target=t)

        if len(self.targets) == 1:
            t = next(iter(self.targets))
        elif patch.target in self.targets:
            t = patch.target
        else:
            return await reject(f"补丁目标 {patch.target} 不存在，可选：{sorted(self.targets)}")
        if t not in allowed:
            return await reject(f"本代只允许修改 {', '.join(allowed)}，不能改 {t}", t)
        judge_change = t == self.judge_target
        if judge_change and not self.anchor:
            return await reject("判别器的修改必须用人工标注集验证，没有配置 loop.anchor", t)
        tgt = self.targets[t]
        await self.checkout(parent_genome)
        patch = patch.model_copy(update={"target": tgt.info.target, "base_snapshot": parent_genome[t]})

        screened = await w.algorithm.screen(patch, parent)
        if screened is not None and not screened.accept:
            return await reject(f"评估前筛掉：{screened.reason}", t)

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
                                                    generation=gen, status="rejected", decision=decision, patch=patch,
                                                    env_cells=self.env_cells))
                await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=blocked,
                                                   accepted=False, proposer=patch.proposer_ref))
                return Candidate(id=blocked, reason=decision.reason, target=t)

        try:
            snap = await tgt.facade.apply(patch)
        except (ToolCallError, ComponentError, ValueError, KeyError) as e:
            return await reject(f"补丁被拒：{e}", t)

        genome = {**parent_genome, t: snap.id}
        cid = genome_id(genome)
        m, _ = await self.score(self.holdout, genome, "holdout")
        m["train_of_parent"] = train_score
        m = self._with_epoch(m, self.metrics[parent].get("judge_epoch", 0.0) + (1.0 if judge_change else 0.0))
        tasks = self.last_task_scores
        decision = await w.gate.check(GateRequest(candidate_id=cid, baseline=self.metrics[parent], candidate=m,
                                                  primary_metric="anchor" if judge_change else None,
                                                  baseline_tasks=self.task_scores.get(parent, {}),
                                                  candidate_tasks=tasks))
        decision.checks += guard_checks  # 护栏的警告也留档
        status = "accepted" if decision.accepted else "rejected"
        await w.lineage.record(LineageEntry(id=cid, target=self.lineage_target, parent_ids=[parent], generation=gen,
                                            status=status, metrics=m, decision=decision, patch=patch,
                                            snapshots=genome, task_scores=tasks, env_cells=self.env_cells))
        if decision.accepted:
            self.genomes[cid], self.metrics[cid], self.task_scores[cid] = genome, m, tasks
        await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=cid,
                                           accepted=decision.accepted, metrics=m, task_scores=tasks,
                                           proposer=patch.proposer_ref))
        return Candidate(id=cid, metrics=m, accepted=decision.accepted, reason=decision.reason or "", target=t)
