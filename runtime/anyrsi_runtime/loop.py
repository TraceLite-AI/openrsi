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

元层进化（loop.meta_every，可进化体类型是 improver 或 curriculum 时）：
  - 元层对象不进 agent 的上下文；普通代只改解题侧，元层代只改一个元层对象（轮流）
  - 元层候选的准入看"下游改进"：同一个父版本、同样的预算，分别用当前和候选的元层状态跑一轮
    练习 → 提案 → 考试，比较考试分的提升；候选必须真的带来提升、而且比当前多 meta_min_gain 以上
  - 接受时连同它产出的那个解题侧改进一起进谱系，这样算法下一代会从它出发
  - curriculum 只影响练习题怎么挑，考试题不变

提案器的反馈（loop.improver_feedback）：每代把"哪个改法在考试上好 / 差"写进 improver 类型的可进化体，
提案器下次出改法时读到。这部分状态跨代保留（live），不随父版本回退，也不走准入——它只影响提案器，
agent 和考试分数不受它直接影响；它有没有用，看提案器出好改法的比例是不是逐代上升。

agent 给自己出改法（loop.self_propose，见 self_propose.py）：解题侧的提案不再交给提案器，而是把
"改进你自己"当成一道题交给当前版本的 agent，它用自己当前的提示、代码、记忆和权重来出改法。agent 变强，
出改法的能力也跟着变强——这是递归改进最直接的形式。和组提案一起用时，每代让 agent 给自己出一组改法。

组提案（loop.group，见 group.py）：
  - 普通的解题侧代改成：提案器一次出一组候选（G 个），父版本和全部候选做同一批考试题（同一道题同一个种子）
  - 考试题分成选择半和确认半：选择半上按策略算组内优势、决定选谁；确认半交给准入组件比较候选和父版本
  - 策略可替换：parent_paired（减父版本，配对检验）、grpo、best_of_n、training_free（把组内比较写成经验进记忆）

判别器参与进化时（loop.judge_every）：
  - agent 侧和判别器轮流进化，同一代只改一边，分数变化才能归因
  - 判别器的修改会改变 holdout 的打分标准，所以它的准入看 anchor（与人工标注的一致率），不看 holdout
  - 每接受一次判别器的修改，版本进入新的"纪元"（judge_epoch）。progress = 2 × 纪元 + holdout，
    新纪元总是优先：旧纪元的高分是在更宽松的标准下得到的
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable

from anyrsi import (AnchorExample, Attribution, AttributionTarget, ComponentError, Episode, Evidence, EvolvableInfo,
                     GateCheck, GateDecision, GateRequest, Item, LineageEntry, Patch, ProposalRequest, Task,
                     TellRequest, ToolCallError, __version__ as ANYRSI_VERSION, content_hash)
from anyrsi.types import canonical_json, for_agent

from .cells import CellError, CellRegistry, task_key
from .config import CellsConfig, LoopConfig
from .group import Member, atomize, describe_op, lesson_patch, load_strategy, merge_patches, rollout_seed, split_tasks
from .momentum import Direction, DirectionLedger, profile
from .self_propose import build_task, parse_patch
from .records import DerivedRecord, EvidenceRecord, RecordError, RecordStore, RunManifest, sanitized_config

Genome = dict[str, str]  # 可进化体 target → 快照 id
META_TYPES = ("improver", "curriculum")  # 元层可进化体：agent 看不到，按下游改进准入


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
    group: dict[str, Any] | None = None  # 组提案模式下这一组的情况


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
        self.meta_targets: list[str] = []
        self.targets: dict[str, _Target] = {}
        self.genomes: dict[str, Genome] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.task_scores: dict[str, dict[str, float]] = {}
        self.last_task_scores: dict[str, float] = {}
        self.live_meta: dict[str, str] = {}  # 提案器反馈：跨代保留的 improver 快照
        self._gen = 0                        # 当前代号（给 agent 自提案的题目编号和种子用）
        self._self_index = 0

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
        self.meta_targets = sorted(t for t, x in self.targets.items() if x.info.type in META_TYPES)
        fb = self.cfg.improver_feedback
        if fb is not None and (fb.target not in self.targets or self.targets[fb.target].info.type != "improver"):
            raise ValueError(f"loop.improver_feedback.target 必须是 improver 类型的可进化体（现有：{sorted(self.targets)}）")
        if self.meta_targets and not self.cfg.meta_every and not (fb and set(self.meta_targets) == {fb.target}):
            self.log(f"元层可进化体 {self.meta_targets} 没有配置 loop.meta_every，本次不进化它们")
        if not self._solver_targets():
            raise ValueError("至少要有一个解题侧的可进化体（memory、prompt、skill、harness……）")
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
            if t == self.judge_target or t in self.meta_targets:
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

    async def train_episodes(self, tasks: list[Task], genome: Genome) -> list[Episode]:
        """练习题按 loop.train_samples 各答若干次；第 2 次起轨迹 id 加 #s<i>，归因和配对不会混。"""
        return [await self.episode(t, genome, sample=i) for t in tasks for i in range(self.cfg.train_samples)]

    async def episode(self, task: Task, genome: Genome, exclude: set[str] | None = None, sample: int = 0) -> Episode:
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
        if sample:
            traj.id = f"{traj.id}#s{sample}"
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
                           components=refs, cells=self.env_cells, anyrsi=ANYRSI_VERSION)

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
        self.last_task_rubric = {self._key(e.task): {r.id: 1.0 if r.result == "pass" else 0.0 for r in e.verdict.rubric
                                                     if r.result in ("pass", "fail")} for e in episodes}
        self.last_task_tags = {self._key(e.task): list(e.task.tags or []) for e in episodes}
        if self.registry and len(self.last_task_scores) != len(episodes):
            raise CellError("逐题分数的键重复：同一道题在同一个格里被评了两次")
        if self.anchor:
            rep = await self.w.judge.calibrate(self.anchor)
            m["anchor"], m["n_anchor"] = rep.agreement, float(rep.n)
        return m, episodes

    @staticmethod
    def _with_epoch(m: dict[str, float], epoch: float) -> dict[str, float]:
        return {**m, "judge_epoch": epoch, "progress": 2 * epoch + m.get("holdout", 0.0)}

    def _solver_targets(self) -> list[str]:
        return [t for t in self.targets if t != self.judge_target and t not in getattr(self, "meta_targets", [])]

    def _allowed_targets(self, gen: int) -> list[str]:
        if self.judge_target is not None and self.cfg.judge_every and gen % self.cfg.judge_every == 0:
            return [self.judge_target]
        return self._solver_targets()

    def _meta_turn(self, gen: int) -> str | None:
        """这一代是不是元层代、轮到哪个元层对象。和判别器代撞上时让给判别器。"""
        if not self.meta_targets or not self.cfg.meta_every or gen % self.cfg.meta_every:
            return None
        if self.judge_target is not None and self.cfg.judge_every and gen % self.cfg.judge_every == 0:
            return None
        return self.meta_targets[(gen // self.cfg.meta_every - 1) % len(self.meta_targets)]

    async def _train_tasks(self, gen: int, genome: Genome) -> list[Task]:
        """取练习题；有 curriculum 时先取整轮的题，再按课程挑、截到 train_limit。"""
        curricula = [t for t in self.meta_targets if self.targets[t].info.type == "curriculum"]
        if not curricula:
            return await self.tasks("train", gen, self.cfg.train_limit)
        from anyrsi_evolvables import apply_curriculum
        pool = await self.tasks("train", gen, None)
        items = []
        for t in curricula:
            exported = await self.targets[t].facade.export(genome[t])
            items += exported if isinstance(exported, list) else []
        return apply_curriculum(pool, items, self.cfg.train_limit)

    async def _improver_hint(self, genome: Genome) -> dict[str, Any]:
        return {t: await self.targets[t].facade.export(genome[t]) for t in self.meta_targets
                if self.targets[t].info.type == "improver"}

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
        self.group_strategy = None
        self.ledger: DirectionLedger | None = None
        if self.cfg.group is not None:
            self.group_strategy = load_strategy(self.cfg.group.strategy)
            mom = self.cfg.group.momentum
            if mom is not None:
                self.ledger = DirectionLedger(mom.decay, mom.max_items, mom.negative)
            if not getattr(self.group_strategy, "selects", True):
                t = self.cfg.group.experience_target
                if not t or t not in self.targets:
                    raise ValueError(f"策略 {self.group_strategy.name} 要写经验，loop.group.experience_target 必须是一个"
                                     f"可进化体（现有：{sorted(self.targets)}）")
            self.holdout_select, self.holdout_confirm = split_tasks(
                [self._key(t) for t in holdout], self.cfg.group.confirm_fraction, self.run_id or "anyrsi")

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
        self._gen = gen
        meta = self._meta_turn(gen)
        if meta is not None:
            return await self._meta_generation(gen, parent, meta, proposer_hint)
        if self.cfg.group is not None and self._allowed_targets(gen) == self._solver_targets():
            return await self._group_generation(gen, parent)
        w = self.w
        parent_genome = self._live(parent)
        await self.checkout(parent_genome)
        train = await self._train_tasks(gen, parent_genome)
        _check_split(train, "train")
        episodes = await self.train_episodes(train, parent_genome)
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
        patches = await self._propose(ProposalRequest(
            target=self.targets[allowed[0]].info, targets=[self.targets[t].info for t in allowed],
            parent_snapshot=parent, episodes=episodes, attributions=attributions,
            state=states[allowed[0]], states=states, max_patches=self.cfg.max_patches,
            hints={**({"proposer": proposer_hint} if proposer_hint else {}),
                   **({"improver": imp} if (imp := await self._improver_hint(parent_genome)) else {})}),
            parent_genome, solver=allowed != [self.judge_target])

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

    # ------------------------------------------------------------ 元层

    async def _guard_blocks(self, patch: Patch, info: EvolvableInfo) -> str | None:
        if not self.w.guard:
            return None
        g = await self.w.guard.inspect(patch, info)
        return None if g.allowed else "护栏拦截：" + ", ".join(sorted({f.rule for f in g.findings if f.severity == "block"}))

    async def _trial(self, gen: int, parent: str, genome: Genome) -> dict[str, Any] | None:
        """在给定的元层状态下跑一轮普通改进：练习 → 归因 → 提案 → 考试。返回解题侧候选和它的考试提升。"""
        w = self.w
        await self.checkout(genome)
        train = await self._train_tasks(gen, genome)
        _check_split(train, "train")
        episodes = await self.train_episodes(train, genome)
        attributions = await w.attribution.explain(episodes) if w.attribution and episodes else []
        states = {t: await tgt.facade.export(genome[t]) for t, tgt in self.targets.items()}
        solver = self._solver_targets()
        improver = await self._improver_hint(genome)
        patches = await self._propose(ProposalRequest(
            target=self.targets[solver[0]].info, targets=[self.targets[t].info for t in solver],
            parent_snapshot=genome_id(genome), episodes=episodes, attributions=attributions,
            state=states[solver[0]], states=states, max_patches=1,
            hints={"role": "solver", **({"improver": improver} if improver else {})}), genome)
        if not patches:
            return None
        patch = patches[0]
        t = solver[0] if len(solver) == 1 else patch.target
        if t not in solver:
            return None
        patch = patch.model_copy(update={"target": self.targets[t].info.target, "base_snapshot": genome[t]})
        if await self._guard_blocks(patch, self.targets[t].info):
            return None
        try:
            snap = await self.targets[t].facade.apply(patch)
        except (ToolCallError, ComponentError, ValueError, KeyError):
            return None
        g2 = {**genome, t: snap.id}
        m, _ = await self.score(self.holdout, g2, "holdout")
        m = self._with_epoch(m, self.metrics[parent].get("judge_epoch", 0.0))
        gain = m["holdout"] - self.metrics[parent].get("holdout", 0.0)
        return {"gain": gain, "genome": g2, "metrics": m, "task_scores": dict(self.last_task_scores), "patch": patch,
                "train_score": fmean(e.verdict.score for e in episodes) if episodes else 0.0}

    def _meta_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """给元层提案器看的近况：最近接受的版本和它们的考试分。"""
        rows = [{"id": cid, "holdout": m.get("holdout"), "meta_gain": m.get("meta_gain_candidate")}
                for cid, m in self.metrics.items()]
        return rows[-limit:]

    async def _meta_generation(self, gen: int, parent: str, meta: str, proposer_hint: str | None) -> GenerationReport:
        w = self.w
        pg = self.genomes[parent]
        await self.checkout(pg)
        train = await self._train_tasks(gen, pg)
        _check_split(train, "train")
        episodes = await self.train_episodes(train, pg)
        report = GenerationReport(generation=gen, parent_id=parent, parent_metrics=self.metrics[parent],
                                  train_score=fmean(e.verdict.score for e in episodes) if episodes else 0.0,
                                  note=f"元层进化代：{meta}")
        attributions = await w.attribution.explain(episodes) if w.attribution and episodes else []
        if self.records is not None:
            self._record_attributions(attributions)
        states = {t: await tgt.facade.export(pg[t]) for t, tgt in self.targets.items()}
        info = self.targets[meta].info
        patches = await self._propose(ProposalRequest(
            target=info, targets=[info], parent_snapshot=parent, episodes=episodes, attributions=attributions,
            state=states[meta], states=states, max_patches=1,
            hints={"role": "meta", "meta_target": meta, "meta_history": self._meta_history(),
                   **({"proposer": proposer_hint} if proposer_hint else {})}), pg, solver=False)

        async def reject(reason: str) -> GenerationReport:
            report.note += f"；{reason}"
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False,
                                               proposer=patches[0].proposer_ref if patches else proposer_hint))
            report.candidates.append(Candidate(id=None, reason=reason, target=meta))
            await self.checkout(pg)
            return report

        if not patches:
            return await reject("提案器没有提出元层修改")
        patch = patches[0].model_copy(update={"target": info.target, "base_snapshot": pg[meta]})
        blocked = await self._guard_blocks(patch, info)
        if blocked:
            return await reject(blocked)
        try:
            snap = await self.targets[meta].facade.apply(patch)
        except (ToolCallError, ComponentError, ValueError, KeyError) as e:
            return await reject(f"元层补丁被拒：{e}")

        base = await self._trial(gen, parent, pg)
        cand = await self._trial(gen, parent, {**pg, meta: snap.id})
        base_gain = base["gain"] if base else 0.0
        cand_gain = cand["gain"] if cand else 0.0
        checks = [GateCheck(name="meta_gain", passed=cand_gain > base_gain + self.cfg.meta_min_gain,
                            detail=f"同预算下游提升：当前 {base_gain:+.4f} → 候选 {cand_gain:+.4f}"
                                   f"（要求多 {self.cfg.meta_min_gain} 以上）"),
                  GateCheck(name="meta_produces_improvement", passed=cand is not None and cand_gain > 0,
                            detail="候选元层状态下产出的解题侧改进必须真的提高考试分")]
        accepted = all(c.passed for c in checks)
        genome = cand["genome"] if cand else {**pg, meta: snap.id}
        cid = genome_id(genome)
        m = {**(cand["metrics"] if cand else self.metrics[parent]), "train_of_parent": report.train_score,
             "meta_gain_base": base_gain, "meta_gain_candidate": cand_gain}
        tasks = cand["task_scores"] if cand else {}
        decision = GateDecision(candidate_id=cid, accepted=accepted, checks=checks,
                                reason="元层候选带来更多下游提升" if accepted else
                                "未通过：" + ", ".join(c.name for c in checks if not c.passed))
        await w.lineage.record(LineageEntry(
            id=cid, target=self.lineage_target, parent_ids=[parent], generation=gen,
            status="accepted" if accepted else "rejected", metrics=m, decision=decision, patch=patch,
            snapshots=genome, task_scores=tasks, env_cells=self.env_cells))
        if accepted:
            self.genomes[cid], self.metrics[cid], self.task_scores[cid] = genome, m, tasks
        await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=cid, accepted=accepted,
                                           metrics=m, task_scores=tasks, proposer=patch.proposer_ref))
        report.candidates.append(Candidate(id=cid, metrics=m, accepted=accepted, reason=decision.reason or "",
                                           target=meta))
        await self.checkout(pg)
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
        if self.cfg.improver_feedback is not None and not judge_change:
            delta = m.get("holdout", 0.0) - self.metrics[parent].get("holdout", 0.0)
            label = "better" if decision.accepted and delta > 0 else "worse" if delta < 0 else None
            if label:
                await self._improver_feedback(gen, [(label, delta, patch.rationale or "")])
        return Candidate(id=cid, metrics=m, accepted=decision.accepted, reason=decision.reason or "", target=t)

    # ------------------------------------------------------------ 组提案

    def _group_solver(self) -> list[str]:
        """组提案模式下提案器可以改的对象：解题侧，除去经验库（经验库只由经验型策略写）。"""
        exp = self.cfg.group.experience_target if self.cfg.group else None
        return [t for t in self._solver_targets() if t != exp]

    def _proposal(self, parent: str, pg: Genome, episodes: list[Episode], attributions: list[Attribution],
                  states: dict[str, Any], hints: dict[str, Any], max_patches: int) -> ProposalRequest:
        solver = self._group_solver()
        return ProposalRequest(target=self.targets[solver[0]].info, targets=[self.targets[t].info for t in solver],
                               parent_snapshot=parent, episodes=episodes, attributions=attributions,
                               state=states[solver[0]], states=states, max_patches=max_patches, hints=hints)

    async def _group_members(self, parent: str, pg: Genome, episodes: list[Episode], attributions: list[Attribution],
                             states: dict[str, Any]) -> list[tuple[Genome, Patch, str, str | None]]:
        """凑一组候选：先按方向（directions 模式），再多采样补足到 G 个。重复、越界、被护栏或可进化体拒绝的不算。
        atomize 时 G 是改法的个数，每个改法拆成一条条小改动，各自是一个组员（最多 max_atoms 个）。"""
        g = self.cfg.group
        solver = self._group_solver()
        out: list[tuple[Genome, Patch, str, str | None]] = []
        seen: set[str] = set()
        improver = await self._improver_hint(pg)
        cap = g.max_atoms if g.atomize else g.size
        got = [0]  # 收到的改法个数（atomize 时用）

        def enough() -> bool:
            return len(out) >= cap or (got[0] if g.atomize else len(out)) >= g.size

        async def take(patches: list[Patch], direction: str | None) -> None:
            for whole in patches:
                if enough():
                    return
                t = solver[0] if len(solver) == 1 else whole.target
                if t not in solver:
                    continue
                got[0] += 1
                for patch in atomize(whole) if g.atomize else [whole]:
                    if len(out) >= cap:
                        return
                    patch = patch.model_copy(update={"target": self.targets[t].info.target, "base_snapshot": pg[t]})
                    key = content_hash([t, patch.ops])
                    if key in seen:
                        continue
                    seen.add(key)
                    if await self._guard_blocks(patch, self.targets[t].info):
                        continue
                    await self.checkout(pg)
                    try:
                        snap = await self.targets[t].facade.apply(patch)
                    except (ToolCallError, ComponentError, ValueError, KeyError):
                        continue
                    out.append(({**pg, t: snap.id}, patch, t, direction))

        base_hints: dict[str, Any] = {"role": "solver", "group_size": g.size, **({"improver": improver} if improver else {})}
        if g.proposals == "directions":
            blamed: dict[str, set[str]] = {}
            for a in attributions:
                for tg in a.targets:
                    if tg.weight < 0:
                        blamed.setdefault(f"{tg.kind}:{tg.ref}", set()).add(a.episode_id)
            dirs = sorted(((len(v), k) for k, v in blamed.items() if len(v) >= g.min_support), key=lambda x: (-x[0], x[1]))
            for i, (n, name) in enumerate(dirs[: g.size]):
                hints = {**base_hints, "group_index": i,
                         "direction": {"id": name, "support": n,
                                       "text": f"fix what attribution blames: {name} (blamed in {n} failed episodes)"}}
                await take(await self._propose(self._proposal(parent, pg, episodes, attributions, states, hints, 1),
                                               pg), name)
        mom = g.momentum
        if mom is not None and self.ledger is not None:  # 方向动量：前 explore 个改法不给方向，其余带上方向
            hint = self.ledger.hint(self._gen)
            for i in range(g.size):
                if enough():
                    break
                guided = hint is not None and i >= mom.explore
                hints = {**base_hints, "group_index": i, "sample": i + 1, **({"momentum": hint} if guided else {})}
                # 还没有方向可给的代记为 no_hint，不算进探索组（否则第 1 代这种最容易涨分的代会让对照偏高）
                label = "guided" if guided else "explore" if hint is not None else "no_hint"
                await take(await self._propose(self._proposal(parent, pg, episodes, attributions, states, hints, 1), pg),
                           label)
            await self.checkout(pg)
            return out
        tries = 0
        while not enough() and tries < 2 * g.size:
            tries += 1
            want = g.size - (got[0] if g.atomize else len(out))
            hints = {**base_hints, "group_index": len(out), "sample": tries}
            patches = await self._propose(self._proposal(parent, pg, episodes, attributions, states, hints,
                                                         want if tries == 1 else 1), pg)
            if not patches and tries >= 2:
                break
            await take(patches, None)
        await self.checkout(pg)
        return out

    def _seeded(self, tasks: list[Task], gen: int) -> list[Task]:
        if not self.cfg.group.common_seed:
            return tasks
        return [t.model_copy(update={"metadata": {**t.metadata, "rollout_seed": rollout_seed(self.run_id, gen,
                                                                                               self._key(t))}})
                for t in tasks]

    @staticmethod
    def _subset(scores: dict[str, float], keys: list[str]) -> dict[str, float]:
        vals = [scores[k] for k in keys if k in scores]
        return {"holdout": fmean(vals) if vals else 0.0, "n_holdout": float(len(vals))}

    async def _group_generation(self, gen: int, parent: str) -> GenerationReport:
        g = self.cfg.group
        strategy = self.group_strategy
        w = self.w
        pg = self._live(parent)
        await self.checkout(pg)
        train = await self._train_tasks(gen, pg)
        _check_split(train, "train")
        episodes = await self.train_episodes(train, pg)
        report = GenerationReport(generation=gen, parent_id=parent, parent_metrics=self.metrics[parent],
                                  train_score=fmean(e.verdict.score for e in episodes) if episodes else 0.0,
                                  note=f"组提案（{strategy.name}）")
        attributions = await w.attribution.explain(episodes) if w.attribution and episodes else []
        if self.records is not None:
            self._record_attributions(attributions)
        states = {t: await tgt.facade.export(pg[t]) for t, tgt in self.targets.items()}
        members = await self._group_members(parent, pg, episodes, attributions, states)

        async def reject(reason: str) -> GenerationReport:
            report.note += f"；{reason}"
            await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, accepted=False))
            await self.checkout(pg)
            return report

        if not members:
            return await reject("提案器没有提出可用的修改")
        tasks = self._seeded(self.holdout, gen)
        sel, conf = self.holdout_select, self.holdout_confirm or self.holdout_select
        parent_m, _ = await self.score(tasks, pg, "holdout")
        parent_scores = dict(self.last_task_scores)
        parent_rubric, task_tags = dict(self.last_task_rubric), dict(self.last_task_tags)
        results = []
        rubrics = []
        for member_genome, *_ in members:
            m, _ = await self.score(tasks, member_genome, "holdout")
            results.append((m, dict(self.last_task_scores)))
            rubrics.append(dict(self.last_task_rubric))
        group = [Member(index=i, scores={k: results[i][1][k] for k in sel}, rationale=members[i][1].rationale or "",
                        direction=members[i][3]) for i in range(len(members))]
        decision = strategy.decide({k: parent_scores[k] for k in sel}, group, g.alpha)
        if self.cfg.improver_feedback is not None and decision.advantage:
            adv = decision.advantage
            hi, lo = max(range(len(adv)), key=lambda i: adv[i]), min(range(len(adv)), key=lambda i: adv[i])
            notes = ([("better", adv[hi], members[hi][1].rationale or "")] if adv[hi] > 0 else []) + \
                    ([("worse", adv[lo], members[lo][1].rationale or "")] if adv[lo] < 0 and lo != hi else [])
            await self._improver_feedback(gen, notes)
            pg = {**pg, **self.live_meta}
        report.group = {**strategy.summary(), "size": len(members), "reason": decision.reason,
                        "chosen": decision.chosen, "parent_select": self._subset(parent_scores, sel)["holdout"],
                        "members": [{"index": i, "target": members[i][2], "direction": members[i][3],
                                     "advantage": decision.advantage[i] if i < len(decision.advantage) else None,
                                     "z": decision.z[i] if i < len(decision.z) else None,
                                     "select": self._subset(results[i][1], sel)["holdout"]}
                                    for i in range(len(members))],
                        "lessons": [x.text for x in decision.lessons]}

        if getattr(strategy, "selects", True):
            if decision.chosen is None:
                if self.ledger is not None:  # 没选出候选，测出来变差的方向照样记下
                    self._record_directions(gen, members, results, rubrics, decision, False, parent_scores,
                                            parent_rubric, task_tags, sel, report)
                return await reject(decision.reason)
            genome, patch, t, direction = members[decision.chosen]
            m, scores = results[decision.chosen]
            merge = [i for i in getattr(decision, "merge", None) or [] if members[i][2] == t]
            if len(merge) > 1:  # 合并型策略：所有变好的方向一起走一步（只合并改同一个可进化体的）
                patch = merge_patches([members[i][1] for i in merge])
                await self.checkout(pg)
                try:
                    snap = await self.targets[t].facade.apply(patch)
                except (ToolCallError, ComponentError, ValueError, KeyError) as e:
                    return await reject(f"合并的改动应用不上：{e}")
                genome = {**pg, t: snap.id}
                m, _ = await self.score(tasks, genome, "holdout")
                scores = dict(self.last_task_scores)
                report.group["merged"] = merge
        else:
            if not decision.lessons:
                return await reject(decision.reason)
            t = g.experience_target
            existing = {d.get("id") for d in states.get(t) or [] if isinstance(d, dict)}
            patch = lesson_patch(self.targets[t].info.target, pg[t], decision.lessons, existing)
            await self.checkout(pg)
            try:
                snap = await self.targets[t].facade.apply(patch)
            except (ToolCallError, ComponentError, ValueError, KeyError) as e:
                return await reject(f"经验写不进 {t}：{e}")
            genome = {**pg, t: snap.id}
            m, _ = await self.score(tasks, genome, "holdout")
            scores = dict(self.last_task_scores)
        cand = await self._group_candidate(gen, parent, genome, patch, t, m, scores, parent_m,
                                           parent_scores, conf, report.train_score, decision.reason)
        report.candidates.append(cand)
        if self.ledger is not None and getattr(strategy, "selects", True):
            self._record_directions(gen, members, results, rubrics, decision, cand.accepted, parent_scores,
                                    parent_rubric, task_tags, sel, report)
        await self.checkout(pg)
        return report

    def _record_directions(self, gen: int, members: list[Any], results: list[Any], rubrics: list[Any], decision: Any,
                           accepted: bool, parent_scores: dict[str, float], parent_rubric: dict[str, Any],
                           task_tags: dict[str, list[str]], sel: list[str], report: GenerationReport) -> None:
        """方向动量：通过准入的那一步里的小改动记为有用的方向，测出来变差的记为有害的方向；
        顺便统计有方向指导和没有方向指导的改法各自的平均优势（对照）。"""
        assert self.ledger is not None
        taken = set(getattr(decision, "merge", None) or ([decision.chosen] if decision.chosen is not None else []))
        by_label: dict[str, list[float]] = {}
        for i, (_, patch, _, label) in enumerate(members):
            adv = decision.advantage[i] if i < len(decision.advantage) else 0.0
            by_label.setdefault(label or "other", []).append(adv)
            helped = accepted and i in taken and adv > 0
            if not helped and adv >= 0:
                continue
            rub, tags = profile(parent_scores, results[i][1], parent_rubric, rubrics[i], task_tags, sel)
            what = " | ".join(describe_op(op) for op in patch.ops)  # 方向以改动内容为准，提案器的理由放后面
            why = (patch.rationale or "").strip()
            text = what if not why or what in why else f"{what} (why: {why[:150]})"
            self.ledger.add(Direction(generation=gen, text=text, advantage=adv, helped=helped, rubric=rub, tags=tags))
        report.group["momentum"] = {
            "by_label": {k: {"n": len(v), "mean_advantage": round(fmean(v), 4), "improved": sum(x > 0 for x in v)}
                         for k, v in by_label.items()},
            "ledger": {"helped": sum(d.helped for d in self.ledger.items),
                       "hurt": sum(not d.helped for d in self.ledger.items)}}

    async def _group_candidate(self, gen: int, parent: str, genome: Genome, patch: Patch, t: str, m: dict[str, float],
                               scores: dict[str, float], parent_m: dict[str, float], parent_scores: dict[str, float],
                               conf: list[str], train_score: float, reason: str) -> Candidate:
        """组里选出的候选（或写了经验的新版本）在确认半上和父版本比，交给准入组件决定。"""
        w = self.w
        cid = genome_id(genome)
        base = {**{k: v for k, v in parent_m.items() if k == "anchor"}, **self._subset(parent_scores, conf)}
        cand = {**{k: v for k, v in m.items() if k == "anchor"}, **self._subset(scores, conf)}
        decision = await w.gate.check(GateRequest(candidate_id=cid, baseline=base, candidate=cand,
                                                  baseline_tasks={k: parent_scores[k] for k in conf},
                                                  candidate_tasks={k: scores[k] for k in conf if k in scores}))
        decision.checks.append(GateCheck(name="group", passed=True,
                                         detail=f"{reason}；确认半 {len(conf)} 道：父版本 {base['holdout']:.3f} → "
                                                f"候选 {cand['holdout']:.3f}"))
        metrics = {**m, "train_of_parent": train_score, "holdout_confirm": cand["holdout"],
                   "parent_holdout_confirm": base["holdout"]}
        metrics = self._with_epoch(metrics, self.metrics[parent].get("judge_epoch", 0.0))
        status = "accepted" if decision.accepted else "rejected"
        await w.lineage.record(LineageEntry(id=cid, target=self.lineage_target, parent_ids=[parent], generation=gen,
                                            status=status, metrics=metrics, decision=decision, patch=patch,
                                            snapshots=genome, task_scores=scores, env_cells=self.env_cells))
        if decision.accepted:
            self.genomes[cid], self.metrics[cid], self.task_scores[cid] = genome, metrics, scores
        await w.algorithm.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=cid,
                                           accepted=decision.accepted, metrics=metrics, task_scores=scores,
                                           proposer=patch.proposer_ref))
        return Candidate(id=cid, metrics=metrics, accepted=decision.accepted, reason=decision.reason or "", target=t)

    # ------------------------------------------------------------ 提案器的反馈

    def _live(self, parent: str) -> Genome:
        """父版本的组合快照，再换上跨代保留的提案器策略（提案器反馈不随父版本回退）。"""
        return {**self.genomes[parent], **self.live_meta}

    async def _improver_feedback(self, gen: int, notes: list[tuple[str, float, str]]) -> None:
        """把 (better / worse, 优势, 改法说明) 写进提案器的策略说明，只保留最近 max_notes 条。"""
        fb = self.cfg.improver_feedback
        notes = [n for n in notes if n[2].strip()]
        if fb is None or not notes:
            return
        if fb.shuffle:  # 对照：好坏随机打乱
            rng = random.Random(f"{self.run_id}:{gen}")
            notes = [(rng.choice(["better", "worse"]), adv, text) for _, adv, text in notes]
        tgt = self.targets[fb.target]
        current = self.live_meta.get(fb.target) or (await tgt.facade.snapshot()).id
        await tgt.facade.rollback(current)
        existing = [d for d in await tgt.facade.export(current) if isinstance(d, dict) and d.get("id", "").startswith("fb-")]
        ops: list[dict[str, Any]] = []
        for i, (label, adv, text) in enumerate(notes):
            word = "BETTER" if label == "better" else "WORSE"
            item = Item(id=f"fb-{gen:05d}-{i}", kind="improver",
                        text=f"{word} on the exam ({adv:+.3f}): {text.strip()[:300]}",
                        tags=["exam-feedback", label], data={"generation": gen, "advantage": adv, "order": 100 + gen})
            ops.append({"op": "add", "item": item.model_dump(mode="json")})
        overflow = len(existing) + len(ops) - fb.max_notes
        for d in sorted(existing, key=lambda d: d["id"])[: max(0, overflow)]:
            ops.append({"op": "remove", "item_id": d["id"]})
        try:
            snap = await tgt.facade.apply(Patch(target=tgt.info.target, base_snapshot=current, ops=ops,
                                                rationale=f"exam feedback from generation {gen}"))
        except (ToolCallError, ComponentError, ValueError, KeyError) as e:
            self.log(f"提案器反馈写不进 {fb.target}：{e}")
            return
        self.live_meta[fb.target] = snap.id

    # ------------------------------------------------------------ agent 给自己出改法

    async def _propose(self, request: ProposalRequest, genome: Genome, solver: bool = True) -> list[Patch]:
        """解题侧的提案：打开 self_propose 时交给当前版本的 agent，否则交给提案器组件。"""
        if not (self.cfg.self_propose and solver):
            if self.w.proposer is None:
                raise ValueError("没有配置 components.proposer：判别器、元层的提案需要它")
            return await self.w.proposer.propose(request)
        out: list[Patch] = []
        await self.checkout(genome)
        gid = genome_id(genome)
        for _ in range(request.max_patches):
            self._self_index += 1
            task = build_task(request, gen=self._gen, index=self._self_index,
                              seed=rollout_seed(self.run_id, "self", self._gen, self._self_index))
            context = await self.context(task, genome)
            traj = await self.w.agent.run(for_agent(task), context, gid)
            patch = parse_patch(traj.output, request, getattr(self.w, "refs", {}).get("agent", "agent"))
            if patch is not None:
                out.append(patch)
        return out
