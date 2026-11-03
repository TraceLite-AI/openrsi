"""组提案：一次出一组候选，在考试题（holdout）上算组内优势（loop.group）。

一代的流程：
  1 父版本做练习题 → 归因（和普通代一样，用来出提案）
  2 提案器一次出 G 个候选：同一提案器多采样（proposals: samples），或沿归因指出的不同方向各出一个
    （proposals: directions）；完全相同的修改只算一次
  3 父版本和 G 个候选做同一批考试题。同一道题给所有版本同一个 rollout_seed（公共随机数，agent 支持时生效），
    差别主要来自版本本身
  4 考试题分两半：选择半用来算组内优势、决定选谁；确认半只用来确认（准入组件在确认半上比较候选和父版本）。
    同一批题既拿来挑又拿来判，挑出来的总会偏高（赢家诅咒），分开用就没有这个问题
  5 怎么算优势、怎么用，由策略决定（strategy），见下面四个内置策略；也可以写自己的策略类接进来

策略接口（GroupStrategy）：
  advantages(parent, members) → 每个候选在每道题上的优势
  decide(parent, members, alpha) → Decision：选中哪个候选（或者不选），以及原因；
  经验型策略（selects = False）不选版本，而是返回要写进记忆的经验条目（lessons）

内置四个：
  parent_paired     优势 = 候选分 − 父版本分（逐题配对）；选平均优势最大、且 z ≥ Φ⁻¹(1 − α/G) 的那个
  grpo              优势 = (分数 − 组内平均) / 组内标准差（GRPO；组里默认包含父版本，normalize: false 时只减平均）；
                    选平均优势最大、且高于父版本的那个
  best_of_n         选选择半上平均分最高的那个（不做检验，偏差交给确认半纠正）
  training_free     Training-Free GRPO 式：不选版本，把组内谁好谁差总结成经验条目，写进记忆类可进化体
                    （loop.group.experience_target），新的记忆版本再走确认半准入
  merge             零阶"梯度"一步：每个候选的优势 = 候选分 − 父版本分（有限差分：往这个方向走一小步，分数变了多少），
                    把所有优势为正（且 z 够大）的候选合并成一步一起走，而不是只挑一个。配合 loop.group.atomize
                    （把每个改法拆成一条条最小改动分别测）使用：提案器出的改法里对的部分和错的部分被分开，
                    弱提案器"偶尔对"的那部分也能被测出来留下。step 是一步最多合并几条（步长）

拆分与合并：atomize(patch) 把一个补丁拆成每条操作一个的小补丁；merge_patches(patches) 按顺序合并，
同一个条目只保留第一条（策略按优势从高到低排好序，冲突时留优势高的）。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from statistics import NormalDist, fmean, pstdev, stdev
from typing import Any

from anyrsi import Item, Patch

# ---------------------------------------------------------------- 统计


def z_threshold(alpha: float, tests: int = 1) -> float:
    """单侧检验的 z 门槛，Bonferroni 分摊到 tests 次检验：Φ⁻¹(1 − α / tests)。"""
    return NormalDist().inv_cdf(1 - alpha / max(1, tests))


def mean_z(diffs: list[float]) -> tuple[float, float]:
    """配对差值的均值和 z（均值 / 标准误）。全部相同且非零时 z = ±inf；少于 2 个时 z = 0。"""
    if not diffs:
        return 0.0, 0.0
    mu = fmean(diffs)
    if len(diffs) < 2:
        return mu, 0.0
    sd = stdev(diffs)
    if sd == 0:
        return mu, (math.inf if mu > 0 else -math.inf if mu < 0 else 0.0)
    return mu, mu / (sd / math.sqrt(len(diffs)))


def split_tasks(keys: list[str], confirm_fraction: float, salt: str) -> tuple[list[str], list[str]]:
    """把考试题按哈希分成选择半和确认半（整个运行固定，同一道题总在同一半）。confirm_fraction = 0 时不分。"""
    if confirm_fraction <= 0:
        return sorted(keys), []
    ordered = sorted(keys, key=lambda k: hashlib.sha256(f"{salt}\0{k}".encode()).hexdigest())
    n_confirm = max(1, min(len(ordered) - 1, round(len(ordered) * confirm_fraction)))
    return sorted(ordered[n_confirm:]), sorted(ordered[:n_confirm])


def rollout_seed(*parts: Any) -> int:
    """同一代里同一道题的公共种子。"""
    return int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- 策略


@dataclass
class Member:
    """组里的一个候选：在选择半上的逐题分数，以及提案器给的说明。"""

    index: int
    scores: dict[str, float]
    rationale: str = ""
    direction: str | None = None


@dataclass
class Decision:
    chosen: int | None
    reason: str
    advantage: list[float] = field(default_factory=list)  # 每个候选的平均优势
    z: list[float] = field(default_factory=list)
    lessons: list[Item] = field(default_factory=list)      # 经验型策略要写进记忆的条目
    merge: list[int] = field(default_factory=list)         # 合并型策略：要合并成一步的候选（按优势从高到低）


class GroupStrategy:
    """组内优势策略的基类。子类实现 advantages()，需要时覆盖 decide()。"""

    name = "base"
    selects = True  # False：不选版本，而是产出经验条目

    def advantages(self, parent: dict[str, float], members: list[Member]) -> list[dict[str, float]]:
        raise NotImplementedError

    def decide(self, parent: dict[str, float], members: list[Member], alpha: float) -> Decision:
        adv = self.advantages(parent, members)
        means = [fmean(a.values()) if a else 0.0 for a in adv]
        zs = [mean_z(list(a.values()))[1] for a in adv]
        best = max(range(len(members)), key=lambda i: means[i])
        if means[best] <= 0:
            return Decision(None, f"组内没有优势为正的候选（最好的 {means[best]:+.3f}）", means, zs)
        return Decision(best, f"候选 {best} 平均优势 {means[best]:+.3f}", means, zs)

    def summary(self) -> dict[str, Any]:
        return {"strategy": self.name}


class ParentPaired(GroupStrategy):
    """优势 = 候选分 − 父版本分（逐题配对）；平均优势最大且 z ≥ Φ⁻¹(1 − α/G) 才选。"""

    name = "parent_paired"

    def advantages(self, parent: dict[str, float], members: list[Member]) -> list[dict[str, float]]:
        return [{t: m.scores[t] - parent[t] for t in parent} for m in members]

    def decide(self, parent: dict[str, float], members: list[Member], alpha: float) -> Decision:
        adv = self.advantages(parent, members)
        stats = [mean_z(list(a.values())) for a in adv]
        means, zs = [s[0] for s in stats], [s[1] for s in stats]
        crit = z_threshold(alpha, len(members))
        best = max(range(len(members)), key=lambda i: (means[i], zs[i]))
        if means[best] > 0 and zs[best] >= crit:
            return Decision(best, f"候选 {best} 比父版本平均高 {means[best]:+.3f}，z = {zs[best]:.2f} ≥ {crit:.2f}",
                            means, zs)
        return Decision(None, f"没有候选显著好过父版本（最好的 {means[best]:+.3f}，z = {zs[best]:.2f}，门槛 {crit:.2f}）",
                        means, zs)


class GRPO(GroupStrategy):
    """GRPO 式：优势 = (分数 − 组内平均) / 组内标准差，每道题单独算。

    with: normalize（默认 true，除以标准差；false 时只减平均）、include_parent（默认 true，组里包含父版本，
    这样"比父版本好"才有意义）。选平均优势最大、且高于父版本平均优势的那个。"""

    name = "grpo"

    def __init__(self, normalize: bool = True, include_parent: bool = True, eps: float = 1e-6):
        self.normalize, self.include_parent, self.eps = normalize, include_parent, eps

    def _group_adv(self, parent: dict[str, float], members: list[Member]) -> tuple[dict[str, float], list[dict[str, float]]]:
        p_adv: dict[str, float] = {}
        m_adv: list[dict[str, float]] = [{} for _ in members]
        for t in parent:
            group = [m.scores[t] for m in members] + ([parent[t]] if self.include_parent else [])
            mu = fmean(group)
            sd = pstdev(group) if self.normalize else 1.0
            scale = sd + self.eps if self.normalize else 1.0
            for i, m in enumerate(members):
                m_adv[i][t] = (m.scores[t] - mu) / scale
            p_adv[t] = (parent[t] - mu) / scale
        return p_adv, m_adv

    def advantages(self, parent: dict[str, float], members: list[Member]) -> list[dict[str, float]]:
        return self._group_adv(parent, members)[1]

    def decide(self, parent: dict[str, float], members: list[Member], alpha: float) -> Decision:
        p_adv, adv = self._group_adv(parent, members)
        means = [fmean(a.values()) if a else 0.0 for a in adv]
        zs = [mean_z([a[t] - p_adv[t] for t in a])[1] for a in adv]
        base = fmean(p_adv.values()) if p_adv else 0.0
        best = max(range(len(members)), key=lambda i: means[i])
        if means[best] > base:
            return Decision(best, f"候选 {best} 组内优势 {means[best]:+.3f}（父版本 {base:+.3f}）", means, zs)
        return Decision(None, f"组内优势最高的候选也不高于父版本（{means[best]:+.3f} vs {base:+.3f}）", means, zs)

    def summary(self) -> dict[str, Any]:
        return {"strategy": self.name, "normalize": self.normalize, "include_parent": self.include_parent}


class BestOfN(GroupStrategy):
    """选选择半上平均分最高的候选，只要求不低于父版本；不做显著性检验，挑选带来的偏差由确认半纠正。"""

    name = "best_of_n"

    def advantages(self, parent: dict[str, float], members: list[Member]) -> list[dict[str, float]]:
        return [{t: m.scores[t] - parent[t] for t in parent} for m in members]

    def decide(self, parent: dict[str, float], members: list[Member], alpha: float) -> Decision:
        raw = [fmean(m.scores[t] for t in parent) for m in members]
        adv = self.advantages(parent, members)
        means = [fmean(a.values()) if a else 0.0 for a in adv]
        zs = [mean_z(list(a.values()))[1] for a in adv]
        best = max(range(len(members)), key=lambda i: raw[i])
        if means[best] < 0:
            return Decision(None, f"最好的候选也低于父版本（{means[best]:+.3f}）", means, zs)
        return Decision(best, f"候选 {best} 选择半平均分最高（{raw[best]:.3f}，比父版本 {means[best]:+.3f}）", means, zs)


class TrainingFreeGRPO(GroupStrategy):
    """Training-Free GRPO 式：不选版本，把组内比较总结成经验，写进记忆。

    每个候选的优势 = 分数 − 组内平均（组里含父版本）。优势显著为正的候选写一条"Better than its group"，
    显著为负的写"Worse than its group"（是相对同组的，不一定是绝对有害），内容是提案器给的修改说明
    （rationale）和优势大小。summarizer 可以换成自己的函数
    （module:function，参数 (parent, members, advantages) → list[Item]），比如让 LLM 读轨迹写经验。

    with: max_lessons（每代最多写几条，默认 3）、min_z（写经验的门槛，默认 1.0）、summarizer"""

    name = "training_free"
    selects = False

    def __init__(self, max_lessons: int = 3, min_z: float = 1.0, summarizer: str | None = None):
        self.max_lessons, self.min_z, self.summarizer = max_lessons, min_z, summarizer

    def advantages(self, parent: dict[str, float], members: list[Member]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = [{} for _ in members]
        for t in parent:
            mu = fmean([parent[t], *(m.scores[t] for m in members)])
            for i, m in enumerate(members):
                out[i][t] = m.scores[t] - mu
        return out

    def decide(self, parent: dict[str, float], members: list[Member], alpha: float) -> Decision:
        adv = self.advantages(parent, members)
        stats = [mean_z(list(a.values())) for a in adv]
        means, zs = [s[0] for s in stats], [s[1] for s in stats]
        if self.summarizer:
            from .wiring import _import

            lessons = list(_import(self.summarizer)(parent, members, adv))[: self.max_lessons]
        else:
            ranked = sorted(range(len(members)), key=lambda i: -abs(zs[i]) if math.isfinite(zs[i]) else -1e9)
            lessons = []
            for i in ranked:
                if len(lessons) >= self.max_lessons:
                    break
                if abs(zs[i]) < self.min_z or not members[i].rationale.strip():
                    continue
                good = means[i] > 0
                text = (f"{'Better' if good else 'Worse'} than its group ({means[i]:+.3f} vs group average): "
                        f"{members[i].rationale.strip()[:400]}")
                lessons.append(Item(id="exp-" + hashlib.sha256(text.encode()).hexdigest()[:12], kind="lesson",
                                    text=text, tags=["group-experience", "better" if good else "worse"],
                                    data={"advantage": means[i], "z": zs[i], "direction": members[i].direction}))
        if not lessons:
            return Decision(None, "组内差别不够大，没有可写的经验", means, zs)
        return Decision(None, f"写入 {len(lessons)} 条经验", means, zs, lessons=lessons)

    def summary(self) -> dict[str, Any]:
        return {"strategy": self.name, "max_lessons": self.max_lessons, "min_z": self.min_z}


BUILTIN = {"parent_paired": ParentPaired, "grpo": GRPO, "best_of_n": BestOfN, "training_free": TrainingFreeGRPO}


class MergePositive(GroupStrategy):
    """零阶"梯度"一步：优势 = 候选分 − 父版本分（逐题配对，即有限差分）；把平均优势 > min_adv 且 z ≥ min_z 的候选
    按优势从高到低取最多 step 个，合并成一步（Decision.merge）。合并后的版本再交给确认半准入。"""

    name = "merge"
    merges = True

    def __init__(self, step: int = 8, min_z: float = 0.0, min_adv: float = 0.0):
        if step < 1:
            raise ValueError("merge 的 step 至少是 1")
        self.step, self.min_z, self.min_adv = step, min_z, min_adv

    def advantages(self, parent: dict[str, float], members: list[Member]) -> list[dict[str, float]]:
        return [{t: m.scores[t] - parent[t] for t in parent} for m in members]

    def decide(self, parent: dict[str, float], members: list[Member], alpha: float) -> Decision:
        adv = self.advantages(parent, members)
        stats = [mean_z(list(a.values())) for a in adv]
        means, zs = [s[0] for s in stats], [s[1] for s in stats]
        good = sorted((i for i in range(len(members)) if means[i] > self.min_adv and zs[i] >= self.min_z),
                      key=lambda i: (-means[i], -zs[i], i))[: self.step]
        if not good:
            best = max(range(len(members)), key=lambda i: (means[i], zs[i]))
            return Decision(None, f"没有往好的方向（最好的 {means[best]:+.3f}，z = {zs[best]:.2f}）", means, zs)
        return Decision(good[0], f"{len(good)} 个方向变好，合并成一步（{', '.join(f'{means[i]:+.3f}' for i in good)}）",
                        means, zs, merge=good)

    def summary(self) -> dict[str, Any]:
        return {"strategy": self.name, "step": self.step, "min_z": self.min_z, "min_adv": self.min_adv}


BUILTIN["merge"] = MergePositive


def op_key(op: dict[str, Any]) -> str:
    """一条操作改的是哪个条目：item.id / item_id / id，都没有时用整条操作的内容。"""
    item = op.get("item") if isinstance(op.get("item"), dict) else {}
    key = item.get("id") or op.get("item_id") or op.get("id") or op.get("path")
    return str(key) if key else hashlib.sha256(repr(sorted(op.items())).encode()).hexdigest()[:16]


def describe_op(op: dict[str, Any]) -> str:
    """一条操作的简短说明：操作 条目: 内容。"""
    item = op.get("item") if isinstance(op.get("item"), dict) else {}
    text = str(item.get("text") or "").replace("\n", " ")
    return f"{op.get('op', '?')} {op_key(op)}" + (f": {text[:120]}" if text else "")


def atomize(patch: Patch) -> list[Patch]:
    """把补丁拆成每条操作一个的小补丁。只有一条操作时原样返回。"""
    if len(patch.ops) <= 1:
        return [patch]
    return [patch.model_copy(update={"ops": [op], "rationale": f"{describe_op(op)} (part {i + 1}/{len(patch.ops)} of: "
                                                              f"{(patch.rationale or '')[:200]})"})
            for i, op in enumerate(patch.ops)]


def merge_patches(patches: list[Patch]) -> Patch:
    """按顺序合并几个改同一个可进化体的补丁；同一个条目只保留第一条操作。"""
    seen: set[str] = set()
    ops: list[dict[str, Any]] = []
    for p in patches:
        for op in p.ops:
            k = op_key(op)
            if k not in seen:
                seen.add(k)
                ops.append(op)
    return patches[0].model_copy(update={"ops": ops, "rationale": " | ".join(describe_op(op) for op in ops)[:1000],
                                         "proposer_ref": patches[0].proposer_ref})


def load_strategy(spec: str | dict[str, Any]) -> GroupStrategy:
    """strategy: 内置名字，或 {name: …, with: {…}}，或 {python: "module:Class", with: {…}}。"""
    if isinstance(spec, str):
        spec = {"name": spec}
    kwargs = dict(spec.get("with") or {})
    if spec.get("python"):
        from .wiring import _import

        obj = _import(spec["python"])(**kwargs)
    elif spec.get("name") in BUILTIN:
        obj = BUILTIN[spec["name"]](**kwargs)
    else:
        raise ValueError(f"未知的组内优势策略 {spec}；内置：{', '.join(BUILTIN)}")
    if not isinstance(obj, GroupStrategy) and not (hasattr(obj, "decide") and hasattr(obj, "advantages")):
        raise ValueError(f"{spec} 不是组内优势策略（要有 advantages() 和 decide()）")
    return obj


def lesson_patch(target: str, base: str, lessons: list[Item], existing: set[str]) -> Patch:
    ops = [{"op": "replace" if i.id in existing else "add", "item": i.model_dump(mode="json")} for i in lessons]
    return Patch(target=target, base_snapshot=base, ops=ops, rationale=f"group experience: {len(lessons)} lessons")


__all__ = ["BUILTIN", "BestOfN", "Decision", "GRPO", "GroupStrategy", "Member", "MergePositive", "ParentPaired",
           "TrainingFreeGRPO", "atomize", "describe_op", "lesson_patch", "load_strategy", "mean_z", "merge_patches", "op_key",
           "rollout_seed", "split_tasks", "z_threshold"]
