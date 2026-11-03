"""方向动量（loop.group.momentum）：把组里测出来的方向交给算子。

组提案 + atomize 时，每条小改动都在考试题上和父版本逐题比过（有限差分）。这里把结果记成"方向"：
  有用的方向   这一代被合并、并通过了准入的小改动（考试上确认过）
  有害的方向   测出来让分数变差的小改动
每条方向带上它让哪些 rubric 项、哪类题（task.tags）变好或变差了多少，所以 rubric 越细、题目标签越清楚，方向越具体。

权重 = |优势| × decay^(距今代数)：方向会用完（比如某条规则补上了，再往那个方向推没用），旧方向逐渐淡出。
每代取权重最高的几条，写进提案请求的 hints["momentum"]，算子沿好方向继续提、避开坏方向。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any


@dataclass
class Direction:
    generation: int
    text: str
    advantage: float
    helped: bool
    rubric: dict[str, float] = field(default_factory=dict)  # rubric 项 → 通过率变化
    tags: dict[str, float] = field(default_factory=dict)    # 题目标签 → 分数变化


def profile(parent_scores: dict[str, float], member_scores: dict[str, float],
            parent_rubric: dict[str, dict[str, float]], member_rubric: dict[str, dict[str, float]],
            tags: dict[str, list[str]], keys: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """一条改动在 keys 这些题上让每个 rubric 项、每类题变了多少（只留不为 0 的）。"""
    by_rubric: dict[str, list[float]] = {}
    by_tag: dict[str, list[float]] = {}
    for k in keys:
        if k not in parent_scores or k not in member_scores:
            continue
        pr, mr = parent_rubric.get(k, {}), member_rubric.get(k, {})
        for rid in set(pr) & set(mr):
            by_rubric.setdefault(rid, []).append(mr[rid] - pr[rid])
        for t in tags.get(k, []):
            by_tag.setdefault(t, []).append(member_scores[k] - parent_scores[k])
    rubric = {r: round(fmean(v), 4) for r, v in by_rubric.items() if fmean(v) != 0}
    tag = {t: round(fmean(v), 4) for t, v in by_tag.items() if fmean(v) != 0}
    return rubric, tag


class DirectionLedger:
    def __init__(self, decay: float = 0.7, max_items: int = 6, negative: bool = True):
        self.decay, self.max_items, self.negative = decay, max_items, negative
        self.items: list[Direction] = []

    def add(self, d: Direction) -> None:
        if d.advantage == 0 or (not d.helped and not self.negative):
            return
        self.items.append(d)

    def weight(self, d: Direction, gen: int) -> float:
        return abs(d.advantage) * self.decay ** max(0, gen - d.generation)

    @staticmethod
    def describe(d: Direction, gen: int) -> str:
        parts = [f"{d.advantage:+.3f} on the exam"]
        ago = gen - d.generation
        parts.append("this generation" if ago <= 0 else f"{ago} generation{'s' if ago > 1 else ''} ago")
        def fmt(xs: dict[str, float]) -> str:
            return ", ".join(f"{k} {v:+.2f}" for k, v in sorted(xs.items(), key=lambda x: -abs(x[1]))[:4])

        if d.tags:
            parts.append("tasks " + fmt(d.tags))
        if d.rubric:
            parts.append("rubric " + fmt(d.rubric))
        return f"[{'; '.join(parts)}] {d.text.strip()[:300]}"

    def hint(self, gen: int) -> dict[str, Any] | None:
        """给算子的方向：helped 沿着走（同类改动用到别处、做得更彻底），hurt 避开。没有记录时返回 None。"""
        def top(helped: bool) -> list[str]:
            ds = sorted((d for d in self.items if d.helped == helped), key=lambda d: -self.weight(d, gen))
            return [self.describe(d, gen) for d in ds[: self.max_items]]
        helped, hurt = top(True), top(False) if self.negative else []
        if not helped and not hurt:
            return None
        return {"helped": helped, "hurt": hurt,
                "how_to_use": "helped: changes verified on held-out tasks to improve the score. They are ALREADY "
                              "APPLIED to the current version, so do not repeat them. Apply the same kind of fix to "
                              "the parts that are still failing (other items, other task types), or extend them. "
                              "hurt: changes that lowered the score; do not make them again. Weighted by size and recency."}


__all__ = ["Direction", "DirectionLedger", "profile"]
