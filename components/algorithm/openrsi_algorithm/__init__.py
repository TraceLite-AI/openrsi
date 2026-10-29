"""进化算法：决定下一代从哪个快照出发，什么时候停。

- GreedyAlgorithm：永远从当前最好的快照出发（爬山）。
- ArchiveAlgorithm：保留所有被接受的快照，按分数加权抽样父代，子代越多越少被选中，
  让搜索不至于困在一条路上（思路类似 Darwin Gödel Machine 的档案式开放进化）。

两者共用停止条件：代数上限、连续多少代没有新纪录、达到目标分数。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from openrsi import Algorithm, AlgorithmStatus, AskResult, TellRequest, serve


@dataclass
class _Node:
    score: float
    children: int = 0


class _Base(Algorithm):
    version = "0.3.0"

    def __init__(self, metric: str = "holdout", max_generations: int = 20, patience: int = 5,
                 target_score: float | None = None):
        self.metric = metric
        self.max_generations = max_generations
        self.patience = patience
        self.target_score = target_score
        self.archive: dict[str, _Node] = {}
        self.generation = 0
        self.best_id: str | None = None
        self._seen_gens: set[int] = set()
        self._last_improved_gen = 0

    @property
    def since_improvement(self) -> int:
        """最近一次刷新纪录之后，又过了多少代。"""
        return sum(1 for g in self._seen_gens if g > self._last_improved_gen)

    @property
    def best_score(self) -> float | None:
        return self.archive[self.best_id].score if self.best_id else None

    def _done(self) -> str | None:
        if not self.archive:
            return "还没有登记根快照（先 tell 一次）"
        if self.generation >= self.max_generations:
            return f"达到代数上限 {self.max_generations}"
        if self.since_improvement >= self.patience:
            return f"连续 {self.patience} 代没有提升"
        if self.target_score is not None and self.best_score is not None and self.best_score >= self.target_score:
            return f"达到目标分数 {self.target_score}"
        return None

    def ask(self) -> AskResult:
        reason = self._done()
        if reason:
            return AskResult(generation=self.generation + 1, parent_id=None, done=True, reason=reason)
        return AskResult(generation=self.generation + 1, parent_id=self._pick())

    def tell(self, result: TellRequest) -> AlgorithmStatus:
        score = result.metrics.get(self.metric)
        if not self.archive:  # 第一次 tell：登记根快照
            if result.candidate_id is None or score is None:
                raise ValueError(f"第一次 tell 必须给出根快照 id 和指标 {self.metric}")
            self.archive[result.candidate_id] = _Node(score)
            self.best_id = result.candidate_id
            self.generation = self._last_improved_gen = result.generation
            return self.status()

        self.generation = max(self.generation, result.generation)
        if result.parent_id in self.archive:
            self.archive[result.parent_id].children += 1
        self._seen_gens.add(result.generation)  # 同一代可能 tell 多次（多个候选），停滞按代计
        if result.accepted and result.candidate_id and score is not None:
            self.archive[result.candidate_id] = _Node(score)
            if self.best_score is None or score > self.best_score:
                self.best_id = result.candidate_id
                self._last_improved_gen = result.generation
        return self.status()

    def status(self) -> AlgorithmStatus:
        reason = self._done() if self.archive else None
        return AlgorithmStatus(generation=self.generation, best_id=self.best_id, best_score=self.best_score,
                               archive_size=len(self.archive), done=reason is not None, reason=reason)

    def _pick(self) -> str:
        raise NotImplementedError


class GreedyAlgorithm(_Base):
    name = "openrsi/greedy"

    def _pick(self) -> str:
        return self.best_id  # type: ignore[return-value]


class ArchiveAlgorithm(_Base):
    """with: temperature（越小越偏向高分快照）、seed"""

    name = "openrsi/archive"

    def __init__(self, temperature: float = 0.1, seed: int = 0, **kw):
        super().__init__(**kw)
        self.temperature = temperature
        self.rng = random.Random(seed)

    def _pick(self) -> str:
        ids = sorted(self.archive)
        top = max(n.score for n in self.archive.values())
        weights = [math.exp((self.archive[i].score - top) / self.temperature) / (1 + self.archive[i].children)
                   for i in ids]
        return self.rng.choices(ids, weights=weights, k=1)[0]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-algorithm")
    ap.add_argument("--kind", choices=["greedy", "archive"], default="archive")
    ap.add_argument("--max-generations", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    cls = GreedyAlgorithm if args.kind == "greedy" else ArchiveAlgorithm
    serve(cls(max_generations=args.max_generations, patience=args.patience))


__all__ = ["ArchiveAlgorithm", "GreedyAlgorithm", "main"]
