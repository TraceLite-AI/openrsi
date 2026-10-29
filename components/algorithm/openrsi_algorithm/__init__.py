"""进化算法：决定下一代从哪个版本出发、用哪个提案器、什么时候停。

| 算法 | 思路来源 | 选父代 |
|---|---|---|
| GreedyAlgorithm | 爬山 | 永远选当前最好的 |
| ArchiveAlgorithm | Darwin Gödel Machine | 按分数加权抽样，子代越多越少被选 |
| ParetoAlgorithm | GEPA、Meta-Harness | 按题比较：在某道题上是最好的才留在前沿，赢的题越多越常被选 |
| ShinkaAlgorithm | ShinkaEvolve | 分数（相对中位数）× 子代惩罚；评估前拒绝和已评估过的太像的补丁；UCB 选提案器 |
| IslandAlgorithm | AlphaEvolve / FunSearch | 多个岛各自爬山，定期把每个岛的最好版本迁移到下一个岛 |
| MapElitesAlgorithm | MAP-Elites | 按行为特征分格，每格留最好的，均匀地从各格选 |

共同的停止条件：代数上限、连续多少代没有新纪录、达到目标分数。
所有算法都只从被接受的版本出发；第一次 tell 登记根版本。
"""
from __future__ import annotations

import bisect
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any

from openrsi import Algorithm, AlgorithmStatus, AskResult, Patch, ScreenResult, TellRequest, serve
from openrsi.types import canonical_json


@dataclass
class Node:
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    task_scores: dict[str, float] = field(default_factory=dict)
    children: int = 0
    order: int = 0  # 进档案的先后，打破平局用


class BaseAlgorithm(Algorithm):
    version = "0.4.0"

    def __init__(self, metric: str = "holdout", max_generations: int = 20, patience: int = 5,
                 target_score: float | None = None, seed: int = 0):
        self.metric = metric
        self.max_generations = max_generations
        self.patience = patience
        self.target_score = target_score
        self.rng = random.Random(seed)
        self.archive: dict[str, Node] = {}
        self.generation = 0
        self.best_id: str | None = None
        self._seen_gens: set[int] = set()
        self._last_improved_gen = 0

    # ------------------------------------------------------------ 停止条件与状态

    @property
    def best_score(self) -> float | None:
        return self.archive[self.best_id].score if self.best_id else None

    @property
    def since_improvement(self) -> int:
        """最近一次刷新纪录之后，又过了多少代。"""
        return sum(1 for g in self._seen_gens if g > self._last_improved_gen)

    def _done(self) -> str | None:
        if not self.archive:
            return "还没有登记根版本（先 tell 一次）"
        if self.generation >= self.max_generations:
            return f"达到代数上限 {self.max_generations}"
        if self.since_improvement >= self.patience:
            return f"连续 {self.patience} 代没有提升"
        if self.target_score is not None and self.best_score is not None and self.best_score >= self.target_score:
            return f"达到目标分数 {self.target_score}"
        return None

    def status(self) -> AlgorithmStatus:
        reason = self._done() if self.archive else None
        return AlgorithmStatus(generation=self.generation, best_id=self.best_id, best_score=self.best_score,
                               archive_size=len(self.archive), done=reason is not None, reason=reason)

    # ------------------------------------------------------------ ask / tell

    def ask(self) -> AskResult:
        reason = self._done()
        if reason:
            return AskResult(generation=self.generation + 1, parent_id=None, done=True, reason=reason)
        return AskResult(generation=self.generation + 1, parent_id=self._pick(), proposer=self._pick_proposer())

    def tell(self, result: TellRequest) -> AlgorithmStatus:
        score = result.metrics.get(self.metric)
        if not self.archive:  # 第一次 tell：登记根版本
            if result.candidate_id is None or score is None:
                raise ValueError(f"第一次 tell 必须给出根版本 id 和指标 {self.metric}")
            self._admit(result.candidate_id, score, result, parent=None)
            self.best_id = result.candidate_id
            self.generation = self._last_improved_gen = result.generation
            return self.status()

        self.generation = max(self.generation, result.generation)
        self._seen_gens.add(result.generation)  # 同一代可能 tell 多次（多个候选），停滞按代计
        if result.parent_id in self.archive:
            self.archive[result.parent_id].children += 1
        improved = False
        if result.accepted and result.candidate_id and score is not None:
            self._admit(result.candidate_id, score, result, parent=result.parent_id)
            if self.best_score is None or score > self.best_score:
                self.best_id, improved = result.candidate_id, True
                self._last_improved_gen = result.generation
        self._observe(result, improved)
        return self.status()

    # ------------------------------------------------------------ 子类覆盖

    def _admit(self, cid: str, score: float, result: TellRequest, parent: str | None) -> None:
        self.archive[cid] = Node(score, dict(result.metrics), dict(result.task_scores), order=len(self.archive))

    def _observe(self, result: TellRequest, improved: bool) -> None:
        """每次 tell 之后的钩子（比如更新提案器的统计）。"""

    def _pick(self) -> str:
        raise NotImplementedError

    def _pick_proposer(self) -> str | None:
        return None

    def _weighted(self, weights: dict[str, float]) -> str:
        ids = sorted(weights)
        total = sum(weights[i] for i in ids)
        if total <= 0:
            return self.best_id  # type: ignore[return-value]
        return self.rng.choices(ids, weights=[weights[i] for i in ids], k=1)[0]


class GreedyAlgorithm(BaseAlgorithm):
    name = "openrsi/greedy"

    def _pick(self) -> str:
        return self.best_id  # type: ignore[return-value]


class ArchiveAlgorithm(BaseAlgorithm):
    """with: temperature（越小越偏向高分版本）"""

    name = "openrsi/archive"

    def __init__(self, temperature: float = 0.1, **kw: Any):
        super().__init__(**kw)
        self.temperature = temperature

    def _pick(self) -> str:
        top = max(n.score for n in self.archive.values())
        return self._weighted({i: math.exp((n.score - top) / self.temperature) / (1 + n.children)
                               for i, n in self.archive.items()})


class ParetoAlgorithm(BaseAlgorithm):
    """GEPA 式的按题前沿：一个版本只要在某道题上达到最好就留在前沿；被选中的概率和它"赢"的题数成正比。
    这样在不同题上各有所长的版本都能保留下来，避免平均分把它们抹掉。没有按题分数时退化为贪心。"""

    name = "openrsi/pareto"

    def frontier(self) -> dict[str, int]:
        tasks = {t for n in self.archive.values() for t in n.task_scores}
        wins: dict[str, int] = {}
        for t in tasks:
            best = max(n.task_scores.get(t, -math.inf) for n in self.archive.values())
            for i, n in self.archive.items():
                if n.task_scores.get(t, -math.inf) == best:
                    wins[i] = wins.get(i, 0) + 1
        # 去掉被支配的：另一个版本在每道题上都不差、且至少一道更好
        def dominated(a: str) -> bool:
            sa = self.archive[a].task_scores
            for b in wins:
                if b == a:
                    continue
                sb = self.archive[b].task_scores
                if all(sb.get(t, -math.inf) >= sa.get(t, -math.inf) for t in tasks) and \
                        any(sb.get(t, -math.inf) > sa.get(t, -math.inf) for t in tasks):
                    return True
            return False

        return {i: w for i, w in wins.items() if not dominated(i)}

    def _pick(self) -> str:
        front = self.frontier()
        if not front:
            return self.best_id  # type: ignore[return-value]
        return self._weighted({i: float(w) for i, w in front.items()})


def _shingles(text: str, n: int = 5) -> set[str]:
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def _sim(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class ShinkaAlgorithm(BaseAlgorithm):
    """ShinkaEvolve 的三个机制：

    - 父代选择：sigmoid(λ·(分数 − 中位数)/离散度) × 1/(1+子代数)，兼顾好成绩和没被充分探索的版本
    - 新颖度筛查（screen）：补丁和已评估过的补丁按字符片段算相似度，超过阈值就不评估，省评估成本
    - 自适应提案器：proposers 列出可选的提案器名字，用 UCB1 选；候选被接受且刷新纪录记 1 分

    with: proposers、novelty_threshold（默认 0.95）、selection_lambda（默认 10）
    """

    name = "openrsi/shinka"

    def __init__(self, proposers: list[str] | None = None, novelty_threshold: float = 0.95,
                 selection_lambda: float = 10.0, **kw: Any):
        super().__init__(**kw)
        self.proposers = list(proposers or [])
        self.novelty_threshold = novelty_threshold
        self.selection_lambda = selection_lambda
        self.seen_patches: list[set[str]] = []
        self.arm_pulls = {p: 0 for p in self.proposers}
        self.arm_reward = {p: 0.0 for p in self.proposers}

    def _pick(self) -> str:
        scores = [n.score for n in self.archive.values()]
        med = statistics.median(scores)
        spread = statistics.median([abs(s - med) for s in scores]) or 1e-6
        return self._weighted({
            i: (1 / (1 + math.exp(-self.selection_lambda * (n.score - med) / spread))) / (1 + n.children)
            for i, n in self.archive.items()})

    def screen(self, patch: Patch, parent_id: str) -> ScreenResult:
        sh = _shingles(canonical_json(patch.ops))
        sim = max((_sim(sh, s) for s in self.seen_patches), default=0.0)
        if sim >= self.novelty_threshold:
            return ScreenResult(accept=False, novelty=round(1 - sim, 4),
                                reason=f"和已评估过的补丁相似度 {sim:.2f} ≥ {self.novelty_threshold}")
        self.seen_patches.append(sh)
        return ScreenResult(accept=True, novelty=round(1 - sim, 4))

    def _pick_proposer(self) -> str | None:
        if not self.proposers:
            return None
        for p in self.proposers:  # 每个先试一次
            if self.arm_pulls[p] == 0:
                return p
        total = sum(self.arm_pulls.values())
        return max(self.proposers, key=lambda p: self.arm_reward[p] / self.arm_pulls[p]
                   + math.sqrt(2 * math.log(total) / self.arm_pulls[p]))

    def _observe(self, result: TellRequest, improved: bool) -> None:
        # CompositeProposer 把成员名写在 proposer_ref 的 # 后面，如 "openrsi/lesson-proposer:0.4.0#lesson"
        arm = result.proposer.rsplit("#", 1)[-1] if result.proposer else None
        arm = arm if arm in self.arm_pulls else None
        if arm:  # 没有产出、被筛掉、被拦下、没通过准入，都算这个提案器的一次失败
            self.arm_pulls[arm] += 1
            self.arm_reward[arm] += 1.0 if improved else 0.0


class IslandAlgorithm(BaseAlgorithm):
    """多个岛各自进化，互不干扰；每隔 migrate_every 代，把每个岛的最好版本复制到下一个岛（环形）。
    保持种群多样性，避免所有搜索都挤到同一个局部最优。with: islands（默认 4）、migrate_every（默认 5）"""

    name = "openrsi/islands"

    def __init__(self, islands: int = 4, migrate_every: int = 5, **kw: Any):
        super().__init__(**kw)
        self.n_islands = islands
        self.migrate_every = migrate_every
        self.members: list[set[str]] = [set() for _ in range(islands)]
        self.island_of_gen: dict[int, int] = {}
        self._migrated_at: set[int] = set()

    def _admit(self, cid: str, score: float, result: TellRequest, parent: str | None) -> None:
        super()._admit(cid, score, result, parent)
        if parent is None:
            for m in self.members:  # 根版本放进每个岛
                m.add(cid)
        else:
            island = self.island_of_gen.get(result.generation, 0)
            self.members[island].add(cid)

    def _best_in(self, island: int) -> str:
        return max(self.members[island], key=lambda i: (self.archive[i].score, -self.archive[i].order))

    def ask(self) -> AskResult:
        gen = self.generation + 1
        if self.migrate_every and gen % self.migrate_every == 0 and gen not in self._migrated_at:
            self._migrated_at.add(gen)
            bests = [self._best_in(k) for k in range(self.n_islands)]
            for k, b in enumerate(bests):
                self.members[(k + 1) % self.n_islands].add(b)
        self.island_of_gen[gen] = (gen - 1) % self.n_islands
        return super().ask()

    def _pick(self) -> str:
        return self._best_in(self.island_of_gen[self.generation + 1])


class MapElitesAlgorithm(BaseAlgorithm):
    """按行为特征把版本分到格子里，每格只留最好的，选父代时各格机会均等。
    descriptors 列出用作特征的指标和分箱边界，比如
    [{"metric": "rubric:tests_pass", "bins": [0.5]}, {"metric": "n_items", "bins": [5, 20]}]。"""

    name = "openrsi/map-elites"

    def __init__(self, descriptors: list[dict[str, Any]] | None = None, **kw: Any):
        super().__init__(**kw)
        self.descriptors = descriptors or []
        self.elites: dict[tuple[int, ...], str] = {}

    def cell(self, metrics: dict[str, float]) -> tuple[int, ...]:
        return tuple(bisect.bisect_right(d["bins"], metrics.get(d["metric"], 0.0)) for d in self.descriptors)

    def _admit(self, cid: str, score: float, result: TellRequest, parent: str | None) -> None:
        super()._admit(cid, score, result, parent)
        c = self.cell(result.metrics)
        cur = self.elites.get(c)
        if cur is None or score > self.archive[cur].score:
            self.elites[c] = cid

    def _pick(self) -> str:
        cells = sorted(self.elites)
        return self.elites[self.rng.choice(cells)] if cells else self.best_id  # type: ignore[return-value]


ALGORITHMS = {"greedy": GreedyAlgorithm, "archive": ArchiveAlgorithm, "pareto": ParetoAlgorithm,
              "shinka": ShinkaAlgorithm, "islands": IslandAlgorithm, "map-elites": MapElitesAlgorithm}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-algorithm")
    ap.add_argument("--kind", choices=sorted(ALGORITHMS), default="archive")
    ap.add_argument("--metric", default="holdout")
    ap.add_argument("--max-generations", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()
    serve(ALGORITHMS[args.kind](metric=args.metric, max_generations=args.max_generations, patience=args.patience))


__all__ = ["ALGORITHMS", "ArchiveAlgorithm", "BaseAlgorithm", "GreedyAlgorithm", "IslandAlgorithm",
           "MapElitesAlgorithm", "ParetoAlgorithm", "ShinkaAlgorithm", "main"]
