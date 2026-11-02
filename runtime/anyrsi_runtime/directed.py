"""有方向的自我改进：用组内比较代替求梯度（设计见《思想与设计》"方向与步长"一页）。

一组 = "不改"对照 + G 个候选方向 × 每个方向 m 个具体修改，全部在同一批练习题上做，按题配对比较：

  组内优势      a(d, j, t) = 修改 (d, j) 在题 t 上的分数 − "不改"在题 t 上的分数（按题扣掉难度）
  胜出        方向相对"不改"的平均优势显著大于 0。门槛按 Bonferroni 分摊到这一组的方向数上；
              父版本没变时，同一方向在前几组里的配对差值会累积起来一起检验；这个方向第 k 次被检验时 α
              再乘 1/(k(k+1))，所有次加起来总的误收率仍不超过 α（反复看数据不会抬高误收率）
  方向成不成立  嵌套方差分析：方向之间的差别要显著大于同一方向内几个修改之间的差别
                （m ≥ 2 才能检验；m = 1 时分不清"方向好"还是"这次写法运气好"）。不成立时胜出的修改
                仍可以去 holdout 准入，但不记作"方向已确定"
  有没有差别    Friedman 检验（以题为区组比较所有版本的排名）只报告不拦截：和对照的逐个比较已经做了
                多重校正，再加一道总体检验只会损失功效（0/1 分数并列很多时尤其明显）
  一致性        按题型（tags / 环境格）分别算平均优势，有题型显著变差时记下适用范围
  动量          m_d ← β·m_d + (1 − β)·Ā_d·max(0, 1 − 1/z²)（按证据强弱收缩，弱证据几乎不动）；
                下一组按它挑候选方向
  收不收        方向是"往哪里多试"，不是"收不收"。默认每组把练习分最好的那个具体修改（相对对照的优势 > 0）
                送去 holdout 准入：提案器质量参差不齐时，挑出好的个例比等"方向平均显著"快得多，
                挑选带来的偏差由另一批数据（holdout）纠正。也可以设 submit: established，只在方向成立时才送
  收敛          连续 patience 组"有把握没有方向能提升 δ 以上"：所有试过的方向，平均优势的单侧置信上界
                都低于 min_effect。只是证据不足（上界还高于 δ）的组不算，免得把"没攒够数据"当成"到头了"

本模块只放纯函数，不依赖 numpy / scipy；循环的接线在 loop.py 的 _directed_generation。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist, fmean, stdev
from typing import Any

# ---------------------------------------------------------------- 分布函数


def _gammainc_q(s: float, x: float) -> float:
    """正则化上不完全伽马函数 Q(s, x)（级数 / 连分式，Numerical Recipes 6.2）。"""
    if x <= 0:
        return 1.0
    gln = math.lgamma(s)
    if x < s + 1:
        ap, total, delta = s, 1.0 / s, 1.0 / s
        for _ in range(1000):
            ap += 1
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + s * math.log(x) - gln))
    b, c, d = x + 1 - s, 1e300, 1.0 / (x + 1 - s)
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + s * math.log(x) - gln) * h)


def chi2_sf(x: float, df: int) -> float:
    """卡方分布的右尾概率 P(X ≥ x)。"""
    return 1.0 if x <= 0 else _gammainc_q(df / 2, x / 2)


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1 / d
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1 + aa / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a, b)。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def f_sf(f: float, d1: int, d2: int) -> float:
    """F 分布的右尾概率 P(F ≥ f)。"""
    if f <= 0:
        return 1.0
    if math.isinf(f):
        return 0.0
    return betainc(d2 / 2, d1 / 2, d2 / (d2 + d1 * f))


def z_threshold(alpha: float, tests: int = 1) -> float:
    """单侧检验的 z 门槛，按 Bonferroni 分摊到 tests 次检验：Φ⁻¹(1 − α / tests)。"""
    return NormalDist().inv_cdf(1 - alpha / max(1, tests))


# ---------------------------------------------------------------- 检验


def _ranks(values: list[float]) -> list[float]:
    """平均秩（并列取平均），从 1 开始。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def friedman(blocks: list[list[float]]) -> tuple[float, float]:
    """Friedman 检验：blocks 每行是一道题上各版本的分数（列 = 版本）。返回 (统计量, p)。

    带并列校正；全部并列的题不提供信息，自动不计。少于 2 个版本或没有有效的题时 p = 1。
    """
    rows = [b for b in blocks if len(set(b)) > 1]
    if not rows or len(rows[0]) < 2:
        return 0.0, 1.0
    k, n = len(rows[0]), len(rows)
    rank_sums = [0.0] * k
    tie_term = 0.0
    for b in rows:
        r = _ranks(b)
        for i in range(k):
            rank_sums[i] += r[i]
        counts: dict[float, int] = {}
        for v in b:
            counts[v] = counts.get(v, 0) + 1
        tie_term += sum(c ** 3 - c for c in counts.values())
    stat = 12 / (n * k * (k + 1)) * sum(s * s for s in rank_sums) - 3 * n * (k + 1)
    stat /= 1 - tie_term / (n * (k ** 3 - k))
    return stat, chi2_sf(stat, k - 1)


def nested_anova(edit_means: dict[str, list[float]], n_tasks: int) -> tuple[float, float]:
    """方向之间 vs 同一方向内部的修改之间。edit_means[方向] = 该方向每个修改在 n_tasks 道题上的平均优势。

    F = MS方向间 / MS方向内修改间，自由度 (G − 1, G(m − 1))。每个方向的修改数要相同且 ≥ 2，方向数 ≥ 2。
    返回 (F, p)；条件不满足时 (nan, nan)。
    """
    groups = list(edit_means.values())
    g = len(groups)
    if g < 2 or any(len(x) != len(groups[0]) for x in groups) or len(groups[0]) < 2:
        return math.nan, math.nan
    m = len(groups[0])
    grand = fmean(v for x in groups for v in x)
    means = [fmean(x) for x in groups]
    ms_dir = n_tasks * m * sum((mu - grand) ** 2 for mu in means) / (g - 1)
    ms_edit = n_tasks * sum((v - mu) ** 2 for x, mu in zip(groups, means) for v in x) / (g * (m - 1))
    if ms_edit == 0:
        return (math.inf, 0.0) if ms_dir > 0 else (0.0, 1.0)
    f = ms_dir / ms_edit
    return f, f_sf(f, g - 1, g * (m - 1))


def mean_z(diffs: list[float]) -> tuple[float, float]:
    """配对差值的均值和 z（均值 / 标准误）。全部相同且非零时 z = ±inf。"""
    if not diffs:
        return 0.0, 0.0
    mu = fmean(diffs)
    if len(diffs) < 2:
        return mu, 0.0
    sd = stdev(diffs)
    if sd == 0:
        return mu, (math.inf if mu > 0 else -math.inf if mu < 0 else 0.0)
    return mu, mu / (sd / math.sqrt(len(diffs)))


# ---------------------------------------------------------------- 一组的结论


@dataclass
class Version:
    """组里的一个版本。direction 为 None 的是"不改"对照。"""

    direction: str | None
    edit: int
    task_scores: dict[str, float]                      # 题 → 分数（多次 rollout 取平均）
    rubric: dict[str, dict[str, float]] = field(default_factory=dict)  # 题 → rubric 项 → 通过率
    step: Any = None


@dataclass
class DirectionResult:
    direction: str
    advantage: float                   # 相对"不改"的平均优势（含累积的历史）
    z: float
    significant: bool
    best_edit: int
    n: int = 0                          # 参与检验的配对差值个数（含累积的历史）
    se: float = 0.0                     # 平均优势的标准误
    z_crit: float = 0.0                 # 这个方向这一次的门槛（按它被检验的次数收紧）
    diffs: list[float] = field(default_factory=list)  # 这一组的逐题差值，供下一组累积
    per_rubric: dict[str, float] = field(default_factory=dict)
    per_group: dict[str, float] = field(default_factory=dict)
    scope: list[str] | None = None     # 有题型显著变差时：只在这些题型上适用


@dataclass
class GroupResult:
    tasks: list[str]
    friedman_stat: float
    friedman_p: float
    anova_f: float
    anova_p: float
    z_crit: float
    directions: list[DirectionResult]
    winner: str | None
    established: bool                  # 方向成立：Friedman、相对对照、（m ≥ 2 时）嵌套方差分析都通过
    reason: str

    def summary(self) -> dict[str, Any]:
        return {"friedman_p": self.friedman_p, "anova_f": self.anova_f, "anova_p": self.anova_p,
                "z_crit": self.z_crit, "winner": self.winner, "established": self.established, "reason": self.reason,
                "directions": [{"direction": d.direction, "advantage": d.advantage, "z": d.z, "n": d.n,
                                "z_crit": d.z_crit,
                                "significant": d.significant, "per_rubric": d.per_rubric, "scope": d.scope}
                               for d in self.directions]}


def analyze_group(control: Version, versions: list[Version], *, alpha: float, tests: int,
                  task_groups: dict[str, str] | None = None, history: dict[str, list[float]] | None = None,
                  looks: dict[str, int] | None = None) -> GroupResult:
    """一组的统计结论。tests = Bonferroni 分摊的方向检验数；history = 同一父版本下各方向以前累积的配对差值；
    looks = 各方向以前被检验过几次（这一次是第 k = looks + 1 次，门槛按 α / (tests · k · (k + 1)) 收紧）。"""
    tasks = sorted(control.task_scores)
    for v in versions:
        if set(v.task_scores) != set(tasks):
            raise ValueError(f"版本 {v.direction}#{v.edit} 做的题和对照不一样，不能配对")
    z_crit = z_threshold(alpha, tests * 2)  # 第一次检验的门槛
    stat, p = friedman([[control.task_scores[t], *(v.task_scores[t] for v in versions)] for t in tasks])

    by_dir: dict[str, list[Version]] = {}
    for v in versions:
        by_dir.setdefault(str(v.direction), []).append(v)
    adv = {(v.direction, v.edit): [v.task_scores[t] - control.task_scores[t] for t in tasks] for v in versions}
    edit_means = {d: [fmean(adv[(d, v.edit)]) for v in vs] for d, vs in by_dir.items()}
    f, fp = nested_anova(edit_means, len(tasks))

    results = []
    for d, vs in by_dir.items():
        per_task = [fmean(adv[(d, v.edit)][i] for v in vs) for i in range(len(tasks))]
        pooled = [*(history or {}).get(d, []), *per_task]
        mu, z = mean_z(pooled)
        se = stdev(pooled) / math.sqrt(len(pooled)) if len(pooled) >= 2 else 0.0
        k = (looks or {}).get(d, 0) + 1
        crit = z_threshold(alpha, tests * k * (k + 1))
        best = max(vs, key=lambda v: fmean(adv[(d, v.edit)]))
        rubric: dict[str, list[float]] = {}
        for v in vs:
            for t in tasks:
                for item, val in v.rubric.get(t, {}).items():
                    base = control.rubric.get(t, {}).get(item)
                    if base is not None:
                        rubric.setdefault(item, []).append(val - base)
        groups: dict[str, list[float]] = {}
        for i, t in enumerate(tasks):
            groups.setdefault((task_groups or {}).get(t, "*"), []).append(per_task[i])
        per_group = {g: fmean(x) for g, x in groups.items()}
        bad = [g for g, x in groups.items() if len(x) >= 3 and mean_z(x)[1] <= -z_threshold(alpha)]
        results.append(DirectionResult(
            direction=d, advantage=mu, z=z, n=len(pooled), se=se, diffs=per_task, z_crit=crit,
            significant=mu > 0 and z >= crit, best_edit=best.edit,
            per_rubric={k: fmean(x) for k, x in sorted(rubric.items())}, per_group=per_group,
            scope=sorted(g for g in groups if g not in bad) if bad else None))
    results.sort(key=lambda r: (-r.significant, -r.advantage))

    sizes = {len(vs) for vs in by_dir.values()}
    sig = [r for r in results if r.significant]

    def done(winner: str | None, established: bool, reason: str) -> GroupResult:
        return GroupResult(tasks, stat, p, f, fp, z_crit, results, winner, established, reason)

    if not sig:
        best = results[0] if results else None
        hint = (f"；最接近的是 {best.direction}（z = {best.z:.2f}，门槛 {best.z_crit:.2f}，累积 {best.n} 个配对）"
                if best else "")
        return done(None, False, f"没有方向显著胜过不改{hint}")
    win = sig[0].direction
    beat = f"{win} 胜过不改（平均优势 {sig[0].advantage:+.3f}，z = {sig[0].z:.2f}）"
    if len(by_dir[win]) < 2:
        return done(win, False, beat + "，但这个方向只有 1 个修改，分不清是方向好还是这次写法好")
    if len(by_dir) >= 2:
        if len(sizes) != 1:
            return done(win, False, beat + "，但各方向的修改数不一样，做不了嵌套方差分析")
        if not fp <= alpha:
            return done(win, False, beat + f"，但方向之间的差别没有显著大于同方向内的差别（F = {f:.2f}, p = {fp:.3g}），"
                                          "可能只是某次写法好")
    return done(win, True, f"方向成立：{beat}")


def flat(group: GroupResult, min_effect: float, confidence: float = 0.95) -> bool:
    """这一组是否"有把握没有方向能提升 min_effect 以上"：每个方向的单侧上界 Ā + z·SE 都低于 min_effect。"""
    if group.winner is not None or not group.directions:
        return False
    za = z_threshold(1 - confidence)
    for r in group.directions:
        if r.n < 2:
            return False  # 数据太少，不能说"有把握"
        if r.advantage + za * r.se >= min_effect:
            return False
    return True


def update_momentum(momentum: dict[str, float], group: GroupResult, beta: float) -> dict[str, float]:
    """m_d ← β·m_d + (1 − β)·Ā_d·max(0, 1 − 1/z²)：证据弱时收缩到 0 附近（正部 James–Stein 收缩）。
    没被试过的方向不变。"""
    out = dict(momentum)
    for r in group.directions:
        shrink = 1.0 if math.isinf(r.z) else max(0.0, 1 - 1 / (r.z * r.z)) if r.z else 0.0
        gain = r.advantage * shrink
        out[r.direction] = beta * out.get(r.direction, 0.0) + (1 - beta) * gain
    return out


def pick_directions(candidates: dict[str, float], momentum: dict[str, float], k: int, explore: int,
                    rng: Any) -> list[str]:
    """挑这一组要试的方向：先按动量、再按归因支持度排序取 k − explore 个，其余名额随机给没排上的方向。"""
    ranked = sorted(candidates, key=lambda d: (-momentum.get(d, 0.0), -candidates[d], d))
    head = ranked[: max(0, k - explore)]
    rest = [d for d in ranked if d not in head]
    rng.shuffle(rest)
    return head + rest[: k - len(head)]


__all__ = ["DirectionResult", "GroupResult", "Version", "analyze_group", "betainc", "chi2_sf", "f_sf", "flat", "friedman",
           "mean_z", "nested_anova", "pick_directions", "update_momentum", "z_threshold"]
