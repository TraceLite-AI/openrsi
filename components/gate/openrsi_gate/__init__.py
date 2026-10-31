"""准入：候选必须在 holdout 上真的变好，才能写进谱系。

检查项（全部通过才接受）：
- improvement   候选的 holdout 比父快照高出 min_gain 以上
- significance  可选：用双比例 z 检验确认提升不是噪声（需要 n_holdout）
- paired_significance
                可选：逐题配对检验。父版本和候选做的是同一批题，配对比较能扣掉题目难度带来的波动，
                比两组独立样本的检验灵敏得多。逐题分数全是 0/1 时用 McNemar 检验（只看结果翻转的题），
                否则用逐题差值的均值 z。候选缺任何一道父版本有的题，直接判为不通过
- anchor        有 anchor 分数时，候选不能比父快照低超过 anchor_tolerance
- no_regression 可选（Self-Harness、AHE 的做法）：逐题比较，分数下降超过 task_tolerance 的题不能多于
                max_task_regressions 道；设为 0 就是"修好失败的同时，任何一道题都不许变差"
- no_cell_regression 可选（环境格模式）：按 <task_id>@<cell_id> 的键逐格求平均，平均分下降超过 cell_tolerance
                的格不能多于 max_cell_regressions 个；设为 0 就是"任何一个环境都不许变差"
- cost          有 cost 且设置了 max_cost 时，不能超预算
"""
from __future__ import annotations

import math

from openrsi import Gate, GateCheck, GateDecision, GateRequest, serve


def two_proportion_z(p1: float, n1: float, p2: float, n2: float) -> float:
    """p2 相对 p1 的提升有多少个标准误。"""
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return math.inf if p2 > p1 else 0.0
    return (p2 - p1) / se


def paired_z(baseline: dict[str, float], candidate: dict[str, float]) -> tuple[float, str]:
    """候选相对父版本的逐题配对提升有多少个标准误，以及给人看的说明。

    两边的题目必须一致（调用方负责检查）。分数全是 0/1 时用 McNemar：z = (修好 − 改坏) / √(修好 + 改坏)；
    否则对逐题差值 d 算 z = mean(d) / (sd(d) / √n)。没有任何差异时 z = 0；差异全同号且大小一样时为 ±inf。
    """
    keys = sorted(baseline)
    pairs = [(baseline[k], candidate[k]) for k in keys]
    n = len(pairs)
    if all(b in (0.0, 1.0) and c in (0.0, 1.0) for b, c in pairs):
        fixed = sum(1 for b, c in pairs if b == 0.0 and c == 1.0)
        broke = sum(1 for b, c in pairs if b == 1.0 and c == 0.0)
        detail = f"McNemar：{n} 道题，修好 {fixed}、改坏 {broke}"
        if fixed + broke == 0:
            return 0.0, detail
        return (fixed - broke) / math.sqrt(fixed + broke), detail
    diffs = [c - b for b, c in pairs]
    mean = math.fsum(diffs) / n
    detail = f"逐题差值：{n} 道题，平均 {mean:+.4f}"
    if n < 2:
        return (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0), detail
    var = math.fsum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0), detail
    return mean / math.sqrt(var / n), detail


class HoldoutGate(Gate):
    name = "openrsi/holdout-gate"
    version = "0.4.2"

    def __init__(self, metric: str = "holdout", min_gain: float = 0.0, significance_z: float | None = None,
                 anchor_tolerance: float = 0.0, max_cost: float | None = None,
                 max_task_regressions: int | None = None, task_tolerance: float = 0.0,
                 paired_significance_z: float | None = None,
                 max_cell_regressions: int | None = None, cell_tolerance: float = 0.0):
        self.paired_significance_z = paired_significance_z
        self.max_cell_regressions, self.cell_tolerance = max_cell_regressions, cell_tolerance
        self.max_task_regressions = max_task_regressions
        self.task_tolerance = task_tolerance
        self.metric = metric
        self.min_gain = min_gain
        self.significance_z = significance_z
        self.anchor_tolerance = anchor_tolerance
        self.max_cost = max_cost

    def check(self, request: GateRequest) -> GateDecision:
        base, cand = request.baseline, request.candidate
        metric = request.primary_metric or self.metric
        checks: list[GateCheck] = []

        if metric not in cand or metric not in base:
            checks.append(GateCheck(name="improvement", passed=False, detail=f"缺少指标 {metric}"))
        else:
            b, c = base[metric], cand[metric]
            ok = c > b + self.min_gain
            checks.append(GateCheck(name="improvement", passed=ok,
                                    detail=f"{metric} {b:.4f} → {c:.4f}（要求提升 > {self.min_gain}）"))
            if self.significance_z is not None:
                n_key = f"n_{metric}"
                if n_key in base and n_key in cand:
                    z = two_proportion_z(b, base[n_key], c, cand[n_key])
                    checks.append(GateCheck(name="significance", passed=z >= self.significance_z,
                                            detail=f"z={z:.2f}（要求 ≥ {self.significance_z}）"))
                else:
                    checks.append(GateCheck(name="significance", passed=False, detail=f"缺少样本量 {n_key}"))

        if self.paired_significance_z is not None:
            checks.append(self._paired_check(request))

        if "anchor" in base and "anchor" in cand:
            ok = cand["anchor"] >= base["anchor"] - self.anchor_tolerance
            checks.append(GateCheck(name="anchor", passed=ok,
                                    detail=f"anchor {base['anchor']:.4f} → {cand['anchor']:.4f}"))

        if self.max_task_regressions is not None:
            if not request.baseline_tasks or not request.candidate_tasks:
                checks.append(GateCheck(name="no_regression", passed=False, detail="缺少逐题分数"))
            else:
                worse = sorted(t for t, b in request.baseline_tasks.items()
                               if request.candidate_tasks.get(t, 0.0) < b - self.task_tolerance)
                ok = len(worse) <= self.max_task_regressions
                shown = ", ".join(worse[:5]) + ("…" if len(worse) > 5 else "")
                checks.append(GateCheck(name="no_regression", passed=ok,
                                        detail=f"{len(worse)} 道题变差（上限 {self.max_task_regressions}）"
                                               + (f"：{shown}" if worse else "")))

        if self.max_cell_regressions is not None:
            checks.append(self._cells(request))

        if self.max_cost is not None and "cost" in cand:
            ok = cand["cost"] <= self.max_cost
            checks.append(GateCheck(name="cost", passed=ok, detail=f"cost {cand['cost']} / 上限 {self.max_cost}"))

        failed = [c.name for c in checks if not c.passed]
        return GateDecision(candidate_id=request.candidate_id, accepted=not failed, checks=checks,
                            reason="全部检查通过" if not failed else f"未通过：{', '.join(failed)}")

    def _paired_check(self, request: GateRequest) -> GateCheck:
        need = self.paired_significance_z
        if not request.baseline_tasks or not request.candidate_tasks:
            return GateCheck(name="paired_significance", passed=False, detail="缺少逐题分数")
        missing = sorted(set(request.baseline_tasks) - set(request.candidate_tasks))
        if missing:
            shown = ", ".join(missing[:5]) + ("…" if len(missing) > 5 else "")
            return GateCheck(name="paired_significance", passed=False,
                             detail=f"候选缺 {len(missing)} 道父版本做过的题：{shown}")
        z, detail = paired_z(request.baseline_tasks, request.candidate_tasks)
        return GateCheck(name="paired_significance", passed=z >= need,
                         detail=f"{detail}；z={z:.2f}（要求 ≥ {need}）")

    def _cells(self, request: GateRequest) -> GateCheck:
        """环境格模式下逐格不退步：逐题分数的键是 <task_id>@<cell_id>，按格求平均，比较父版本和候选。"""
        def by_cell(scores: dict[str, float]) -> dict[str, float]:
            acc: dict[str, list[float]] = {}
            for key, v in scores.items():
                if "@" in key:
                    acc.setdefault(key.rsplit("@", 1)[1], []).append(v)
            return {c: sum(vs) / len(vs) for c, vs in acc.items()}

        base, cand = by_cell(request.baseline_tasks), by_cell(request.candidate_tasks)
        if not base or set(base) != set(cand):
            return GateCheck(name="no_cell_regression", passed=False,
                             detail="缺少按格的逐题分数，或父版本和候选的格不一致")
        worse = sorted(c for c in base if cand[c] < base[c] - self.cell_tolerance)
        shown = ", ".join(f"{c} {base[c]:.3f}→{cand[c]:.3f}" for c in worse[:5])
        return GateCheck(name="no_cell_regression", passed=len(worse) <= self.max_cell_regressions,
                         detail=f"{len(worse)} 个格变差（上限 {self.max_cell_regressions}）" + (f"：{shown}" if worse else ""))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-gate")
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--significance-z", type=float)
    ap.add_argument("--max-task-regressions", type=int)
    ap.add_argument("--paired-significance-z", type=float)
    args = ap.parse_args()
    serve(HoldoutGate(min_gain=args.min_gain, significance_z=args.significance_z,
                      max_task_regressions=args.max_task_regressions,
                      paired_significance_z=args.paired_significance_z))


__all__ = ["HoldoutGate", "main", "paired_z", "two_proportion_z"]
