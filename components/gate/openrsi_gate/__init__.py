"""准入：候选必须在 holdout 上真的变好，才能写进谱系。

检查项（全部通过才接受）：
- improvement   候选的 holdout 比父快照高出 min_gain 以上
- significance  可选：用双比例 z 检验确认提升不是噪声（需要 n_holdout）
- anchor        有 anchor 分数时，候选不能比父快照低超过 anchor_tolerance
- no_regression 可选（Self-Harness、AHE 的做法）：逐题比较，分数下降超过 task_tolerance 的题不能多于
                max_task_regressions 道；设为 0 就是"修好失败的同时，任何一道题都不许变差"
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


class HoldoutGate(Gate):
    name = "openrsi/holdout-gate"
    version = "0.4.0"

    def __init__(self, metric: str = "holdout", min_gain: float = 0.0, significance_z: float | None = None,
                 anchor_tolerance: float = 0.0, max_cost: float | None = None,
                 max_task_regressions: int | None = None, task_tolerance: float = 0.0):
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

        if self.max_cost is not None and "cost" in cand:
            ok = cand["cost"] <= self.max_cost
            checks.append(GateCheck(name="cost", passed=ok, detail=f"cost {cand['cost']} / 上限 {self.max_cost}"))

        failed = [c.name for c in checks if not c.passed]
        return GateDecision(candidate_id=request.candidate_id, accepted=not failed, checks=checks,
                            reason="全部检查通过" if not failed else f"未通过：{', '.join(failed)}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-gate")
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--significance-z", type=float)
    ap.add_argument("--max-task-regressions", type=int)
    args = ap.parse_args()
    serve(HoldoutGate(min_gain=args.min_gain, significance_z=args.significance_z,
                      max_task_regressions=args.max_task_regressions))


__all__ = ["HoldoutGate", "main", "two_proportion_z"]
