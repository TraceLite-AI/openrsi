"""基于证据的归因：把判决里没通过的 rubric 项，顺着它引用的证据追到轨迹步骤；把成败记到用过的 memory 条目上。

不需要重跑 agent，代价很低，适合每代都跑。需要反事实重放的归因（去掉一条 memory 重跑）留给运行时做。
"""
from __future__ import annotations

import re
from collections import defaultdict

from openrsi import Attribution, AttributionTarget, Attributor, Episode, serve

_STEP_SOURCE = re.compile(r"^trajectory:(?P<traj>.+):step:(?P<step>[^:]+)$")


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


class RubricEvidenceAttributor(Attributor):
    """每个 episode 产出一条 Attribution：

    - rubric：每个没通过的项记一份责任（权重按项的权重分摊）
    - step：没通过的项引用的证据如果来自轨迹某一步，那一步记责任
    - item：用过的 memory 条目，成功记功劳、失败记责任，按条目数分摊
    - component memory:missing：失败但一条 memory 都没用上，提示可能缺少相关知识
    """

    name = "openrsi/rubric-evidence-attribution"
    version = "0.4.0"

    def explain(self, episodes: list[Episode]) -> list[Attribution]:
        return [self._one(ep) for ep in episodes]

    def _one(self, ep: Episode) -> Attribution:
        v, traj = ep.verdict, ep.trajectory
        sources = {e.id: e.source for e in [*ep.evidence, *v.collected]}
        targets: list[AttributionTarget] = []

        bad = [r for r in v.rubric if r.result in ("fail", "error")]
        total_w = sum(r.weight for r in bad) or 1.0
        step_blame: dict[str, float] = defaultdict(float)
        for r in bad:
            share = r.weight / total_w
            targets.append(AttributionTarget(kind="rubric", ref=r.id, weight=_clip(-share), note=r.note))
            steps = []
            for eid in r.evidence_ids:
                m = _STEP_SOURCE.match(sources.get(eid, ""))
                if m and m["traj"] == traj.id:
                    steps.append(m["step"])
            for sid in steps:
                step_blame[sid] += share / len(steps)
        for sid, w in sorted(step_blame.items()):
            targets.append(AttributionTarget(kind="step", ref=sid, weight=_clip(-w)))

        signal = 2 * v.score - 1 if v.passed or bad else 0.0  # 通过 → 正，全错 → -1
        if traj.used_items:
            per = signal / len(traj.used_items)
            for item_id in traj.used_items:
                targets.append(AttributionTarget(kind="item", ref=item_id, weight=_clip(per)))
        elif not v.passed:
            targets.append(AttributionTarget(kind="component", ref="memory:missing", weight=-0.5,
                                             note="失败时没有用到任何 memory 条目，可能缺少相关知识"))

        why = "; ".join(f"{r.id}: {r.note}" for r in bad if r.note) or ("passed" if v.passed else "failed")
        return Attribution(episode_id=traj.id, targets=targets, method="rubric-evidence", rationale=why)


def item_scores(attributions: list[Attribution]) -> dict[str, tuple[float, int]]:
    """跨 episode 汇总每个 memory 条目的净权重和出现次数，提案器可以用它淘汰有害条目。"""
    acc: dict[str, list[float]] = defaultdict(list)
    for a in attributions:
        for t in a.targets:
            if t.kind == "item":
                acc[t.ref].append(t.weight)
    return {k: (sum(v), len(v)) for k, v in acc.items()}


def main() -> None:
    serve(RubricEvidenceAttributor())


__all__ = ["RubricEvidenceAttributor", "item_scores", "main"]
