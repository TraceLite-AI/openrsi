"""基于证据的归因：把判决里没通过的 rubric 项，顺着它引用的证据追到轨迹步骤；把成败记到用过的 memory 条目上。

不需要重跑 agent，代价很低，适合每代都跑。需要反事实重放的归因（去掉一条 memory 重跑）留给运行时做。
CellContrastAttributor 在此之上做跨格对比：同一道题只在某些环境格失败时，把责任归到环境坐标上。
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


def _cell_info(ep: Episode) -> tuple[str | None, dict[str, str]]:
    """这个 episode 所在的格和坐标：优先读编排方记录的 env_cell 证据，没有就只用 task.env_ref。"""
    for e in ep.evidence:
        if e.type == "env_cell" and isinstance(e.content, dict) and e.verify():
            return str(e.content.get("cell")), {str(k): str(v) for k, v in (e.content.get("coords") or {}).items()}
    return ep.task.env_ref, {}


class CellContrastAttributor(RubricEvidenceAttributor):
    """在证据归因之上做跨格对比（环境格模式，0.5 草案）。同一道题在几个格里都做了时：

    - 有的格过、有的格挂：挂的那几个 episode 记 env 责任，指向"挂的格和所有过的格都不一样"的坐标
      （ref 写成 <坐标键>=<值>）；找不到这样的坐标就指向格本身（cell:<格 id>）
    - 所有格都挂：记 component agent:capability 责任，说明和环境无关
    - 所有格都过，或者只在一个格里做过：不加东西，和 RubricEvidenceAttributor 一样

    同一道题在不同格里的结果是天然的对照实验，不用额外重跑。坐标只是相关，不证明是它导致的。
    """

    name = "openrsi/cell-contrast-attribution"
    version = "0.4.1"

    def __init__(self, env_weight: float = 0.8, capability_weight: float = 0.5):
        self.env_weight, self.capability_weight = env_weight, capability_weight

    def explain(self, episodes: list[Episode]) -> list[Attribution]:
        out = super().explain(episodes)
        groups: dict[str, list[int]] = defaultdict(list)
        for i, ep in enumerate(episodes):
            if _cell_info(ep)[0] is not None:
                groups[ep.task.id].append(i)
        for idx in groups.values():
            cells = {_cell_info(episodes[i])[0] for i in idx}
            if len(cells) < 2:
                continue
            passed = [i for i in idx if episodes[i].verdict.passed]
            failed = [i for i in idx if not episodes[i].verdict.passed]
            if not failed:
                continue
            if not passed:
                for i in failed:
                    out[i].targets.append(AttributionTarget(
                        kind="component", ref="agent:capability", weight=-self.capability_weight,
                        note=f"这道题在全部 {len(cells)} 个格里都失败，和环境无关"))
                continue
            ok_cells = sorted({_cell_info(episodes[i])[0] for i in passed})
            ok_coords = [_cell_info(episodes[i])[1] for i in passed]
            for i in failed:
                cell, coords = _cell_info(episodes[i])
                odd = sorted(k for k, v in coords.items() if all(c.get(k) != v for c in ok_coords))
                note = f"同一道题在 {', '.join(ok_cells)} 通过，在 {cell} 失败"
                if odd:
                    share = self.env_weight / len(odd)
                    out[i].targets += [AttributionTarget(kind="env", ref=f"{k}={coords[k]}", weight=-share, note=note)
                                       for k in odd]
                else:
                    out[i].targets.append(AttributionTarget(kind="env", ref=f"cell:{cell}",
                                                            weight=-self.env_weight, note=note))
        return out


def item_scores(attributions: list[Attribution]) -> dict[str, tuple[float, int]]:
    """跨 episode 汇总每个 memory 条目的净权重和出现次数，提案器可以用它淘汰有害条目。"""
    acc: dict[str, list[float]] = defaultdict(list)
    for a in attributions:
        for t in a.targets:
            if t.kind == "item":
                acc[t.ref].append(t.weight)
    return {k: (sum(v), len(v)) for k, v in acc.items()}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-attribution")
    ap.add_argument("--kind", choices=["rubric-evidence", "cell-contrast"], default="rubric-evidence")
    args = ap.parse_args()
    serve(CellContrastAttributor() if args.kind == "cell-contrast" else RubricEvidenceAttributor())


__all__ = ["CellContrastAttributor", "RubricEvidenceAttributor", "item_scores", "main"]
