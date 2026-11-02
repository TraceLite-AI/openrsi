"""rubric 怎么影响"找方向"：多项 rubric、判别器噪声、rubric 有偏差。

玩具模型（12 个位置，取值 good / neutral / bad）：

  rubric 4 项，每项的通过率由几个位置决定：
    correct   s0 s1 s2 s3（s11 改好会伤害它：权重 −0.25）
    tool_use  s4 s5 s6
    format    s7 s8 s9
    proxy     s10 s11（比如"回答够长"：s11 改好它就更容易过）
  真实目标 = Σ uᵢ·pᵢ，u = correct 0.5、tool_use 0.3、format 0.2、proxy 0（proxy 本身不值钱）。
  判别器分数 = Σ rᵢ·观察到的第 i 项是否通过 / Σ rᵢ。
    无偏 rubric：r = u（proxy 权重 0）
    有偏 rubric：r = correct 0.3、tool_use 0.2、format 0.15、proxy 0.35 —— 它奖励 s11，而 s11 实际上有害
  判别器噪声 ε：每一项的判定以 ε 的概率判反。
  归因：rubric 型——第 i 项没过，就在影响第 i 项的位置里追责（attr_acc 的概率按真实危害挑，其余在组内随便挑）；
        只有对错型——整题没过，在全部 12 个位置里追责。

三个测试：
  1 多项 rubric：用"整题对错"打分 vs 用 rubric 加权分打分，组内比较找对方向的概率；归因按整题 vs 按 rubric 项
  2 判别器噪声：ε 从 0 到 0.3，组内比较和归因的命中率怎么掉
  3 rubric 有偏差：组内比较会不会稳定地选中有害的 s11；连续改进 25 步，rubric 分和真实目标各自怎么走；
    加一道"人工标注集"检查（20 道题按真实目标打分，候选不能显著变差）能不能拦住

    python benchmarks/directed_toy/rubric_probe.py --trials 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "directed_toy"))

from anyrsi_runtime.directed import Version, analyze_group, mean_z  # noqa: E402
from direction_probe import Model as EditModel  # noqa: E402
from directed_toy import SLOTS, START, VALUE  # noqa: E402

ITEMS = {
    "correct": {"s0": 0.30, "s1": 0.25, "s2": 0.20, "s3": 0.15, "s11": -0.25},
    "tool_use": {"s4": 0.30, "s5": 0.20, "s6": 0.10},
    "format": {"s7": 0.20, "s8": 0.15, "s9": 0.10},
    "proxy": {"s10": 0.10, "s11": 0.40},
}
TRUE_U = {"correct": 0.5, "tool_use": 0.3, "format": 0.2, "proxy": 0.0}
BIASED_R = {"correct": 0.3, "tool_use": 0.2, "format": 0.15, "proxy": 0.35}
REQUIRED = ("correct", "tool_use", "format")  # 整题对错 = 这三项都过


def item_p(state: dict[str, str], item: str, difficulty: float) -> float:
    w = ITEMS[item]
    s = sum(c * VALUE[state[k]] for k, c in w.items()) / sum(abs(c) for c in w.values())
    return min(0.97, max(0.03, 0.5 + 0.35 * s + difficulty))


class World:
    def __init__(self, rubric: dict[str, float] | None = None, judge_noise: float = 0.0, attr_acc: float = 0.5,
                 q: float = 0.6, collateral: float = 0.5, break_bias: float = 0.0, seed: int = 0):
        self.r = dict(rubric or TRUE_U)
        self.eps, self.attr_acc = judge_noise, attr_acc
        self.edits = EditModel(q=q, collateral=collateral, break_bias=break_bias, seed=seed)
        self.difficulties = self.edits.difficulties

    # 期望值（真实目标、rubric 分）
    def true_value(self, state: dict[str, str]) -> float:
        return fmean(sum(TRUE_U[i] * item_p(state, i, d) for i in ITEMS) for d in self.difficulties)

    def rubric_value(self, state: dict[str, str]) -> float:
        tot = sum(self.r.values())
        return fmean(sum(self.r[i] * self._observed_p(item_p(state, i, d)) for i in ITEMS) / tot
                     for d in self.difficulties)

    def _observed_p(self, p: float) -> float:
        return p * (1 - self.eps) + (1 - p) * self.eps

    def gains(self, state: dict[str, str], draws: int = 300, seed: int = 0) -> tuple[dict[str, float], dict[str, float]]:
        """每个方向的真实效果和 rubric 效果（沿它改一次的期望变化）。"""
        rng = random.Random(f"g:{seed}")
        t0, r0 = self.true_value(state), self.rubric_value(state)
        tg, rg = {}, {}
        for s in SLOTS:
            after = [self.edits.edit(state, s, rng) for _ in range(draws // 10)]
            tg[s] = fmean(self.true_value(a) for a in after) - t0
            rg[s] = fmean(self.rubric_value(a) for a in after) - r0
        return tg, rg

    # 一次做题
    def outcomes(self, state: dict[str, str], d: float, us: dict[str, float] | None,
                 rng: random.Random) -> dict[str, bool]:
        return {i: ((us[i] if us else rng.random()) < item_p(state, i, d)) for i in ITEMS}

    def judged(self, o: dict[str, bool], rng: random.Random) -> dict[str, bool]:
        return {i: (v != (rng.random() < self.eps)) for i, v in o.items()}

    def score(self, obs: dict[str, bool], mode: str) -> float:
        if mode == "binary":
            return float(all(obs[i] for i in REQUIRED))
        return sum(self.r[i] * obs[i] for i in ITEMS) / sum(self.r.values())

    def true_score(self, o: dict[str, bool]) -> float:
        return sum(TRUE_U[i] * o[i] for i in ITEMS)

    def rollout(self, state: dict[str, str], tasks: list[float], mode: str, rng: random.Random,
                us: list[dict[str, float]] | None = None) -> tuple[dict[str, float], list[dict[str, bool]]]:
        scores, judged = {}, []
        for t, d in enumerate(tasks):
            o = self.outcomes(state, d, us[t] if us else None, rng)
            j = self.judged(o, rng)
            judged.append(j)
            scores[f"t{t}"] = self.score(j, mode)
        return scores, judged

    def blame(self, state: dict[str, str], judged: list[dict[str, bool]], mode: str,
              rng: random.Random) -> dict[str, int]:
        counts = dict.fromkeys(SLOTS, 0)

        def pick(pool: dict[str, float]) -> str:
            harm = {k: max(0.0, c) * (1 - VALUE[state[k]]) + max(0.0, -c) * (1 + VALUE[state[k]])
                    for k, c in pool.items()}
            if rng.random() < self.attr_acc and sum(harm.values()) > 0:
                return rng.choices(list(harm), weights=list(harm.values()))[0]
            return rng.choice(list(pool))

        everything = {k: c for w in ITEMS.values() for k, c in w.items()}
        for j in judged:
            if mode == "binary":
                if not all(j[i] for i in REQUIRED):
                    counts[pick(everything)] += 1
            else:
                for i in ITEMS:
                    if self.r[i] > 0 and not j[i]:
                        counts[pick(ITEMS[i])] += 1
        return counts


def _crn(n: int, rng: random.Random) -> list[dict[str, float]]:
    return [{i: rng.random() for i in ITEMS} for _ in range(n)]


def identify(world: World, state: dict[str, str], tg: dict[str, float], *, mode: str, N: int, m: int, crn: bool,
             trials: int, G: int = 3, seed: int = 0, include: str | None = None) -> dict[str, Any]:
    """单组：G 个方向里挑"真实效果最好"的那个。返回组内比较和归因的命中率，以及选中方向的真实效果。"""
    rng = random.Random(f"id:{seed}:{mode}:{N}:{m}:{crn}")
    hit = att_hit = 0
    chosen_true, att_true, picked_include = [], [], 0
    for _ in range(trials):
        pool = [s for s in SLOTS if s != include]
        dirs = rng.sample(pool, G - (1 if include else 0)) + ([include] if include else [])
        best = max(dirs, key=lambda s: tg[s])
        tasks = [rng.choice(world.difficulties) for _ in range(N)]
        us = _crn(N, rng) if crn else None
        cs, cj = world.rollout(state, tasks, mode, rng, us)
        vs = [Version(d, j, world.rollout(world.edits.edit(state, d, rng), tasks, mode, rng, us)[0])
              for d in dirs for j in range(m)]
        g = analyze_group(Version(None, 0, cs), vs, alpha=0.05, tests=G)
        top = max(g.directions, key=lambda r: r.advantage).direction
        hit += top == best
        chosen_true.append(tg[top])
        picked_include += top == include
        counts = world.blame(state, cj, mode, rng)
        att = max(dirs, key=lambda s: (counts[s], rng.random()))
        att_hit += att == best
        att_true.append(tg[att])
    out = {"mode": mode, "N": N, "m": m, "crn": crn, "组内比较命中": hit / trials, "归因命中": att_hit / trials,
           "组内比较选中的真实效果": fmean(chosen_true), "归因选中的真实效果": fmean(att_true)}
    if include:
        out[f"组内比较选中 {include} 的比例"] = picked_include / trials
    return out


def goodhart(world: World, *, steps: int, trials: int, anchor: int, N: int = 60, holdout: int = 60,
             seed: int = 0, item_guard: bool = False, guard_z: float = -1.0) -> dict[str, Any]:
    """连续改进 steps 步：每步一组（按 rubric 归因挑 2 个方向 + 1 个随机），每方向 1 个修改，公共随机数；
    rubric 优势最大且 z ≥ 1.645 的修改，再过 holdout（同一个 rubric，配对 z > 0），可选再过人工标注集
    （anchor 道题按真实目标打分，不许显著变差：z > −1），可选再过分项不回退（item_guard：holdout 上
    REQUIRED 里每一项的配对差值都不许显著为负，z > guard_z）。记录真实目标和 rubric 分的期望值。"""
    true_curve = [[] for _ in range(steps + 1)]
    rub_curve = [[] for _ in range(steps + 1)]
    harmful = accepted = 0
    for t in range(trials):
        rng = random.Random(f"gh:{seed}:{t}:{anchor}")
        state = dict(START)
        true_curve[0].append(world.true_value(state))
        rub_curve[0].append(world.rubric_value(state))
        blame_total = dict.fromkeys(SLOTS, 0)
        for k in range(steps):
            ranked = sorted(SLOTS, key=lambda s: (-blame_total[s], rng.random()))
            dirs = ranked[:2] + [rng.choice(ranked[2:])]
            tasks = [rng.choice(world.difficulties) for _ in range(N)]
            us = _crn(N, rng)
            cs, cj = world.rollout(state, tasks, "graded", rng, us)
            for s, c in world.blame(state, cj, "graded", rng).items():
                blame_total[s] += c
            cand = []
            for d in dirs:
                new = world.edits.edit(state, d, rng)
                vs, _ = world.rollout(new, tasks, "graded", rng, us)
                mu, z = mean_z([vs[x] - cs[x] for x in cs])
                cand.append((mu, z, new))
            mu, z, new = max(cand, key=lambda c: c[0])
            ok = mu > 0 and z >= 1.645
            if ok:  # holdout：同一个 rubric
                ht = [rng.choice(world.difficulties) for _ in range(holdout)]
                hu = _crn(holdout, rng)
                a, ja = world.rollout(state, ht, "graded", rng, hu)
                b, jb = world.rollout(new, ht, "graded", rng, hu)
                ok = mean_z([b[x] - a[x] for x in a])[0] > 0
                if ok and item_guard:  # 分项不回退：核心项不能显著变差
                    for item in REQUIRED:
                        mu_i, z_i = mean_z([float(y[item]) - float(x[item]) for x, y in zip(ja, jb)])
                        if mu_i < 0 and z_i <= guard_z:
                            ok = False
            if ok and anchor:  # 人工标注集：按真实目标打分
                at = [rng.choice(world.difficulties) for _ in range(anchor)]
                au = _crn(anchor, rng)
                diffs = []
                for i, d in enumerate(at):
                    o0 = world.outcomes(state, d, au[i], rng)
                    o1 = world.outcomes(new, d, au[i], rng)
                    diffs.append(world.true_score(o1) - world.true_score(o0))
                mu_a, z_a = mean_z(diffs)
                ok = not (mu_a < 0 and z_a <= -1.0)
            if ok:
                accepted += 1
                harmful += world.true_value(new) < world.true_value(state)
                state = new
            true_curve[k + 1].append(world.true_value(state))
            rub_curve[k + 1].append(world.rubric_value(state))
    return {"anchor": anchor, "item_guard": item_guard, "guard_z": guard_z, "true": [round(fmean(x), 4) for x in true_curve],
            "rubric": [round(fmean(x), 4) for x in rub_curve],
            "接受次数/次运行": round(accepted / trials, 2),
            "接受里真实变差的比例": round(harmful / accepted, 3) if accepted else 0.0}


def run(trials: int, out: Path) -> dict[str, Any]:
    rep: dict[str, Any] = {"label": "synthetic toy，rubric 测试", "trials": trials}
    # 1 多项 rubric（无偏、无噪声）
    w = World()
    tg, _ = w.gains(START)
    rep["多项rubric"] = [identify(w, START, tg, mode=mode, N=n, m=m, crn=crn, trials=trials)
                        for mode in ("binary", "graded") for n in (30, 120) for crn in (False, True) for m in (1, 3)
                        if not (n == 120 and m == 3 and not crn)]
    # 2 判别器噪声
    rep["判别器噪声"] = []
    for eps in (0.0, 0.1, 0.2, 0.3):
        w = World(judge_noise=eps)
        tg, _ = w.gains(START)
        for crn, m in ((False, 1), (True, 3)):
            rep["判别器噪声"].append({"ε": eps, **identify(w, START, tg, mode="graded", N=60, m=m, crn=crn,
                                                         trials=trials)})
    # 3 rubric 有偏差
    rep["有偏"] = {}
    for name, rub in (("无偏", TRUE_U), ("有偏", BIASED_R)):
        w = World(rubric=rub)
        tg, rg = w.gains(START)
        entry = {"s11 真实效果": round(tg["s11"], 4), "s11 rubric 效果": round(rg["s11"], 4),
                 "rubric 上最好的方向": max(rg, key=rg.get), "真实最好的方向": max(tg, key=tg.get),
                 "单组（s11 必在候选里）": identify(w, START, tg, mode="graded", N=60, m=3, crn=True,
                                            trials=trials, include="s11")}
        entry["连续改进·只有 holdout"] = goodhart(w, steps=25, trials=max(40, trials // 4), anchor=0)
        entry["连续改进·加人工标注集"] = goodhart(w, steps=25, trials=max(40, trials // 4), anchor=20)
        entry["连续改进·加分项不回退（z ≤ −1 拦）"] = goodhart(w, steps=25, trials=max(40, trials // 4), anchor=0,
                                                       item_guard=True, guard_z=-1.0)
        entry["连续改进·加分项不回退（z ≤ −1.645 拦）"] = goodhart(w, steps=25, trials=max(40, trials // 4), anchor=0,
                                                          item_guard=True, guard_z=-1.645)
        rep["有偏"][name] = entry
    out.mkdir(parents=True, exist_ok=True)
    (out / "rubric_probe.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "rubric_probe.md").write_text(markdown(rep), encoding="utf-8")
    return rep


def _f(x: Any) -> str:
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def markdown(r: dict[str, Any]) -> str:
    lines = [f"# rubric 测试（{r['label']}，每格 {r['trials']} 次）", "", "## 1 多项 rubric（无偏、判别器无噪声）", "",
             "| 打分 | 题数 | 每方向修改 | 公共随机数 | 组内比较命中 | 归因命中 |", "| --- | --- | --- | --- | --- | --- |"]
    for x in r["多项rubric"]:
        lines.append(f"| {'整题对错' if x['mode'] == 'binary' else 'rubric 加权'} | {x['N']} | {x['m']} | "
                     f"{'是' if x['crn'] else '否'} | {_f(x['组内比较命中'])} | {_f(x['归因命中'])} |")
    lines += ["", "## 2 判别器噪声（rubric 加权分，60 道题）", "",
              "| ε | 设置 | 组内比较命中 | 归因命中 |", "| --- | --- | --- | --- |"]
    for x in r["判别器噪声"]:
        lines.append(f"| {x['ε']} | {'公共随机数 + 3 个修改' if x['crn'] else '独立随机 + 1 个修改'} | "
                     f"{_f(x['组内比较命中'])} | {_f(x['归因命中'])} |")
    lines += ["", "## 3 rubric 有偏差", ""]
    for name, e in r["有偏"].items():
        one = e["单组（s11 必在候选里）"]
        lines += [f"### {name}", "",
                  f"- s11 的真实效果 {e['s11 真实效果']:+.4f}，rubric 上的效果 {e['s11 rubric 效果']:+.4f}；"
                  f"rubric 上最好的方向 {e['rubric 上最好的方向']}，真实最好的方向 {e['真实最好的方向']}",
                  f"- 单组（3 个方向里必有 s11，60 道题、公共随机数、每方向 3 个修改）：组内比较选中 s11 的比例 "
                  f"{_f(one['组内比较选中 s11 的比例'])}，选中方向的平均真实效果 {one['组内比较选中的真实效果']:+.4f}", "",
                  "| 连续改进 25 步 | 真实目标 起点 → 终点 | rubric 分 起点 → 终点 | 接受次数 | 接受里真实变差 |",
                  "| --- | --- | --- | --- | --- |"]
        for key in [k for k in e if k.startswith("连续改进")]:
            g = e[key]
            lines.append(f"| {key.split('·')[1]} | {g['true'][0]:.3f} → {g['true'][-1]:.3f} | "
                         f"{g['rubric'][0]:.3f} → {g['rubric'][-1]:.3f} | {g['接受次数/次运行']} | "
                         f"{g['接受里真实变差的比例']:.1%} |")
        lines.append("")
    lines.append("命中 = 3 个方向里选中了真实效果最好的那个（随机挑约 0.33）。")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("out/directed_toy"))
    args = ap.parse_args(argv)
    run(args.trials, args.out)
    print((args.out / "rubric_probe.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
