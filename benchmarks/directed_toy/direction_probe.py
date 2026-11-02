"""方向识别测试：组内比较能不能找到正确的优化方向？

用 directed_toy 的玩具问题（每个方向的真实效果可以精确算出来），直接调用 anyrsi_runtime.directed 的统计函数，
不跑完整循环，所以能做几千次试验。三个测试：

  1  单组识别：随机给 G 个方向，一组比较选出来的是不是真实效果最好的那个。
     对比"按归因次数挑"（现在的提案器相当于这样做）和"随机挑"。
  2  影响因素：题目数 N、每题次数 k、提案器质量 q、归因准确率，分别扫一遍。
  3  多组累积：按真实循环的策略（动量 + 探索名额 + 证据累积）连续做 g 组，看动量排序和真实排序的
     Spearman 相关，以及排第一的是不是全局最好的方向。对比累积同样多错题的归因次数排序。

"方向"的真实效果 = 沿它写一个修改（按提案器模型随机产生）以后，真实答对率的期望变化。

    python benchmarks/directed_toy/direction_probe.py --trials 400
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

from anyrsi_runtime.directed import Version, analyze_group, pick_directions, update_momentum  # noqa: E402
from directed_toy import SLOTS, START, VALUE, WEIGHTS, true_p  # noqa: E402


class Model:
    """玩具问题的生成模型：题目难度、提案器怎么改、agent 怎么答、归因怎么指。"""

    def __init__(self, q: float = 0.6, collateral: float = 0.5, break_bias: float = 0.0, attr_acc: float = 0.5,
                 n_difficulty: int = 400, seed: int = 0):
        self.q, self.collateral, self.break_bias, self.attr_acc = q, collateral, break_bias, attr_acc
        rng = random.Random(f"difficulty:{seed}")
        self.difficulties = [rng.gauss(0, 0.12) for _ in range(n_difficulty)]

    def edit(self, state: dict[str, str], slot: str, rng: random.Random) -> dict[str, str]:
        out = dict(state)
        out[slot] = "good" if rng.random() < self.q else rng.choice(list(VALUE))
        if rng.random() < self.collateral:
            other = rng.choice([s for s in SLOTS if s != slot])
            out[other] = "bad" if rng.random() < self.break_bias else rng.choice(list(VALUE))
        return out

    def mean_p(self, state: dict[str, str]) -> float:
        return fmean(true_p(state, d) for d in self.difficulties)

    def true_gains(self, state: dict[str, str], draws: int = 3000, seed: int = 0) -> dict[str, float]:
        """每个方向的真实效果：沿它改一次后，平均答对率的期望变化（蒙特卡洛，固定种子）。"""
        rng = random.Random(f"gains:{seed}")
        base = self.mean_p(state)
        out = {}
        for s in SLOTS:
            out[s] = fmean(self.mean_p(self.edit(state, s, rng)) for _ in range(draws // 10)) - base
        return out

    def rollout(self, state: dict[str, str], tasks: list[float], repeats: int, rng: random.Random,
                us: list[list[float]] | None = None) -> dict[str, float]:
        """us 不为空时用公共随机数：同一道题、同一次重复，所有版本用同一个随机数（相当于 LLM 固定采样种子）。"""
        return {f"t{i}": fmean(float((us[i][k] if us else rng.random()) < true_p(state, d)) for k in range(repeats))
                for i, d in enumerate(tasks)}

    def blames(self, state: dict[str, str], tasks: list[float], rng: random.Random) -> dict[str, int]:
        """对照在这批题上答错时的归因次数（现在的提案器就是看这个）。"""
        harm = {k: WEIGHTS[k] * (1 - VALUE[state[k]]) for k in SLOTS}
        counts = dict.fromkeys(SLOTS, 0)
        for d in tasks:
            if rng.random() >= true_p(state, d):
                if rng.random() < self.attr_acc and sum(harm.values()) > 0:
                    counts[rng.choices(SLOTS, weights=[harm[k] for k in SLOTS])[0]] += 1
                else:
                    counts[rng.choice(SLOTS)] += 1
        return counts


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman 相关；并列取平均秩（很多方向的动量都是 0，不取平均秩会被下标顺序带偏）。"""
    from anyrsi_runtime.directed import _ranks

    ra, rb = _ranks(a), _ranks(b)
    ma, mb = fmean(ra), fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def eb_posterior(support: dict[str, float], history: dict[str, list[float]]) -> dict[str, float]:
    """经验贝叶斯：归因份额当先验，组内比较测到的优势当观测，合成每个方向效果的后验均值。

    模型：真实效果 θ_d = b·a_d + e_d，e_d ~ N(0, τ²)；观测 Ā_d ~ N(θ_d, se_d²)。a_d 是归因份额。
    b、τ² 用试过的方向按矩估计（加权最小二乘过原点，再用残差方差减去平均 se²）。没试过的方向后验就是 b·a_d。
    """
    total = sum(support.values()) or 1.0
    share = {d: support.get(d, 0.0) / total for d in support}
    obs = {d: (fmean(x), (sum((v - fmean(x)) ** 2 for v in x) / (len(x) - 1) / len(x)) if len(x) > 1 else 1.0)
           for d, x in history.items() if x}
    tried = [d for d in obs if d in share]
    if len(tried) < 2:
        b, tau2 = 0.0, 1.0
    else:
        tau2 = max(1e-6, fmean(obs[d][0] ** 2 for d in tried))
        b = 0.0
        for _ in range(20):
            w = {d: 1 / (obs[d][1] + tau2) for d in tried}
            den = sum(w[d] * share[d] ** 2 for d in tried)
            b = max(0.0, sum(w[d] * share[d] * obs[d][0] for d in tried) / den) if den else 0.0
            resid = fmean((obs[d][0] - b * share[d]) ** 2 - obs[d][1] for d in tried)
            tau2 = max(1e-6, resid)
    out = {}
    for d in support:
        prior = b * share[d]
        if d in obs:
            mean, se2 = obs[d]
            out[d] = (prior / tau2 + mean / se2) / (1 / tau2 + 1 / se2)
        else:
            out[d] = prior
    return out


# ---------------------------------------------------------------- 1、2：单组识别


def single_group(model: Model, state: dict[str, str], gains: dict[str, float], *, G: int, N: int, m: int,
                 repeats: int, trials: int, seed: int = 0, alpha: float = 0.05, crn: bool = False) -> dict[str, Any]:
    rng = random.Random(f"single:{seed}")
    res = {"winner_is_best": 0, "winner_positive": 0, "winner_harmful": 0, "no_winner": 0,
           "top_mean_is_best": 0, "attribution_is_best": 0, "random_is_best": 0, "regret_top_mean": [],
           "regret_attribution": []}
    for _ in range(trials):
        dirs = rng.sample(SLOTS, G)
        best = max(dirs, key=lambda d: gains[d])
        tasks = [rng.choice(model.difficulties) for _ in range(N)]
        us = [[rng.random() for _ in range(repeats)] for _ in range(N)] if crn else None
        control = Version(None, 0, model.rollout(state, tasks, repeats, rng, us))
        vs = [Version(d, j, model.rollout(model.edit(state, d, rng), tasks, repeats, rng, us))
              for d in dirs for j in range(m)]
        g = analyze_group(control, vs, alpha=alpha, tests=G)
        top = max(g.directions, key=lambda r: r.advantage).direction
        res["top_mean_is_best"] += top == best
        res["regret_top_mean"].append(gains[best] - gains[top])
        if g.winner is None:
            res["no_winner"] += 1
        else:
            res["winner_is_best"] += g.winner == best
            res["winner_positive"] += gains[g.winner] > 0
            res["winner_harmful"] += gains[g.winner] < 0
        counts = model.blames(state, tasks, rng)
        att = max(dirs, key=lambda d: (counts[d], rng.random()))
        res["attribution_is_best"] += att == best
        res["regret_attribution"].append(gains[best] - gains[att])
        res["random_is_best"] += rng.choice(dirs) == best
    n = trials
    won = n - res["no_winner"]
    return {"G": G, "N": N, "m": m, "repeats": repeats, "crn": crn, "trials": n,
            "P(有胜出者)": won / n,
            "P(胜出者=最好 | 有胜出者)": res["winner_is_best"] / won if won else None,
            "P(胜出者真实有害 | 有胜出者)": res["winner_harmful"] / won if won else None,
            "P(平均优势最高=最好)": res["top_mean_is_best"] / n,
            "P(归因最多=最好)": res["attribution_is_best"] / n,
            "P(随机=最好)": res["random_is_best"] / n,
            "遗憾·平均优势最高": fmean(res["regret_top_mean"]),
            "遗憾·归因最多": fmean(res["regret_attribution"])}


# ---------------------------------------------------------------- 3：多组累积


def accumulate(model: Model, state: dict[str, str], gains: dict[str, float], *, groups: int, G: int, N: int,
               explore: int, trials: int, seed: int = 0, beta: float = 0.7, rank_by: str = "momentum"
               ) -> list[dict[str, Any]]:
    """父版本不变，连续做 groups 组。每组后记录：动量排序、累积平均优势排序、累积归因次数排序 vs 真实排序。"""
    truth = [gains[s] for s in SLOTS]
    best = max(SLOTS, key=lambda s: gains[s])
    rows = [{"momentum_top": 0, "pooled_top": 0, "attr_top": 0, "eb_top": 0, "rho_momentum": [], "rho_pooled": [],
             "rho_attr": [], "rho_eb": []} for _ in range(groups)]
    for t in range(trials):
        momentum: dict[str, float] = {}
        history: dict[str, list[float]] = {}
        looks: dict[str, int] = {}
        blame_total = dict.fromkeys(SLOTS, 0)
        trng = random.Random(f"acc:{seed}:{t}")
        for gi in range(groups):
            if rank_by == "eb":
                post = eb_posterior({s: float(blame_total[s]) for s in SLOTS}, history)
                chosen = pick_directions(post, {}, G, explore, trng)
            else:
                chosen = pick_directions({s: float(blame_total[s]) for s in SLOTS}, momentum, G, explore, trng)
            tasks = [trng.choice(model.difficulties) for _ in range(N)]
            control = Version(None, 0, model.rollout(state, tasks, 1, trng))
            for s, c in model.blames(state, tasks, trng).items():
                blame_total[s] += c
            vs = [Version(d, 0, model.rollout(model.edit(state, d, trng), tasks, 1, trng)) for d in chosen]
            g = analyze_group(control, vs, alpha=0.05, tests=G, history=history, looks=looks)
            for r in g.directions:
                history.setdefault(r.direction, []).extend(r.diffs)
                looks[r.direction] = looks.get(r.direction, 0) + 1
            momentum = update_momentum(momentum, g, beta)
            mom = [momentum.get(s, 0.0) for s in SLOTS]
            pooled = [fmean(history[s]) if history.get(s) else 0.0 for s in SLOTS]
            attr = [float(blame_total[s]) for s in SLOTS]
            post = eb_posterior({s: float(blame_total[s]) for s in SLOTS}, history)
            eb = [post[s] for s in SLOTS]
            row = rows[gi]
            row["eb_top"] += max(SLOTS, key=lambda s: (post[s], trng.random())) == best
            row["rho_eb"].append(spearman(eb, truth))
            row["momentum_top"] += max(SLOTS, key=lambda s: (momentum.get(s, 0.0), trng.random())) == best
            row["pooled_top"] += max(SLOTS, key=lambda s: (pooled[SLOTS.index(s)], trng.random())) == best
            row["attr_top"] += max(SLOTS, key=lambda s: (blame_total[s], trng.random())) == best
            row["rho_momentum"].append(spearman(mom, truth))
            row["rho_pooled"].append(spearman(pooled, truth))
            row["rho_attr"].append(spearman(attr, truth))
    return [{"组数": i + 1, "P(动量第一=最好)": r["momentum_top"] / trials,
             "P(累积优势第一=最好)": r["pooled_top"] / trials, "P(归因第一=最好)": r["attr_top"] / trials,
             "P(合并第一=最好)": r["eb_top"] / trials,
             "ρ·动量": fmean(r["rho_momentum"]), "ρ·累积优势": fmean(r["rho_pooled"]), "ρ·归因": fmean(r["rho_attr"]),
             "ρ·合并": fmean(r["rho_eb"])}
            for i, r in enumerate(rows)]


# ---------------------------------------------------------------- 报告


def run(trials: int, out: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"label": "synthetic toy，方向识别测试", "trials": trials}
    mid = {s: ("good" if i in (3, 5, 8) else "bad" if i in (0, 4) else "neutral") for i, s in enumerate(SLOTS)}
    states = {"起点": dict(START), "中途": mid}
    proposers = {"较好": dict(q=0.6, collateral=0.5, break_bias=0.0), "很差": dict(q=0.2, collateral=0.9, break_bias=0.9)}
    report["states"] = {}
    for sname, state in states.items():
        for pname, pk in proposers.items():
            model = Model(**pk)
            gains = model.true_gains(state)
            key = f"{sname}·提案器{pname}"
            entry: dict[str, Any] = {"true_gains": {s: round(v, 4) for s, v in sorted(gains.items(),
                                                                                      key=lambda x: -x[1])}}
            entry["单组"] = [single_group(model, state, gains, G=3, N=n, m=m, repeats=1, trials=trials, crn=crn)
                            for n in (30, 120) for crn in (False, True) for m in (1, 3)]
            entry["累积"] = accumulate(model, state, gains, groups=15, G=3, N=30, explore=1, trials=max(50, trials // 4))
            report["states"][key] = entry
    attr_sweep = []
    for acc in (0.0, 0.25, 0.5, 0.75, 1.0):
        model = Model(q=0.6, collateral=0.5, attr_acc=acc)
        gains = model.true_gains(START)
        r = single_group(model, START, gains, G=3, N=60, m=1, repeats=1, trials=trials)
        attr_sweep.append({"归因准确率": acc, "P(平均优势最高=最好)": r["P(平均优势最高=最好)"],
                           "P(归因最多=最好)": r["P(归因最多=最好)"]})
    report["归因准确率扫描"] = attr_sweep
    out.mkdir(parents=True, exist_ok=True)
    (out / "direction_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "direction_probe.md").write_text(markdown(report), encoding="utf-8")
    return report


def _f(x: Any) -> str:
    return "—" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


def markdown(r: dict[str, Any]) -> str:
    lines = [f"# 方向识别测试（{r['label']}，每格 {r['trials']} 次试验）", ""]
    for key, e in r["states"].items():
        lines += [f"## {key}", "", "真实方向效果（从大到小）：" + "，".join(f"{k} {v:+.3f}" for k, v in e["true_gains"].items()),
                  "", "| 设置 | P(平均优势最高=最好) | P(归因最多=最好) | P(随机=最好) | P(有胜出者) | "
                  "P(胜出者=最好) | P(胜出者有害) | 遗憾·组内比较 | 遗憾·归因 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        rows = [(f"N={x['N']}，每方向 {x['m']} 个修改，{'公共随机数' if x['crn'] else '独立随机'}", x) for x in e["单组"]]
        for name, x in rows:
            lines.append(f"| {name} | {_f(x['P(平均优势最高=最好)'])} | {_f(x['P(归因最多=最好)'])} | "
                         f"{_f(x['P(随机=最好)'])} | {_f(x['P(有胜出者)'])} | {_f(x['P(胜出者=最好 | 有胜出者)'])} | "
                         f"{_f(x['P(胜出者真实有害 | 有胜出者)'])} | {x['遗憾·平均优势最高']:.3f} | {x['遗憾·归因最多']:.3f} |")
        lines += ["", "多组累积（每组 3 个方向、30 道题、1 个探索名额）：", "",
                  "| 组数 | P(动量第一=最好) | P(累积优势第一=最好) | P(归因第一=最好) | ρ·动量 | ρ·累积优势 | ρ·归因 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for x in e["累积"]:
            if x["组数"] in (1, 3, 5, 10, 15):
                lines.append(f"| {x['组数']} | {_f(x['P(动量第一=最好)'])} | {_f(x['P(累积优势第一=最好)'])} | "
                             f"{_f(x['P(归因第一=最好)'])} | {_f(x['ρ·动量'])} | {_f(x['ρ·累积优势'])} | {_f(x['ρ·归因'])} |")
        lines.append("")
    lines += ["## 归因准确率扫描（起点，提案器较好，N=60，3 个方向）", "",
              "| 归因准确率 | P(组内比较=最好) | P(归因最多=最好) |", "| --- | --- | --- |"]
    for x in r["归因准确率扫描"]:
        lines.append(f"| {x['归因准确率']} | {_f(x['P(平均优势最高=最好)'])} | {_f(x['P(归因最多=最好)'])} |")
    lines += ["", "遗憾 = 真实最好方向的效果 − 选中方向的效果（越小越好）。ρ 是和真实排序的 Spearman 相关（1 表示完全一致）。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("out/directed_toy"))
    args = ap.parse_args(argv)
    run(args.trials, args.out)
    print((args.out / "direction_probe.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
