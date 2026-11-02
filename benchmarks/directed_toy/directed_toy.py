"""已知最优解的玩具问题：现在的循环是不是随机游走，有方向的循环能不能收敛。

被进化的东西是 12 个"位置"（prompt 可进化体里的 s0..s11），每个取 good / neutral / bad。
真实答对率由位置决定，权重不同（有的位置重要，有的完全无关）：

    p = clamp(0.55 + 0.3 · Σ wᵢ·vᵢ / Σ wᵢ + 题目难度, 0.03, 0.97)，v(good)=1, v(neutral)=0, v(bad)=−1

单个位置的效果小（改对最重要的一个也只涨几个百分点），考试题只有 30 道，噪声和效果同一个量级——
这正是真实实验里"随机游走"的情形。起点 p ≈ 0.39，最优 p ≈ 0.85。每次做题按 p 掷硬币，自带二项噪声。
agent 答错时会"说出"一个原因（归因），只有 attr_acc 的概率指对，其余是随便一个位置。
提案器模拟 LLM：沿给定方向改，有 q 的概率改对；还经常顺手改坏别的位置（collateral）。

四组对照（同样的 agent 调用预算，benchmarks/directed_toy/README.md）：
    A  现在的循环：看一道错题的归因就改，考试准入只看"分数变高"
    B  A + 统计准入（考试题上逐题配对检验，z ≥ 1.645）
    C  有方向：每组 3 个方向 × 1 个修改，带"不改"对照在同一批练习题上比，最好的修改送去考试准入
    D  完整：每组 3 个方向 × 2 个修改 + 嵌套方差分析 + 线搜索 + 收敛判据
A、C、D 的考试准入一样（分数变高就收），差别只在候选怎么产生、怎么挑：A 每次 1 个，C、D 每组从几个里挑最好的，
并用动量决定往哪些方向多试。

提案器质量是参数：q 是"沿方向改对"的概率，collateral 是"顺手改坏别处"的概率。
q 高时改动本身就往好的方向漂，随机游走也会上升；q 低时改动接近随机，这才是真实实验里看到的情形。

    python benchmarks/directed_toy/directed_toy.py --budget 6000 --seeds 0 1 2 3 4 --q 0.35 --collateral 0.7 \
        --break-bias 0.8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "benchmarks" / "directed_toy") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "directed_toy"))

from anyrsi import (Agent, Attribution, AttributionTarget, Environment, Item, Patch, ProposalRequest,  # noqa: E402
                    Proposer, Step, Task, Trajectory)
from anyrsi import Attributor as _Attributor  # noqa: E402

SLOTS = [f"s{i}" for i in range(12)]
WEIGHTS = dict(zip(SLOTS, [0.16, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.0, 0.0]))
VALUE = {"good": 1.0, "neutral": 0.0, "bad": -1.0}
START = {s: ("bad" if i % 2 == 0 else "neutral") for i, s in enumerate(SLOTS)}
SYNTHETIC = "synthetic toy，已知最优解，非 benchmark"


def true_p(values: dict[str, str], difficulty: float = 0.0) -> float:
    total = sum(WEIGHTS.values())
    s = sum(WEIGHTS[k] * VALUE[values.get(k, "neutral")] for k in SLOTS) / total
    return min(0.97, max(0.03, 0.55 + 0.3 * s + difficulty))


def values_of(items: list[Item] | list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for i in items:
        d = i if isinstance(i, dict) else i.model_dump()
        if d.get("id") in SLOTS:
            out[d["id"]] = d.get("text", "neutral")
    return out


class ToyEnv(Environment):
    name, version = "toy/slots-env", "0.1.0"

    def __init__(self, seed: int = 0, train_per_round: int = 30, holdout: int = 30):
        self.seed, self.train_per_round, self.n_holdout = seed, train_per_round, holdout

    def _task(self, split: str, i: int) -> Task:
        rng = random.Random(f"{self.seed}:{split}:{i}")
        return Task(id=f"{split}-{i}", split=split, input={"i": i}, reference="right",
                    tags=["easy" if i % 2 else "hard"], metadata={"difficulty": rng.gauss(0, 0.12)})

    def tasks(self, split: str, round: int = 0, limit: int | None = None, cell: str | None = None) -> list[Task]:
        if split == "train":
            out = [self._task("train", round * 1000 + i) for i in range(self.train_per_round)]
        elif split in ("holdout", "final"):
            out = [self._task(split, i) for i in range(self.n_holdout)]
        else:
            out = []
        return out[:limit] if limit else out


class ToyAgent(Agent):
    """按真实答对率掷硬币；答错时说一个原因（attr_acc 的概率是真正最拖后腿的位置之一）。

    common_noise=True 时同一道题每次用同一个随机数（公共随机数）：两个答对率一样的版本在同一道题上结果一定相同。
    """

    name, version = "toy/slots-agent", "0.1.0"

    def __init__(self, seed: int = 0, attr_acc: float = 0.5, common_noise: bool = False):
        self.rng, self.attr_acc, self.common_noise = random.Random(f"agent:{seed}"), attr_acc, common_noise
        self.seed = seed

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        values = values_of(context)
        p = true_p(values, task.metadata.get("difficulty", 0.0))
        u = random.Random(f"{self.seed}:{task.id}").random() if self.common_noise else self.rng.random()
        right = u < p
        blame = None
        if not right:
            harm = {k: WEIGHTS[k] * (1 - VALUE[values.get(k, "neutral")]) for k in SLOTS}
            if self.rng.random() < self.attr_acc and sum(harm.values()) > 0:
                blame = self.rng.choices(SLOTS, weights=[harm[k] for k in SLOTS])[0]
            else:
                blame = self.rng.choice(SLOTS)
        return Trajectory(id=f"traj-{task.id}-{snapshot_id}-{self.rng.random():.6f}", task_id=task.id,
                          snapshot_id=snapshot_id, output="right" if right else "wrong",
                          steps=[Step(type="observation", content=f"p={p:.3f}")],
                          metadata={"blame": blame, "true_p": p})


class ToyAttributor(_Attributor):
    name, version = "toy/slots-attributor", "0.1.0"

    def explain(self, episodes: list[Any]) -> list[Attribution]:
        out = []
        for ep in episodes:
            blame = ep.trajectory.metadata.get("blame")
            if blame and not ep.verdict.passed:
                out.append(Attribution(episode_id=ep.trajectory.id, method="toy",
                                       targets=[AttributionTarget(kind="item", ref=blame, weight=-1.0)]))
        return out


class ToyProposer(Proposer):
    """模拟 LLM 提案器。

    有方向（hints.direction = item:sN）：把 sN 改成 good 的概率是 q，否则随机取值。
    没有方向（现在的循环）：随便挑一道错题，按它的归因去改（等于看一道题的"文本梯度"）。
    步长（hints.step = 1/2/3…）：一次改几个位置：方向上的那个，再加 step − 1 个随手改的位置（随机取值）。
    collateral：每次还有这么大的概率顺手改动另一个位置；break_bias 是这种顺手改动把它改成 bad 的概率
    （其余情况随机取值）。LLM 改代码时顺手改坏别处很常见，break_bias 越高，改动整体越接近"平均有害"。
    """

    name, version = "toy/slots-proposer", "0.1.0"

    def __init__(self, seed: int = 0, q: float = 0.6, collateral: float = 0.5, break_bias: float = 0.0):
        self.rng, self.q, self.collateral, self.break_bias = random.Random(f"proposer:{seed}"), q, collateral, break_bias

    def _side_effect(self) -> str:
        return "bad" if self.rng.random() < self.break_bias else self.rng.choice(list(VALUE))

    def propose(self, request: ProposalRequest) -> list[Patch]:
        current = values_of(request.state or [])
        direction = (request.hints or {}).get("direction")
        if direction:
            slot = str(direction["id"]).split(":", 1)[-1]
        else:
            blames = [t.ref for a in request.attributions for t in a.targets if t.weight < 0]
            slot = self.rng.choice(blames) if blames else self.rng.choice(SLOTS)
        if slot not in SLOTS:
            return []
        change = {slot: "good" if self.rng.random() < self.q else self.rng.choice(list(VALUE))}
        step = int((request.hints or {}).get("step") or 1)
        others = [s for s in SLOTS if s != slot]
        for s in self.rng.sample(others, min(len(others), max(0, step - 1))):
            change[s] = self._side_effect()
        if self.rng.random() < self.collateral:
            s = self.rng.choice(others)
            change.setdefault(s, self._side_effect())
        ops = [{"op": "replace", "item": {"id": k, "kind": "prompt", "text": v}}
               for k, v in sorted(change.items()) if current.get(k) != v]
        if not ops:
            return []
        return [Patch(target=request.target.target, base_snapshot=request.parent_snapshot, ops=ops,
                      rationale=f"set {', '.join(f'{k}={v}' for k, v in sorted(change.items()))}",
                      proposer_ref=self.ref)]


# ---------------------------------------------------------------- 实验

GROUPS = ("A", "B", "C", "D")


class BudgetExhausted(RuntimeError):
    pass


class _Counting:
    def __init__(self, inner: Any, budget: int):
        self.inner, self.budget, self.calls = inner, budget, 0
        self.kinds = getattr(inner, "kinds", ["agent"])

    async def run(self, *a: Any, **k: Any) -> Any:
        if self.calls >= self.budget:
            raise BudgetExhausted(f"预算 {self.budget} 次 agent 调用用完")
        self.calls += 1
        return await self.inner.run(*a, **k)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def write_config(base: Path, group: str, seed: int, holdout: int, q: float = 0.6, collateral: float = 0.5,
                 break_bias: float = 0.0, alpha: float = 0.05) -> Path:
    gate: dict[str, Any] = {"min_gain": 0.0}
    if group == "B":
        gate["paired_significance_z"] = 1.645
    loop: dict[str, Any] = {"max_patches": 1, "train_limit": 30}
    if group == "C":
        loop["directed"] = {"group_size": 3, "edits_per_direction": 1, "explore": 1, "seed": seed, "patience": 4,
                            "alpha": alpha}
    if group == "D":
        loop["directed"] = {"group_size": 3, "edits_per_direction": 2, "explore": 1, "seed": seed, "patience": 4,
                            "alpha": alpha, "steps": [1, 2, 3]}
    comps = {
        "env": {"python": "directed_toy:ToyEnv", "with": {"seed": seed, "holdout": holdout}},
        "agent": {"python": "directed_toy:ToyAgent", "with": {"seed": seed}},
        "judge": {"python": "anyrsi_judge:EvidenceJudge",
                  "with": {"rubric": {"items": [{"id": "correct", "check": "output_equals", "required": True}]}}},
        "attribution": {"python": "directed_toy:ToyAttributor"},
        "proposer": {"python": "directed_toy:ToyProposer", "with": {"seed": seed, "q": q, "collateral": collateral,
                                                                       "break_bias": break_bias}},
        "algorithm": {"python": "anyrsi_algorithm:GreedyAlgorithm",
                      "with": {"max_generations": 10_000, "patience": 10_000}},
        "gate": {"python": "anyrsi_gate:HoldoutGate", "with": gate},
        "lineage": {"python": "anyrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
    }
    data = {"protocol": "anyrsi/0.4", "name": f"directed-toy-{group}", "plugins": ["directed_toy"],
            "components": comps, "loop": loop,
            "evolvables": {"prompt": {"python": "anyrsi_evolvables:PromptEvolvable",
                                      "with": {"path": "state/prompt",
                                               "initial": [{"id": k, "text": v} for k, v in START.items()]}}}}
    base.mkdir(parents=True, exist_ok=True)
    (base / "anyrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "anyrsi.yaml"


async def run_group(group: str, base: Path, seed: int, budget: int, holdout: int = 30, q: float = 0.6,
                    collateral: float = 0.5, break_bias: float = 0.0, alpha: float = 0.05) -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(write_config(base, group, seed, holdout, q, collateral, break_bias, alpha))
    start = time.monotonic()
    async with Wiring(cfg) as w:
        counter = _Counting(w.components["agent"], budget)
        w.components["agent"] = counter
        runner = Runner(w, cfg.loop)
        try:
            await runner.run()
            stopped = runner.converged or "算法停止"
        except BudgetExhausted as e:
            stopped = str(e)
        log = await w.lineage.log()
        prompt = w.evolvables["prompt"]

        async def truth(entry: Any) -> float:
            return true_p(values_of(await prompt.export(entry.snapshots["prompt"])))

        root = next(e for e in log if e.status == "root")
        accepted = [e for e in log if e.status == "accepted"]
        best = max([root, *accepted], key=lambda e: e.metrics.get("holdout", 0.0))
        curve, bad = [], 0
        for e in accepted:
            parent = next(x for x in log if x.id == e.parent_ids[0])
            gain = await truth(e) - await truth(parent)
            bad += gain < 0
            curve.append(round(await truth(e), 4))
        return {"group": group, "seed": seed, "stopped": stopped, "calls": counter.calls,
                "accepted": len(accepted), "harmful_accepts": bad,
                "true_start": round(await truth(root), 4), "true_final": round(await truth(best), 4),
                "true_optimum": round(true_p({k: "good" for k in SLOTS}), 4), "curve": curve,
                "seconds": round(time.monotonic() - start, 2)}


async def experiment(out: Path, seeds: list[int], budget: int, groups: tuple[str, ...] = GROUPS, q: float = 0.6,
                     collateral: float = 0.5, break_bias: float = 0.0, alpha: float = 0.05) -> dict[str, Any]:
    runs = [await run_group(g, out / f"seed{s}" / g, s, budget, q=q, collateral=collateral, break_bias=break_bias,
                            alpha=alpha) for s in seeds for g in groups]
    summary = {}
    for g in groups:
        rs = [r for r in runs if r["group"] == g]
        finals = [r["true_final"] for r in rs]
        acc = sum(r["accepted"] for r in rs)
        summary[g] = {"true_final_mean": round(fmean(finals), 4),
                      "true_final_sd": round(stdev(finals), 4) if len(finals) > 1 else 0.0,
                      "harmful_accept_rate": round(sum(r["harmful_accepts"] for r in rs) / acc, 3) if acc else 0.0,
                      "accepted_mean": round(acc / len(rs), 2),
                      "converged_runs": sum(1 for r in rs if "收敛" in r["stopped"])}
    report = {"label": SYNTHETIC, "budget": budget, "seeds": seeds, "q": q, "collateral": collateral,
              "break_bias": break_bias, "alpha": alpha,
              "summary": summary, "runs": runs}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# 有方向 vs 随机游走（{report['label']}）", "",
             f"预算：每组每个种子 {report['budget']} 次 agent 调用；种子 {report['seeds']}；"
             f"提案器 q = {report['q']}，collateral = {report['collateral']}，break_bias = {report['break_bias']}；"
             f"C、D 的 α = {report['alpha']}。"
             f"真实答对率起点 {report['runs'][0]['true_start']}，最优 {report['runs'][0]['true_optimum']}。", "",
             "| 组 | 最终真实答对率（均值 ± 标准差） | 接受的改动里真实变差的比例 | 平均接受次数 | 判定收敛的运行 |",
             "| --- | --- | --- | --- | --- |"]
    for g, s in report["summary"].items():
        lines.append(f"| {g} | {s['true_final_mean']:.3f} ± {s['true_final_sd']:.3f} | {s['harmful_accept_rate']:.1%} | "
                     f"{s['accepted_mean']} | {s['converged_runs']} |")
    lines += ["", "最终版本 = 开发期 holdout 最高的已接受版本；真实答对率按已知公式计算，不是再考一次。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--groups", default="ABCD")
    ap.add_argument("--q", type=float, default=0.6, help="提案器沿方向改对的概率")
    ap.add_argument("--collateral", type=float, default=0.5, help="提案器顺手改动别处的概率")
    ap.add_argument("--break-bias", type=float, default=0.0, help="顺手改动是把东西改坏的概率")
    ap.add_argument("--alpha", type=float, default=0.05, help="C、D 组的显著性水平")
    ap.add_argument("--out", type=Path, default=Path("out/directed_toy"))
    args = ap.parse_args(argv)
    asyncio.run(experiment(args.out, args.seeds, args.budget, tuple(args.groups), args.q, args.collateral,
                           args.break_bias, args.alpha))
    print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

