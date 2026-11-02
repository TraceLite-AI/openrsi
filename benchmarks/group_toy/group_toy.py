"""组提案的玩具实验：一组候选在考试题上比，四种策略和"现在的循环"比，最优解已知。

被进化的是 12 个"位置"（prompt 可进化体里的 s0..s11），每个取 good / neutral / bad，权重不同（两个无关）。
真实答对率 p = clamp(0.55 + 0.3 · Σ wᵢvᵢ / Σ wᵢ + 题目难度, 0.03, 0.97)；起点 ≈ 0.39，最优 0.85。
agent 按 p 掷硬币；题目带 rollout_seed 时用它当随机数（公共随机数：同一代同一道题，各版本掷同一枚硬币）。
答错时说一个原因（attr_acc 的概率是真正拖后腿的位置）。提案器模拟 LLM：看一道错题的归因去改，
有 q 的概率改对，还会顺手改动别处（collateral，其中 break_bias 的概率是改坏）。
经验型策略（training_free）把"比同组好的改法"写进 experience 可进化体，agent 照着做（改动说明里的 sK=V）。

对照（同样的 agent 调用预算）：
    A          现在的循环：每代 1 个提案，考试分变高就收
    paired     组提案 + parent_paired（减父版本、配对检验）
    grpo       组提案 + GRPO 式组内优势
    best_of_n  组提案 + 挑最高分
    training_free  组提案 + 把组内比较写成经验
组提案都是每组 4 个候选，考试题一半挑、一半确认（准入只看确认半，分数变高就收）。

    python benchmarks/group_toy/group_toy.py --budget 6000 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "benchmarks" / "group_toy") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))

from anyrsi import (Agent, Attribution, AttributionTarget, Attributor, Environment, Item, Patch,  # noqa: E402
                    ProposalRequest, Proposer, Step, Task, Trajectory)

SLOTS = [f"s{i}" for i in range(12)]
WEIGHTS = dict(zip(SLOTS, [0.16, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.0, 0.0]))
VALUE = {"good": 1.0, "neutral": 0.0, "bad": -1.0}
START = {s: ("bad" if i % 2 == 0 else "neutral") for i, s in enumerate(SLOTS)}
SET = re.compile(r"\b(s\d+)=(good|neutral|bad)\b")
SYNTHETIC = "synthetic toy，已知最优解，非 benchmark"


def true_p(values: dict[str, str], difficulty: float = 0.0) -> float:
    s = sum(WEIGHTS[k] * VALUE[values.get(k, "neutral")] for k in SLOTS) / sum(WEIGHTS.values())
    return min(0.97, max(0.03, 0.55 + 0.3 * s + difficulty))


def effective(items: list[Any]) -> dict[str, str]:
    """位置的取值，再叠加经验条目（"Better than its group …: set sK=V"）：经验说好的改法，agent 照着做。"""
    values, lessons = {}, []
    for i in items:
        d = i if isinstance(i, dict) else i.model_dump()
        if d.get("id") in SLOTS:
            values[d["id"]] = d.get("text", "neutral")
        elif d.get("kind") == "lesson" and str(d.get("text", "")).startswith("Better"):
            lessons.append(d)
    for d in sorted(lessons, key=lambda x: x.get("data", {}).get("advantage", 0)):
        for k, v in SET.findall(d["text"]):
            values[k] = v
    return values


class ToyEnv(Environment):
    name, version = "toy/group-env", "0.1.0"

    def __init__(self, seed: int = 0, train_per_round: int = 30, holdout: int = 60):
        self.seed, self.train_per_round, self.n_holdout = seed, train_per_round, holdout

    def _task(self, split: str, i: int) -> Task:
        rng = random.Random(f"{self.seed}:{split}:{i}")
        return Task(id=f"{split}-{i}", split=split, input={"i": i}, reference="right",
                    metadata={"difficulty": rng.gauss(0, 0.12)})

    def tasks(self, split: str, round: int = 0, limit: int | None = None, cell: str | None = None) -> list[Task]:
        if split == "train":
            out = [self._task("train", round * 1000 + i) for i in range(self.train_per_round)]
        elif split in ("holdout", "final"):
            out = [self._task(split, i) for i in range(self.n_holdout)]
        else:
            out = []
        return out[:limit] if limit else out


class ToyAgent(Agent):
    name, version = "toy/group-agent", "0.1.0"

    def __init__(self, seed: int = 0, attr_acc: float = 0.5):
        self.rng, self.attr_acc = random.Random(f"agent:{seed}"), attr_acc

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        values = effective(context)
        p = true_p(values, task.metadata.get("difficulty", 0.0))
        seed = task.metadata.get("rollout_seed")
        u = random.Random(seed).random() if isinstance(seed, int) else self.rng.random()
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
                          steps=[Step(type="observation", content=f"p={p:.3f}")], metadata={"blame": blame})


class ToyAttributor(Attributor):
    name, version = "toy/group-attributor", "0.1.0"

    def explain(self, episodes: list[Any]) -> list[Attribution]:
        return [Attribution(episode_id=ep.trajectory.id, method="toy",
                            targets=[AttributionTarget(kind="item", ref=ep.trajectory.metadata["blame"], weight=-1.0)])
                for ep in episodes if ep.trajectory.metadata.get("blame") and not ep.verdict.passed]


class ToyProposer(Proposer):
    """有方向（hints.direction = item:sN）就改 sN；否则随便挑一道错题，按它的归因改。改对的概率 q，
    顺手改动别处的概率 collateral（其中 break_bias 是改坏）。一次可以返回多个提案（max_patches）。"""

    name, version = "toy/group-proposer", "0.1.0"

    def __init__(self, seed: int = 0, q: float = 0.6, collateral: float = 0.5, break_bias: float = 0.0):
        self.rng, self.q, self.collateral, self.break_bias = random.Random(f"proposer:{seed}"), q, collateral, break_bias

    def _one(self, request: ProposalRequest) -> Patch | None:
        current = effective(request.state or [])
        direction = (request.hints or {}).get("direction")
        if direction:
            slot = str(direction["id"]).split(":", 1)[-1]
        else:
            blames = [t.ref for a in request.attributions for t in a.targets if t.weight < 0]
            slot = self.rng.choice(blames) if blames else self.rng.choice(SLOTS)
        change = {slot: "good" if self.rng.random() < self.q else self.rng.choice(list(VALUE))}
        if self.rng.random() < self.collateral:
            other = self.rng.choice([s for s in SLOTS if s != slot])
            change[other] = "bad" if self.rng.random() < self.break_bias else self.rng.choice(list(VALUE))
        ops = [{"op": "replace", "item": {"id": k, "kind": "prompt", "text": v}}
               for k, v in sorted(change.items()) if current.get(k) != v]
        if not ops:
            return None
        return Patch(target=request.target.target, base_snapshot=request.parent_snapshot, ops=ops,
                     rationale="set " + ", ".join(f"{k}={v}" for k, v in sorted(change.items())), proposer_ref=self.ref)

    def propose(self, request: ProposalRequest) -> list[Patch]:
        out = [p for p in (self._one(request) for _ in range(request.max_patches)) if p]
        return out


# ---------------------------------------------------------------- 实验

ARMS = ("A", "paired", "grpo", "best_of_n", "training_free")


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


def write_config(base: Path, arm: str, seed: int, *, q: float = 0.6, collateral: float = 0.5,
                 break_bias: float = 0.0, holdout: int = 60, size: int = 4) -> Path:
    loop: dict[str, Any] = {"max_patches": 1, "train_limit": 30}
    evolvables: dict[str, Any] = {"prompt": {"python": "anyrsi_evolvables:PromptEvolvable",
                                             "with": {"target": "prompt", "path": "state/prompt",
                                                      "initial": [{"id": k, "text": v} for k, v in START.items()]}}}
    if arm != "A":
        loop["group"] = {"size": size, "strategy": arm if arm != "paired" else "parent_paired",
                         "confirm_fraction": 0.5}
    if arm == "training_free":
        evolvables["experience"] = {"python": "anyrsi_evolvables:PromptEvolvable",
                                    "with": {"target": "experience", "path": "state/experience"}}
        loop["group"]["experience_target"] = "experience"
    comps = {
        "env": {"python": "group_toy:ToyEnv", "with": {"seed": seed, "holdout": holdout}},
        "agent": {"python": "group_toy:ToyAgent", "with": {"seed": seed}},
        "judge": {"python": "anyrsi_judge:EvidenceJudge",
                  "with": {"rubric": {"items": [{"id": "correct", "check": "output_equals", "required": True}]}}},
        "attribution": {"python": "group_toy:ToyAttributor"},
        "proposer": {"python": "group_toy:ToyProposer",
                     "with": {"seed": seed, "q": q, "collateral": collateral, "break_bias": break_bias}},
        "algorithm": {"python": "anyrsi_algorithm:GreedyAlgorithm",
                      "with": {"max_generations": 10_000, "patience": 10_000}},
        "gate": {"python": "anyrsi_gate:HoldoutGate", "with": {"min_gain": 0.0}},
        "lineage": {"python": "anyrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
    }
    data = {"protocol": "anyrsi/0.4", "name": f"group-toy-{arm}", "plugins": ["group_toy"],
            "components": comps, "loop": loop, "evolvables": evolvables}
    base.mkdir(parents=True, exist_ok=True)
    (base / "anyrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "anyrsi.yaml"


async def run_arm(arm: str, base: Path, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(write_config(base, arm, seed, **kw))
    start = time.monotonic()
    async with Wiring(cfg) as w:
        counter = _Counting(w.components["agent"], budget)
        w.components["agent"] = counter
        runner = Runner(w, cfg.loop)
        try:
            await runner.run()
        except BudgetExhausted:
            pass
        log = await w.lineage.log()

        async def truth(entry: Any) -> float:
            items: list[Any] = []
            for t, sid in entry.snapshots.items():
                items += await w.evolvables[t].export(sid)
            return true_p(effective(items))

        root = next(e for e in log if e.status == "root")
        accepted = [e for e in log if e.status == "accepted"]
        final = accepted[-1] if accepted else root  # 贪心链上最后接受的就是当前版本
        harmful = 0
        for e in accepted:
            parent = next(x for x in log if x.id == e.parent_ids[0])
            harmful += await truth(e) < await truth(parent)
        return {"arm": arm, "seed": seed, "calls": counter.calls, "accepted": len(accepted), "harmful": harmful,
                "true_start": round(await truth(root), 4), "true_final": round(await truth(final), 4),
                "seconds": round(time.monotonic() - start, 2)}


async def experiment(out: Path, seeds: list[int], budget: int, arms: tuple[str, ...] = ARMS,
                     **kw: Any) -> dict[str, Any]:
    runs = [await run_arm(a, out / f"seed{s}" / a, s, budget, **kw) for s in seeds for a in arms]
    by = {(r["seed"], r["arm"]): r for r in runs}
    summary = {}
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        finals = [r["true_final"] for r in rs]
        acc = sum(r["accepted"] for r in rs)
        row = {"true_final_mean": round(fmean(finals), 4),
               "true_final_sd": round(stdev(finals), 4) if len(finals) > 1 else 0.0,
               "accepted_mean": round(acc / len(rs), 2),
               "harmful_rate": round(sum(r["harmful"] for r in rs) / acc, 3) if acc else 0.0}
        if a != "A" and "A" in arms and len(seeds) > 1:
            d = [by[(s, a)]["true_final"] - by[(s, "A")]["true_final"] for s in seeds]
            row["vs_A"] = round(fmean(d), 4)
            row["vs_A_se"] = round(stdev(d) / len(d) ** 0.5, 4)
        summary[a] = row
    report = {"label": SYNTHETIC, "budget": budget, "seeds": seeds, "settings": kw, "summary": summary, "runs": runs}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(r: dict[str, Any]) -> str:
    lines = [f"# 组提案（{r['label']}）", "",
             f"预算每组每个种子 {r['budget']} 次 agent 调用；种子 {r['seeds']}；设置 {r['settings']}。", "",
             "| 做法 | 最终真实答对率 | 相对 A（配对） | 平均接受次数 | 接受里真实变差 |", "| --- | --- | --- | --- | --- |"]
    for a, s in r["summary"].items():
        vs = f"{s['vs_A']:+.3f} ± {s['vs_A_se']:.3f}" if "vs_A" in s else "—"
        lines.append(f"| {a} | {s['true_final_mean']:.3f} ± {s['true_final_sd']:.3f} | {vs} | {s['accepted_mean']} | "
                     f"{s['harmful_rate']:.1%} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--collateral", type=float, default=0.5)
    ap.add_argument("--break-bias", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=Path("out/group_toy"))
    args = ap.parse_args(argv)
    asyncio.run(experiment(args.out, args.seeds, args.budget, tuple(args.arms.split(",")), q=args.q,
                           collateral=args.collateral, break_bias=args.break_bias))
    print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
