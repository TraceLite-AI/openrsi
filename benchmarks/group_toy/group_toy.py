"""组提案的玩具实验：一组候选在考试题上比，四种策略和"现在的循环"比，最优解已知。

被进化的是 12 个"位置"（prompt 可进化体里的 s0..s11），每个取 good / neutral / bad，权重不同（两个无关）。
标签只是名字：s1、s4、s7、s9 这四个位置"bad"反而最好，所以提案器要知道每个位置该选哪个（改对的概率 q）。
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

递归改进实验（--recursive）：提案器会读 loop.improver_feedback 写回来的"考试上好 / 差的改法"，
下次改同一个位置时偏向赢过的取值、避开输过的顺手改动。看提案器出好改法的比例是不是逐代上升：
    A / A+fb / group / group+fb / group+fb-shuffled（好坏标签随机打乱的对照）

    python benchmarks/group_toy/group_toy.py --recursive --budget 6000 --seeds 0 1 2 3 4 5 6 7 8 9

agent 给自己出改法（--self，loop.self_propose）：agent 收到"改进你自己"的题时，出改法的水平跟它自己的能力挂钩
（真实答对率越接近最优，改对的概率越高、顺手改坏的越少）——这是递归改进的核心假设，玩具里是设定的。对照：
    group:fixed   组提案 + 外部提案器，水平固定在 agent 起点的水平（只有 agent 自进化）
    group:self    组提案 + agent 给自己出改法（递归）
    A:fixed / A:self   同上，但用原来的循环（每代 1 个改法，分数变高就收）
    group:strong  组提案 + 一个一直很强的外部提案器（上限参考）

    python benchmarks/group_toy/group_toy.py --self --budget 6000 --seeds 0 1 2 3 4 5 6 7 8 9
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
# 标签只是名字：有几个位置"bad"反而最好（极性反过来），所以"改成 good"不总是对的，
# 提案器要知道每个位置该选哪个（改对的概率 q），反馈里"哪个取值赢了"才真的带信息
FLIPPED = {"s1", "s4", "s7", "s9"}
BEST = {s: ("bad" if s in FLIPPED else "good") for s in SLOTS}
WORST = {s: ("good" if s in FLIPPED else "bad") for s in SLOTS}
START = {s: (WORST[s] if i % 2 == 0 else "neutral") for i, s in enumerate(SLOTS)}


def val(slot: str, label: str) -> float:
    return -VALUE[label] if slot in FLIPPED else VALUE[label]
SET = re.compile(r"\b(s\d+)=(good|neutral|bad)\b")
REPLACE = re.compile(r"\breplace (s\d+): (good|neutral|bad)\b")
SYNTHETIC = "synthetic toy，已知最优解，非 benchmark"


def true_p(values: dict[str, str], difficulty: float = 0.0) -> float:
    s = sum(WEIGHTS[k] * val(k, values.get(k, "neutral")) for k in SLOTS) / sum(WEIGHTS.values())
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


P_START, P_BEST = 0.39, 0.85


def capability(values: dict[str, str]) -> float:
    """agent 的能力：真实答对率在起点和最优之间走到了哪（0～1）。"""
    return min(1.0, max(0.0, (true_p(values) - P_START) / (P_BEST - P_START)))


class ToyAgent(Agent):
    """做题：按真实答对率掷硬币。收到"改进你自己"的题（loop.self_propose）时出一个改法，水平跟自己的能力挂钩：
    改对的概率 self_q_lo → self_q_hi，顺手改坏的概率 self_break → 0（能力从 0 到 1 线性变化）。"""

    name, version = "toy/group-agent", "0.1.0"

    def __init__(self, seed: int = 0, attr_acc: float = 0.5, self_q_lo: float = 0.2, self_q_hi: float = 0.9,
                 self_collateral: float = 0.5, self_break: float = 0.8):
        self.rng, self.attr_acc = random.Random(f"agent:{seed}"), attr_acc
        self.self_q = (self_q_lo, self_q_hi)
        self.self_collateral, self.self_break = self_collateral, self_break

    def _improve_self(self, task: Task, context: list[Item], snapshot_id: str | None) -> Trajectory:
        values = effective(context)
        cap = capability(values)
        q = self.self_q[0] + (self.self_q[1] - self.self_q[0]) * cap
        seed = task.metadata.get("rollout_seed")
        rng = random.Random(seed) if isinstance(seed, int) else self.rng
        blames = [a["ref"] for a in task.input.get("attributions", []) if a.get("weight", 0) < 0]
        direction = (task.input.get("hints") or {}).get("direction")
        change = make_edit(rng, values, blames, direction, q, self.self_collateral, self.self_break * (1 - cap))
        ops = [{"op": "replace", "item": {"id": k, "kind": "prompt", "text": v}}
               for k, v in sorted(change.items()) if values.get(k) != v]
        out = json.dumps({"target": "prompt", "ops": ops,
                          "rationale": "set " + ", ".join(f"{k}={v}" for k, v in sorted(change.items()))})
        return Trajectory(id=f"traj-{task.id}-{snapshot_id}", task_id=task.id, snapshot_id=snapshot_id, output=out,
                          metadata={"capability": cap, "q": q})

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        if isinstance(task.input, dict) and task.input.get("kind") == "self_improvement":
            return self._improve_self(task, context, snapshot_id)
        values = effective(context)
        p = true_p(values, task.metadata.get("difficulty", 0.0))
        seed = task.metadata.get("rollout_seed")
        u = random.Random(seed).random() if isinstance(seed, int) else self.rng.random()
        right = u < p
        blame = None
        if not right:
            harm = {k: WEIGHTS[k] * (1 - val(k, values.get(k, "neutral"))) for k in SLOTS}
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


def make_edit(rng: random.Random, current: dict[str, str], blames: list[str], direction: Any, q: float,
              collateral: float, break_bias: float, score: dict[tuple[str, str], int] | None = None) -> dict[str, str]:
    """出一个改法：挑一个位置（有方向按方向，否则随便挑一个被归因的位置），改对的概率 q；
    还有 collateral 的概率顺手改动另一个位置，其中 break_bias 的概率是改坏。score 是提案器反馈学到的偏好。"""
    score = score or {}
    if direction:
        slot = str(direction["id"]).split(":", 1)[-1]
    else:
        slot = rng.choice(blames) if blames else rng.choice(SLOTS)
    if slot not in SLOTS:
        slot = rng.choice(SLOTS)
    known = {v: score[(slot, v)] for v in VALUE if score.get((slot, v), 0) > 0}
    if known and rng.random() < 0.9:
        change = {slot: max(known, key=known.get)}
    else:
        change = {slot: BEST[slot] if rng.random() < q else rng.choice(list(VALUE))}
    if rng.random() < collateral:
        other = rng.choice([s for s in SLOTS if s != slot])
        value = WORST[other] if rng.random() < break_bias else rng.choice(list(VALUE))
        if score.get((other, value), 0) >= 0:
            change[other] = value
    return change


class ToyProposer(Proposer):
    """有方向（hints.direction = item:sN）就改 sN；否则随便挑一道错题，按它的归因改。改对的概率 q，
    顺手改动别处的概率 collateral（其中 break_bias 是改坏）。一次可以返回多个提案（max_patches）。

    会读提案器的策略说明（hints.improver，由 loop.improver_feedback 写入）："BETTER … set sK=V" 记一分，
    "WORSE …" 扣一分。要改的位置有得分为正的取值时，九成照着改；顺手改动如果是得分为负的取值，就不做。
    没有策略说明时和原来完全一样。"""

    name, version = "toy/group-proposer", "0.1.0"

    def __init__(self, seed: int = 0, q: float = 0.6, collateral: float = 0.5, break_bias: float = 0.0):
        self.rng, self.q, self.collateral, self.break_bias = random.Random(f"proposer:{seed}"), q, collateral, break_bias

    @staticmethod
    def learned(request: ProposalRequest) -> dict[tuple[str, str], int]:
        score: dict[tuple[str, str], int] = {}
        for items in ((request.hints or {}).get("improver") or {}).values():
            for d in items or []:
                text = str(d.get("text", "")) if isinstance(d, dict) else ""
                sign = 1 if text.startswith("BETTER") else -1 if text.startswith("WORSE") else 0
                for k, v in SET.findall(text):
                    score[(k, v)] = score.get((k, v), 0) + sign
        mom = (request.hints or {}).get("momentum") or {}  # 方向动量：helped 沿着走，hurt 避开
        for key, sign in (("helped", 1), ("hurt", -1)):
            for text in mom.get(key) or []:
                for k, v in set(REPLACE.findall(str(text))) | set(SET.findall(str(text))):
                    score[(k, v)] = score.get((k, v), 0) + sign
        return score

    def _one(self, request: ProposalRequest) -> Patch | None:
        current = effective(request.state or [])
        blames = [t.ref for a in request.attributions for t in a.targets if t.weight < 0]
        change = make_edit(self.rng, current, blames, (request.hints or {}).get("direction"), self.q,
                           self.collateral, self.break_bias, self.learned(request))
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
    """arm：A、paired、grpo、best_of_n、training_free，后面可以加 +fb（提案器反馈）或 +fb-shuffled（打乱的对照）；
    group 等于 best_of_n。也可以写成 <A|group>:<fixed|self|strong>（--self 实验）：fixed 是水平固定在 agent 起点的
    外部提案器，strong 是一直很强的外部提案器，self 是 agent 给自己出改法。"""
    proposer_mode = None
    if ":" in arm:
        arm, proposer_mode = arm.split(":", 1)
        if proposer_mode == "fixed":
            q, collateral, break_bias = 0.2, 0.5, 0.8
        elif proposer_mode == "strong":
            q, collateral, break_bias = 0.9, 0.5, 0.0
    base_arm, _, fb = arm.partition("+")
    base_arm = "best_of_n" if base_arm == "group" else base_arm
    loop: dict[str, Any] = {"max_patches": 1, "train_limit": 30}
    evolvables: dict[str, Any] = {"prompt": {"python": "anyrsi_evolvables:PromptEvolvable",
                                             "with": {"target": "prompt", "path": "state/prompt",
                                                      "initial": [{"id": k, "text": v} for k, v in START.items()]}}}
    if fb:
        evolvables["improver"] = {"python": "anyrsi_evolvables:ImproverEvolvable",
                                  "with": {"target": "improver", "path": "state/improver"}}
        loop["improver_feedback"] = {"target": "improver", "shuffle": fb == "fb-shuffled"}
    arm = base_arm
    if arm == "grad":  # 拆小合并：每个改法拆成单条改动逐条测，变好的合并成一步
        loop["group"] = {"size": size, "strategy": "merge", "confirm_fraction": 0.5, "atomize": True,
                         "max_atoms": 4 * size}
    elif arm == "mom":  # 拆小合并 + 方向动量
        loop["group"] = {"size": size, "strategy": "merge", "confirm_fraction": 0.5, "atomize": True,
                         "max_atoms": 4 * size, "momentum": {"decay": 0.7, "max_items": 6, "explore": 1}}
    elif arm != "A":
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
    if proposer_mode == "self":
        loop["self_propose"] = True
        del comps["proposer"]
    data = {"protocol": "anyrsi/0.4", "name": f"group-toy-{arm}", "plugins": ["group_toy"],
            "components": comps, "loop": loop, "evolvables": evolvables}
    base.mkdir(parents=True, exist_ok=True)
    (base / "anyrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "anyrsi.yaml"


async def run_arm(arm: str, base: Path, seed: int, budget: int, **kw: Any) -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(write_config(base, arm, seed, **kw))
    start = time.monotonic()
    quality: list[float] = []  # 每个提案的真实效果（相对提案时的父版本）
    lift: list[float] = []     # 同一时刻：读反馈的提案器 − 不读反馈的同款提案器，出好改法的比例之差
    ghost = ToyProposer(seed=seed + 7919, q=kw.get("q", 0.6), collateral=kw.get("collateral", 0.5),
                        break_bias=kw.get("break_bias", 0.0))
    curve: list[tuple[int, str | None]] = []  # (已用调用数, 当时最新接受的版本)
    async with Wiring(cfg) as w:
        agent = w.components["agent"]

        class MeasuredAgent:  # agent 给自己出改法时，记下改法的真实效果
            kinds = agent.kinds

            async def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Any:
                traj = await agent.run(task, context, snapshot_id)
                if isinstance(task.input, dict) and task.input.get("kind") == "self_improvement":
                    try:
                        ops = json.loads(traj.output)["ops"]
                    except (TypeError, ValueError, KeyError):
                        ops = None
                    if ops:
                        before = effective(context)
                        after = {**before, **{o["item"]["id"]: o["item"]["text"] for o in ops
                                              if o.get("op") == "replace" and o["item"]["id"] in SLOTS}}
                        quality.append(true_p(after) - true_p(before))
                return traj

            def __getattr__(self, name: str) -> Any:
                return getattr(agent, name)

        counter = _Counting(MeasuredAgent(), budget)
        w.components["agent"] = counter
        inner = w.components.get("proposer")

        class Measured:
            kinds = getattr(inner, "kinds", ["proposer"])

            async def propose(self, req: ProposalRequest) -> list[Patch]:
                patches = await inner.propose(req)
                before = effective(req.state or [])

                def gain(p: Patch | None) -> float:
                    if p is None:
                        return 0.0
                    after = dict(before)
                    for op in p.ops:
                        if op.get("op") == "replace" and op["item"]["id"] in SLOTS:
                            after[op["item"]["id"]] = op["item"]["text"]
                    return true_p(after) - true_p(before)

                plain = req.model_copy(update={"hints": {k: v for k, v in (req.hints or {}).items()
                                                         if k != "improver"}})
                for p in patches:
                    g = gain(p)
                    quality.append(g)
                    if (req.hints or {}).get("improver"):
                        base = fmean(gain(ghost._one(plain)) > 0 for _ in range(8))
                        lift.append(float(g > 0) - base)
                return patches

            def __getattr__(self, name: str) -> Any:
                return getattr(inner, name)

        if inner is not None:
            w.components["proposer"] = Measured()
        runner = Runner(w, cfg.loop)
        latest: list[str | None] = [None]

        def track(report: Any) -> None:
            for c in report.candidates:
                if c.accepted:
                    latest[0] = c.id
            curve.append((counter.calls, latest[0]))

        runner.on_generation = track
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
        third = max(1, len(quality) // 3)
        by_id = {e.id: e for e in log}
        points = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            upto = [cid for calls, cid in curve if calls <= frac * budget]
            cid = next((c for c in reversed(upto) if c), None)
            points.append(round(await truth(by_id[cid]) if cid else await truth(root), 4))
        return {"arm": arm, "seed": seed, "curve": points, "calls": counter.calls, "accepted": len(accepted), "harmful": harmful,
                "true_start": round(await truth(root), 4), "true_final": round(await truth(final), 4),
                "proposals": len(quality),
                "good_early": round(fmean(q > 0 for q in quality[:third]), 4) if quality else 0.0,
                "good_late": round(fmean(q > 0 for q in quality[-third:]), 4) if quality else 0.0,
                "lift_early": round(fmean(lift[: max(1, len(lift) // 3)]), 4) if lift else None,
                "lift_late": round(fmean(lift[-max(1, len(lift) // 3):]), 4) if lift else None,
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
               "good_early": round(fmean(r["good_early"] for r in rs), 3),
               "good_late": round(fmean(r["good_late"] for r in rs), 3),
               "lift_early": round(fmean(r["lift_early"] for r in rs), 3) if rs[0]["lift_early"] is not None else None,
               "lift_late": round(fmean(r["lift_late"] for r in rs), 3) if rs[0]["lift_late"] is not None else None,
               "true_final_sd": round(stdev(finals), 4) if len(finals) > 1 else 0.0,
               "accepted_mean": round(acc / len(rs), 2),
               "harmful_rate": round(sum(r["harmful"] for r in rs) / acc, 3) if acc else 0.0,
               "curve": [round(fmean(r["curve"][i] for r in rs), 3) for i in range(4)]}
        base = arms[0]  # 第一组是对照
        if a != base and len(seeds) > 1:
            d = [by[(s, a)]["true_final"] - by[(s, base)]["true_final"] for s in seeds]
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
             f"| 做法 | 最终真实答对率 | 相对 {next(iter(r['summary']))}（配对） | 平均接受次数 | 接受里真实变差 | "
             "好提案比例 前 1/3 → 后 1/3 | 读反馈带来的提升 前 1/3 → 后 1/3 | 真实答对率 @25/50/75/100% 预算 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a, s in r["summary"].items():
        vs = f"{s['vs_A']:+.3f} ± {s['vs_A_se']:.3f}" if "vs_A" in s else "—"
        lf = f"{s['lift_early']:+.2f} → {s['lift_late']:+.2f}" if s.get("lift_early") is not None else "—"
        cv = " / ".join(f"{x:.2f}" for x in s["curve"])
        lines.append(f"| {a} | {s['true_final_mean']:.3f} ± {s['true_final_sd']:.3f} | {vs} | {s['accepted_mean']} | "
                     f"{s['harmful_rate']:.1%} | {s['good_early']:.2f} → {s['good_late']:.2f} | {lf} | {cv} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--collateral", type=float, default=0.5)
    ap.add_argument("--break-bias", type=float, default=0.0)
    ap.add_argument("--recursive", action="store_true", help="提案器反馈实验：提案器读考试反馈")
    ap.add_argument("--self", dest="self_", action="store_true", help="递归改进实验：agent 给自己出改法")
    ap.add_argument("--out", type=Path, default=Path("out/group_toy"))
    args = ap.parse_args(argv)
    arms = ("A", "A+fb", "group", "group+fb", "group+fb-shuffled") if args.recursive else tuple(args.arms.split(","))
    if args.self_:
        arms = ("A:fixed", "A:self", "group:fixed", "group:self", "group:strong")
    asyncio.run(experiment(args.out, args.seeds, args.budget, arms, q=args.q,
                           collateral=args.collateral, break_bias=args.break_bias))
    print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
