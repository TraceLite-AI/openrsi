"""真实 LLM 上跑完整的 AnyRSI 循环：agent 自己出改法（loop.self_propose）vs 独立的提示改写器（PromptRewriter）。

任务是 probe.py 里的难版秘密运算符：8 个符号各是一个较复杂的运算，系统提示里没写规则就按 a+b 算。
各组用同一个模型做题、出改法。组名写成 <循环>:<谁出改法>，只写一个词时等于 A:<词>：
  谁出改法
    self      当前版本的 agent 收到"改进你自己"的题，回一个 JSON 补丁（可以加、改、删提示段落）
    rewriter  anyrsi_llm:PromptRewriter，同一个模型以"提示改写器"的身份重写整段系统提示
    patch     anyrsi_llm:PatchProposer，同一道"改进"题由同一个模型以普通助手身份回答，JSON 补丁
  循环
    A         每代 1 个改法，考试分变高就收
    bon       组提案：一次 G 个改法，考试选择半上挑最好的一个，确认半准入
    grad      组提案 + 拆小合并：G 个改法拆成一条条小改动，逐条在选择半上测（有限差分），
              所有变好的合并成一步，确认半准入（loop.group.atomize + merge 策略）

    python benchmarks/self_propose_llm/loop_llm.py --model gpt-5.4-mini --seeds 0 1 2 --generations 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe import BASE, FORMAT, HARD, OPS, SYMBOLS  # noqa: E402

from anyrsi import Environment, Task  # noqa: E402


class HardOpsEnv(Environment):
    name, version = "bench/hard-secret-ops", "0.1.0"

    def __init__(self, seed: int = 0, symbols_per_round: int = 4, per_symbol: int = 3, holdout_per_symbol: int = 3,
                 easy: bool = False):
        rng = random.Random(seed)
        self.ops = {k: v for k, v in OPS.items() if k != "add"} if easy else HARD
        self.secret = dict(zip(SYMBOLS, rng.sample(sorted(self.ops), len(SYMBOLS))))
        self.seed, self.k, self.per, self.hold = seed, symbols_per_round, per_symbol, holdout_per_symbol

    def _task(self, split: str, key: str, sym: str, a: int, b: int) -> Task:
        return Task(id=f"{split}-{key}", split=split, tags=[sym], input={"question": f"Compute: {a} {sym} {b}\n{FORMAT}"},
                    reference={"expected": str(self.ops[self.secret[sym]](a, b))})

    def tasks(self, split: str, round: int = 0, limit: int | None = None) -> list[Task]:
        if split == "holdout":
            rng = random.Random(f"{self.seed}-holdout")
            out = [self._task("holdout", f"{s}{i}", s, rng.randint(2, 9), rng.randint(2, 9))
                   for s in SYMBOLS for i in range(self.hold)]
        elif split == "train":
            rng = random.Random(f"{self.seed}-train-{round}")
            out = [self._task("train", f"{round}.{s}{i}", s, rng.randint(2, 9), rng.randint(2, 9))
                   for s in rng.sample(SYMBOLS, self.k) for i in range(self.per)]
        else:
            out = []
        return out[:limit] if limit else out


def fresh_test(prompt: list[dict[str, Any]], seed: int, model: str, key_env: str, per_symbol: int = 6,
               easy: bool = False) -> float:
    """最终版本在一批没用过的新题上的分数（考试题被用来挑过，分数会偏高）。"""
    import os
    from concurrent.futures import ThreadPoolExecutor

    from anyrsi import Item
    from anyrsi_llm import chat, render_prompt

    env = HardOpsEnv(seed=seed, easy=easy)
    rng = random.Random(f"{seed}-fresh-test")
    tasks = [env._task("final", f"{s}{i}", s, rng.randint(2, 9), rng.randint(2, 9))
             for s in SYMBOLS for i in range(per_symbol)]
    system = render_prompt([Item.model_validate(d) for d in prompt]) or BASE

    def one(t: Task) -> bool:
        for attempt in range(4):
            try:
                out = chat("https://api.openai.com/v1", model, [{"role": "system", "content": system},
                                                                {"role": "user", "content": t.input["question"]}],
                           temperature=None, max_tokens=4000, api_key=os.environ.get(key_env), timeout=600)["text"]
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return False
                time.sleep(2 ** (attempt + 1))
        line = out.strip().splitlines()[0] if out.strip() else ""
        import re
        m = re.findall(r"=\s*(-?\d+)\s*$", line)
        return bool(m) and m[-1] == t.reference["expected"]

    with ThreadPoolExecutor(8) as pool:
        return sum(pool.map(one, tasks)) / len(tasks)


def write_config(base: Path, arm: str, seed: int, model: str, key_env: str, generations: int, size: int = 4,
                 easy: bool = False, confirm: float = 0.5) -> Path:
    mode, _, who = arm.rpartition(":")
    mode = mode or "A"
    llm = {"base_url": "https://api.openai.com/v1", "model": model, "api_key_env": key_env}
    comps: dict[str, Any] = {
        "env": {"python": "loop_llm:HardOpsEnv", "with": {"seed": seed, "easy": easy}},
        "agent": {"python": "anyrsi_llm:ChatAgent", "with": {**llm, "temperature": None, "max_tokens": 4000}},
        "judge": {"python": "anyrsi_judge:EvidenceJudge",
                  "with": {"rubric": {"items": [{"id": "correct", "check": "output_equals", "required": True,
                                                 "params": {"extract": r"=\s*(-?\d+)\s*$"}}]}}},
        "evidence": {"same_as": "judge"},
        "algorithm": {"python": "anyrsi_algorithm:GreedyAlgorithm",
                      "with": {"max_generations": generations, "patience": generations, "target_score": 1.0}},
        "gate": {"python": "anyrsi_gate:HoldoutGate", "with": {"min_gain": 0.0}},
        "lineage": {"python": "anyrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
    }
    loop: dict[str, Any] = {"max_patches": 1}
    if mode == "bon":
        loop["group"] = {"size": size, "strategy": "best_of_n", "confirm_fraction": confirm}
    elif mode == "grad":
        loop["group"] = {"size": size, "strategy": "merge", "confirm_fraction": confirm, "atomize": True,
                         "max_atoms": 4 * size}
    elif mode != "A":
        raise ValueError(f"unknown loop mode {mode}")
    if who == "self":
        loop["self_propose"] = True
    elif who == "rewriter":
        comps["proposer"] = {"python": "anyrsi_llm:PromptRewriter",
                             "with": {**llm, "temperature": None, "max_tokens": 4000, "item_id": "instructions"}}
    elif who == "patch":
        comps["proposer"] = {"python": "anyrsi_llm:PatchProposer",
                             "with": {**llm, "temperature": None, "max_tokens": 4000}}
    else:
        raise ValueError(f"unknown proposer {who}")
    data = {"protocol": "anyrsi/0.4", "name": f"llm-self-{arm}", "plugins": ["loop_llm"], "components": comps,
            "loop": loop,
            "evolvables": {"prompt": {"python": "anyrsi_evolvables:PromptEvolvable",
                                      "with": {"target": "prompt", "path": "state/prompt",
                                               "initial": [{"id": "instructions", "text": BASE}]}}}}
    base.mkdir(parents=True, exist_ok=True)
    (base / "anyrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "anyrsi.yaml"


async def run_arm(arm: str, base: Path, seed: int, model: str, key_env: str, generations: int,
                  size: int = 4, easy: bool = False, confirm: float = 0.5) -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(write_config(base, arm, seed, model, key_env, generations, size, easy, confirm))
    curve: list[float] = []
    calls_curve: list[int] = []
    async with Wiring(cfg) as w:
        agent = w.components["agent"]
        calls = [0]

        class Counting:
            kinds = agent.kinds

            async def run(self, task: Task, context: Any, snapshot_id: str | None = None) -> Any:
                calls[0] += 1
                out = agent.run(task, context, snapshot_id)
                return await out if asyncio.iscoroutine(out) else out

            def __getattr__(self, name: str) -> Any:
                return getattr(agent, name)

        w.components["agent"] = Counting()
        runner = Runner(w, cfg.loop)
        best = [None]

        def track(report: Any) -> None:
            for c in report.candidates:
                if c.accepted and c.metrics:
                    best[0] = c.metrics.get("holdout")
            curve.append(best[0] if best[0] is not None else report.parent_metrics.get("holdout"))
            calls_curve.append(calls[0])

        runner.on_generation = track
        await runner.run()
        log = await w.lineage.log()
        root = next(e for e in log if e.status == "root")
        accepted = [e for e in log if e.status == "accepted"]
        final = accepted[-1] if accepted else root
        prompt = await w.evolvables["prompt"].export(final.snapshots["prompt"])
    proposals = [e for e in log if e.status in ("accepted", "rejected")]
    test = fresh_test(prompt, seed, model, key_env, easy=easy)
    return {"arm": arm, "seed": seed, "model": model, "task": "easy" if easy else "hard", "confirm": confirm,
            "generations": generations, "start": root.metrics.get("holdout"),
            "final": final.metrics.get("holdout"), "test": test, "curve": curve, "accepted": len(accepted),
            "proposals": len(proposals), "calls": calls_curve[-1] if calls_curve else 0,
            "calls_curve": calls_curve, "final_prompt": [d["text"] for d in prompt]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")  # 先用便宜的模型
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--arms", nargs="+", default=["self", "rewriter"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--size", type=int, default=4, help="组提案每代几个改法（G）")
    ap.add_argument("--easy", action="store_true", help="简单版运算（a+b+1、max 这类）")
    ap.add_argument("--confirm", type=float, default=0.5, help="组提案的确认半比例；0 表示挑和判用同一批考试题")
    ap.add_argument("--workdir", default="loop_runs")
    ap.add_argument("--out", default="loop_results.jsonl")
    args = ap.parse_args()

    async def one(arm: str, seed: int) -> None:
        t = time.monotonic()
        r = await run_arm(arm, Path(args.workdir) / f"{args.model}-{arm.replace(':', '_')}-{seed}", seed, args.model,
                          args.api_key_env, args.generations, args.size, args.easy, args.confirm)
        r["secs"] = round(time.monotonic() - t)
        with open(args.out, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{args.model} {arm:8s} seed {seed}: {r['start']:.3f} -> {r['final']:.3f} (new tasks {r['test']:.3f})  "
              f"accepted {r['accepted']}/{r['proposals']}  calls {r['calls']}  curve {[round(x or 0, 2) for x in r['curve']]}  "
              f"({r['secs']}s)", flush=True)

    async def all_() -> None:
        await asyncio.gather(*(one(a, s) for a in args.arms for s in args.seeds))

    asyncio.run(all_())


if __name__ == "__main__":
    main()
