"""真实 LLM 上跑完整的 AnyRSI 循环：agent 自己出改法（loop.self_propose）vs 独立的提示改写器（PromptRewriter）。

任务是 probe.py 里的难版秘密运算符：8 个符号各是一个较复杂的运算，系统提示里没写规则就按 a+b 算。
两组用同一个模型做题、出改法，只有"谁来出改法"不同：
    self      当前版本的 agent 收到"改进你自己"的题，回一个 JSON 补丁（可以加、改、删提示段落）
    rewriter  anyrsi_llm:PromptRewriter，同一个模型以"提示改写器"的身份重写整段系统提示

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
from probe import BASE, FORMAT, HARD, SYMBOLS  # noqa: E402

from anyrsi import Environment, Task  # noqa: E402


class HardOpsEnv(Environment):
    name, version = "bench/hard-secret-ops", "0.1.0"

    def __init__(self, seed: int = 0, symbols_per_round: int = 4, per_symbol: int = 3, holdout_per_symbol: int = 3):
        rng = random.Random(seed)
        self.secret = dict(zip(SYMBOLS, rng.sample(sorted(HARD), len(SYMBOLS))))
        self.seed, self.k, self.per, self.hold = seed, symbols_per_round, per_symbol, holdout_per_symbol

    def _task(self, split: str, key: str, sym: str, a: int, b: int) -> Task:
        return Task(id=f"{split}-{key}", split=split, tags=[sym], input={"question": f"Compute: {a} {sym} {b}\n{FORMAT}"},
                    reference={"expected": str(HARD[self.secret[sym]](a, b))})

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


def write_config(base: Path, arm: str, seed: int, model: str, key_env: str, generations: int) -> Path:
    llm = {"base_url": "https://api.openai.com/v1", "model": model, "api_key_env": key_env}
    comps: dict[str, Any] = {
        "env": {"python": "loop_llm:HardOpsEnv", "with": {"seed": seed}},
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
    if arm == "self":
        loop["self_propose"] = True
    else:
        comps["proposer"] = {"python": "anyrsi_llm:PromptRewriter",
                             "with": {**llm, "temperature": None, "max_tokens": 4000, "item_id": "instructions"}}
    data = {"protocol": "anyrsi/0.4", "name": f"llm-self-{arm}", "plugins": ["loop_llm"], "components": comps,
            "loop": loop,
            "evolvables": {"prompt": {"python": "anyrsi_evolvables:PromptEvolvable",
                                      "with": {"target": "prompt", "path": "state/prompt",
                                               "initial": [{"id": "instructions", "text": BASE}]}}}}
    base.mkdir(parents=True, exist_ok=True)
    (base / "anyrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "anyrsi.yaml"


async def run_arm(arm: str, base: Path, seed: int, model: str, key_env: str, generations: int) -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(write_config(base, arm, seed, model, key_env, generations))
    curve: list[float] = []
    async with Wiring(cfg) as w:
        runner = Runner(w, cfg.loop)
        best = [None]

        def track(report: Any) -> None:
            for c in report.candidates:
                if c.accepted and c.metrics:
                    best[0] = c.metrics.get("holdout")
            curve.append(best[0] if best[0] is not None else report.parent_metrics.get("holdout"))

        runner.on_generation = track
        await runner.run()
        log = await w.lineage.log()
        root = next(e for e in log if e.status == "root")
        accepted = [e for e in log if e.status == "accepted"]
        final = accepted[-1] if accepted else root
        prompt = await w.evolvables["prompt"].export(final.snapshots["prompt"])
    proposals = [e for e in log if e.status in ("accepted", "rejected")]
    return {"arm": arm, "seed": seed, "model": model, "start": root.metrics.get("holdout"),
            "final": final.metrics.get("holdout"), "curve": curve, "accepted": len(accepted),
            "proposals": len(proposals), "final_prompt": [d["text"] for d in prompt]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")  # 先用便宜的模型
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--arms", nargs="+", default=["self", "rewriter"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--workdir", default="loop_runs")
    ap.add_argument("--out", default="loop_results.jsonl")
    args = ap.parse_args()

    async def one(arm: str, seed: int) -> None:
        t = time.monotonic()
        r = await run_arm(arm, Path(args.workdir) / f"{args.model}-{arm}-{seed}", seed, args.model,
                          args.api_key_env, args.generations)
        r["secs"] = round(time.monotonic() - t)
        with open(args.out, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{args.model} {arm:8s} seed {seed}: {r['start']:.3f} -> {r['final']:.3f}  "
              f"accepted {r['accepted']}/{r['proposals']}  curve {[round(x or 0, 2) for x in r['curve']]}  "
              f"({r['secs']}s)", flush=True)

    async def all_() -> None:
        await asyncio.gather(*(one(a, s) for a in args.arms for s in args.seeds))

    asyncio.run(all_())


if __name__ == "__main__":
    main()
