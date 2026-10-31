"""OpenRSI 原生完整循环的 A/B/C 对照实验（spec/drafts/two-loop-evolution.md §8B，P3）。

    A  冻结初始 agent，不做任何改进                     → 能力与成本基线
    B  等搜索预算，每一代都从根版本出发（IndependentAlgorithm）→ 排除"只是多试几次"
    C  等搜索预算，下一代可以继承已接受的改进（GreedyAlgorithm） → 检验累积改进

规则（都由本脚本保证）：
  - 搜索预算按 agent 调用次数算，覆盖 train、开发期 holdout 打分、反事实重跑、失败重试；到上限立刻停，
    正在评估的候选记为未完成，不删行
  - 每组锁定最终候选以后才各做一次封存考试（final 划分），封存考试的调用另计、各组相同；
    搜索期间任何组件请求 final 题都会让实验失败
  - 报告逐题结果、通过率、两两配对差值（修好 / 改坏）、接受和拒绝的原因、调用次数和耗时

默认用 secret_ops 示例做合成冒烟实验，报告会标明"synthetic smoke，非 benchmark"。

    python benchmarks/native_loop/native_loop.py --budget 400 --seeds 0 1 2 --out out/native
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

ROOT = Path(__file__).resolve().parents[2]
for p in ("examples/secret_ops", "benchmarks/native_loop"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

from openrsi import Task  # noqa: E402
from openrsi_runtime import Runner, Wiring, load_config  # noqa: E402
from secret_ops import SYMBOLS, SecretOpsEnv  # noqa: E402

SYNTHETIC = "synthetic smoke，非 benchmark"
EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
GROUPS = {"A": None, "B": "openrsi_algorithm:IndependentAlgorithm", "C": "openrsi_algorithm:GreedyAlgorithm"}


class BudgetExhausted(RuntimeError):
    """搜索预算用完。"""


class SealedEnv(SecretOpsEnv):
    """secret_ops 环境加一套封存考试（final）：数字范围和 train / holdout 都不重叠。记下谁要过哪种题。"""

    name, version = "bench/sealed-secret-ops", "0.1.0"
    requested: list[str] = []

    def __init__(self, seed: int = 0, final_per_symbol: int = 5):
        super().__init__(seed=seed)
        self.final_per_symbol = final_per_symbol

    def tasks(self, split: str, round: int = 0, limit: int | None = None) -> list[Task]:
        SealedEnv.requested.append(split)
        if split != "final":
            return super().tasks(split, round, limit)
        rng = random.Random(f"{self.seed}-final")
        out = [self._task("final", str(i), s, rng.randint(100, 999), rng.randint(100, 999))
               for s in SYMBOLS for i in range(self.final_per_symbol)]
        return out[:limit] if limit else out


class _CountingAgent:
    """包住 agent 门面：每次 run 记一次调用，超过预算就停。"""

    def __init__(self, inner: Any, budget: int | None):
        self.inner, self.budget, self.calls = inner, budget, 0
        self.kinds = getattr(inner, "kinds", ["agent"])

    async def run(self, *args: Any, **kw: Any) -> Any:
        if self.budget is not None and self.calls >= self.budget:
            raise BudgetExhausted(f"搜索预算 {self.budget} 次 agent 调用已用完")
        self.calls += 1
        return await self.inner.run(*args, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def write_config(base: Path, seed: int, algorithm: str | None) -> Path:
    comps = {
        "env": {"python": "native_loop:SealedEnv", "with": {"seed": seed}},
        "agent": {"python": "secret_ops:RuleAgent"},
        "proposer": {"python": "secret_ops:RuleInducer"},
        "judge": {"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
        "evidence": {"same_as": "judge"},
        "attribution": {"python": "openrsi_attribution:RubricEvidenceAttributor"},
        "algorithm": {"python": algorithm or "openrsi_algorithm:GreedyAlgorithm",
                      "with": {"max_generations": 10_000, "patience": 10_000, "seed": seed}},
        "gate": {"python": "openrsi_gate:HoldoutGate", "with": {"min_gain": 0.0}},
        "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
        "memory": {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}},
    }
    base.mkdir(parents=True, exist_ok=True)
    data = {"protocol": "openrsi/0.4", "name": "native-loop", "plugins": ["native_loop", "secret_ops"],
            "components": comps, "loop": {"retrieve_k": 5, "max_patches": 1}}
    (base / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "openrsi.yaml"


async def run_group(name: str, base: Path, seed: int, budget: int) -> dict[str, Any]:
    cfg = load_config(write_config(base, seed, GROUPS[name]))
    start = time.monotonic()
    async with Wiring(cfg) as w:
        counter = _CountingAgent(w.components["agent"], budget)
        w.components["agent"] = counter
        runner = Runner(w, cfg.loop)
        stopped, incomplete = "", False
        try:
            # A 组不搜索：只建根版本（它的开发期 holdout 打分也计入调用，三组相同）
            await runner.run(0 if name == "A" else None)
            stopped = "A 组不做改进" if name == "A" else "算法停止"
        except BudgetExhausted as e:
            stopped, incomplete = str(e), True
        search_calls = counter.calls
        log = await w.lineage.log()
        if "final" in SealedEnv.requested:
            raise RuntimeError("搜索阶段有组件请求了封存考试题（final），实验无效")
        # 锁定最终候选：开发期 holdout 最高的已接受版本；A 组就是根版本
        root = next(e for e in log if e.status == "root")
        final_entry = root if name == "A" else (await w.lineage.best("holdout", runner.lineage_target) or root)
        counter.budget = None  # 封存考试另计
        before = counter.calls
        final_tasks = await w.env.tasks("final", 0, None)
        m, episodes = await runner.score(final_tasks, runner._genome_of(final_entry), "final")
        sealed_calls = counter.calls - before
    accepted = [e for e in log if e.status == "accepted"]
    rejected = [e for e in log if e.status == "rejected"]
    reasons: dict[str, int] = {}
    for e in rejected:
        key = (e.decision.reason if e.decision else "") or "未知"
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "group": name, "seed": seed, "stopped": stopped, "incomplete_generation": incomplete,
        "search_agent_calls": search_calls, "budget": budget, "sealed_agent_calls": sealed_calls,
        "generations": max((e.generation for e in log), default=0), "accepted": len(accepted),
        "rejected": len(rejected), "reject_reasons": reasons,
        "locked_version": final_entry.id, "dev_holdout": final_entry.metrics.get("holdout"),
        "final": m["final"], "final_pass": m["final_pass"],
        "final_tasks": {e.task.id: e.verdict.score for e in episodes},
        "seconds": round(time.monotonic() - start, 2),
    }


def paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """b 相对 a 的逐题配对比较：修好、改坏、平均差值。"""
    keys = sorted(a)
    fixed = sum(1 for k in keys if a[k] < 0.5 <= b[k])
    broke = sum(1 for k in keys if b[k] < 0.5 <= a[k])
    diff = sum(b[k] - a[k] for k in keys) / len(keys) if keys else 0.0
    return {"fixed": fixed, "broke": broke, "mean_diff": round(diff, 4)}


async def experiment(out: Path, seeds: list[int], budget: int) -> dict[str, Any]:
    runs = []
    for seed in seeds:
        for name in GROUPS:
            SealedEnv.requested = []
            runs.append(await run_group(name, out / f"seed{seed}" / name, seed, budget))
    by = {(r["seed"], r["group"]): r for r in runs}
    pairs = [{"seed": s, "pair": f"{y}-{x}", **paired(by[(s, x)]["final_tasks"], by[(s, y)]["final_tasks"])}
             for s in seeds for x, y in (("A", "B"), ("A", "C"), ("B", "C"))]
    report = {"label": SYNTHETIC, "budget_agent_calls": budget, "seeds": seeds, "runs": runs, "paired": pairs}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [f"# 原生循环 A/B/C（{report['label']}）", "",
             f"搜索预算：每组 {report['budget_agent_calls']} 次 agent 调用；种子 {report['seeds']}。", "",
             "| 种子 | 组 | 搜索调用 | 代数 | 接受 / 拒绝 | 停止原因 | 开发期 holdout | 封存考试通过率 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in report["runs"]:
        lines.append(f"| {r['seed']} | {r['group']} | {r['search_agent_calls']} | {r['generations']} | "
                     f"{r['accepted']} / {r['rejected']} | {r['stopped']} | {r['dev_holdout']:.3f} | "
                     f"{r['final_pass']:.3f} |")
    lines += ["", "| 种子 | 比较 | 修好 | 改坏 | 平均差值 |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {p['seed']} | {p['pair']} | {p['fixed']} | {p['broke']} | {p['mean_diff']:+.4f} |"
              for p in report["paired"]]
    lines += ["", "封存考试在每组锁定最终版本之后各做一次，调用另计；逐题结果见 report.json。"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budget", type=int, default=400, help="每组的搜索预算（agent 调用次数）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, default=Path("out/native_loop"))
    args = ap.parse_args(argv)
    report = asyncio.run(experiment(args.out, args.seeds, args.budget))
    print((args.out / "report.md").read_text(encoding="utf-8"))
    return 0 if report["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
