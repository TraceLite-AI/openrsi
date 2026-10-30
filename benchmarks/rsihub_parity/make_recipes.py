"""生成对照实验的 RSIHub 配方，写进 RSIHub 的 recipes/ 下。

以 RSIHub 公开的 recipes/hyperagents（MiniSWE 目标 agent、Codex 变异、GPT-5.4）为底，评测器改成
Terminal-Bench 2 全部 89 题，按 50 / 19 / 20 划分 train / gate / sealed（和 RSIHub README 结果表一致），
最后在 sealed 上做一次 anchor 评估。各组只在下面列出的算子上不同，其余完全相同：

  baseline   原样：validate=hyperagents（编译检查）、gate=parent_eligible（评估完整即入档）
  openrsi    validate=openrsi_guard（同样的编译检查 + 护栏，评估器受保护）
             gate=openrsi_no_regression（总分不降，且逐题变差不超过 N 道）
  shinka     在 openrsi 基础上 select=openrsi_shinka（可选）

用法：python make_recipes.py /path/to/RSIHub [--generations 10] [--max-task-regressions 2] [--arms baseline,openrsi]
"""
import argparse
import copy
import hashlib
import math
from pathlib import Path

import yaml

SPLIT = {"train": 0.5618, "gate": 0.2135, "sealed": 0.2247, "seed": 0}
EXPECTED = {"train": 50, "gate": 19, "sealed": 20}


def split_counts(n: int, ratios: dict) -> dict:
    """复现 RSIHub splits._assign 的计数规则，确认 89 题会分成 50 / 19 / 20。"""
    names = ["train", "gate", "sealed"]
    raw = {k: n * ratios[k] for k in names}
    counts = {k: math.floor(raw[k]) for k in names}
    for k in sorted(names, key=lambda k: (-(raw[k] - counts[k]), names.index(k)))[: n - sum(counts.values())]:
        counts[k] += 1
    return counts


def base_recipe(rsihub: Path, generations: int) -> dict:
    recipe = yaml.safe_load((rsihub / "recipes" / "hyperagents" / "evolve.yaml").read_text())
    recipe["experiment"]["max_generations"] = generations
    ev = recipe["evaluator"]
    for key in ("task_scope", "evaluation_split"):
        ev.pop(key, None)
    ev.update({"dataset": "terminal-bench@2.0", "split": dict(SPLIT), "tasks_per_round": 50,
               "anchor": {"final": True, "every_rounds": 0}})
    return recipe


def arm(recipe: dict, name: str, max_regressions: int) -> dict:
    r = copy.deepcopy(recipe)
    r["experiment"]["id"] = f"openrsi-parity-{name}"
    ops = r["operators"]
    if name in ("openrsi", "shinka"):
        ops["validate"] = {"operator": "openrsi_guard", "timeout_s": 300, "config": {
            "protected_paths": ["evaluator/**"],  # Hyperagents 允许改 operators/，所以只保护评估器
            "compile_globs": ["operators/mutate.py", "target/**/*.py"]}}  # 和 hyperagents validate 一样的编译检查
        ops["gate"] = {"operator": "openrsi_no_regression", "config": {
            "max_task_regressions": max_regressions, "task_tolerance": 0.0, "allow_equal": True}}
    if name == "shinka":
        ops["select"] = {"operator": "openrsi_shinka", "config": {"selection_lambda": 10.0}}
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rsihub")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--max-task-regressions", type=int, default=2)
    ap.add_argument("--arms", default="baseline,openrsi")
    args = ap.parse_args()
    rsihub = Path(args.rsihub).resolve()
    counts = split_counts(89, SPLIT)
    assert counts == EXPECTED, counts
    base = base_recipe(rsihub, args.generations)
    for name in args.arms.split(","):
        out = rsihub / "recipes" / f"openrsi_parity_{name}" / "evolve.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(arm(base, name, args.max_task_regressions), sort_keys=False, allow_unicode=True)
        out.write_text(text)
        print(f"{name:9s} {out}  sha256={hashlib.sha256(text.encode()).hexdigest()[:12]}")
    print(f"划分：{counts}（89 题，seed 0）")


if __name__ == "__main__":
    main()
