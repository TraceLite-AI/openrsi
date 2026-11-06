"""挑考试题：从测噪声的结果里找"能正常判分、起点 agent 又常做不出来"的题。

考试题（gate）决定每个改法收不收。一道题如果判分器跑不起来、或者起点 agent 每次都能做对，它对准入没有任何贡献，
只增加花费；起点 agent 时对时错的题还会带来噪声。上一次 Terminal-Bench 实验的 19 道考试题里，
13 道判分器没跑起来、4 道每次都过、2 道时过时不过——没有一道能证明进化。

    python benchmarks/noise/noise.py anyrsi.yaml --repeats 3 --split all --out noise_all.json
    python benchmarks/terminal_preflight/headroom.py noise_all.json --size 19

每道题按多遍结果分成：
  broken     有无法评分的结果（判分环境问题，先修环境）
  saturated  每次都通过（没有提升空间）
  flaky      时过时不过（有空间，但有噪声）
  failing    每次都失败（有空间；可能超出模型能力）
考试题按 flaky、failing 的顺序选（flaky 里通过率低的优先），只从练习题和原考试题里选，封存题不碰。
输出一段可以直接放进 RSIHubTaskEnv 配置的 gate 列表。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def classify(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for task, runs in report["per_task"].items():
        if report.get("splits", {}).get(task) == "final":
            continue
        scored = [r for r in runs if "unscorable" not in r]
        passes = sum(bool(r.get("passed")) for r in scored)
        if len(scored) < len(runs) or not scored:
            kind = "broken"
        elif passes == len(scored):
            kind = "saturated"
        elif passes == 0:
            kind = "failing"
        else:
            kind = "flaky"
        out[task] = {"kind": kind, "runs": len(runs), "scored": len(scored), "passes": passes,
                     "pass_rate": passes / len(scored) if scored else None}
    return out


def pick(classes: dict[str, dict[str, Any]], size: int, include_failing: bool = True) -> list[str]:
    flaky = sorted((t for t, c in classes.items() if c["kind"] == "flaky"), key=lambda t: (classes[t]["pass_rate"], t))
    failing = sorted(t for t, c in classes.items() if c["kind"] == "failing") if include_failing else []
    return (flaky + failing)[:size]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", type=Path, nargs="+", help="noise.py 的输出（可以多份，同一道题的结果会合并）")
    ap.add_argument("--size", type=int, default=19, help="想要几道考试题")
    ap.add_argument("--min-size", type=int, default=8, help="少于这么多道就提醒：考试题太少测不出改进")
    ap.add_argument("--no-failing", action="store_true", help="不选每次都失败的题（只选时过时不过的）")
    args = ap.parse_args()
    merged: dict[str, Any] = {"per_task": {}, "splits": {}}
    for path in args.reports:
        rep = json.loads(path.read_text(encoding="utf-8"))
        merged["splits"].update(rep.get("splits", {}))
        for t, runs in rep["per_task"].items():
            merged["per_task"].setdefault(t, []).extend(runs)
    classes = classify(merged)
    counts: dict[str, int] = {}
    for c in classes.values():
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    print("题目分类：" + "，".join(f"{k} {counts.get(k, 0)}" for k in ("broken", "saturated", "flaky", "failing")))
    chosen = pick(classes, args.size, not args.no_failing)
    for t in chosen:
        c = classes[t]
        print(f"  {t}: {c['kind']}，{c['passes']}/{c['scored']} 次通过")
    if len(chosen) < args.min_size:
        print(f"\n只有 {len(chosen)} 道有提升空间的题（少于 {args.min_size}）：考试题太少，测不出改进。"
              "先修好判分环境（broken 的题），或者换一个弱一点的做题模型。")
    broken = sorted(t for t, c in classes.items() if c["kind"] == "broken")
    if broken:
        print(f"\n判分环境有问题的题 {len(broken)} 道（先修环境再用）：{broken[:10]}")
    print("\n放进 RSIHubTaskEnv 的 with 里：\n  gate: " + json.dumps(chosen, ensure_ascii=False))


if __name__ == "__main__":
    main()
