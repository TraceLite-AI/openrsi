"""测量噪声：同一个版本把考试题（holdout）跑 N 遍，看结果有多不稳定。在做任何方法对比之前先跑它。

    python benchmarks/noise/noise.py path/to/anyrsi.yaml --repeats 3

报告：
  翻转的题      N 遍结果不一致（有的通过有的没过）的题；比一个改动能修好的题还多，就测不出单条改动的效果
  部分得分      分数不是 0 或 1 的题数；有的话可以拆成更细的 rubric
  rubric 项     判别器给出的 rubric 条目（越多，方向越细）
  题目标签      task.tags（拆小合并、方向记录会按标签分解提升）
不进化、不写谱系。结果另存一份 JSON（--out）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import fmean
from typing import Any


async def measure(config: Path, repeats: int, limit: int | None = None) -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(config)
    async with Wiring(cfg) as w:
        r = Runner(w, cfg.loop)
        await r._load_targets()
        holdout = await r.tasks("holdout", 0, limit or cfg.loop.holdout_limit)
        genome = {t: (await tgt.facade.snapshot()).id for t, tgt in r.targets.items()}
        runs: dict[str, list[dict[str, Any]]] = {}
        tags: dict[str, list[str]] = {}
        totals = []
        for i in range(repeats):
            m, episodes = await r.score(holdout, genome, "holdout")
            totals.append(m["holdout"])
            for e in episodes:
                k = r._key(e.task)
                tags[k] = list(e.task.tags or [])
                runs.setdefault(k, []).append({"score": e.verdict.score, "passed": e.verdict.passed,
                                               "rubric": {x.id: x.result for x in e.verdict.rubric}})
            print(f"run {i + 1}/{repeats}: holdout {m['holdout']:.3f}", flush=True)
    flips = sorted(k for k, v in runs.items() if len({x["passed"] for x in v}) > 1)
    partial = sorted(k for k, v in runs.items() if any(x["score"] not in (0.0, 1.0) for x in v))
    rubric_ids = sorted({rid for v in runs.values() for x in v for rid in x["rubric"]})
    return {"config": str(config), "repeats": repeats, "tasks": len(runs), "holdout_per_run": totals,
            "holdout_mean": fmean(totals), "flipped": flips, "partial_score": partial, "rubric_items": rubric_ids,
            "tags": sorted({t for ts in tags.values() for t in ts}), "per_task": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="只用前几道考试题")
    ap.add_argument("--out", type=Path, default=Path("noise_report.json"))
    args = ap.parse_args()
    rep = asyncio.run(measure(args.config, args.repeats, args.limit))
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    n = rep["tasks"]
    print(f"\n{n} 道考试题 × {args.repeats} 遍：每遍分数 {', '.join(f'{x:.3f}' for x in rep['holdout_per_run'])}")
    print(f"翻转的题：{len(rep['flipped'])}/{n}  {rep['flipped'][:10]}")
    print(f"部分得分的题：{len(rep['partial_score'])}/{n}")
    print(f"rubric 项（{len(rep['rubric_items'])}）：{rep['rubric_items'][:10]}")
    print(f"题目标签（{len(rep['tags'])}）：{rep['tags'][:10]}")
    print(f"详细结果：{args.out}")


if __name__ == "__main__":
    main()
