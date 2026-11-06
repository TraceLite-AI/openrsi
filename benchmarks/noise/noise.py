"""测量噪声：同一个版本把考试题（holdout）跑 N 遍，看结果有多不稳定。在做任何方法对比之前先跑它。

    python benchmarks/noise/noise.py path/to/anyrsi.yaml --repeats 3
    python benchmarks/noise/noise.py path/to/anyrsi.yaml --repeats 3 --split all   # 练习题 + 考试题，用来挑考试题

--split：holdout（默认，考试题）、train（全部练习题，按代轮换的题会合并）、all（两者）。封存题（final）不能测：
拿它来挑题就把它泄漏了。

报告：
  翻转的题      N 遍结果不一致（有的通过有的没过）的题；比一个改动能修好的题还多，就测不出单条改动的效果
  部分得分      分数不是 0 或 1 的题数；有的话可以拆成更细的 rubric
  rubric 项     判别器给出的 rubric 条目（越多，方向越细）
  题目标签      task.tags（拆小合并、方向记录会按标签分解提升）
  无法评分      判分器没跑起来的题（判别器把必需的 rubric 项标成 error，见 anyrsi_runtime.scoring），单独记，不算失败
不进化、不写谱系，也不因无法评分的题太多而中断（这里就是要看清楚）。结果另存一份 JSON（--out），
可以交给 benchmarks/terminal_preflight/headroom.py 挑考试题。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import fmean
from typing import Any


async def pool(r: Any, split: str, limit: int | None) -> list[Any]:
    if split == "holdout":
        return await r.tasks("holdout", 0, limit)
    if split == "train":  # 按代轮换的练习题：一直取到不再出现新题为止
        seen: dict[str, Any] = {}
        stale = 0
        for rnd in range(1000):
            new = [t for t in await r.tasks("train", rnd, None) if t.id not in seen]
            seen.update({t.id: t for t in new})
            stale = 0 if new else stale + 1
            if stale >= 20:
                break
        return list(seen.values())[:limit] if limit else list(seen.values())
    if split == "all":
        return [*(await pool(r, "train", None)), *(await pool(r, "holdout", None))][: limit or None]
    raise ValueError("--split 只能是 holdout、train、all（封存题不能拿来挑题）")


async def measure(config: Path, repeats: int, limit: int | None = None, split: str = "holdout") -> dict[str, Any]:
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(config)
    cfg.loop.max_unscorable = None  # 诊断：全部记下来，不中断
    async with Wiring(cfg) as w:
        r = Runner(w, cfg.loop)
        await r._load_targets()
        holdout = await pool(r, split, limit or (cfg.loop.holdout_limit if split == "holdout" else None))
        genome = {t: (await tgt.facade.snapshot()).id for t, tgt in r.targets.items()}
        runs: dict[str, list[dict[str, Any]]] = {}
        tags: dict[str, list[str]] = {r._key(t): list(t.tags or []) for t in holdout}
        splits = {r._key(t): t.split for t in holdout}
        totals = []
        for i in range(repeats):
            m, episodes = await r.score(holdout, genome, "holdout")
            totals.append(m["holdout"])
            for e in episodes:
                k = r._key(e.task)
                runs.setdefault(k, []).append({"score": e.verdict.score, "passed": e.verdict.passed,
                                               "rubric": {x.id: x.result for x in e.verdict.rubric}})
            for k, why in r.last_unscorable.items():
                runs.setdefault(k, []).append({"unscorable": why})
            print(f"run {i + 1}/{repeats}: score {m['holdout']:.3f} on {int(m['n_holdout'])} scorable, "
                  f"{len(r.last_unscorable)} unscorable", flush=True)
    scored = {k: [x for x in v if "unscorable" not in x] for k, v in runs.items()}
    flips = sorted(k for k, v in scored.items() if len({x["passed"] for x in v}) > 1)
    partial = sorted(k for k, v in scored.items() if any(x["score"] not in (0.0, 1.0) for x in v))
    rubric_ids = sorted({rid for v in scored.values() for x in v for rid in x["rubric"]})
    broken = sorted(k for k, v in runs.items() if any("unscorable" in x for x in v))
    return {"config": str(config), "split": split, "repeats": repeats, "tasks": len(runs), "splits": splits,
            "unscorable": broken, "holdout_per_run": totals,
            "holdout_mean": fmean(totals), "flipped": flips, "partial_score": partial, "rubric_items": rubric_ids,
            "tags": sorted({t for ts in tags.values() for t in ts}), "per_task": runs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="只用前几道题")
    ap.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    ap.add_argument("--out", type=Path, default=Path("noise_report.json"))
    args = ap.parse_args()
    rep = asyncio.run(measure(args.config, args.repeats, args.limit, args.split))
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    n = rep["tasks"]
    print(f"\n{n} 道题（{args.split}）× {args.repeats} 遍：每遍分数（只算能评分的题）"
          f"{', '.join(f'{x:.3f}' for x in rep['holdout_per_run'])}")
    print(f"无法评分的题：{len(rep['unscorable'])}/{n}  {rep['unscorable'][:10]}")
    print(f"翻转的题：{len(rep['flipped'])}/{n}  {rep['flipped'][:10]}")
    print(f"部分得分的题：{len(rep['partial_score'])}/{n}")
    print(f"rubric 项（{len(rep['rubric_items'])}）：{rep['rubric_items'][:10]}")
    print(f"题目标签（{len(rep['tags'])}）：{rep['tags'][:10]}")
    print(f"详细结果：{args.out}")


if __name__ == "__main__":
    main()
