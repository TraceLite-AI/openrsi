"""手写补丁测试：一个你认为最好的修改，在考试题上到底能不能测出提升？

用已在源题上验证的手写修改，检查准入题能否检测到它的效果。

    python benchmarks/terminal_preflight/handpatch.py anyrsi.yaml --diff my_change.diff --repeats 3
    python benchmarks/terminal_preflight/handpatch.py anyrsi.yaml --patch my_patch.json --repeats 3

--diff   对被进化的代码仓库（harness）的 unified diff（git diff 的输出）
--patch  JSON：{"target": "<可进化体>", "ops": [...]}，ops 用该可进化体的补丁格式（提示、记忆等也能测）
起点版本和打了补丁的版本交替各跑 --repeats 遍同一批题（默认考试题；--split train / all 可换）。不进化、不写谱系，
补丁只在测试期间生效，结束后切回起点版本。封存题不能用。

只使用同题同轮两个版本都有分的配对。先对每题取平均，再汇总题间差值。
多题时对题目及其配对轮次重采样，给出近似 95% 区间和粗略灵敏度。单题只报告观察值。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "noise"))


def load_patch(args: argparse.Namespace, targets: list[str]) -> tuple[str, list[dict[str, Any]], str]:
    if args.diff:
        target = args.target or ("harness" if "harness" in targets else targets[0])
        return target, [{"op": "diff", "unified": Path(args.diff).read_text(encoding="utf-8")}], str(args.diff)
    data = json.loads(Path(args.patch).read_text(encoding="utf-8"))
    target = data.get("target") or args.target or targets[0]
    if not isinstance(data.get("ops"), list) or not data["ops"]:
        raise SystemExit("--patch 文件里要有非空的 ops 列表")
    return target, data["ops"], str(args.patch)


def analyse(base: dict[str, list[float | None]], cand: dict[str, list[float | None]], seed: int = 0,
            n_boot: int = 4000) -> dict[str, Any]:
    """List indices are repeat IDs. Missing verdicts retain their slot as None."""
    paired = {t: [(a, b) for a, b in zip(base[t], cand.get(t, [])) if a is not None and b is not None]
              for t in sorted(base)}
    tasks = [t for t in paired if paired[t]]
    counts = {t: len(v) for t, v in paired.items()}
    if not tasks:
        return {"tasks": 0, "paired_counts": counts, "verdict": "没有两个版本都能评分的题"}
    b = {t: fmean(a for a, _ in paired[t]) for t in tasks}
    c = {t: fmean(v for _, v in paired[t]) for t in tasks}
    deltas = [[v - a for a, v in paired[t]] for t in tasks]
    mean = fmean(fmean(ds) for ds in deltas)
    lo = hi = se = None
    if len(tasks) > 1:
        rng = random.Random(seed)
        boots = sorted(fmean(fmean(rng.choices(ds, k=len(ds))) for ds in rng.choices(deltas, k=len(tasks)))
                       for _ in range(n_boot))
        lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
        se = stdev(boots)
    eps = 1e-9
    verdict = "单题，仅报告观察差值"
    if lo is not None and hi is not None:
        verdict = "测出提升" if lo > eps else "测出变差" if hi < -eps else "测不出差别"
    return {"tasks": len(tasks), "baseline": fmean(b.values()), "patched": fmean(c.values()), "diff": mean,
            "ci95": [lo, hi] if lo is not None else None, "se": se,
            "min_detectable": 2.8 * se if se else None, "paired_counts": counts,
            "better": sorted(t for t in tasks if c[t] > b[t] + eps), "worse": sorted(t for t in tasks if c[t] < b[t] - eps),
            "same": sum(abs(c[t] - b[t]) <= eps for t in tasks),
            "baseline_flaky": sorted(t for t in tasks if len({round(x, 6) for x in base[t] if x is not None}) > 1),
            "verdict": verdict}


async def run(config: Path, args: argparse.Namespace) -> dict[str, Any]:
    from anyrsi import Patch
    from anyrsi_runtime import Runner, Wiring, load_config
    from noise import pool

    cfg = load_config(config)
    cfg.loop.max_unscorable = None  # 诊断：全部记下来，不中断
    async with Wiring(cfg) as w:
        r = Runner(w, cfg.loop)
        await r._load_targets()
        target, ops, source = load_patch(args, sorted(r.targets))
        if target not in r.targets:
            raise SystemExit(f"配置里没有可进化体 {target}（现有：{sorted(r.targets)}）")
        tasks = await pool(r, args.split, args.limit or (cfg.loop.holdout_limit if args.split == "holdout" else None))
        parent = {t: (await tgt.facade.snapshot()).id for t, tgt in r.targets.items()}
        tgt = r.targets[target]
        snap = await tgt.facade.apply(Patch(target=tgt.info.target, base_snapshot=parent[target], ops=ops,
                                            rationale=f"hand-written patch {source}"))
        patched = {**parent, target: snap.id}
        if snap.id == parent[target]:
            raise SystemExit("补丁没有改变任何东西")
        keys = [r._key(task) for task in tasks]
        scores: dict[str, dict[str, list[float | None]]] = {
            name: {key: [None] * args.repeats for key in keys} for name in ("baseline", "patched")}
        broken: dict[str, set[str]] = {"baseline": set(), "patched": set()}
        try:
            for i in range(args.repeats):
                order = [("baseline", parent), ("patched", patched)]
                for name, genome in (order if i % 2 == 0 else order[::-1]):  # 交替先后，抵消时间上的漂移
                    m, _ = await r.score(tasks, genome, "holdout")
                    for k, v in r.last_task_scores.items():
                        scores[name][k][i] = v
                    broken[name] |= set(r.last_unscorable)
                    print(f"run {i + 1}/{args.repeats} {name:8s}: {m['holdout']:.3f} on {int(m['n_holdout'])} scorable, "
                          f"{len(r.last_unscorable)} unscorable", flush=True)
        finally:
            await r.checkout(parent)  # 切回起点版本
    rep = analyse(scores["baseline"], scores["patched"], seed=args.seed)
    rep.update({"config": str(config), "patch": source, "target": target, "split": args.split, "repeats": args.repeats,
                "unscorable": {k: sorted(v) for k, v in broken.items()}, "per_task": scores})
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--diff", type=Path)
    src.add_argument("--patch", type=Path)
    ap.add_argument("--target", help="补丁改哪个可进化体（--diff 默认 harness）")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--split", default="holdout", choices=["holdout", "train", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("handpatch_report.json"))
    args = ap.parse_args(argv)
    if args.repeats < 1:
        ap.error("--repeats 必须为正数")
    rep = asyncio.run(run(args.config, args))
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    if not rep["tasks"]:
        print(rep["verdict"])
        return 1
    print(f"有效配对 {sum(rep['paired_counts'].values())}/{len(rep['paired_counts']) * args.repeats}，"
          f"覆盖 {rep['tasks']} 道题。")
    print(f"起点 {rep['baseline']:.3f} → 打补丁后 {rep['patched']:.3f}，差值 {rep['diff']:+.3f}")
    if rep["ci95"] is not None:
        print(f"近似 95% 区间 {rep['ci95'][0]:+.3f} ～ {rep['ci95'][1]:+.3f}")
    print(f"变好 {len(rep['better'])} 道 {rep['better'][:8]}；变差 {len(rep['worse'])} 道 {rep['worse'][:8]}；"
          f"不变 {rep['same']} 道")
    print(f"起点自己多遍不一致的题：{len(rep['baseline_flaky'])} 道 {rep['baseline_flaky'][:8]}")
    if rep["min_detectable"] is not None:
        print(f"这批题、这个遍数下，大约 {rep['min_detectable']:.3f} 以上的真实提升才测得出来")
    n_bad = len(set(rep["unscorable"]["baseline"]) | set(rep["unscorable"]["patched"]))
    if n_bad:
        print(f"判分器没跑起来的题 {n_bad} 道（缺失轮次已剔除）")
    print(f"结论：{rep['verdict']}。详细结果：{args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
