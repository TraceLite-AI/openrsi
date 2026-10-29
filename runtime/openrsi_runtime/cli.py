"""命令行。

    openrsi run  [openrsi.yaml] [--generations N]   跑自我改进循环（有谱系就接着跑）
    openrsi log  [openrsi.yaml]                     列出谱系，包括被拒绝的候选
    openrsi best [openrsi.yaml]                     当前最佳快照
    openrsi check [openrsi.yaml]                    只检查配置和组件能否接通
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from .config import load_config
from .loop import GenerationReport, Runner
from .wiring import Wiring


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _print_generation(r: GenerationReport) -> None:
    base = r.parent_metrics.get("holdout")
    if not r.candidates:
        print(f"{r.generation:>4}  {r.parent_id[:20]:<20} {_fmt(r.train_score):>6} {_fmt(base):>7}       -  {r.note}")
    for c in r.candidates:
        verdict = "接受" if c.accepted else "拒绝"
        cid = c.id or "-"
        extra = ""
        if "anchor" in c.metrics and "anchor" in r.parent_metrics:
            extra = f"  anchor {_fmt(r.parent_metrics['anchor'])}→{_fmt(c.metrics['anchor'])}"
        if r.note:
            extra += f"  [{r.note}]"
        print(f"{r.generation:>4}  {r.parent_id[:20]:<20} {_fmt(r.train_score):>6} {_fmt(base):>7} "
              f"{_fmt(c.metrics.get('holdout')):>7}  {verdict} {cid}  {c.reason}{extra}")


async def _run(path: str, generations: int | None) -> int:
    cfg = load_config(path)
    async with Wiring(cfg) as w:
        print(f"[{cfg.name}]")
        for slot, ref in w.refs.items():
            print(f"  {slot:<11} {ref}")
        print(f"\n{'gen':>4}  {'parent':<20} {'train':>6} {'base':>7} {'cand':>7}  result")
        runner = Runner(w, cfg.loop, on_generation=_print_generation, log=lambda s: print(f"  · {s}"))
        await runner.run(generations)
        best = await w.lineage.best(cfg.loop.metric)
        if best:
            print(f"\n最佳快照 {best.id}  {cfg.loop.metric}={_fmt(best.metrics.get(cfg.loop.metric))}")
    return 0


async def _log(path: str) -> int:
    cfg = load_config(path)
    async with Wiring(cfg, slots=["lineage"]) as w:
        for e in await w.lineage.log():
            parent = e.parent_ids[0] if e.parent_ids else "-"
            reason = e.decision.reason if e.decision else ""
            anchor = f" anchor={_fmt(e.metrics['anchor'])}" if "anchor" in e.metrics else ""
            print(f"{e.generation:>4}  {e.status:<8} {e.id:<24} ← {parent:<24} "
                  f"holdout={_fmt(e.metrics.get('holdout'))}{anchor}  {reason}")
    return 0


async def _best(path: str) -> int:
    cfg = load_config(path)
    async with Wiring(cfg, slots=["lineage"]) as w:
        best = await w.lineage.best(cfg.loop.metric)
        if not best:
            print("谱系为空")
            return 1
        print(best.model_dump_json(indent=2))
    return 0


async def _check(path: str) -> int:
    cfg = load_config(path)
    async with Wiring(cfg) as w:
        for slot, ref in w.refs.items():
            print(f"  ok  {slot:<11} {ref}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openrsi")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "log", "best", "check"):
        p = sub.add_parser(name)
        p.add_argument("config", nargs="?", default="openrsi.yaml")
        if name == "run":
            p.add_argument("--generations", type=int)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_run(args.config, args.generations))
    return asyncio.run({"log": _log, "best": _best, "check": _check}[args.cmd](args.config))


if __name__ == "__main__":
    sys.exit(main())
