"""组件化的代价：同一个组件在进程内、子进程（stdio）、HTTP 三种方式下，每次调用多花多少时间；
以及 secret_ops 示例整轮运行在全部进程内和全部子进程两种方式下的耗时。

用法：python benchmarks/composition/overhead.py [--calls 300]
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

import test_composition as tc  # noqa: E402

from openrsi import GateRequest, Task, Trajectory, connect  # noqa: E402
from openrsi_gate import HoldoutGate  # noqa: E402
from openrsi_judge import EvidenceJudge  # noqa: E402


def stats(samples: list[float]) -> dict:
    s = sorted(samples)
    return {"median_ms": round(1000 * statistics.median(s), 3), "p95_ms": round(1000 * s[int(0.95 * (len(s) - 1))], 3)}


async def bench_remote(target, calls: int) -> dict:
    req = GateRequest(candidate_id="c", baseline={"holdout": 0.5}, candidate={"holdout": 0.6})
    task = Task(id="t", input="x", split="train", reference={"expected": "a"})
    traj = Trajectory(id="tr", task_id="t", output="a", steps=[])
    out = {}
    async with connect(target["gate"], env=target["env"]) as gate, \
            connect(target["judge"], env=target["env"]) as judge:
        for name, fn in (("gate.check", lambda: gate.gate.check(req)),
                         ("judge.collect+evaluate", lambda: _judge(judge, task, traj))):
            await fn()
            samples = []
            for _ in range(calls):
                t0 = time.perf_counter()
                await fn()
                samples.append(time.perf_counter() - t0)
            out[name] = stats(samples)
    return out


async def _judge(rc, task, traj):
    ev = await rc.evidence.collect(task, traj)
    return await rc.judge.evaluate(task, traj, ev)


def bench_local(calls: int) -> dict:
    gate, judge = HoldoutGate(), EvidenceJudge(tc.EQ)
    req = GateRequest(candidate_id="c", baseline={"holdout": 0.5}, candidate={"holdout": 0.6})
    task = Task(id="t", input="x", split="train", reference={"expected": "a"})
    traj = Trajectory(id="tr", task_id="t", output="a", steps=[])
    out = {}
    for name, fn in (("gate.check", lambda: gate.check(req)),
                     ("judge.collect+evaluate", lambda: judge.evaluate(task, traj, judge.collect(task, traj)))):
        samples = []
        for _ in range(calls):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        out[name] = stats(samples)
    return out


def loop_time(remote: bool) -> dict:
    base = Path(tempfile.mkdtemp())
    comps = {s: (tc.stdio_spec(*spec, base) if remote else tc.python_spec(*spec)) for s, spec in tc.SLOTS.items()}
    t0 = time.perf_counter()
    reports, log = tc.run(tc.write_config(base, comps))
    return {"seconds": round(time.perf_counter() - t0, 2), "generations": len(reports),
            "best_holdout": max(e.metrics["holdout"] for e in log)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=300)
    args = ap.parse_args()
    env = {**os.environ, "PYTHONPATH": tc.PYTHONPATH}
    launch = [sys.executable, "-m", "openrsi.launch"]
    stdio = {"gate": [*launch, "openrsi_gate:HoldoutGate"],
             "judge": [*launch, "openrsi_judge:EvidenceJudge", "--with", json.dumps({"rubric": tc.EQ})], "env": env}
    base = Path(tempfile.mkdtemp())
    procs = []
    try:
        gp, gurl = tc.start_http("openrsi_gate:HoldoutGate", {}, base)
        jp, jurl = tc.start_http("openrsi_judge:EvidenceJudge", {"rubric": tc.EQ}, base)
        procs = [gp, jp]
        result = {
            "python": sys.version.split()[0], "calls": args.calls,
            "per_call": {"in_process": bench_local(args.calls),
                         "stdio": asyncio.run(bench_remote(stdio, args.calls)),
                         "http": asyncio.run(bench_remote({"gate": gurl, "judge": jurl, "env": None}, args.calls))},
            "secret_ops_loop": {"in_process": loop_time(False), "all_stdio": loop_time(True)},
        }
    finally:
        for p in procs:
            p.terminate()
    out = ROOT / "benchmarks" / "composition" / "out"
    out.mkdir(exist_ok=True)
    (out / "overhead.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
