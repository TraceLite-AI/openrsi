"""跑前体检：确认判分器真的在判分，再开始花钱跑进化。

对抽到的每道题跑两次 Harbor：
  oracle   题目自带的标准答案（Harbor 的 oracle agent）→ 应该通过，而且测试真的跑了
  nop      什么都不做 → 应该不通过，而且测试真的跑了（不是因为判分器没起来才得 0）
任何一项不对就列出来、退出码 1。上一次 Terminal-Bench 实验里 1061 次评测有 643 次判分器没跑到测试，
这一步能在第一次调用模型之前发现它。

    python benchmarks/terminal_preflight/preflight.py examples/rsihub_weights/anyrsi.step1.yaml --per-split 5

配置里的 agent 必须是 anyrsi_rsihub:HarborAgent，judge 用 HarborRewardJudge（配置里没有就用默认的）。
oracle / nop 的 Harbor 参数默认是 --agent oracle / --agent nop，你的 Harbor 版本不一样时用 --oracle-args / --nop-args 改。
封存题默认也体检（只验证判分器，不让任何 agent 学到东西），--splits 可以改。
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shlex
import sys
from pathlib import Path
from typing import Any


async def _call(x: Any) -> Any:
    return await x if asyncio.iscoroutine(x) else x


async def check(config: Path, splits: list[str], per_split: int, oracle_args: list[str], nop_args: list[str] | None,
                ) -> list[dict[str, Any]]:
    from anyrsi_rsihub import HarborAgent, HarborRewardJudge
    from anyrsi_runtime import Wiring, load_config
    from anyrsi_runtime.scoring import unscorable

    cfg = load_config(config)
    rows = []
    async with Wiring(cfg) as w:
        unwrap = lambda c: getattr(c, "_obj", c)  # noqa: E731  进程内组件外面有一层适配
        agent, judge, env = unwrap(w.components["agent"]), unwrap(w.components.get("judge")), unwrap(w.components["env"])
        if not isinstance(agent, HarborAgent):
            raise SystemExit("体检只支持 anyrsi_rsihub:HarborAgent（配置里的 agent 不是）")
        if not isinstance(judge, HarborRewardJudge):
            judge = HarborRewardJudge()
        runs = [("oracle", oracle_args, True)] + ([("nop", nop_args, False)] if nop_args is not None else [])
        for split in splits:
            for task in (await _call(env.tasks(split, 0, per_split)))[:per_split]:
                for label, args, should_pass in runs:
                    probe = copy.copy(agent)
                    probe.agent_args = list(args)
                    probe.extra_args = []  # 候选代码、模型这些参数是给被进化的 agent 的，体检用不到
                    traj = await asyncio.to_thread(probe.run, task, [], f"preflight-{label}")
                    v = judge.evaluate(task, traj, [])
                    started = (v.collected[0].content.get("tests_started") if v.collected else None)
                    broken = unscorable(v)
                    problems = []
                    if broken:
                        problems.append(f"判分器没跑到测试：{broken[:160]}")
                    elif started is None:
                        problems.append("看不到判分器输出（verifier/ 目录不存在），无法确认测试跑了")
                    if not broken and v.passed != should_pass:
                        problems.append(f"{'标准答案没通过' if should_pass else '什么都不做却通过了'}："
                                        f"{(v.notes or [''])[0][:160]}")
                    if traj.metadata.get("error"):
                        problems.append(f"Harbor 没跑起来：{traj.metadata['error'][:160]}")
                    rows.append({"task": task.id, "split": split, "probe": label, "passed": v.passed,
                                 "tests_started": started, "ok": not problems, "problems": problems})
                    print(f"{'OK ' if not problems else 'BAD'} {split:7s} {task.id:40s} {label:6s} "
                          f"{'; '.join(problems)}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "holdout", "final"])
    ap.add_argument("--per-split", type=int, default=5, help="每个划分抽几道题")
    ap.add_argument("--oracle-args", default="--agent oracle")
    ap.add_argument("--nop-args", default="--agent nop", help="空字符串表示不跑 nop")
    ap.add_argument("--out", type=Path, default=Path("preflight_report.json"))
    args = ap.parse_args(argv)
    nop = shlex.split(args.nop_args) if args.nop_args.strip() else None
    rows = asyncio.run(check(args.config, args.splits, args.per_split, shlex.split(args.oracle_args), nop))
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = [r for r in rows if not r["ok"]]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} 项正常。详细结果：{args.out}")
    if bad:
        print("判分环境有问题，先修好再跑进化（不要把这些结果当成 agent 的成绩）。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
