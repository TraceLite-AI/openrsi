"""无法评分的题（判分器没跑起来）：不算分、不参与准入比较、不交给提案器；太多就停下报错。"""
import asyncio
import sys
from pathlib import Path

import pytest

from anyrsi import RubricResult, Verdict
from anyrsi_runtime.scoring import UnscorableError, unscorable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))
import group_toy as gt  # noqa: E402


def test_unscorable_rule():
    ok = Verdict(passed=False, score=0.0, judge_ref="j", rubric=[RubricResult(id="a", result="fail", required=True)])
    assert unscorable(ok) is None
    soft = Verdict(passed=False, score=0.0, judge_ref="j", rubric=[RubricResult(id="a", result="error")])
    assert unscorable(soft) is None  # 不是必需项，只当作普通信息
    bad = Verdict(passed=False, score=0.0, judge_ref="j",
                  rubric=[RubricResult(id="started", result="error", required=True, note="no python")])
    assert unscorable(bad) == "no python"


def run(tmp_path, broken, budget=1500, arm="A", max_unscorable=None, **cfg_kw):
    """玩具循环：题号在 broken 里的题判分器"坏了"。返回 runner、谱系、准入请求、提案器看到的练习题。"""
    import yaml

    from anyrsi_runtime import Runner, Wiring, load_config

    path = gt.write_config(tmp_path / arm, arm, 0, **cfg_kw)
    data = yaml.safe_load(path.read_text())
    data["loop"]["max_unscorable"] = max_unscorable
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)
    gate_calls, seen = [], []

    async def go():
        async with Wiring(cfg) as w:
            judge, gate, proposer = w.components["judge"], w.components["gate"], w.components["proposer"]

            class BrokenJudge:
                kinds = judge.kinds

                async def evaluate(self, task, traj, evidence=None):
                    if task.id in broken:
                        return Verdict(passed=False, score=0.0, judge_ref="broken", rubric=[
                            RubricResult(id="verifier_started", result="error", required=True, note="read-only")])
                    return await judge.evaluate(task, traj, evidence)

                def __getattr__(self, n):
                    return getattr(judge, n)

            class GateSpy:
                kinds = gate.kinds

                async def check(self, req):
                    gate_calls.append(req)
                    return await gate.check(req)

                def __getattr__(self, n):
                    return getattr(gate, n)

            class ProposerSpy:
                kinds = proposer.kinds

                async def propose(self, req):
                    seen.extend(e.task.id for e in req.episodes)
                    return await proposer.propose(req)

                def __getattr__(self, n):
                    return getattr(proposer, n)

            w.components.update({"judge": BrokenJudge(), "gate": GateSpy(), "proposer": ProposerSpy(),
                                 "agent": gt._Counting(w.components["agent"], budget)})
            runner = Runner(w, cfg.loop)
            try:
                await runner.run()
            except gt.BudgetExhausted:
                pass
            return runner, await w.lineage.log()

    runner, log = asyncio.run(go())
    return runner, log, gate_calls, seen


def broken_ids(n_holdout=10, n_train=3):
    env = gt.ToyEnv(seed=0)
    hold = [t.id for t in env.tasks("holdout")][:n_holdout]
    train = [t.id for r in range(40) for t in env.tasks("train", r)][: n_train * 40]
    return set(hold), set(train[::13])


@pytest.mark.parametrize("arm", ["A", "best_of_n"])
def test_unscorable_tasks_are_left_out_of_scores_gates_and_proposals(tmp_path, arm):
    hold, train = broken_ids()
    runner, log, gate_calls, seen = run(tmp_path, hold | train, arm=arm, max_unscorable=0.5)
    root = next(e for e in log if e.status == "root")
    assert root.metrics["n_holdout"] == 50 and root.metrics["n_holdout_unscorable"] == 10
    assert not set(root.task_scores) & hold
    assert gate_calls and all(not (set(r.baseline_tasks) | set(r.candidate_tasks)) & hold for r in gate_calls)
    assert all(set(r.baseline_tasks) == set(r.candidate_tasks) for r in gate_calls)
    assert seen and not set(seen) & train  # 提案器看不到判分器坏掉的练习题


def test_too_many_unscorable_stops_the_run(tmp_path):
    hold, _ = broken_ids(n_holdout=20)
    with pytest.raises(UnscorableError, match="max_unscorable"):
        run(tmp_path, hold, max_unscorable=0.2)
    runner, log, *_ = run(tmp_path / "off", hold, max_unscorable=None)  # 不设上限：只剔除
    assert next(e for e in log if e.status == "root").metrics["n_holdout"] == 40


def test_gate_compares_only_tasks_scored_on_both_sides(tmp_path):
    """父版本和候选能评分的题不一样时，只在两边都有分的题上比，均值按这些题重算。"""
    from types import SimpleNamespace

    from anyrsi_runtime.loop import Runner

    fake = SimpleNamespace(_subset=Runner._subset)
    base_t, cand_t = {"a": 1.0, "b": 0.0, "c": 0.0}, {"a": 1.0, "b": 1.0, "d": 1.0}
    bm, bt, cm, ct = Runner._paired(fake, {"holdout": 1 / 3, "x": 7}, base_t, {"holdout": 1.0}, cand_t)
    assert set(bt) == set(ct) == {"a", "b"} and bm["holdout"] == 0.5 and cm["holdout"] == 1.0 and bm["x"] == 7
    same = Runner._paired(fake, {"holdout": 0.5}, {"a": 1.0}, {"holdout": 0.0}, {"a": 0.0})
    assert same == ({"holdout": 0.5}, {"a": 1.0}, {"holdout": 0.0}, {"a": 0.0})
