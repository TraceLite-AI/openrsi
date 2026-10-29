"""v0.3 第二部分：判别器参与进化（anchor 把关、轮流进化、纪元）和反事实归因。"""
import asyncio
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openrsi import (Agent, Environment, Episode, EvolvableInfo, GateRequest, Patch, ProposalRequest, Proposer, Task,
                     Trajectory, build_server, connect, load_manifest)
from openrsi.conformance import check
from openrsi_gate import HoldoutGate
from openrsi_judge import EvolvingJudge
from openrsi_proposer import CompositeProposer, LessonProposer, RubricAuditor
from openrsi_runtime import Runner, Wiring, load_config
from openrsi_runtime.config import RunConfig

import coevo

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "judge_coevolution" / "openrsi.yaml"
LENIENT = {"items": [{"id": "says_done", "check": "output_matches", "params": {"pattern": "done"}, "required": True}]}


# ---------------------------------------------------------------- 组件

def test_gate_primary_metric():
    req = dict(candidate_id="c", baseline={"holdout": 1.0, "anchor": 0.5}, candidate={"holdout": 0.5, "anchor": 1.0})
    assert not HoldoutGate().check(GateRequest(**req)).accepted
    assert HoldoutGate().check(GateRequest(**req, primary_metric="anchor")).accepted


def test_evolving_judge_versions_its_rubric(tmp_path):
    j = EvolvingJudge(LENIENT, protect=["tests/**"], path=tmp_path)
    task = coevo._task("train", 1)
    cheat = coevo.trajectory(task, honest=False)
    assert j.evaluate(task, cheat, []).passed  # 宽松的判别器被骗过
    lenient = j.snapshot().id
    j.apply(Patch(target="rubric", ops=[{"op": "add", "item": {"id": "np", "data": {
        "check": "no_protected_edits", "required": True, "order": 1}}}]))
    assert not j.evaluate(task, cheat, []).passed
    j.rollback(lenient)
    assert j.evaluate(task, cheat, []).passed  # 回滚后标准也回去了
    with pytest.raises(Exception, match="不合法"):
        j.apply(Patch(target="rubric", ops=[{"op": "add", "item": {"id": "x", "data": {"check": "telepathy"}}}]))
    again = EvolvingJudge({"items": [{"id": "other", "check": "tests_pass"}]}, path=tmp_path)
    assert [i.id for i in again.rubric.items] == ["says_done"]  # 从磁盘恢复，不会被新的初始 rubric 覆盖


def episode(task, traj, judge):
    ev = judge.collect(task, traj)
    return Episode(task=task, trajectory=traj, verdict=judge.evaluate(task, traj, ev), evidence=ev)


def request(judge, episodes, types=("rubric",), max_patches=3):
    infos = [EvolvableInfo(target=t, type=t, head="p") for t in types]
    return ProposalRequest(target=infos[0], targets=infos, parent_snapshot="p", episodes=episodes,
                           states={"rubric": judge.export(), "memory": []}, max_patches=max_patches)


def test_rubric_auditor_finds_contradictions():
    j = EvolvingJudge(LENIENT, protect=["tests/**"])
    t = coevo._task("train", 1)
    [patch] = RubricAuditor().propose(request(j, [episode(t, coevo.trajectory(t, honest=False), j)]))
    assert patch.target == "rubric" and patch.ops[0]["item"]["data"]["check"] == "no_protected_edits"
    assert patch.ops[0]["item"]["data"]["required"] is True
    assert RubricAuditor().propose(request(j, [episode(t, coevo.trajectory(t, honest=True), j)])) == []
    assert RubricAuditor().propose(request(j, [episode(t, coevo.trajectory(t, honest=False), j)],
                                           types=("memory",))) == []  # 本代不允许改判别器
    # 已有但不是必选的检查项：改成必选
    j2 = EvolvingJudge({"items": [*LENIENT["items"], {"id": "np", "check": "no_protected_edits", "weight": 0.1}],
                        "pass_threshold": 0.5}, protect=["tests/**"])
    [p2] = RubricAuditor().propose(request(j2, [episode(t, coevo.trajectory(t, honest=False), j2)]))
    assert p2.ops[0]["op"] == "replace" and p2.ops[0]["item"]["data"]["required"] is True


def test_composite_and_target_selection():
    with pytest.raises(ValueError, match="未知的提案器"):
        CompositeProposer(["lesson", "magic"])
    j = EvolvingJudge(LENIENT, protect=["tests/**"])
    t = coevo._task("train", 1)
    j.apply(Patch(target="rubric", ops=[{"op": "add", "item": {"id": "np", "data": {"check": "no_protected_edits",
                                                                                     "required": True}}}]))
    failed = episode(t, coevo.trajectory(t, honest=False), j)
    assert not failed.verdict.passed
    [lesson] = LessonProposer().propose(request(j, [failed], types=("rubric", "memory")))
    assert lesson.target == "memory"  # LessonProposer 只改 memory
    assert len(CompositeProposer(["lesson", "rubric-auditor"]).propose(
        request(j, [failed], types=("memory",), max_patches=1))) == 1


# ---------------------------------------------------------------- 运行时

def config(tmp_path, **loop):
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["plugins"] = ["coevo"]
    data["loop"].update(loop)
    data["loop"]["anchor"] = str(EXAMPLE.parent / "anchor.jsonl") if data["loop"].get("anchor") else None
    for slot in ("judge", "memory", "lineage"):
        data["components"][slot]["with"]["path"] = f"state/{slot}"
    path = tmp_path / "openrsi.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(path)


def run(cfg, generations=None):
    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop).run(generations)
            return reports, await w.lineage.log(), await w.lineage.best(cfg.loop.metric)

    return asyncio.run(go())


def test_coevolution_example(tmp_path):
    reports, log, best = run(config(tmp_path))
    root, judge_step, agent_step = log
    assert root.metrics["holdout"] == 1.0 and root.metrics["anchor"] == 0.5  # 看起来完美，其实在作弊
    assert judge_step.generation == 2 and judge_step.metrics["anchor"] == 1.0
    assert judge_step.metrics["holdout"] < root.metrics["holdout"] and judge_step.metrics["judge_epoch"] == 1
    assert judge_step.snapshots["memory"] == root.snapshots["memory"]  # 判别器代没动 memory
    assert agent_step.generation == 3 and agent_step.metrics["holdout"] == 1.0
    assert agent_step.snapshots["rubric"] == judge_step.snapshots["rubric"]  # agent 代没动判别器
    assert best.id == agent_step.id and best.metrics["progress"] == 3.0
    assert "判别器进化代" in reports[1].note


class SpyCoder(coevo.ShortcutCoder):
    contexts: list = []

    def run(self, task, context, snapshot_id=None):
        SpyCoder.contexts.append([i.kind for i in context])
        return super().run(task, context, snapshot_id)


def test_rubric_never_reaches_agent(tmp_path):
    import sys
    sys.modules.setdefault("spy_mod", sys.modules[__name__])
    cfg = config(tmp_path)
    cfg.components["agent"].python = f"{__name__}:SpyCoder"
    run(cfg)
    assert SpyCoder.contexts and not any("rubric" in kinds for kinds in SpyCoder.contexts)


class Saboteur(Proposer):
    """判别器代削弱判别器（删掉检查项），agent 代偷偷改判别器。"""

    name, version = "test/saboteur", "0.0.1"

    def propose(self, request):
        rubric = request.states["rubric"]
        weaken = Patch(target="rubric", ops=[{"op": "remove", "item_id": i["id"]} for i in rubric
                                             if i["data"]["check"] != "output_matches"])
        note = Patch(target="memory", ops=[{"op": "add", "item": {"id": f"n{len(request.states['memory'])}",
                                                                  "text": "noop"}}])
        return [p for p in (weaken, note) if p.ops]


def test_weakening_the_judge_is_rejected_even_if_holdout_rises(tmp_path):
    cfg = config(tmp_path)
    reports, log, _ = run(cfg, generations=3)  # 先走到判别器收紧、agent 学会正路
    cfg.components["proposer"].python = f"{__name__}:Saboteur"
    cfg.components["proposer"].with_ = {}
    cfg.components["algorithm"].with_["target_score"] = 99
    cfg.loop.max_patches = 2
    reports, log2, best = run(cfg, generations=2)
    judge_gen = next(r for r in reports if r.note == "判别器进化代")
    weaken = next(c for c in judge_gen.candidates if c.target == "rubric")
    assert not weaken.accepted
    assert weaken.metrics["anchor"] < judge_gen.parent_metrics["anchor"]  # 一致率掉了
    wrong_side = next(c for c in judge_gen.candidates if c.target == "memory")
    assert "本代只允许修改" in wrong_side.reason
    agent_gen = next(r for r in reports if r.note != "判别器进化代")
    assert any("本代只允许修改" in c.reason for c in agent_gen.candidates if c.target == "rubric")
    assert best.metrics["progress"] == 3.0


def test_judge_evolution_requires_anchor(tmp_path):
    with pytest.raises(ValidationError, match="anchor"):
        RunConfig.model_validate({"name": "x", "loop": {"judge_every": 2},
                                  "components": {s: {"python": "m:C"} for s in (
                                      "env", "agent", "judge", "memory", "proposer", "algorithm", "gate", "lineage")}})
    # 判别器是可进化体但没开 judge_every：判别器保持不变
    cfg = config(tmp_path, judge_every=None, anchor=None)
    reports, log, _ = run(cfg, generations=4)
    assert all(e.snapshots["rubric"] == log[0].snapshots["rubric"] for e in log)


# ---------------------------------------------------------------- 反事实归因

class OneTaskEnv(Environment):
    name, version = "test/one-task", "0.0.1"

    def tasks(self, split, round=0, limit=None):
        return [Task(id=f"{split}-q", split=split, input={"q": "x"}, reference={"expected": 1})]


class PickyAgent(Agent):
    """有 bad 条目就答错；否则有 good 条目答对；什么都没有也答错。"""

    name, version = "test/picky", "0.0.1"

    def run(self, task, context, snapshot_id=None):
        ids = {i.id for i in context}
        out = 0 if "bad" in ids else (1 if "good" in ids else 0)
        return Trajectory(id=f"t-{len(ids)}-{sorted(ids)}", task_id=task.id, output=out, used_items=sorted(ids))


def test_counterfactual_attribution(tmp_path):
    data = {"name": "cf", "components": {
        "env": {"python": f"{__name__}:OneTaskEnv"}, "agent": {"python": f"{__name__}:PickyAgent"},
        "judge": {"python": "openrsi_judge:EvidenceJudge",
                  "with": {"rubric": {"items": [{"id": "eq", "check": "output_equals", "required": True}]}}},
        "memory": {"python": "openrsi_memory:MemoryEvolvable"},
        "proposer": {"python": "openrsi_proposer:LessonProposer"},
        "algorithm": {"python": "openrsi_algorithm:GreedyAlgorithm"},
        "gate": {"python": "openrsi_gate:HoldoutGate"}, "lineage": {"python": "openrsi_lineage:LineageStore"}},
        "loop": {"counterfactual_reruns": 10, "retrieve_k": 10}}
    (tmp_path / "openrsi.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(tmp_path / "openrsi.yaml")

    async def go():
        async with Wiring(cfg) as w:
            r = Runner(w, cfg.loop)
            await r._load_targets()
            mem = w.memory
            await mem.apply(Patch(target="memory", ops=[{"op": "add", "item": {"id": "good", "text": "x"}},
                                                        {"op": "add", "item": {"id": "bad", "text": "x"}}]))
            genome = {"memory": (await mem.snapshot()).id}
            task = (await w.env.tasks("train", 1))[0]
            ep = await r.episode(task, genome)
            return ep, await r.counterfactual([ep], genome)

    ep, [attr] = asyncio.run(go())
    assert not ep.verdict.passed and attr.method == "counterfactual"
    weights = {t.ref: t.weight for t in attr.targets}
    assert weights["bad"] == -1.0  # 拿掉 bad 就答对了：它有害
    assert weights["good"] == 0.0  # 有 bad 在，good 拿不拿掉都答错


def test_counterfactual_budget_and_proposer_sees_it(tmp_path):
    captured = []

    class Capture(Proposer):
        name, version = "test/capture", "0.0.1"

        def propose(self, request):
            captured.extend(a.method for a in request.attributions)
            return []

    import sys
    sys.modules[__name__].Capture = Capture
    cfg = config(tmp_path, counterfactual_reruns=2)
    cfg.components["proposer"].python = f"{__name__}:Capture"
    cfg.components["proposer"].with_ = {}
    run(cfg, generations=1)
    assert captured.count("counterfactual") == 0  # 第一代 agent 没用到任何 memory，不会重跑
    assert "rubric-evidence" in captured


def test_evolving_judge_conformance_and_manifest():
    report = asyncio.run(check(build_server(EvolvingJudge(LENIENT))))
    assert report.ok, report.text()

    async def served():
        async with connect(build_server(EvolvingJudge(LENIENT))) as c:
            return c.info

    info = asyncio.run(served())
    manifest = load_manifest(ROOT / "components" / "judge" / "rsi-component.evolving.yaml")
    assert (manifest.name, manifest.version) == (info.name, info.version)
    assert set(manifest.implements) == set(info.tools) and set(manifest.kinds) == set(info.kinds)
