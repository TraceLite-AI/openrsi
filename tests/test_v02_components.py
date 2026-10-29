"""v0.2 新组件：归因、提案器、进化算法、准入、谱系。"""
import asyncio
from pathlib import Path

import pytest

from openrsi import (Episode, Evidence, EvolvableInfo, GateRequest, Item, LineageEntry, ProposalRequest, RubricResult,
                     Task, TellRequest, Trajectory, Verdict, build_server, connect, load_manifest)
from openrsi.conformance import check
from openrsi_algorithm import ArchiveAlgorithm, GreedyAlgorithm
from openrsi_attribution import RubricEvidenceAttributor, item_scores
from openrsi_gate import HoldoutGate, two_proportion_z
from openrsi_lineage import LineageStore
from openrsi_proposer import ClaudeProposer, LessonProposer, ProposedItem, ProposedPatch

ROOT = Path(__file__).resolve().parents[1]


def episode(tid="t1", passed=False, used=(), tags=("x",), note="wrong answer", score=None):
    step_ev = Evidence.make("tool_result", {"content": "42"}, f"trajectory:traj-{tid}:step:s1", "test")
    out_ev = Evidence.make("output", {"value": 42}, f"trajectory:traj-{tid}:output", "test")
    rubric = [RubricResult(id="correct", result="pass" if passed else "fail", evidence_ids=[step_ev.id, out_ev.id],
                           required=True, note=None if passed else note)]
    v = Verdict(score=score if score is not None else (1.0 if passed else 0.0), passed=passed, rubric=rubric,
                judge_ref="test:0", evidence_ids=[step_ev.id, out_ev.id])
    return Episode(task=Task(id=tid, tags=list(tags)), verdict=v, evidence=[step_ev, out_ev],
                   trajectory=Trajectory(id=f"traj-{tid}", task_id=tid, used_items=list(used)))


# ---------------------------------------------------------------- 归因

def test_attribution_blames_rubric_and_step():
    [a] = RubricEvidenceAttributor().explain([episode()])
    by_kind = {t.kind: t for t in a.targets}
    assert by_kind["rubric"].ref == "correct" and by_kind["rubric"].weight == -1.0
    assert by_kind["step"].ref == "s1" and by_kind["step"].weight < 0  # 只有来自轨迹步骤的证据会追到 step
    assert by_kind["component"].ref == "memory:missing"  # 失败且没用 memory
    assert "wrong answer" in a.rationale


def test_attribution_items_credit_and_blame():
    attrs = RubricEvidenceAttributor().explain([
        episode("a", passed=True, used=["good"]), episode("b", passed=False, used=["bad"]),
        episode("c", passed=False, used=["bad", "good"]),
    ])
    scores = item_scores(attrs)
    assert scores["good"][0] > 0 > scores["bad"][0]
    assert scores["bad"][1] == 2


# ---------------------------------------------------------------- 提案器

def req(*eps, state=None, attributions=None):
    return ProposalRequest(target=EvolvableInfo(target="memory", type="memory", head="p"), parent_snapshot="p",
                           episodes=list(eps), state=state or [], attributions=attributions or [])


def test_lesson_proposer_writes_lessons_and_prunes():
    eps = [episode("a", used=["harmful"], note="off by one"), episode("b", used=["harmful"], note="off by one")]
    state = [Item(id="harmful", text="always add one", tags=["x"]).model_dump(mode="json")]
    [patch] = LessonProposer().propose(req(*eps, state=state, attributions=RubricEvidenceAttributor().explain(eps)))
    assert patch.base_snapshot == "p" and patch.target == "memory"
    ops = {o["op"]: o for o in patch.ops}
    assert ops["remove"]["item_id"] == "harmful"
    assert ops["add"]["item"]["id"] == "lesson:x" and "off by one" in ops["add"]["item"]["text"]
    assert LessonProposer().propose(req(episode(passed=True))) == []  # 全部通过就不提


class FakeClaude:
    def __init__(self, parsed, stop="end_turn"):
        self.kwargs = None
        outer = self

        class Msgs:
            def parse(self, **kw):
                outer.kwargs = kw
                return type("R", (), {"stop_reason": stop, "parsed_output": parsed})()

        self.beta = type("B", (), {"messages": Msgs()})()


def test_claude_proposer_normalizes_ops_and_hides_reference():
    parsed = ProposedPatch(rationale="r", ops=[
        ProposedItem(op="add", id="existing", text="updated"),  # 已存在 → 应改成 replace
        ProposedItem(op="replace", id="new", text="fresh"),     # 不存在 → 应改成 add
        ProposedItem(op="remove", id="ghost"),                  # 不存在 → 丢弃
    ])
    fake = FakeClaude(parsed)
    ep = episode()
    ep.task.reference = {"expected": "SECRET-ANSWER"}
    state = [Item(id="existing", text="old").model_dump(mode="json")]
    [patch] = ClaudeProposer(client=fake).propose(req(ep, state=state))
    assert [(o["op"], o["item"]["id"]) for o in patch.ops] == [("replace", "existing"), ("add", "new")]
    assert fake.kwargs["model"] == "claude-opus-5-5" and fake.kwargs["fallbacks"] == "default"
    assert "SECRET-ANSWER" not in fake.kwargs["messages"][0]["content"]
    assert ClaudeProposer(client=FakeClaude(None, stop="refusal")).propose(req(episode())) == []


# ---------------------------------------------------------------- 进化算法

def tell(alg, gen, parent, cand, ok, score):
    return alg.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=cand, accepted=ok,
                                metrics={"holdout": score}))


def test_greedy_tracks_best_and_stops_on_patience():
    alg = GreedyAlgorithm(patience=2, max_generations=50)
    assert alg.ask().done  # 没登记根之前不能开始
    tell(alg, 0, None, "root", True, 0.1)
    assert alg.ask().parent_id == "root"
    tell(alg, 1, "root", "a", True, 0.5)
    tell(alg, 2, "a", "b", False, 0.9)  # 被拒绝的不进档案
    assert alg.ask().parent_id == "a"
    tell(alg, 2, "a", "c", False, 0.2)  # 同一代多个候选只算一代
    assert not alg.ask().done
    tell(alg, 3, "a", None, False, 0.0)
    ask = alg.ask()
    assert ask.done and "没有提升" in ask.reason


def test_target_and_generation_limits():
    alg = GreedyAlgorithm(target_score=0.8)
    tell(alg, 0, None, "root", True, 0.9)
    assert alg.ask().done
    alg = GreedyAlgorithm(max_generations=1)
    tell(alg, 0, None, "root", True, 0.1)
    tell(alg, 1, "root", "a", True, 0.2)
    assert alg.ask().done


def test_archive_samples_only_accepted_and_is_seeded():
    def picks(seed):
        alg = ArchiveAlgorithm(seed=seed, temperature=1.0, max_generations=100, patience=100)
        tell(alg, 0, None, "root", True, 0.1)
        tell(alg, 1, "root", "a", True, 0.5)
        tell(alg, 2, "a", "rej", False, 0.99)
        return [alg.ask().parent_id for _ in range(30)]

    assert picks(1) == picks(1)
    assert set(picks(1)) <= {"root", "a"} and len(set(picks(1))) == 2  # 温度高时两者都会被选到


# ---------------------------------------------------------------- 准入

def gate_req(b, c, **extra):
    return GateRequest(candidate_id="c", baseline={"holdout": b, **extra.get("base", {})},
                       candidate={"holdout": c, **extra.get("cand", {})})


def test_gate_basic_and_min_gain():
    assert HoldoutGate().check(gate_req(0.5, 0.6)).accepted
    assert not HoldoutGate().check(gate_req(0.5, 0.5)).accepted  # 持平不算提升
    assert not HoldoutGate(min_gain=0.2).check(gate_req(0.5, 0.6)).accepted
    d = HoldoutGate().check(GateRequest(candidate_id="c", baseline={}, candidate={"holdout": 1.0}))
    assert not d.accepted and "缺少指标" in d.checks[0].detail


def test_gate_significance_anchor_cost():
    small = gate_req(0.5, 0.6, base={"n_holdout": 10}, cand={"n_holdout": 10})
    large = gate_req(0.5, 0.6, base={"n_holdout": 2000}, cand={"n_holdout": 2000})
    g = HoldoutGate(significance_z=1.64)
    assert not g.check(small).accepted and g.check(large).accepted  # 同样提升，样本少就不算数
    assert not HoldoutGate().check(gate_req(0.5, 0.9, base={"anchor": 0.9}, cand={"anchor": 0.7})).accepted
    assert not HoldoutGate(max_cost=1.0).check(gate_req(0.5, 0.9, cand={"cost": 2.0})).accepted
    assert two_proportion_z(0.5, 100, 0.5, 100) == 0


# ---------------------------------------------------------------- 谱系

def test_lineage_persistence_and_best(tmp_path):
    path = tmp_path / "l.jsonl"
    s = LineageStore(path)
    s.record(LineageEntry(id="r", target="m", status="root", metrics={"holdout": 0.1}))
    s.record(LineageEntry(id="a", target="m", parent_ids=["r"], status="accepted", metrics={"holdout": 0.6}))
    s.record(LineageEntry(id="x", target="m", parent_ids=["r"], status="rejected", metrics={"holdout": 0.9}))
    s.record(LineageEntry(id="b", target="m", parent_ids=["a"], status="accepted", metrics={"holdout": 0.6}))
    again = LineageStore(path)
    assert [e.id for e in again.log()] == ["r", "a", "x", "b"]
    assert again.best().id == "a"  # 平分取更早的；被拒绝的 0.9 不算
    assert [e.id for e in again.log(status="rejected")] == ["x"]
    again.record(LineageEntry(id="r", target="m", status="root", metrics={"holdout": 0.2}))
    assert LineageStore(path).log()[-1].id == "r"


# ---------------------------------------------------------------- 协议层

OFFICIAL = {
    "attribution": RubricEvidenceAttributor,
    "proposer": LessonProposer,
    "algorithm": ArchiveAlgorithm,
    "gate": HoldoutGate,
    "lineage": LineageStore,
}


@pytest.mark.parametrize("name", sorted(OFFICIAL))
def test_conformance_and_manifest(name):
    comp = OFFICIAL[name]()
    report = asyncio.run(check(build_server(comp)))
    assert report.ok, report.text()

    async def served():
        async with connect(build_server(OFFICIAL[name]())) as c:
            return c.info

    info = asyncio.run(served())
    manifest = load_manifest(ROOT / "components" / name / "rsi-component.yaml")
    assert (manifest.name, manifest.version) == (info.name, info.version)
    assert set(manifest.implements) == set(info.tools) and set(manifest.kinds) == set(info.kinds)
