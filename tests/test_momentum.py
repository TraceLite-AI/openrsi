"""方向动量（loop.group.momentum）：测出来的方向交给算子，带衰减，留探索对照。"""
import sys
from pathlib import Path

import pytest

from anyrsi_runtime.momentum import Direction, DirectionLedger, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))
sys.path.insert(0, str(ROOT / "tests"))
import group_toy as gt  # noqa: E402


def test_profile_breaks_the_gain_down_by_rubric_and_task_tag():
    parent = {"a": 0.0, "b": 1.0, "c": 0.0, "z": 0.0}
    member = {"a": 1.0, "b": 1.0, "c": 0.0, "z": 1.0}
    pr = {"a": {"fmt": 1, "ok": 0}, "b": {"fmt": 1, "ok": 1}, "c": {"fmt": 0, "ok": 0}, "z": {"ok": 0}}
    mr = {"a": {"fmt": 1, "ok": 1}, "b": {"fmt": 0, "ok": 1}, "c": {"fmt": 0, "ok": 0}, "z": {"ok": 1}}
    tags = {"a": ["⊕"], "b": ["⊕"], "c": ["⊗"], "z": ["⊛"]}
    rubric, tag = profile(parent, member, pr, mr, tags, ["a", "b", "c"])  # z 不在这批题里
    assert rubric == {"ok": pytest.approx(1 / 3, abs=1e-3), "fmt": pytest.approx(-1 / 3, abs=1e-3)}
    assert tag == {"⊕": 0.5}  # ⊗ 没变，不列出


def test_ledger_weights_decay_and_negative_directions():
    led = DirectionLedger(decay=0.5, max_items=2)
    assert led.hint(0) is None
    led.add(Direction(0, "replace s1: bad", 0.4, True, tags={"⊕": 0.5}))
    led.add(Direction(2, "replace s2: good", 0.2, True, rubric={"ok": 0.1}))
    led.add(Direction(2, "replace s3: bad", 0.05, True))
    led.add(Direction(2, "replace s4: good", -0.3, False))
    led.add(Direction(2, "nothing", 0.0, True))  # 没效果的不记
    assert len(led.items) == 4
    h = led.hint(2)
    # 权重：s1 0.4×0.25 = 0.1，s2 0.2，s3 0.05 → 取前 2 条：s2、s1
    assert [x.split("] ")[1] for x in h["helped"]] == ["replace s2: good", "replace s1: bad"]
    assert "2 generations ago" in h["helped"][1] and "tasks ⊕ +0.50" in h["helped"][1]
    assert "this generation" in h["helped"][0] and "rubric ok +0.10" in h["helped"][0]
    assert h["hurt"] == [h["hurt"][0]] and "replace s4: good" in h["hurt"][0] and "-0.300" in h["hurt"][0]
    quiet = DirectionLedger(negative=False)
    quiet.add(Direction(0, "x", -0.3, False))
    assert quiet.hint(0) is None


def test_loop_hands_verified_directions_to_the_proposer(tmp_path):
    from test_group import run_with_spies

    from anyrsi_runtime import Runner, Wiring, load_config  # noqa: F401

    seen = []
    orig = gt.ToyProposer._one

    def spy(self, request):
        seen.append(dict(request.hints or {}))
        return orig(self, request)

    gt.ToyProposer._one = spy
    try:
        runner, reports, log, *_ = run_with_spies(tmp_path, "mom", budget=3000, q=0.35, collateral=0.7, break_bias=0.8)
    finally:
        gt.ToyProposer._one = orig
    groups = [r.group for r in reports if r.group]
    assert groups and all("momentum" in g for g in groups)
    # 每代第 1 个改法不给方向（探索），有记录之后其余的都带方向
    guided = [h for h in seen if "momentum" in h]
    assert guided and all(h["group_index"] >= 1 for h in guided)
    assert any(h["group_index"] == 0 and "momentum" not in h for h in seen)
    labels = [{m["direction"] for m in g["members"]} for g in groups]
    assert all(ls <= {"guided", "explore", "no_hint"} for ls in labels)
    assert labels[0] == {"no_hint"}  # 第 1 代还没有方向，不算进对照
    assert any("guided" in ls and "explore" in ls for ls in labels)
    assert all("no_hint" not in ls or ls == {"no_hint"} for ls in labels)
    # 有用的方向只来自通过准入的那一步
    accepted_text = {op["item"]["id"] + ": " + op["item"]["text"] for e in log if e.status == "accepted" for op in e.patch.ops}
    helped = [d for d in runner.ledger.items if d.helped]
    assert helped and all(any(t in d.text for t in accepted_text) for d in helped)
    assert all(d.advantage < 0 for d in runner.ledger.items if not d.helped)


def test_toy_proposer_follows_momentum():
    from anyrsi import EvolvableInfo, ProposalRequest

    info = EvolvableInfo(target="prompt", type="prompt", head="h")
    state = [{"id": k, "kind": "prompt", "text": v} for k, v in gt.START.items()]
    att = [gt.Attribution(episode_id="e", method="t", targets=[gt.AttributionTarget(kind="item", ref="s4", weight=-1)])]
    p = gt.ToyProposer(seed=1, q=0.2, collateral=0.0)
    plain = ProposalRequest(target=info, parent_snapshot="p", state=state, attributions=att)
    led = DirectionLedger()
    led.add(Direction(0, "replace s4: bad (part 1/2 of: ...)", 0.2, True))
    led.add(Direction(0, "replace s4: good", -0.2, False))
    taught = plain.model_copy(update={"hints": {"momentum": led.hint(1)}})
    rate = lambda req: sum("s4=bad" in x.rationale for x in (p._one(req) for _ in range(200)) if x) / 200  # noqa: E731
    assert rate(taught) > 0.9 > 0.75 > rate(plain)


def test_momentum_reaches_the_self_improvement_task():
    from anyrsi import EvolvableInfo, ProposalRequest
    from anyrsi.self_improve import build_task

    info = EvolvableInfo(target="prompt", type="prompt", head="h")
    req = ProposalRequest(target=info, parent_snapshot="p", state=[], hints={"momentum": {"helped": ["x"], "hurt": []}})
    t = build_task(req, gen=1, index=1)
    assert t.input["hints"]["momentum"]["helped"] == ["x"] and "hints.momentum" in t.input["instruction"]


def test_only_gate_confirmed_edits_count_as_helped_and_losses_are_kept():
    from types import SimpleNamespace

    from anyrsi import Patch
    from anyrsi_runtime.group import Decision
    from anyrsi_runtime.loop import GenerationReport, Runner

    def member(i):
        return ({}, Patch(target="p", base_snapshot="b", ops=[{"op": "replace", "item": {"id": f"s{i}", "text": "x"}}]),
                "p", "guided" if i else "explore")

    members = [member(i) for i in range(3)]
    parent = {"t1": 0.0, "t2": 1.0}
    results = [({}, {"t1": 1.0, "t2": 1.0}), ({}, {"t1": 0.0, "t2": 0.0}), ({}, {"t1": 1.0, "t2": 1.0})]
    rubrics = [{"t1": {"ok": 1.0}, "t2": {"ok": 1.0}}, {"t1": {"ok": 0.0}, "t2": {"ok": 0.0}},
               {"t1": {"ok": 1.0}, "t2": {"ok": 1.0}}]
    decision = Decision(chosen=0, reason="", advantage=[0.5, -0.5, 0.5], merge=[0, 2])
    pr = {"t1": {"ok": 0.0}, "t2": {"ok": 1.0}}
    tags = {"t1": ["⊕"], "t2": ["⊗"]}

    def record(accepted, applied=(0, 2)):
        fake = SimpleNamespace(ledger=DirectionLedger())
        report = GenerationReport(generation=3, parent_id="root", parent_metrics={}, train_score=0.0,
                                  group={"applied_members": list(applied)})
        Runner._record_directions(fake, 3, members, results, rubrics, decision, accepted, parent, pr, tags,
                                  ["t1", "t2"], report)
        return fake.ledger.items, report.group["momentum"]

    items, stats = record(True)
    assert sorted((d.text.split(":")[0], d.helped) for d in items) == [("replace s0", True), ("replace s1", False),
                                                                        ("replace s2", True)]
    hurt = next(d for d in items if not d.helped)
    assert hurt.advantage == -0.5 and hurt.tags == {"⊗": -1.0} and hurt.rubric == {"ok": -0.5}
    assert stats["by_label"]["guided"]["n"] == 2 and stats["by_label"]["explore"]["improved"] == 1
    items, _ = record(True, applied=(0,))  # a conflicting edit omitted from the merge gets no credit
    assert not any("s2:" in d.text for d in items)
    items, _ = record(False)  # 准入没过：合并的那几条不算有用的方向，变差的照样记
    assert [(d.text.split(":")[0], d.helped) for d in items] == [("replace s1", False)]
