"""零阶"梯度"一步：loop.group.atomize（拆成小改动逐条测）+ merge 策略（变好的合并成一步）+ PatchProposer。"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from anyrsi import EvolvableInfo, Patch, ProposalRequest
from anyrsi_runtime.group import Member, MergePositive, atomize, load_strategy, merge_patches, op_key

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))
sys.path.insert(0, str(ROOT / "tests"))
import group_toy as gt  # noqa: E402


def rep(i, text):
    return {"op": "replace", "item": {"id": i, "kind": "prompt", "text": text}}


def test_atomize_and_merge():
    p = Patch(target="prompt", base_snapshot="b", ops=[rep("s1", "good"), rep("s2", "bad"), {"op": "remove", "item_id": "s3"}],
              rationale="three edits", proposer_ref="x")
    parts = atomize(p)
    assert [q.ops for q in parts] == [[op] for op in p.ops]
    assert all(q.base_snapshot == "b" and q.proposer_ref == "x" for q in parts)
    assert "replace s1: good" in parts[0].rationale and "part 1/3" in parts[0].rationale
    one = Patch(target="prompt", base_snapshot="b", ops=[rep("s1", "x")])
    assert atomize(one) == [one]
    assert [op_key(o) for o in p.ops] == ["s1", "s2", "s3"]
    # 同一个条目只留第一条（策略已按优势从高到低排序）
    m = merge_patches([Patch(target="prompt", base_snapshot="b", ops=[rep("s1", "good")]),
                       Patch(target="prompt", base_snapshot="b", ops=[rep("s1", "bad"), rep("s4", "good")])])
    assert m.ops == [rep("s1", "good"), rep("s4", "good")] and m.base_snapshot == "b"


def test_merge_strategy_takes_every_improving_direction_up_to_step():
    parent = {f"t{i}": 0.0 for i in range(6)}

    def member(i, wins):
        return Member(index=i, scores={f"t{k}": 1.0 if k < wins else 0.0 for k in range(6)})

    members = [member(0, 1), member(1, 3), member(2, 0), member(3, 2)]
    members.append(Member(index=4, scores={**{f"t{k}": 0.0 for k in range(6)}, "t0": -1.0}))
    d = MergePositive().decide(parent, members, 0.05)
    assert d.merge == [1, 3, 0] and d.chosen == 1  # 优势从高到低，不变和变差的不要
    assert d.advantage[1] == pytest.approx(0.5) and d.advantage[4] < 0
    assert MergePositive(step=2).decide(parent, members, 0.05).merge == [1, 3]
    assert MergePositive(min_adv=0.4).decide(parent, members, 0.05).merge == [1]
    assert MergePositive(min_z=1.2).decide(parent, members, 0.05).merge == [1, 3]  # z：1/6 → 1.0，2/6 → 1.58
    none = MergePositive().decide(parent, [member(0, 0), members[4]], 0.05)
    assert none.chosen is None and none.merge == []
    assert isinstance(load_strategy("merge"), MergePositive)
    assert load_strategy({"name": "merge", "with": {"step": 3}}).step == 3
    with pytest.raises(ValueError):
        MergePositive(step=0)


def test_loop_tests_single_edits_and_merges_the_good_ones(tmp_path):
    from test_group import run_with_spies

    runner, reports, log, _, gate_calls, _ = run_with_spies(tmp_path, "grad", budget=3000, q=0.35, collateral=0.7,
                                                            break_bias=0.8)
    groups = [r.group for r in reports if r.group]
    assert groups and all(g["strategy"] == "merge" for g in groups)
    assert max(g["size"] for g in groups) > 4  # 4 个改法拆出了不止 4 条小改动
    conf = set(runner.holdout_confirm)
    assert gate_calls and all(set(req.candidate_tasks) == conf for req in gate_calls)
    merged = [g for g in groups if g.get("merged")]
    assert merged, "应当有把几条小改动合并成一步的代"
    accepted = [e for e in log if e.status == "accepted"]
    assert accepted and any(len(e.patch.ops) > 1 for e in accepted)
    # 合并的就是选中的那几条：每条被合并的小改动优势都为正
    for g in merged:
        assert all(g["members"][i]["advantage"] > 0 for i in g["merged"])


def test_weak_proposer_still_finds_the_direction_on_the_toy(tmp_path):
    kw = {"q": 0.2, "collateral": 0.8, "break_bias": 0.8}
    a = asyncio.run(gt.run_arm("A", tmp_path / "a", 1, 4000, **kw))
    g = asyncio.run(gt.run_arm("grad", tmp_path / "g", 1, 4000, **kw))
    assert g["harmful"] == 0
    assert g["true_final"] > a["true_final"] + 0.05 and g["true_final"] > g["true_start"] + 0.25


def test_patch_proposer_asks_the_self_improvement_task(monkeypatch):
    import anyrsi_llm
    from anyrsi_llm import PatchProposer

    reply = {"target": "prompt", "ops": [rep("rule", "a ⊕ b = a*b"), rep("base", "be careful")], "rationale": "two"}
    seen = []

    def fake_chat(base_url, model, messages, **kw):
        seen.append(messages)
        return {"text": "Sure: " + json.dumps(reply), "usage": {}, "finish_reason": "stop"}

    monkeypatch.setattr(anyrsi_llm, "chat", fake_chat)
    info = EvolvableInfo(target="prompt", type="prompt", head="h")
    req = ProposalRequest(target=info, targets=[info], parent_snapshot="p", state=[], states={"prompt": []},
                          max_patches=2)
    out = PatchProposer(base_url="http://127.0.0.1:9/v1", model="m").propose(req)
    assert len(out) == 2 and out[0].ops == reply["ops"] and out[0].proposer_ref.startswith("anyrsi/patch-proposer")
    sys_msg, user = seen[0][0]["content"], seen[0][1]["content"]
    assert "improves other AI assistants" in sys_msg and "You are improving yourself" in user
