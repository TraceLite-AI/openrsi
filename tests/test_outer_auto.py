"""外循环自动出候选（OuterLoop.auto_round）：自动生成、筛选、验证，但从不自动升级。"""
import pytest

from openrsi import Episode, Evidence, Task, Trajectory, Verdict
from openrsi_attribution import RubricEvidenceAttributor
from openrsi_runtime.outer import (AttributionFixture, CandidateSpec, OuterError, OuterLoop, ValidationPlan,
                                   grid_candidates, judge_check_candidates, qualification_hash)

from test_outer_loop import BASE_RUBRIC, QUAL, example, judge

DEV = [example(i, k) for k in ("honest", "wrong", "cheat") for i in range(100, 104)]  # 和 QUAL 不重叠


def plan(qual=QUAL):
    return ValidationPlan(qualification_hash=qualification_hash(qual), min_examples=5)


def test_auto_round_finds_the_missing_check_but_does_not_promote(tmp_path):
    loop = OuterLoop(tmp_path)
    cands = judge_check_candidates(BASE_RUBRIC, judge_kwargs={"protect": ["tests/**"]})
    assert {c.ref for c in cands} >= {"judge+no_protected_edits", "judge+no_tool_errors"}
    result = loop.auto_round("judge", "judge@base", judge(BASE_RUBRIC), cands, plan(), QUAL, dev=DEV, top_k=3)
    assert result is not None
    proposal, validation = result
    assert proposal.candidate_ref == "judge+no_protected_edits" and validation.accepted
    assert validation.candidate_metrics["false_pass_rate"] == 0
    assert loop.active() == {}  # 自动化只到报告为止
    loop.promote(proposal, validation, approver="Aizs")
    assert loop.active()["judge"]["ref"] == "judge+no_protected_edits"


def test_dev_screening_limits_how_many_candidates_touch_qualification(tmp_path):
    loop = OuterLoop(tmp_path)
    cands = judge_check_candidates(BASE_RUBRIC, judge_kwargs={"protect": ["tests/**"]})
    loop.auto_round("judge", "judge@base", judge(BASE_RUBRIC), cands, plan(), QUAL, dev=DEV, top_k=1)
    kinds = [h["kind"] for h in loop.history()]
    assert kinds.count("outer_validation") == 1 and kinds.count("outer_proposal") == 1


def test_dev_and_qualification_must_not_overlap(tmp_path):
    with pytest.raises(OuterError, match="开发集和资格集"):
        OuterLoop(tmp_path).auto_round("judge", "judge@base", judge(BASE_RUBRIC),
                                       judge_check_candidates(BASE_RUBRIC), plan(), QUAL, dev=QUAL[:3])


def test_no_candidate_good_enough_returns_none_and_keeps_records(tmp_path):
    loop = OuterLoop(tmp_path)
    cands = judge_check_candidates(BASE_RUBRIC, checks=["no_tool_errors"])  # 加了也不改变任何判决
    assert loop.auto_round("judge", "judge@base", judge(BASE_RUBRIC), cands, plan(), QUAL) is None
    v = [h for h in loop.history() if h["kind"] == "outer_validation"]
    assert len(v) == 1 and not v[0]["accepted"]


def test_crashing_candidate_is_recorded_and_skipped(tmp_path):
    def boom():
        raise RuntimeError("构造失败")

    loop = OuterLoop(tmp_path)
    good = judge_check_candidates(BASE_RUBRIC, checks=["no_protected_edits"], judge_kwargs={"protect": ["tests/**"]})
    cands = [CandidateSpec(ref="broken", content={"x": 1}, build=boom), *good]
    proposal, _ = loop.auto_round("judge", "judge@base", judge(BASE_RUBRIC), cands, plan(), QUAL)
    assert proposal.candidate_ref == "judge+no_protected_edits"
    assert any(h["kind"] == "outer_error" and "构造失败" in h["error"] for h in loop.history())


def cell_fixture(i: int) -> AttributionFixture:
    def ep(cell, coords, passed):
        task = Task(id=f"t{i}", env_ref=cell, split="train", input={"q": i})
        traj = Trajectory(id=f"traj{i}@{cell}", task_id=task.id, steps=[])
        ev = Evidence.make("env_cell", {"cell": cell, "coords": coords, "fingerprint": "f"}, f"runtime:cell:{cell}",
                           "openrsi/runtime")
        return Episode(task=task, trajectory=traj, evidence=[ev], verdict=Verdict(
            task_id=task.id, trajectory_id=traj.id, passed=passed, score=float(passed), judge_ref="j"))

    return AttributionFixture(episodes=[ep("utf8", {"E4.1:默认编码": "utf-8"}, True),
                                        ep("cp1252", {"E4.1:默认编码": "cp1252"}, False)],
                              culprit_kind="env", culprit_ref="E4.1:默认编码=cp1252")


def test_attribution_grid_search_picks_cross_cell_contrast(tmp_path):
    qual = [cell_fixture(i) for i in range(10)]
    cands = grid_candidates("openrsi_attribution:CellContrastAttributor", {"env_weight": [0.3, 0.8]})
    assert [c.ref for c in cands] == ["openrsi_attribution:CellContrastAttributor{'env_weight': 0.3}",
                                      "openrsi_attribution:CellContrastAttributor{'env_weight': 0.8}"]
    loop = OuterLoop(tmp_path)
    proposal, v = loop.auto_round("attribution", "rubric-evidence", RubricEvidenceAttributor(), cands, plan(qual), qual)
    assert v.accepted and v.base_metrics["localization"] == 0 and v.candidate_metrics["localization"] == 1
    assert proposal.candidate_ref.endswith("0.8}")  # 0.3 时责任比"缺记忆"小，定位不到
