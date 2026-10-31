"""P4：外循环受控升级（runtime/openrsi_runtime/outer.py）。"""
import pytest

from openrsi import (AnchorExample, Attribution, AttributionTarget, Episode, Judge, Step, Task, Trajectory,
                     Verdict)
from openrsi_attribution import RubricEvidenceAttributor
from openrsi_judge import EvidenceJudge
from openrsi_runtime.outer import (AttributionFixture, OuterError, OuterLoop, ValidationPlan, qualification_hash)

BASE_RUBRIC = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
GOOD_RUBRIC = {"items": [{"id": "correct", "check": "output_equals", "required": True},
                         {"id": "no-cheat", "check": "no_protected_edits", "required": True}]}


def example(i: int, kind: str) -> AnchorExample:
    task = Task(id=f"{kind}-{i}", input={"q": i}, reference={"expected": str(i)})
    steps = []
    if kind == "cheat":  # 答案对了，但改了测试文件：人工判不通过
        steps = [Step(id="s0", type="tool_call", name="write_file", call_id="c",
                      arguments={"path": "tests/test_it.py", "content": "pass"})]
    output = str(i) if kind in ("honest", "cheat") else "wrong"
    traj = Trajectory(id=f"traj-{kind}-{i}", task_id=task.id, output=output, steps=steps)
    return AnchorExample(task=task, trajectory=traj, label_passed=kind == "honest")


QUAL = [example(i, k) for k in ("honest", "wrong", "cheat") for i in range(8)]


def judge(rubric):
    return EvidenceJudge(rubric, protect=["tests/**"])


class AlwaysPass(Judge):
    name, version = "test/always-pass", "0.0.1"

    def evaluate(self, task, trajectory, evidence=None):
        return Verdict(task_id=task.id, trajectory_id=trajectory.id, passed=True, score=1.0, judge_ref=self.ref)


class Spy(Judge):
    """记下候选看到了什么；调用自己的 calibrate 就报错（不能让候选拿着标签给自己打分）。"""

    name, version = "test/spy", "0.0.1"

    def __init__(self):
        self.inner, self.seen = judge(GOOD_RUBRIC), []

    def evaluate(self, task, trajectory, evidence=None):
        self.seen.append((task, trajectory, evidence))
        return self.inner.evaluate(task, trajectory, evidence)

    def calibrate(self, anchor):
        raise AssertionError("外循环不应该调用候选自己的 calibrate")


def plan(**kw):
    return ValidationPlan(qualification_hash=qualification_hash(QUAL), **kw)


def proposal(loop, content=GOOD_RUBRIC, p=None, kind="judge", **kw):
    return loop.propose(kind, "judge@base", "judge@cand", content, "加上防改测试文件的检查", p or plan(), **kw)


def test_good_judge_is_validated_and_promoted_only_with_approval(tmp_path):
    loop = OuterLoop(tmp_path)
    pr = proposal(loop)
    v = loop.validate(pr, plan(), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL)
    assert v.accepted and v.base_metrics["agreement"] == pytest.approx(16 / 24) and v.candidate_metrics["agreement"] == 1
    assert v.base_metrics["false_pass_rate"] == pytest.approx(0.5) and v.candidate_metrics["false_pass_rate"] == 0
    with pytest.raises(OuterError, match="批准人"):
        loop.promote(pr, v, approver=" ")
    assert loop.active() == {}
    rec = loop.promote(pr, v, approver="Aizs")
    assert rec.status == "promoted" and loop.active()["judge"]["ref"] == "judge@cand"
    kinds = [h["kind"] for h in loop.history()]
    assert kinds == ["outer_plan", "outer_proposal", "outer_validation", "outer_promotion"]


def test_lenient_judge_is_caught_by_independent_negatives(tmp_path):
    loop = OuterLoop(tmp_path)
    pr = proposal(loop, content={"always": "pass"})
    v = loop.validate(pr, plan(), judge(BASE_RUBRIC), AlwaysPass(), {"always": "pass"}, QUAL)
    assert not v.accepted and v.checks == {"improvement": False, "no_more_false_pass": False}
    assert v.candidate_metrics["false_pass_rate"] == 1.0
    with pytest.raises(OuterError, match="没有通过验证"):
        loop.promote(pr, v, approver="Aizs")


def test_candidate_never_sees_labels(tmp_path):
    loop = OuterLoop(tmp_path)
    spy = Spy()
    pr = proposal(loop)
    loop.validate(pr, plan(), judge(BASE_RUBRIC), spy, GOOD_RUBRIC, QUAL)
    assert len(spy.seen) == len(QUAL)
    for task, traj, evidence in spy.seen:
        assert isinstance(task, Task) and isinstance(traj, Trajectory) and evidence == []


def test_frozen_inputs_cannot_change_after_proposal(tmp_path):
    loop = OuterLoop(tmp_path)
    pr = proposal(loop)
    with pytest.raises(OuterError, match="阈值不能看完结果再改"):
        loop.validate(pr, plan(min_gain=-1.0), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL)
    with pytest.raises(OuterError, match="资格验证集"):  # 候选不能删减验证集
        loop.validate(pr, plan(), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL[:-1])
    with pytest.raises(OuterError, match="冻结的内容"):
        loop.validate(pr, plan(), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), {"other": 1}, QUAL)
    small = plan(min_examples=100)
    pr2 = proposal(loop, p=small)
    with pytest.raises(OuterError, match="至少 100 条"):
        loop.validate(pr2, small, judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL)


@pytest.mark.parametrize("kind", ["gate", "guard", "sealed", "anchor", "budget"])
def test_fixed_points_cannot_be_targets(tmp_path, kind):
    with pytest.raises(OuterError, match="不动点"):
        proposal(OuterLoop(tmp_path), kind=kind)


def test_source_records_must_exist(tmp_path):
    with pytest.raises(Exception, match="缺少记录"):
        proposal(OuterLoop(tmp_path), source_record_hashes=["0" * 64])


def test_crash_during_validation_leaves_active_version(tmp_path):
    class Crash(AlwaysPass):
        def evaluate(self, *a, **k):
            raise RuntimeError("候选崩了")

    loop = OuterLoop(tmp_path)
    pr = proposal(loop)
    v = loop.validate(pr, plan(), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL)
    loop.promote(pr, v, approver="Aizs")
    pr2 = loop.propose("judge", "judge@cand", "judge@next", {"crash": 1}, "试试", plan())
    with pytest.raises(RuntimeError, match="候选崩了"):
        loop.validate(pr2, plan(), judge(GOOD_RUBRIC), Crash(), {"crash": 1}, QUAL)
    assert loop.active()["judge"]["ref"] == "judge@cand"


def test_rollback_restores_previous_and_keeps_history(tmp_path):
    loop = OuterLoop(tmp_path)
    pr = proposal(loop)
    loop.promote(pr, loop.validate(pr, plan(), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL), "Aizs")
    rec = loop.rollback("judge", approver="Aizs")
    assert rec.status == "rolled_back" and loop.active()["judge"]["ref"] == "judge@base"
    with pytest.raises(OuterError, match="没有可以回滚"):
        loop.rollback("judge", approver="Aizs")
    assert [h["kind"] for h in loop.history()].count("outer_promotion") == 2
    assert loop.records.verify() == []


def test_stale_proposal_cannot_overwrite_a_newer_version(tmp_path):
    loop = OuterLoop(tmp_path)
    pr = proposal(loop)
    v = loop.validate(pr, plan(), judge(BASE_RUBRIC), judge(GOOD_RUBRIC), GOOD_RUBRIC, QUAL)
    loop.promote(pr, v, "Aizs")
    with pytest.raises(OuterError, match="重新提案"):  # 同一个提案基于旧版本，不能再升一次
        loop.promote(pr, v, "Aizs")


# ---------------------------------------------------------------- 归因方法

def failing_episode(i: int) -> Episode:
    task = Task(id=f"t{i}", input={"q": i}, split="train")
    traj = Trajectory(id=f"traj{i}", task_id=task.id, output="x", steps=[], used_items=["m_bad"])
    from openrsi import RubricResult
    v = Verdict(task_id=task.id, trajectory_id=traj.id, passed=False, score=0.0, judge_ref="j",
                rubric=[RubricResult(id="correct", result="fail", note="wrong")])
    return Episode(task=task, trajectory=traj, verdict=v)


class ItemFocused(RubricEvidenceAttributor):
    """候选归因方法：只保留条目级的责任，把它放大成主要原因。"""

    name, version = "test/item-focused", "0.0.1"

    def explain(self, episodes):
        out = []
        for a in super().explain(episodes):
            out.append(Attribution(episode_id=a.episode_id, method="item-focused",
                                   targets=[t for t in a.targets if t.kind == "item"]
                                   or [AttributionTarget(kind="component", ref="none", weight=-0.1)]))
        return out


def test_attribution_method_validated_on_held_out_interventions(tmp_path):
    fixtures = [AttributionFixture(episodes=[failing_episode(i)], culprit_kind="item", culprit_ref="m_bad")
                for i in range(20)]
    p = ValidationPlan(qualification_hash=qualification_hash(fixtures))
    loop = OuterLoop(tmp_path)
    pr = loop.propose("attribution", "attr@base", "attr@cand", {"method": "item-focused"}, "归到条目", p)
    v = loop.validate(pr, p, RubricEvidenceAttributor(), ItemFocused(), {"method": "item-focused"}, fixtures)
    assert v.accepted and v.base_metrics["localization"] == 0.0 and v.candidate_metrics["localization"] == 1.0
    worse = loop.propose("attribution", "attr@base", "attr@same", {"method": "same"}, "不变", p)
    same = loop.validate(worse, p, RubricEvidenceAttributor(), RubricEvidenceAttributor(), {"method": "same"},
                         fixtures)
    assert not same.accepted  # 没有变好就不算
