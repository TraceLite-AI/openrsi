import hashlib

from openrsi import AnchorExample, Step, Task, Trajectory
from openrsi_judge import EvidenceJudge, LLMCheckResult, Rubric, RubricJudge, TrajectoryCollector, parse_test_summary

import judge_standalone


def rubric(*items, threshold=0.5):
    return Rubric.model_validate({"items": list(items), "pass_threshold": threshold})


def traj(*steps, output=None, tid="t"):
    return Trajectory(id=f"traj-{tid}", task_id=tid, steps=[Step(id=f"s{i}", **s) for i, s in enumerate(steps)],
                      output=output)


TESTS_OK = [dict(type="tool_call", name="run_tests", call_id="c", arguments={}),
            dict(type="tool_result", name="run_tests", call_id="c", content="3 passed in 0.1s")]


def test_parse_test_summary():
    assert parse_test_summary("1 failed, 4 passed in 0.31s") == {
        "passed": 4, "failed": 1, "errors": 0, "skipped": 0, "ok": False}
    assert parse_test_summary("5 passed")["ok"] is True
    assert parse_test_summary("2 passed, 1 error")["ok"] is False
    assert parse_test_summary("no tests ran") is None


def test_catches_test_file_edit():
    judge_standalone.main()  # 不抛错即可
    j = EvidenceJudge(judge_standalone.RUBRIC, protect=["tests/**"])
    t = judge_standalone.from_messages(judge_standalone.MESSAGES, task_id="x")
    v = j.evaluate(judge_standalone.TASK, t, j.collect(judge_standalone.TASK, t))
    by_id = {r.id: r for r in v.rubric}
    assert by_id["tests_pass"].result == "pass"
    assert by_id["no_test_edits"].result == "fail"
    assert not v.passed  # 必选项没过，分数再高也不通过
    edit_ev = [e for e in v.collected or j.collect(judge_standalone.TASK, t) if e.type == "protected_edit"]
    assert edit_ev and edit_ev[0].id in by_id["no_test_edits"].evidence_ids


def test_every_conclusion_cites_known_evidence():
    j = EvidenceJudge(rubric({"id": "tp", "check": "tests_pass"}, {"id": "ne", "check": "no_tool_errors"}))
    task = Task(id="t")
    t = traj(*TESTS_OK)
    ev = j.collect(task, t)
    v = j.evaluate(task, t, ev)
    known = {e.id for e in ev}
    assert v.passed and v.score == 1.0
    for r in v.rubric:
        assert set(r.evidence_ids) <= known and r.evidence_ids


def test_self_collects_when_no_evidence_given():
    j = EvidenceJudge(rubric({"id": "tp", "check": "tests_pass", "required": True}))
    v = j.evaluate(Task(id="t"), traj(*TESTS_OK), [])
    assert v.passed and v.collected
    assert set(v.rubric[0].evidence_ids) <= {e.id for e in v.collected}


def test_rejects_tampered_evidence():
    j = EvidenceJudge(rubric({"id": "tp", "check": "tests_pass", "required": True}))
    task = Task(id="t")
    t = traj(dict(type="tool_result", name="run_tests", content="1 failed, 2 passed"))
    ev = j.collect(task, t)
    forged = [e.model_copy(update={"content": {**e.content, "ok": True, "failed": 0}}) if e.type == "test_result" else e
              for e in ev]
    v = j.evaluate(task, t, forged)
    assert not v.passed
    assert any("哈希不符" in n for n in v.notes)


def test_no_test_run_is_not_a_pass():
    j = RubricJudge(rubric({"id": "tp", "check": "tests_pass", "required": True}), collector=TrajectoryCollector())
    v = j.evaluate(Task(id="t"), traj(dict(type="message", role="assistant", content="all tests pass!")), [])
    assert not v.passed and v.rubric[0].result == "fail"


def test_output_equals_and_skip():
    j = EvidenceJudge(rubric({"id": "eq", "check": "output_equals", "required": True}))
    assert j.evaluate(Task(id="t", input={"expected": 7}), traj(output=7), []).passed
    assert not j.evaluate(Task(id="t", input={"expected": 7}), traj(output=8), []).passed
    v = j.evaluate(Task(id="t", input="no dict"), traj(output=8), [])
    assert v.rubric[0].result == "skip" and not v.passed  # 没有任何结论不能算通过


def test_workspace_file_hashes(tmp_path):
    (tmp_path / "tests").mkdir()
    f = tmp_path / "tests" / "test_a.py"
    f.write_text("assert 1 == 1\n")
    baseline = {"tests/test_a.py": hashlib.sha256(f.read_bytes()).hexdigest()}
    j = EvidenceJudge(rubric({"id": "ne", "check": "no_protected_edits", "required": True}),
                      protect=["tests/**"], workspace=tmp_path)
    task = Task(id="t", metadata={"file_hashes": baseline})
    assert j.evaluate(task, traj(), []).passed
    f.write_text("assert True\n")
    v = j.evaluate(task, traj(), [])
    assert not v.passed and "tests/test_a.py" in v.rubric[0].note


class FakeChecker:
    def __init__(self, passed):
        self.passed, self.calls = passed, []

    def check(self, question, task, evidence):
        self.calls.append((question, [e.type for e in evidence]))
        return LLMCheckResult(passed=self.passed, reason="because", cited_evidence_ids=[evidence[0].id])


def test_llm_item_uses_checker_and_filters_evidence():
    fake = FakeChecker(passed=True)
    j = EvidenceJudge(rubric({"id": "why", "check": "llm",
                              "params": {"question": "Explains root cause?", "evidence_types": ["output"]}}),
                      checker=fake)
    v = j.evaluate(Task(id="t"), traj(*TESTS_OK, output="root cause was X"), [])
    assert v.passed and v.confidence is None
    assert fake.calls == [("Explains root cause?", ["output"])]
    assert v.rubric[0].evidence_ids and v.rubric[0].note == "because"


def test_calibrate():
    j = EvidenceJudge(rubric({"id": "eq", "check": "output_equals", "required": True}))
    good, bad = traj(output=1, tid="a"), traj(output=2, tid="b")
    rep = j.calibrate([
        AnchorExample(task=Task(id="a", input={"expected": 1}), trajectory=good, label_passed=True),
        AnchorExample(task=Task(id="b", input={"expected": 1}), trajectory=bad, label_passed=True),  # 人工认为可以
    ])
    assert rep.n == 2 and rep.agreement == 0.5 and rep.false_fail == 1 and rep.disagreements == ["b"]


def test_claude_checker_request_shape():
    """不调用真实 API：检查请求参数，以及拒答和未知证据 id 的处理。"""
    from openrsi_judge import ClaudeChecker, LLMCheckError
    from openrsi import Evidence

    class FakeMessages:
        def __init__(self, response):
            self.response, self.kwargs = response, None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return self.response

    def client(response):
        msgs = FakeMessages(response)
        return SimpleNamespaceClient(msgs), msgs

    class SimpleNamespaceClient:
        def __init__(self, msgs):
            self.beta = type("B", (), {"messages": msgs})()

    ev = [Evidence.make("output", {"value": "x"}, "s", "c")]
    ok = type("R", (), {"stop_reason": "end_turn",
                        "parsed_output": LLMCheckResult(passed=True, reason="r", cited_evidence_ids=[ev[0].id, "bogus"])})()
    c, msgs = client(ok)
    result = ClaudeChecker(client=c).check("Q?", Task(id="t"), ev)
    assert result.cited_evidence_ids == [ev[0].id]  # 编造的 id 被过滤
    k = msgs.kwargs
    assert k["model"] == "claude-opus-5-5" and k["output_format"] is LLMCheckResult
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]
    assert ev[0].id in k["messages"][0]["content"]

    refused = type("R", (), {"stop_reason": "refusal", "parsed_output": None})()
    c, _ = client(refused)
    import pytest
    with pytest.raises(LLMCheckError):
        ClaudeChecker(client=c).check("Q?", Task(id="t"), ev)
