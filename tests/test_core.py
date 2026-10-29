from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from openrsi import Evidence, Step, Trajectory, Verdict, content_hash, from_messages
from openrsi.protocol import kinds_from_tools, required_tools


def test_evidence_hash_and_tamper():
    e = Evidence.make("output", {"value": 42}, "trajectory:t:output", "test")
    assert e.verify()
    assert e.id == Evidence.make("output", {"value": 42}, "trajectory:t:output", "other").id  # id 与收集者无关
    assert not e.model_copy(update={"content": {"value": 43}}).verify()
    assert e.id != Evidence.make("output", {"value": 42}, "trajectory:t2:output", "test").id


def test_content_hash_is_key_order_independent():
    assert content_hash({"a": 1, "b": [1, 2]}) == content_hash({"b": [1, 2], "a": 1})


def test_verdict_score_bounds():
    with pytest.raises(ValidationError):
        Verdict(score=1.5, passed=True, judge_ref="x")


def test_kinds_from_tools():
    assert kinds_from_tools(required_tools("judge")) == ["judge"]
    assert set(kinds_from_tools(required_tools("memory"))) == {"evolvable", "memory"}
    assert kinds_from_tools({"rsi_judge_evaluate"}) == []  # 缺 info


def test_from_messages_openai():
    t = from_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_tests", "arguments": '{"cmd": "pytest"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "1 passed"},
        {"role": "assistant", "content": "done"},
    ], task_id="t")
    assert [s.type for s in t.steps] == ["message", "tool_call", "tool_result", "message"]
    assert t.steps[1].arguments == {"cmd": "pytest"}
    assert t.steps[2].name == "run_tests"  # 通过 call_id 补上工具名
    assert t.output == "done"
    assert [s.id for s in t.steps] == ["s0", "s1", "s2", "s3"]


def test_from_messages_langchain_objects():
    msgs = [
        SimpleNamespace(type="human", content="go"),
        SimpleNamespace(type="ai", content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "k"}]),
        SimpleNamespace(type="tool", content="result", tool_call_id="k", name="search", status="error"),
        SimpleNamespace(type="ai", content="answer", tool_calls=[]),
    ]
    t = from_messages(msgs, task_id="t")
    assert [(s.type, s.role) for s in t.steps] == [
        ("message", "user"), ("tool_call", "assistant"), ("tool_result", "tool"), ("message", "assistant")]
    assert t.steps[2].is_error is True
    assert t.output == "answer"


def test_from_messages_anthropic_blocks():
    t = from_messages([
        {"role": "assistant", "content": [{"type": "text", "text": "running"},
                                          {"type": "tool_use", "id": "u1", "name": "bash", "input": {"cmd": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "a.py", "is_error": False}]},
    ], task_id="t")
    assert [s.type for s in t.steps] == ["message", "tool_call", "tool_result"]
    assert t.steps[2].name == "bash"


def test_trajectory_roundtrip_json():
    t = Trajectory(id="x", task_id="t", steps=[Step(type="message", role="user", content="hi")])
    assert Trajectory.model_validate_json(t.model_dump_json()) == t
