from types import SimpleNamespace

from openrsi import Task, Trajectory
from openrsi_dspy import feedback_text, judge_metric
from openrsi_judge import EvidenceJudge, Rubric
from openrsi_langgraph import evaluate_state, trajectory_from_state

import memory_evolution

EQ = Rubric.model_validate({"items": [{"id": "answer", "check": "output_equals", "required": True}]})


def test_dspy_metric_returns_score_and_feedback():
    metric = judge_metric(
        EvidenceJudge(EQ),
        to_task=lambda ex: Task(id=ex.id, input={"expected": ex.answer}),
        to_trajectory=lambda ex, pred: Trajectory(id=f"p-{ex.id}", task_id=ex.id, output=pred.answer),
    )
    gold = SimpleNamespace(id="q1", answer="Paris")
    good = metric(gold, SimpleNamespace(answer="Paris"))
    bad = metric(gold, SimpleNamespace(answer="Lyon"))
    score = lambda r: r["score"] if isinstance(r, dict) else r.score  # noqa: E731
    fb = lambda r: r["feedback"] if isinstance(r, dict) else r.feedback  # noqa: E731
    assert score(good) == 1.0 and score(bad) == 0.0
    assert "answer" in fb(bad) and "Lyon" in fb(bad)
    assert "All rubric items passed" in fb(good)


def test_feedback_text_marks_required():
    v = EvidenceJudge(EQ).evaluate(Task(id="t", input={"expected": 1}), Trajectory(id="x", task_id="t", output=2), [])
    assert "(required)" in feedback_text(v)


def test_langgraph_state():
    state = {"messages": [
        SimpleNamespace(type="human", content="capital of France?"),
        SimpleNamespace(type="ai", content="Paris", tool_calls=[]),
    ]}
    t = trajectory_from_state(state, task_id="q")
    assert t.output == "Paris" and len(t.steps) == 2
    assert evaluate_state(EvidenceJudge(EQ), state, Task(id="q", input={"expected": "Paris"})).passed


def test_memory_evolution_example_reaches_full_holdout():
    log = memory_evolution.run(rounds=10, verbose=False)
    assert log[0]["baseline"] == 0.0
    accepted = [r for r in log if r["accepted"]]
    assert accepted and accepted[-1]["candidate"] == 1.0
    for r in log:  # 被接受的候选一定严格提升了 holdout
        if r["accepted"]:
            assert r["candidate"] > r["baseline"]
