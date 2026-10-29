"""把 OpenRSI 判别器变成 DSPy 的 metric。

    from openrsi_dspy import judge_metric
    metric = judge_metric(judge, to_task=..., to_trajectory=...)
    dspy.GEPA(metric=metric, ...)            # GEPA 会用到 feedback 文本
    dspy.MIPROv2(metric=metric, ...)          # 其他优化器只看分数

不依赖 dspy 本身：装了 dspy 就返回 dspy.Prediction(score, feedback)，没装就返回 dict。
"""
from __future__ import annotations

from typing import Any, Callable

from openrsi import EvidenceCollector, Judge, Task, Trajectory, Verdict


def feedback_text(verdict: Verdict) -> str:
    """把没通过的 rubric 项写成一段反馈，给 GEPA 这类反思式优化器读。"""
    lines = [f"score={verdict.score:.2f} passed={verdict.passed}"]
    for r in verdict.rubric:
        if r.result != "pass":
            lines.append(f"- [{r.result}] {r.id}{' (required)' if r.required else ''}: {r.note or ''}".rstrip())
    if len(lines) == 1:
        lines.append("All rubric items passed.")
    return "\n".join(lines)


def judge_metric(
    judge: Judge,
    *,
    to_task: Callable[[Any], Task],
    to_trajectory: Callable[[Any, Any], Trajectory],
    collector: EvidenceCollector | None = None,
    with_feedback: bool = True,
) -> Callable[..., Any]:
    """to_task(example) 和 to_trajectory(example, prediction) 负责把 DSPy 对象转成协议格式。
    collector 不给时，判别器若自带取证会自己收集。"""

    def metric(gold: Any, pred: Any, trace: Any = None, pred_name: Any = None, pred_trace: Any = None) -> Any:
        task = to_task(gold)
        traj = to_trajectory(gold, pred)
        evidence = collector.collect(task, traj) if collector else []
        verdict = judge.evaluate(task, traj, evidence)
        if not with_feedback:
            return verdict.score
        text = feedback_text(verdict)
        try:
            import dspy
        except ImportError:
            return {"score": verdict.score, "feedback": text}
        return dspy.Prediction(score=verdict.score, feedback=text)

    return metric


__all__ = ["feedback_text", "judge_metric"]
