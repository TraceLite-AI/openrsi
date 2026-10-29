"""从 LangGraph 的状态里取出消息，转成 OpenRSI 轨迹，再交给任何 OpenRSI 判别器。

    from openrsi_langgraph import trajectory_from_state, evaluate_state
    final_state = graph.invoke({"messages": [...]})
    verdict = evaluate_state(judge, final_state, task)

不依赖 langgraph 本身：消息对象按 LangChain 的字段（type、content、tool_calls、tool_call_id）读取。
"""
from __future__ import annotations

from typing import Any, Mapping

from openrsi import EvidenceCollector, Judge, Task, Trajectory, Verdict, from_messages


def trajectory_from_state(state: Mapping[str, Any], *, task_id: str, messages_key: str = "messages",
                          output_key: str | None = None, trajectory_id: str | None = None) -> Trajectory:
    output = state.get(output_key) if output_key else None
    return from_messages(state[messages_key], task_id=task_id, trajectory_id=trajectory_id, output=output)


def evaluate_state(judge: Judge, state: Mapping[str, Any], task: Task, *,
                   collector: EvidenceCollector | None = None, **kwargs: Any) -> Verdict:
    traj = trajectory_from_state(state, task_id=task.id, **kwargs)
    evidence = collector.collect(task, traj) if collector else []
    return judge.evaluate(task, traj, evidence)


__all__ = ["evaluate_state", "trajectory_from_state"]
