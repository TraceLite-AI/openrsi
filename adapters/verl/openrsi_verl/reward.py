"""把 OpenRSI 判别器变成 verl 的奖励函数。

verl 的自定义奖励函数签名是 compute_score(data_source, solution_str, ground_truth, extra_info=None)，
在配置里用 custom_reward_function.path 指向文件、custom_reward_function.name 指定函数名。

两种用法：

1. 在自己的奖励文件里调用 make_compute_score(judge) 生成函数；
2. 直接把 custom_reward_function.path 指向本文件、name 设为 compute_score，
   再用环境变量 OPENRSI_JUDGE 指向一个 YAML：
       python: "openrsi_judge:EvidenceJudge"
       with: {rubric: {items: [{id: correct, check: output_equals, required: true}]}}

判别器在进程内运行，每次奖励计算不走网络。
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from openrsi import EvidenceCollector, Judge, Step, Task, Trajectory


def reference_from(ground_truth: Any) -> Any:
    """ground_truth 是 JSON 对象字符串就解析成 dict，否则当作 {"expected": ground_truth}。"""
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {"expected": ground_truth}


def default_task(data_source: str, ground_truth: Any, extra_info: dict) -> Task:
    return Task(id=str(extra_info.get("index", "verl")), input=extra_info.get("input"), split="train",
                tags=[data_source, *extra_info.get("tags", [])], reference=reference_from(ground_truth))


def default_trajectory(task: Task, solution_str: str, extra_info: dict) -> Trajectory:
    return Trajectory(id=f"verl-{task.id}", task_id=task.id, output=solution_str,
                      steps=[Step(id="s0", type="message", role="assistant", content=solution_str)])


def make_compute_score(
    judge: Judge,
    *,
    task_from: Callable[[str, Any, dict], Task] = default_task,
    trajectory_from: Callable[[Task, str, dict], Trajectory] = default_trajectory,
    return_dict: bool = False,
) -> Callable[..., Any]:
    """return_dict=True 时返回 {"score": ..., "passed": ..., "rubric/<id>": ...}（新版 verl 支持把额外字段记进日志）。"""

    def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> Any:
        extra = extra_info or {}
        task = task_from(data_source, ground_truth, extra)
        traj = trajectory_from(task, solution_str, extra)
        evidence = judge.collect(task, traj) if isinstance(judge, EvidenceCollector) else []
        v = judge.evaluate(task, traj, evidence)
        if not return_dict:
            return float(v.score)
        out: dict[str, float] = {"score": float(v.score), "passed": 1.0 if v.passed else 0.0}
        for r in v.rubric:
            if r.result in ("pass", "fail"):
                out[f"rubric/{r.id}"] = 1.0 if r.result == "pass" else 0.0
        return out

    return compute_score


def load_judge(spec_path: str | Path) -> Judge:
    import yaml

    spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    module, _, cls = spec["python"].partition(":")
    return getattr(importlib.import_module(module), cls)(**(spec.get("with") or {}))


_CACHED: Callable[..., Any] | None = None


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> Any:
    """供 verl 的 custom_reward_function 直接引用。判别器由环境变量 OPENRSI_JUDGE 指定，首次调用时加载。"""
    global _CACHED
    if _CACHED is None:
        path = os.environ.get("OPENRSI_JUDGE")
        if not path:
            raise RuntimeError("请设置环境变量 OPENRSI_JUDGE，指向判别器的 YAML 配置")
        _CACHED = make_compute_score(load_judge(path), return_dict=os.environ.get("OPENRSI_REWARD_DICT") == "1")
    return _CACHED(data_source, solution_str, ground_truth, extra_info)
