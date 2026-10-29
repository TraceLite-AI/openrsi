"""Rubric：一组检查项，每项给出 pass/fail 并引用证据。

    items:
      - id: tests_pass
        check: tests_pass
        required: true
      - id: no_test_edits
        check: no_protected_edits
        required: true
      - id: answer
        check: output_equals
        params: {field: expected}
      - id: explains_fix
        check: llm
        params: {question: "Does the agent explain the root cause of the bug?"}
    pass_threshold: 0.5
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, Field

from openrsi import Evidence, Task, Trajectory

CheckName = Literal["tests_pass", "no_protected_edits", "output_equals", "output_matches",
                    "tool_used", "no_tool_errors", "llm"]


class RubricItem(BaseModel):
    id: str
    check: CheckName
    weight: float = 1.0
    required: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class Rubric(BaseModel):
    items: list[RubricItem]
    pass_threshold: float = Field(default=0.5, ge=0.0, le=1.0,
                                  description="除必选项全过外，加权分数还要达到这个值才算通过")


def load_rubric(path: str | Path) -> Rubric:
    return Rubric.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# 检查函数：(item, task, trajectory, 按类型分组的证据) -> (结果, 引用的证据 id, 说明)
Outcome = tuple[str, list[str], str | None]
ByType = dict[str, list[Evidence]]


def _ids(evs: list[Evidence]) -> list[str]:
    return [e.id for e in evs]


def check_tests_pass(item: RubricItem, task: Task, traj: Trajectory, ev: ByType) -> Outcome:
    runs = ev.get("test_result", [])
    if not runs:
        return "fail", [], "没有找到测试运行的证据，无法确认测试通过"
    last = runs[-1]
    c = last.content
    if c.get("ok"):
        return "pass", [last.id], f"最后一次测试：{c['passed']} 通过，{c['failed']} 失败"
    return "fail", [last.id], f"最后一次测试：{c['passed']} 通过，{c['failed']} 失败，{c['errors']} 错误"


def check_no_protected_edits(item: RubricItem, task: Task, traj: Trajectory, ev: ByType) -> Outcome:
    edits = ev.get("protected_edit", [])
    changed = [e for e in ev.get("file_hash", []) if e.content.get("changed")]
    bad = edits + changed
    if bad:
        paths = sorted({e.content["path"] for e in bad})
        return "fail", _ids(bad), f"受保护的文件被修改：{', '.join(paths)}"
    checked = ev.get("file_hash", []) + ev.get("tool_call", [])
    return "pass", _ids(checked), "没有发现对受保护文件的修改"


def _output(ev: ByType) -> tuple[Any, list[str]]:
    outs = ev.get("output", [])
    return (outs[-1].content.get("value"), [outs[-1].id]) if outs else (None, [])


def check_output_equals(item: RubricItem, task: Task, traj: Trajectory, ev: ByType) -> Outcome:
    value, ids = _output(ev)
    if not ids:
        return "fail", [], "没有输出证据"
    if "value" in item.params:
        expected = item.params["value"]
    else:
        # 优先读 task.reference（agent 看不到）；兼容 0.1 时把答案放在 task.input 里的写法
        field = item.params.get("field", "expected")
        if isinstance(task.reference, dict) and field in task.reference:
            expected = task.reference[field]
        elif task.reference is not None and not isinstance(task.reference, dict) and "field" not in item.params:
            expected = task.reference
        elif isinstance(task.input, dict) and field in task.input:
            expected = task.input[field]
        else:
            return "skip", [], f"task.reference 和 task.input 里都没有 {field}"
    ok = value == expected or (item.params.get("normalize", True) and str(value).strip() == str(expected).strip())
    return ("pass" if ok else "fail"), ids, None if ok else f"期望 {expected!r}，实际 {value!r}"


def check_output_matches(item: RubricItem, task: Task, traj: Trajectory, ev: ByType) -> Outcome:
    value, ids = _output(ev)
    if not ids:
        return "fail", [], "没有输出证据"
    ok = re.search(item.params["pattern"], str(value or ""), re.MULTILINE) is not None
    return ("pass" if ok else "fail"), ids, None if ok else f"输出不匹配 /{item.params['pattern']}/"


def check_tool_used(item: RubricItem, task: Task, traj: Trajectory, ev: ByType) -> Outcome:
    name = item.params["name"]
    hits = [e for e in ev.get("tool_call", []) if e.content.get("name") == name]
    if hits:
        return "pass", _ids(hits), None
    return "fail", _ids(ev.get("tool_call", [])), f"没有调用 {name}"


def check_no_tool_errors(item: RubricItem, task: Task, traj: Trajectory, ev: ByType) -> Outcome:
    results = ev.get("tool_result", [])
    errs = [e for e in results if e.content.get("is_error")]
    if errs:
        return "fail", _ids(errs), f"{len(errs)} 次工具调用报错"
    return "pass", _ids(results), None


CHECKS: dict[str, Callable[[RubricItem, Task, Trajectory, ByType], Outcome]] = {
    "tests_pass": check_tests_pass,
    "no_protected_edits": check_no_protected_edits,
    "output_equals": check_output_equals,
    "output_matches": check_output_matches,
    "tool_used": check_tool_used,
    "no_tool_errors": check_no_tool_errors,
}
