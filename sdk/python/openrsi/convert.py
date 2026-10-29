"""把常见框架的消息列表转成 OpenRSI 轨迹。

支持两种输入，可以混用：
- OpenAI / Anthropic 风格的 dict：role、content、tool_calls、tool_call_id
- LangChain 消息对象：type（human/ai/tool/system）、content、tool_calls、tool_call_id、name
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from .types import Step, Trajectory

_ROLE_FROM_LC_TYPE = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}


def _get(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _role(msg: Any) -> str:
    role = _get(msg, "role")
    if role:
        return role
    return _ROLE_FROM_LC_TYPE.get(_get(msg, "type", ""), _get(msg, "type", "unknown"))


def _text(content: Any) -> Any:
    """Anthropic 风格的 content 块列表里只取文本；其他保持原样。"""
    if isinstance(content, list) and all(isinstance(b, dict) and "type" in b for b in content):
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        return "\n".join(texts) if texts else content
    return content


def _tool_calls(msg: Any) -> list[tuple[str | None, str, dict]]:
    calls = []
    for tc in _get(msg, "tool_calls", None) or []:
        if "function" in (tc if isinstance(tc, dict) else {}):  # OpenAI 格式
            fn = tc["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append((tc.get("id"), fn.get("name", ""), args))
        else:  # LangChain 格式
            calls.append((_get(tc, "id"), _get(tc, "name", ""), _get(tc, "args", {}) or {}))
    # Anthropic 风格：content 里的 tool_use 块
    content = _get(msg, "content")
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                calls.append((b.get("id"), b.get("name", ""), b.get("input") or {}))
    return calls


def from_messages(messages: Iterable[Any], *, task_id: str, trajectory_id: str | None = None,
                  output: Any = None, **metadata: Any) -> Trajectory:
    steps: list[Step] = []
    call_names: dict[str, str] = {}
    last_assistant_text: Any = None

    for msg in messages:
        role = _role(msg)
        content = _get(msg, "content")

        if role == "tool":
            call_id = _get(msg, "tool_call_id")
            steps.append(Step(type="tool_result", role="tool", content=_text(content), call_id=call_id,
                              name=_get(msg, "name") or call_names.get(call_id or ""),
                              is_error=_get(msg, "status") == "error" or None))
            continue

        # Anthropic 风格：user 消息里的 tool_result 块
        if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    cid = b.get("tool_use_id")
                    steps.append(Step(type="tool_result", role="tool", content=_text(b.get("content")),
                                      call_id=cid, name=call_names.get(cid or ""),
                                      is_error=b.get("is_error") or None))
            continue

        text = _text(content)
        if text not in (None, "", []):
            steps.append(Step(type="message", role=role, content=text))
            if role == "assistant":
                last_assistant_text = text
        for call_id, name, args in _tool_calls(msg):
            if call_id:
                call_names[call_id] = name
            steps.append(Step(type="tool_call", role=role, name=name, arguments=args, call_id=call_id))

    for i, s in enumerate(steps):
        s.id = f"s{i}"
    return Trajectory(id=trajectory_id or f"traj-{task_id}", task_id=task_id, steps=steps,
                      output=output if output is not None else last_assistant_text, metadata=metadata)
