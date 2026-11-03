"""agent 给自己出改法（loop.self_propose）：递归改进最直接的形式。

不再用单独的提案器，而是把"改进你自己"当成一道题，交给当前版本的 agent。题目里有：练习题里做错的轨迹、
归因结果、它自己当前的可改内容（提示段落、记忆条目、代码文件……），以及输出格式。agent 用的就是自己当前的
提示、代码、记忆和权重，所以 agent 变强，它出改法的能力也跟着变强——改进者和被改进者是同一个。

agent 的输出是一个 JSON 对象（可以夹在文字里）：
    {"target": "<改哪个可进化体>", "ops": [...补丁操作...], "rationale": "<一句话说明>"}
ops 的格式和该可进化体的补丁一样，题目里给了每个可进化体的示例。解析不出来就当作这次没出改法。

数据隔离不变：题目只含 train 的轨迹；出的补丁照样过护栏、照样在考试题上准入。
"""
from __future__ import annotations

import json
from typing import Any

from anyrsi import Patch, ProposalRequest, Task

KIND = "self_improvement"

INSTRUCTION = (
    "You are improving yourself. Below are your current editable parts, tasks you recently failed (with what you did "
    "and why it was judged wrong), and an analysis of what caused the failures. Propose ONE change to your own "
    "editable parts that would make you succeed on such tasks in general. Never write answers to specific tasks. "
    "Reply with one JSON object only: {\"target\": <one of the targets>, \"ops\": [<patch operations in the format of "
    "that target's examples>], \"rationale\": <one sentence>}."
)


def _text(x: Any, limit: int) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s[:limit]


def build_task(request: ProposalRequest, *, gen: int, index: int, seed: int | None = None,
               max_failures: int = 6, max_state_chars: int = 6000) -> Task:
    """把提案请求变成一道"改进你自己"的题。split 是 train：它只用到练习题的信息。"""
    targets = []
    for info in request.targets or [request.target]:
        state = request.states.get(info.target) if request.states else request.state
        targets.append({"target": info.target, "type": info.type, "surfaces": info.surfaces,
                        "example_ops": info.example_ops, "current": state})
    failures = []
    for ep in [e for e in request.episodes if not e.verdict.passed][:max_failures]:
        why = "; ".join(r.note or r.id for r in ep.verdict.rubric if r.result == "fail") or "judged wrong"
        failures.append({"episode": ep.trajectory.id, "task": _text(ep.task.input, 800),
                         "output": _text(ep.trajectory.output, 800), "why_wrong": why})
    attributions = [{"episode": a.episode_id, "kind": t.kind, "ref": t.ref, "weight": t.weight, "note": t.note}
                    for a in request.attributions for t in a.targets][:50]
    hints = {k: v for k, v in (request.hints or {}).items() if k in ("direction", "step", "group_index", "improver")}
    body = {"targets": targets, "failures": failures, "attributions": attributions, "hints": hints}
    listing = _text(body, max_state_chars * max(1, len(targets)) + 8000)
    metadata: dict[str, Any] = {"self_improvement": True, "generation": gen, "index": index}
    if seed is not None:
        metadata["rollout_seed"] = seed
    return Task(id=f"self-improve-g{gen}-{index}", split="train", tags=[KIND], metadata=metadata,
                input={"kind": KIND, "instruction": f"{INSTRUCTION}\n\n{listing}", **body})


def _json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def parse_patch(output: Any, request: ProposalRequest, agent_ref: str) -> Patch | None:
    """agent 的输出 → 补丁。没有 ops、target 不在可改对象里、格式不对，都返回 None。"""
    obj = output if isinstance(output, dict) else _json_object(output) if isinstance(output, str) else None
    if not obj or not isinstance(obj.get("ops"), list) or not obj["ops"]:
        return None
    allowed = {t.target for t in (request.targets or [request.target])}
    target = obj.get("target") or request.target.target
    if target not in allowed or not all(isinstance(op, dict) for op in obj["ops"]):
        return None
    return Patch(target=target, base_snapshot=request.parent_snapshot, ops=obj["ops"],
                 rationale=str(obj.get("rationale") or "")[:500], proposer_ref=f"self:{agent_ref}")


__all__ = ["INSTRUCTION", "KIND", "build_task", "parse_patch"]
