"""提案器：把 train 上的失败变成 memory 补丁。

- LessonProposer：不需要 LLM 的基线。把失败原因写成按标签归类的"教训"条目，并删掉多次只带来失败的条目。
- ClaudeProposer：让 Claude 读失败的轨迹、判决和归因，写出可复用的经验条目。
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from openrsi import Item, Patch, ProposalRequest, Proposer, serve

DEFAULT_MODEL = "claude-opus-5-5"


def _current_items(request: ProposalRequest) -> dict[str, Item]:
    if isinstance(request.state, list):
        return {d["id"]: Item.model_validate(d) for d in request.state if isinstance(d, dict) and "id" in d}
    return {}


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", text).strip("-").lower()[:48] or "general"


def _item_net(request: ProposalRequest) -> dict[str, tuple[float, int]]:
    acc: dict[str, list[float]] = defaultdict(list)
    for a in request.attributions:
        for t in a.targets:
            if t.kind == "item":
                acc[t.ref].append(t.weight)
    return {k: (sum(v), len(v)) for k, v in acc.items()}


class LessonProposer(Proposer):
    """with: prune_after（某条目至少被用多少次且从未帮上忙才删除，默认 2）"""

    name = "openrsi/lesson-proposer"
    version = "0.3.0"

    def __init__(self, prune_after: int = 2):
        self.prune_after = prune_after

    def propose(self, request: ProposalRequest) -> list[Patch]:
        items = _current_items(request)
        ops: list[dict] = []

        # 1. 删掉被多次使用、净效果为负、从未带来成功的条目
        for item_id, (net, n) in sorted(_item_net(request).items()):
            if item_id in items and n >= self.prune_after and net < 0:
                positive = any(t.kind == "item" and t.ref == item_id and t.weight > 0
                               for a in request.attributions for t in a.targets)
                if not positive:
                    ops.append({"op": "remove", "item_id": item_id})

        # 2. 失败原因按标签归类，写成教训
        notes: dict[str, list[str]] = defaultdict(list)
        tags_for: dict[str, list[str]] = {}
        for ep in request.episodes:
            if ep.verdict.passed:
                continue
            key = "-".join(sorted(ep.task.tags)) or "general"
            tags_for[key] = sorted(ep.task.tags)
            for r in ep.verdict.rubric:
                if r.result in ("fail", "error") and r.note and r.note not in notes[key]:
                    notes[key].append(r.note)
        removed = {o["item_id"] for o in ops}
        for key, lines in sorted(notes.items()):
            item_id = f"lesson:{_slug(key)}"
            text = "Past failures: " + " | ".join(lines[:5])
            item = Item(id=item_id, kind="lesson", text=text, tags=tags_for[key],
                        data={"source": "train-failures", "count": len(lines)})
            op = "replace" if item_id in items and item_id not in removed else "add"
            if item_id in removed:
                ops = [o for o in ops if o.get("item_id") != item_id]
                op = "replace"
            ops.append({"op": op, "item": item.model_dump(mode="json")})

        if not ops:
            return []
        return [Patch(target=request.target.target, base_snapshot=request.parent_snapshot, ops=ops,
                      rationale=f"{len(ops)} memory ops from {sum(not e.verdict.passed for e in request.episodes)} failures",
                      proposer_ref=self.ref)]


class ProposedItem(BaseModel):
    op: Literal["add", "replace", "remove"]
    id: str = Field(description="Item id. Reuse an existing id to replace or remove it.")
    text: str = Field(default="", description="The reusable lesson, written as guidance for future attempts")
    tags: list[str] = Field(default_factory=list, description="Task tags this lesson applies to")


class ProposedPatch(BaseModel):
    ops: list[ProposedItem]
    rationale: str


SYSTEM = (
    "You improve an AI agent's memory. You see the agent's recent training episodes (task, trajectory, verdict "
    "with rubric results and the evidence they cite), an attribution of what caused each failure, and the "
    "current memory items. Write a small number of memory changes that would help the agent succeed on "
    "similar future tasks. Lessons must generalize: never store the answer to a specific task. Prefer "
    "replacing or removing items that attribution shows were harmful. Return no ops if nothing would help."
)


class ClaudeProposer(Proposer):
    """with: model（默认 claude-opus-5-5）、effort、max_ops。需要 anthropic 和 API 凭证。"""

    name = "openrsi/claude-proposer"
    version = "0.3.0"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "high", max_ops: int = 5, client=None):
        self.model, self.effort, self.max_ops = model, effort, max_ops
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic  # 需要 pip install "openrsi-proposer[llm]"

            self._client = anthropic.Anthropic()
        return self._client

    def _payload(self, request: ProposalRequest) -> str:
        episodes = [{
            "task": ep.task.model_dump(mode="json", exclude={"reference", "metadata"}),
            "steps": [s.model_dump(mode="json", exclude_none=True) for s in ep.trajectory.steps],
            "output": ep.trajectory.output,
            "used_items": ep.trajectory.used_items,
            "verdict": {"passed": ep.verdict.passed, "score": ep.verdict.score,
                        "rubric": [r.model_dump(mode="json", exclude_none=True) for r in ep.verdict.rubric]},
        } for ep in request.episodes]
        attributions = [a.model_dump(mode="json", exclude_none=True) for a in request.attributions]
        return json.dumps({"memory": request.state, "episodes": episodes, "attributions": attributions},
                          ensure_ascii=False)

    def propose(self, request: ProposalRequest) -> list[Patch]:
        if all(ep.verdict.passed for ep in request.episodes):
            return []
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=SYSTEM,
            messages=[{"role": "user", "content": f"At most {self.max_ops} ops.\n\n{self._payload(request)}"}],
            output_format=ProposedPatch,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return []
        items = _current_items(request)
        ops = []
        for p in response.parsed_output.ops[: self.max_ops]:
            # 模型可能把 add/replace 弄反，按当前内容纠正，避免补丁整体被拒
            if p.op == "remove":
                if p.id in items:
                    ops.append({"op": "remove", "item_id": p.id})
                continue
            op = "replace" if p.id in items else "add"
            ops.append({"op": op, "item": Item(id=p.id, kind="lesson", text=p.text, tags=p.tags).model_dump(mode="json")})
        if not ops:
            return []
        return [Patch(target=request.target.target, base_snapshot=request.parent_snapshot, ops=ops,
                      rationale=response.parsed_output.rationale, proposer_ref=self.ref)]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-proposer")
    ap.add_argument("--kind", choices=["lesson", "claude"], default="lesson")
    args = ap.parse_args()
    serve(LessonProposer() if args.kind == "lesson" else ClaudeProposer())


__all__ = ["ClaudeProposer", "LessonProposer", "ProposedPatch", "main"]
