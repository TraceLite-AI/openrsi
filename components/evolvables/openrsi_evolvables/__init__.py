"""memory 之外的两种可进化体：prompt 库和 skill 库。都基于 SDK 的 ItemStore，有快照树、可回滚。

- PromptEvolvable：agent 指令的各个段落。render() 按顺序拼成完整 prompt。
- SkillLibrary：agent 能调用的 Python 函数。条目的 data.code 是源码，data.entry 是入口函数名（默认 run）。
  invoke() 在独立子进程里执行（见 sandbox.py 的限制说明）。
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from openrsi import ComponentError, Item, SkillResult, Skills, serve
from openrsi.itemstore import ItemStore

from .sandbox import run_skill


class PromptEvolvable(ItemStore):
    """with: max_chars（单个段落上限，默认 4000）"""

    name = "openrsi/prompt"
    version = "0.4.0"
    evolvable_type = "prompt"
    surfaces = ["sections"]

    def __init__(self, target: str = "prompt", path: str | Path | None = None, max_chars: int = 4000):
        self.max_chars = max_chars
        super().__init__(target, path)

    def validate_item(self, item: Item) -> None:
        if not item.text.strip():
            raise ComponentError(f"prompt 段落 {item.id} 是空的")
        if len(item.text) > self.max_chars:
            raise ComponentError(f"prompt 段落 {item.id} 有 {len(item.text)} 字符，超过上限 {self.max_chars}")

    def example_ops(self) -> list[dict[str, Any]]:
        return [{"op": "add", "item": {"id": "__openrsi_probe__", "kind": "prompt",
                                       "text": "openrsi conformance probe", "tags": ["probe"]}}]

    def render(self, snapshot_id: str | None = None) -> str:
        """按 data.order（默认 0）和 id 排序，拼成完整的 prompt。"""
        items = sorted(self.items(snapshot_id).values(), key=lambda i: (i.data.get("order", 0), i.id))
        return "\n\n".join(i.text for i in items)


def check_skill_code(code: str, entry: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ComponentError(f"skill 代码有语法错误：{e}") from e
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if entry not in names:
        raise ComponentError(f"skill 代码没有在顶层定义函数 {entry}")


class SkillLibrary(ItemStore, Skills):
    """with: timeout（秒）、cpu_seconds、memory_mb"""

    name = "openrsi/skills"
    version = "0.4.0"
    evolvable_type = "skill"
    surfaces = ["skills"]

    def __init__(self, target: str = "skills", path: str | Path | None = None, timeout: float = 5.0,
                 cpu_seconds: int = 5, memory_mb: int = 512):
        self.limits = {"timeout": timeout, "cpu_seconds": cpu_seconds, "memory_mb": memory_mb}
        super().__init__(target, path)

    def validate_item(self, item: Item) -> None:
        code = item.data.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ComponentError(f"skill {item.id} 缺少 data.code")
        check_skill_code(code, item.data.get("entry", "run"))

    def example_ops(self) -> list[dict[str, Any]]:
        return [{"op": "add", "item": {"id": "__openrsi_probe__", "kind": "skill", "text": "openrsi conformance probe",
                                       "tags": ["probe"],
                                       "data": {"code": "def run(x=1):\n    return x + 1\n", "entry": "run"}}}]

    def invoke(self, skill_id: str, arguments: dict[str, Any] | None = None,
               snapshot_id: str | None = None) -> SkillResult:
        items = self.items(snapshot_id)
        if skill_id not in items:
            raise KeyError(f"快照里没有 skill {skill_id}")
        item = items[skill_id]
        return run_skill(item.data["code"], arguments, entry=item.data.get("entry", "run"), **self.limits)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-evolvables")
    ap.add_argument("--kind", choices=["prompt", "skills"], required=True)
    ap.add_argument("--path", help="持久化目录")
    ap.add_argument("--target", help="补丁里的 target 名（默认与 kind 相同）")
    args = ap.parse_args()
    cls = PromptEvolvable if args.kind == "prompt" else SkillLibrary
    serve(cls(target=args.target or args.kind, path=args.path))


__all__ = ["PromptEvolvable", "SkillLibrary", "check_skill_code", "main", "run_skill"]
