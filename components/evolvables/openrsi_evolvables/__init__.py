"""memory 之外的两种可进化体：prompt 库和 skill 库。都基于 SDK 的 ItemStore，有快照树、可回滚。

- PromptEvolvable：agent 指令的各个段落。render() 按顺序拼成完整 prompt。
- SkillLibrary：agent 能调用的 Python 函数。条目的 data.code 是源码，data.entry 是入口函数名（默认 run）。
  invoke() 在独立子进程里执行（见 sandbox.py 的限制说明）。

元层可进化体（Improver 层，运行时按 loop.meta_every 进化，agent 看不到，按"下游改进"准入）：
- ImproverEvolvable：提案器的策略说明（type improver）。提案器从 hints["improver"] 读到它
- CurriculumEvolvable：练习题怎么挑（type curriculum）。条目 data = {tag, weight}，只影响 train 取题，不碰考试题
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from openrsi import ComponentError, Item, Patch, SkillResult, Skills, serve
from openrsi.itemstore import ItemStore

from .sandbox import run_skill


class PromptEvolvable(ItemStore):
    """with: max_chars（单个段落上限，默认 4000）；initial（可选，新建时作为第一个版本的段落列表，
    每项 {id, text, data: {order}}；已有存储时忽略）"""

    name = "openrsi/prompt"
    version = "0.4.0"
    evolvable_type = "prompt"
    surfaces = ["sections"]

    def __init__(self, target: str = "prompt", path: str | Path | None = None, max_chars: int = 4000,
                 initial: list[dict[str, Any]] | None = None):
        self.max_chars = max_chars
        super().__init__(target, path)
        if initial and not self.items():
            ops = [{"op": "add", "item": {"kind": "prompt", **i}} for i in initial]
            self.apply(Patch(target=target, ops=ops, rationale="initial prompt"))

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


class ImproverEvolvable(PromptEvolvable):
    """提案器的策略（Improver × Harness）：几段说明或参数，提案器自己决定怎么读。
    with: 同 PromptEvolvable；条目的 data 可以放结构化参数（如 {"max_rules": 2}）。"""

    name = "openrsi/improver"
    version = "0.4.1"
    evolvable_type = "improver"
    surfaces = ["strategy"]

    def __init__(self, target: str = "improver", path: str | Path | None = None, max_chars: int = 4000,
                 initial: list[dict[str, Any]] | None = None):
        super().__init__(target, path, max_chars, initial)

    def example_ops(self) -> list[dict[str, Any]]:
        return [{"op": "add", "item": {"id": "__openrsi_probe__", "kind": "improver",
                                       "text": "openrsi conformance probe", "tags": ["probe"]}}]


class CurriculumEvolvable(ItemStore):
    """练习题的挑选（环境 × Solver 的搜索侧）：每个条目给一个标签的权重。
    运行时取 train 题时，按标签权重排序、去掉权重为 0 的标签，再截到 train_limit；没写的标签权重是 1。
    考试题（holdout）不受影响。with: initial（可选，{tag: weight}）"""

    name = "openrsi/curriculum"
    version = "0.4.1"
    evolvable_type = "curriculum"
    surfaces = ["tag_weights"]

    def __init__(self, target: str = "curriculum", path: str | Path | None = None,
                 initial: dict[str, float] | None = None):
        super().__init__(target, path)
        if initial and not self.items():
            ops = [{"op": "add", "item": {"id": f"tag:{t}", "kind": "curriculum", "text": f"{t} weight {w}",
                                          "data": {"tag": t, "weight": w}}} for t, w in sorted(initial.items())]
            self.apply(Patch(target=target, ops=ops, rationale="initial curriculum"))

    def validate_item(self, item: Item) -> None:
        tag, weight = item.data.get("tag"), item.data.get("weight")
        if not isinstance(tag, str) or not tag:
            raise ComponentError(f"课程条目 {item.id} 缺少 data.tag")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0:
            raise ComponentError(f"课程条目 {item.id} 的 data.weight 必须是非负数")

    def example_ops(self) -> list[dict[str, Any]]:
        return [{"op": "add", "item": {"id": "tag:__probe__", "kind": "curriculum", "text": "probe",
                                       "data": {"tag": "__probe__", "weight": 1}}}]


def apply_curriculum(tasks: list[Any], items: list[dict[str, Any]], limit: int | None) -> list[Any]:
    """按课程挑练习题：权重 0 的标签去掉，其余按（最大标签权重降序，原顺序）排，截到 limit。"""
    weights = {d["data"]["tag"]: float(d["data"]["weight"]) for d in items
               if isinstance(d, dict) and isinstance(d.get("data"), dict) and "tag" in d["data"]}

    def w(task: Any) -> float:
        tags = list(getattr(task, "tags", []) or [])
        return max((weights.get(t, 1.0) for t in tags), default=1.0)

    ranked = [t for _, _, t in sorted((-w(t), i, t) for i, t in enumerate(tasks) if w(t) > 0)]
    return ranked[:limit] if limit else ranked


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


__all__ = ["CurriculumEvolvable", "ImproverEvolvable", "PromptEvolvable", "SkillLibrary", "apply_curriculum",
           "check_skill_code", "main", "run_skill"]
