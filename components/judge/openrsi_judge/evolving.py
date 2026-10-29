"""可进化的判别器：rubric 存成带版本的条目，判别时总是用当前快照里的 rubric。

它同时是取证组件、判别器和可进化体。修改 rubric 也走补丁，可以回滚。
判别器的修改会改变打分标准，所以运行时不用 holdout 评价它，而是看它在人工标注的 anchor 集上
和人工判断的一致率（见 PROTOCOL.md 第 9 节）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from openrsi import ComponentError, Item, Patch
from openrsi.itemstore import ItemStore

from .collector import TrajectoryCollector
from .judge import RubricJudge
from .llm import Checker
from .rubric import Rubric, RubricItem, load_rubric


def as_rubric(rubric: Rubric | dict | str | Path) -> Rubric:
    if isinstance(rubric, Rubric):
        return rubric
    if isinstance(rubric, dict):
        return Rubric.model_validate(rubric)
    return load_rubric(rubric)


def rubric_item_to_item(ri: RubricItem, order: int = 0) -> dict[str, Any]:
    data = ri.model_dump(mode="json", exclude={"id"})
    data["order"] = order
    return Item(id=ri.id, kind="rubric", text=ri.description or ri.check, data=data).model_dump(mode="json")


class EvolvingJudge(ItemStore, TrajectoryCollector, RubricJudge):
    """with: rubric（初始 rubric）、target（默认 rubric）、path、protect、workspace、test_tools 等取证参数"""

    name = "openrsi/evolving-judge"
    version = "0.4.0"
    evolvable_type = "rubric"
    surfaces = ["rubric"]

    def __init__(self, rubric: Rubric | dict | str | Path, *, target: str = "rubric", path: str | Path | None = None,
                 checker: Checker | None = None, **collector_config: Any):
        TrajectoryCollector.__init__(self, **collector_config)
        initial = as_rubric(rubric)
        self.pass_threshold = initial.pass_threshold
        self.collector = self
        self._checker = checker
        ItemStore.__init__(self, target, path)
        if self.snapshot().id == "snap_root" and not self.items():  # 新建时写入初始 rubric；从磁盘恢复时不重写
            ops = [{"op": "add", "item": rubric_item_to_item(ri, i)} for i, ri in enumerate(initial.items)]
            self.apply(Patch(target=target, ops=ops, rationale="initial rubric"))

    @property
    def rubric(self) -> Rubric:  # type: ignore[override]
        items = sorted(self.items().values(), key=lambda i: (i.data.get("order", 0), i.id))
        return Rubric(items=[self._to_rubric_item(i) for i in items], pass_threshold=self.pass_threshold)

    @staticmethod
    def _to_rubric_item(item: Item) -> RubricItem:
        data = {k: v for k, v in item.data.items() if k != "order"}
        return RubricItem.model_validate({**data, "id": item.id})

    def validate_item(self, item: Item) -> None:
        try:
            self._to_rubric_item(item)
        except ValidationError as e:
            raise ComponentError(f"rubric 条目 {item.id} 不合法：{e.errors()[0]['msg']}") from e

    def example_ops(self) -> list[dict[str, Any]]:
        return [{"op": "add", "item": rubric_item_to_item(
            RubricItem(id="__openrsi_probe__", check="no_tool_errors", description="openrsi conformance probe"), 99)}]
