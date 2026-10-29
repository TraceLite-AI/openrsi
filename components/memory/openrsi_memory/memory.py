"""可进化 memory：所有修改都经过补丁，每个补丁生成一个快照，可以检索任意快照、随时回滚。

快照、补丁、回滚、持久化由 SDK 的 ItemStore 提供，这里只加上检索。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from openrsi import Item, Memory
from openrsi.itemstore import ItemStore

_WORD = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    """英文按词、中文按字切分，再加上中文相邻字的二元组。"""
    raw = _WORD.findall((text or "").lower())
    bigrams = [a + b for a, b in zip(raw, raw[1:]) if len(a) == len(b) == 1 and "\u4e00" <= a <= "\u9fff"]
    return raw + bigrams


class MemoryEvolvable(ItemStore, Memory):
    name = "openrsi/memory"
    version = "0.3.0"
    evolvable_type = "memory"

    def __init__(self, target: str = "memory", path: str | Path | None = None):
        super().__init__(target, path)

    def retrieve(self, query: str = "", tags: list[str] | None = None, k: int = 5,
                 snapshot_id: str | None = None) -> list[Item]:
        """标签是过滤条件，查询只用来排序。

        - 给了标签：返回带这些标签的条目，按与查询的相关度排序（不相关的也保留，排在后面）
        - 只给查询：返回与查询有词重叠的条目（词重叠打分，带 IDF）
        - 都不给：返回前 k 条
        """
        items = sorted(self.items(snapshot_id).values(), key=lambda it: it.id)
        want = set(tags or [])
        if want:
            items = [it for it in items if want & set(it.tags)]
        q = set(_tokens(query))
        if not q:
            return items[:k]
        docs = {it.id: Counter(_tokens(it.text + " " + " ".join(it.tags))) for it in items}
        df = Counter(t for c in docs.values() for t in c)
        n = len(items) or 1
        scored = [(sum(math.log(1 + n / df[t]) for t in q if t in docs[it.id]), it) for it in items]
        if not want:
            scored = [(s, it) for s, it in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], x[1].id))
        return [it for _, it in scored[:k]]
