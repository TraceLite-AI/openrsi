"""可进化 memory：所有修改都经过补丁，每个补丁生成一个快照，可以检索任意快照、随时回滚。

快照组成一棵树（不是一条链），同一个父快照上可以长出多个候选，供进化算法挑选。
给了 path 就持久化到目录：snapshots/<id>.json 保存快照和补丁，HEAD 记录当前快照。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from openrsi import Diff, EvolvableInfo, Item, Memory, MemoryOp, Patch, Snapshot, StaleSnapshotError, content_hash
from openrsi.types import canonical_json, now_iso

PROBE = {"op": "add", "item": {"id": "__openrsi_probe__", "text": "openrsi conformance probe", "tags": ["probe"]}}
_WORD = re.compile(r"[a-z0-9_]+|[一-鿿]")


def _tokens(text: str) -> list[str]:
    """英文按词、中文按字切分，再加上中文相邻字的二元组。"""
    raw = _WORD.findall((text or "").lower())
    bigrams = [a + b for a, b in zip(raw, raw[1:]) if len(a) == len(b) == 1 and "一" <= a <= "鿿"]
    return raw + bigrams


def _items_hash(items: dict[str, Item]) -> str:
    return content_hash([items[k].model_dump(mode="json") for k in sorted(items)])


class MemoryEvolvable(Memory):
    name = "openrsi/memory"
    version = "0.1.0"

    def __init__(self, target: str = "memory", path: str | Path | None = None):
        self.target = target
        self.path = Path(path) if path else None
        self._snapshots: dict[str, Snapshot] = {}
        self._ops: dict[str, list[MemoryOp]] = {}
        self._cache: dict[str, dict[str, Item]] = {}
        self._head: str | None = None
        if self.path and (self.path / "HEAD").exists():
            self._load()
        else:
            root = Snapshot(id="snap_root", target=target, content_hash=_items_hash({}),
                            created_at=now_iso(), message="empty memory")
            self._store(root, [])
            self._set_head(root.id)

    # ------------------------------------------------------------ 持久化

    def _load(self) -> None:
        for f in (self.path / "snapshots").glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            snap = Snapshot.model_validate(data["snapshot"])
            self._snapshots[snap.id] = snap
            self._ops[snap.id] = [MemoryOp.model_validate(o) for o in data["ops"]]
        self._head = (self.path / "HEAD").read_text().strip()

    def _store(self, snap: Snapshot, ops: list[MemoryOp]) -> None:
        self._snapshots[snap.id] = snap
        self._ops[snap.id] = ops
        if self.path:
            d = self.path / "snapshots"
            d.mkdir(parents=True, exist_ok=True)
            body = {"snapshot": snap.model_dump(mode="json"), "ops": [o.model_dump(mode="json") for o in ops]}
            (d / f"{snap.id}.json").write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    def _set_head(self, snap_id: str) -> None:
        self._head = snap_id
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / "HEAD").write_text(snap_id)

    # ------------------------------------------------------------ 内容

    def items(self, snapshot_id: str | None = None) -> dict[str, Item]:
        sid = snapshot_id or self._head
        if sid not in self._snapshots:
            raise KeyError(f"没有快照 {sid}")
        if sid not in self._cache:
            snap = self._snapshots[sid]
            base = dict(self.items(snap.parent_ids[0])) if snap.parent_ids else {}
            self._cache[sid] = _apply_ops(base, self._ops[sid])
        return self._cache[sid]

    # ------------------------------------------------------------ Evolvable

    def describe(self) -> EvolvableInfo:
        return EvolvableInfo(target=self.target, type="memory", head=self._head, surfaces=["items"],
                             size=len(self.items()), example_ops=[PROBE])

    def snapshot(self) -> Snapshot:
        return self._snapshots[self._head]

    def apply(self, patch: Patch) -> Snapshot:
        if patch.target != self.target:
            raise ValueError(f"补丁目标是 {patch.target}，本组件是 {self.target}")
        if patch.base_snapshot and patch.base_snapshot != self._head:
            raise StaleSnapshotError(f"补丁基于 {patch.base_snapshot}，当前是 {self._head}")
        parent = self._head
        ops = [MemoryOp.model_validate(o) for o in patch.ops]
        new_items = _apply_ops(dict(self.items(parent)), ops)  # 不合法的 op 在这里抛错，状态不变
        sid = "snap_" + hashlib.sha256(f"{parent}|{canonical_json(patch.ops)}".encode()).hexdigest()[:16]
        if sid not in self._snapshots:
            snap = Snapshot(id=sid, target=self.target, parent_ids=[parent], patch_id=patch.id,
                            content_hash=_items_hash(new_items), created_at=now_iso(), message=patch.rationale)
            self._store(snap, ops)
            self._cache[sid] = new_items
        self._set_head(sid)
        return self._snapshots[sid]

    def rollback(self, snapshot_id: str) -> Snapshot:
        if snapshot_id not in self._snapshots:
            raise KeyError(f"没有快照 {snapshot_id}")
        self._set_head(snapshot_id)
        return self._snapshots[snapshot_id]

    def diff(self, from_id: str, to_id: str) -> Diff:
        a, b = self.items(from_id), self.items(to_id)
        ops: list[dict] = [{"op": "remove", "item_id": k} for k in sorted(set(a) - set(b))]
        for k in sorted(b):
            if k not in a:
                ops.append({"op": "add", "item": b[k].model_dump(mode="json")})
            elif a[k] != b[k]:
                ops.append({"op": "replace", "item": b[k].model_dump(mode="json")})
        return Diff(from_id=from_id, to_id=to_id, ops=ops)

    def history(self, snapshot_id: str | None = None) -> list[Snapshot]:
        """从某个快照沿第一个父快照回溯到根，新的在前。"""
        out, cur = [], snapshot_id or self._head
        while cur:
            snap = self._snapshots[cur]
            out.append(snap)
            cur = snap.parent_ids[0] if snap.parent_ids else None
        return out

    # ------------------------------------------------------------ Memory

    def retrieve(self, query: str = "", tags: list[str] | None = None, k: int = 5,
                 snapshot_id: str | None = None) -> list[Item]:
        """简单的词重叠打分（带 IDF）+ 标签加分。只给标签不给查询时按标签过滤。"""
        items = list(self.items(snapshot_id).values())
        want = set(tags or [])
        if want:
            items = [it for it in items if want & set(it.tags)] or []
        q = _tokens(query)
        if not q:
            return items[:k]
        docs = {it.id: Counter(_tokens(it.text + " " + " ".join(it.tags))) for it in items}
        df = Counter(t for c in docs.values() for t in c)
        n = len(items) or 1
        scored = []
        for it in items:
            c = docs[it.id]
            s = sum(math.log(1 + n / df[t]) for t in set(q) if t in c)
            if s > 0:
                scored.append((s, it.id, it))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [it for _, _, it in scored[:k]]


def _apply_ops(items: dict[str, Item], ops: list[MemoryOp]) -> dict[str, Item]:
    for op in ops:
        if op.op == "add":
            if op.item is None:
                raise ValueError("add 需要 item")
            if op.item.id in items:
                raise ValueError(f"条目 {op.item.id} 已存在，修改请用 replace")
            items[op.item.id] = op.item
        elif op.op == "replace":
            if op.item is None or op.item.id not in items:
                raise ValueError(f"replace 的条目不存在: {op.item.id if op.item else None}")
            items[op.item.id] = op.item
        elif op.op == "remove":
            if op.item_id not in items:
                raise ValueError(f"remove 的条目不存在: {op.item_id}")
            del items[op.item_id]
    return items
