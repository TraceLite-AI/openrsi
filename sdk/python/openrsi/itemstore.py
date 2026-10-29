"""按条目组织的可进化体的通用实现：快照树、原子补丁、回滚、比较、导出、持久化。

memory、prompt、skill 库都是"一组带 id 的条目"，区别只在怎么校验条目、怎么交给 agent 用。
继承 ItemStore，按需覆盖 validate_item / describe 即可。

快照组成一棵树（不是一条链），同一个父快照上可以长出多个候选。
给了 path 就持久化到目录：snapshots/<id>.json 保存快照和补丁，HEAD 记录当前快照。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .components import Evolvable, StaleSnapshotError
from .types import Diff, EvolvableInfo, Item, MemoryOp, Patch, Snapshot, canonical_json, content_hash, now_iso


def items_hash(items: dict[str, Item]) -> str:
    return content_hash([items[k].model_dump(mode="json") for k in sorted(items)])


class ItemStore(Evolvable):
    evolvable_type = "items"
    surfaces = ["items"]

    def __init__(self, target: str, path: str | Path | None = None):
        self.target = target
        self.path = Path(path).resolve() if path else None  # 构造时定下绝对路径，之后工作目录变了也不影响
        self._snapshots: dict[str, Snapshot] = {}
        self._ops: dict[str, list[MemoryOp]] = {}
        self._cache: dict[str, dict[str, Item]] = {}
        self._head: str | None = None
        if self.path and (self.path / "HEAD").exists():
            self._load()
        else:
            root = Snapshot(id="snap_root", target=target, content_hash=items_hash({}),
                            created_at=now_iso(), message=f"empty {self.evolvable_type}")
            self._store(root, [])
            self._set_head(root.id)

    # ------------------------------------------------------------ 子类可覆盖

    def validate_item(self, item: Item) -> None:
        """apply 前对每个新增或替换的条目调用；不合格就抛 ComponentError，整个补丁不生效。"""

    def example_ops(self) -> list[dict[str, Any]]:
        return [{"op": "add", "item": {"id": "__openrsi_probe__", "text": "openrsi conformance probe",
                                       "tags": ["probe"]}}]

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
            self._cache[sid] = apply_ops(base, self._ops[sid])
        return self._cache[sid]

    # ------------------------------------------------------------ Evolvable

    def describe(self) -> EvolvableInfo:
        return EvolvableInfo(target=self.target, type=self.evolvable_type, head=self._head, surfaces=self.surfaces,
                             size=len(self.items()), example_ops=self.example_ops())

    def snapshot(self) -> Snapshot:
        return self._snapshots[self._head]

    def apply(self, patch: Patch) -> Snapshot:
        if patch.target != self.target:
            raise ValueError(f"补丁目标是 {patch.target}，本组件是 {self.target}")
        if patch.base_snapshot and patch.base_snapshot != self._head:
            raise StaleSnapshotError(f"补丁基于 {patch.base_snapshot}，当前是 {self._head}")
        parent = self._head
        ops = [MemoryOp.model_validate(o) for o in patch.ops]
        for op in ops:
            if op.item is not None:
                self.validate_item(op.item)
        new_items = apply_ops(dict(self.items(parent)), ops)  # 不合法的 op 在这里抛错，状态不变
        sid = "snap_" + hashlib.sha256(f"{parent}|{canonical_json(patch.ops)}".encode()).hexdigest()[:16]
        if sid not in self._snapshots:
            snap = Snapshot(id=sid, target=self.target, parent_ids=[parent], patch_id=patch.id,
                            content_hash=items_hash(new_items), created_at=now_iso(), message=patch.rationale)
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

    def export(self, snapshot_id: str | None = None) -> list[dict]:
        """快照里的全部条目，按 id 排序。"""
        items = self.items(snapshot_id)
        return [items[k].model_dump(mode="json") for k in sorted(items)]

    def history(self, snapshot_id: str | None = None) -> list[Snapshot]:
        """从某个快照沿第一个父快照回溯到根，新的在前。"""
        out, cur = [], snapshot_id or self._head
        while cur:
            snap = self._snapshots[cur]
            out.append(snap)
            cur = snap.parent_ids[0] if snap.parent_ids else None
        return out


def apply_ops(items: dict[str, Item], ops: list[MemoryOp]) -> dict[str, Item]:
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
