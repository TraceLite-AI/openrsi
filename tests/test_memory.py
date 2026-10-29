import pytest

from openrsi import Item, Patch, StaleSnapshotError
from openrsi_memory import MemoryEvolvable


def add(*items):
    return [{"op": "add", "item": Item(**i).model_dump(mode="json")} for i in items]


def test_apply_rollback_diff():
    m = MemoryEvolvable()
    s0 = m.snapshot()
    s1 = m.apply(Patch(target="memory", base_snapshot=s0.id, ops=add({"id": "a", "text": "alpha"})))
    assert s1.parent_ids == [s0.id] and m.snapshot().id == s1.id
    assert [i.id for i in m.retrieve("alpha")] == ["a"]
    assert m.diff(s0.id, s1.id).ops == [{"op": "add", "item": {"id": "a", "text": "alpha", "tags": [],
                                                               "kind": "note", "data": {}}}]
    assert m.diff(s1.id, s0.id).ops == [{"op": "remove", "item_id": "a"}]
    assert m.rollback(s0.id).content_hash == s0.content_hash
    assert m.retrieve("alpha") == []
    assert [i.id for i in m.retrieve("alpha", snapshot_id=s1.id)] == ["a"]  # 旧快照仍可读


def test_snapshot_tree_and_idempotent_ids():
    m = MemoryEvolvable()
    root = m.snapshot().id
    a = m.apply(Patch(target="memory", ops=add({"id": "a"})))
    m.rollback(root)
    b = m.apply(Patch(target="memory", ops=add({"id": "b"})))
    assert a.id != b.id and a.parent_ids == b.parent_ids == [root]  # 同一父快照上的两个候选
    m.rollback(root)
    assert m.apply(Patch(target="memory", ops=add({"id": "a"}))).id == a.id  # 同父同补丁 → 同一快照


def test_stale_and_invalid_patches_leave_state_unchanged():
    m = MemoryEvolvable()
    s0 = m.snapshot()
    s1 = m.apply(Patch(target="memory", ops=add({"id": "a"})))
    with pytest.raises(StaleSnapshotError):
        m.apply(Patch(target="memory", base_snapshot=s0.id, ops=add({"id": "b"})))
    with pytest.raises(ValueError):
        m.apply(Patch(target="memory", ops=add({"id": "c"}) + add({"id": "a"})))  # 第二个 op 冲突
    with pytest.raises(ValueError):
        m.apply(Patch(target="other", ops=add({"id": "d"})))
    assert m.snapshot().id == s1.id and set(m.items()) == {"a"}


def test_replace_and_remove():
    m = MemoryEvolvable()
    m.apply(Patch(target="memory", ops=add({"id": "a", "text": "old"})))
    m.apply(Patch(target="memory", ops=[{"op": "replace", "item": {"id": "a", "text": "new"}}]))
    assert m.items()["a"].text == "new"
    m.apply(Patch(target="memory", ops=[{"op": "remove", "item_id": "a"}]))
    assert m.items() == {}
    with pytest.raises(ValueError):
        m.apply(Patch(target="memory", ops=[{"op": "remove", "item_id": "a"}]))


def test_persistence(tmp_path):
    m = MemoryEvolvable(path=tmp_path)
    s1 = m.apply(Patch(target="memory", ops=add({"id": "a", "text": "persisted"})))
    m2 = MemoryEvolvable(path=tmp_path)
    assert m2.snapshot().id == s1.id
    assert [i.id for i in m2.retrieve("persisted")] == ["a"]
    assert [s.id for s in m2.history()] == [s1.id, "snap_root"]


def test_retrieve_ranking_tags_and_chinese():
    m = MemoryEvolvable()
    m.apply(Patch(target="memory", ops=add(
        {"id": "py", "text": "pytest fixtures share setup", "tags": ["testing"]},
        {"id": "zh", "text": "运行测试前先安装依赖", "tags": ["setup"]},
        {"id": "x", "text": "unrelated note"},
    )))
    assert m.retrieve("how do pytest fixtures work")[0].id == "py"
    assert m.retrieve("测试依赖")[0].id == "zh"
    assert [i.id for i in m.retrieve(tags=["setup"])] == ["zh"]
    assert m.retrieve("nothing matches here") == []
