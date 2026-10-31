"""不可变记录与运行清单（two-loop-evolution 设计 §4.1、§4.3，P2）。

RecordStore 是按内容寻址、只追加的存储：
  - 每个对象按规范 JSON 的 SHA-256 存成 objects/<前两位>/<哈希>.json；同一个哈希再写一次必须内容相同
  - 读的时候重新算哈希，对不上就报错（发现篡改）；verify() 把整个库检查一遍
  - index.jsonl 只追加，记录每个对象的种类和所属运行
原始记录（题目、轨迹、证据、执行回执）写成 EvidenceRecord；判决、归因这类可以重算的东西写成
DerivedRecord，带上来源哈希。重算只会追加新的 DerivedRecord（supersedes 指向旧的），不改原始记录。

哈希只能发现篡改，不能证明写入者可信；真实的对抗实验还需要独立的存储权限。

RunManifest 记下一次运行的身份：配置（不含环境变量的值）、组件名和版本、环境格指纹、协议版本。
续跑时清单必须一致；严格模式（配置了 records）下不一致就拒绝，否则只给警告。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from openrsi import content_hash
from openrsi.types import canonical_json, now_iso


class RecordError(RuntimeError):
    """记录缺失、哈希对不上或试图覆盖。"""


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "evidence_record"
    run_id: str
    episode_id: str
    task_digest: str
    cell_fingerprint: str | None = None
    execution_receipt_hash: str | None = None
    raw_trajectory_hash: str
    raw_artifact_hashes: list[str] = Field(default_factory=list, description="各条证据的哈希")
    producer_ref: str


class DerivedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "derived_record"
    run_id: str
    what: str = Field(description="verdict / attribution 等")
    source_hashes: list[str]
    method_ref: str
    output_hash: str
    supersedes: str | None = None
    created_at: str = Field(default_factory=now_iso)


class RecordStore:
    def __init__(self, path: str | Path):
        self.root = Path(path)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def _file(self, digest: str) -> Path:
        return self.root / "objects" / digest[:2] / f"{digest}.json"

    def put(self, obj: Any, kind: str, run_id: str = "") -> str:
        """写入一个对象，返回它的哈希。已有同哈希的对象时内容必须一致。"""
        data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
        text = canonical_json(data)
        digest = content_hash(data)
        f = self._file(digest)
        if f.exists():
            if f.read_text(encoding="utf-8") != text:
                raise RecordError(f"记录 {digest[:12]} 已存在且内容不同：库被改过")
            return digest
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
        with self.index.open("a", encoding="utf-8") as idx:
            idx.write(json.dumps({"hash": digest, "kind": kind, "run_id": run_id}, ensure_ascii=False) + "\n")
        return digest

    def get(self, digest: str) -> Any:
        f = self._file(digest)
        if not f.exists():
            raise RecordError(f"缺少记录 {digest[:12]}")
        data = json.loads(f.read_text(encoding="utf-8"))
        if content_hash(data) != digest:
            raise RecordError(f"记录 {digest[:12]} 的内容和哈希对不上：被篡改")
        return data

    def entries(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.index.exists():
            return []
        rows = [json.loads(line) for line in self.index.read_text(encoding="utf-8").split("\n") if line.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]

    def verify(self) -> list[str]:
        """检查索引里的每个对象：返回出问题的哈希（缺失或被改）。"""
        bad = []
        for row in self.entries():
            try:
                self.get(row["hash"])
            except RecordError:
                bad.append(row["hash"])
        return bad


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    protocol: str
    config_hash: str = Field(description="脱敏后的配置：组件写法、构造参数、循环、环境格；环境变量只记名字")
    components: dict[str, str] = Field(description="位置 → 组件名:版本（不含传输方式）")
    cells: dict[str, str] = Field(default_factory=dict, description="格 id → 指纹")
    openrsi: str

    @property
    def manifest_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def sanitized_config(config: Any) -> dict[str, Any]:
    """配置里和结果有关的部分。command 组件的环境变量只保留名字（可能是凭据）；传输方式不算在内。"""
    def spec(s: Any) -> dict[str, Any]:
        d = s.model_dump(mode="json", by_alias=True, exclude_none=True)
        d["env"] = sorted((d.get("env") or {}))
        for k in ("python", "command", "url"):
            d.pop(k, None)
        return d
    return {"protocol": config.protocol, "plugins": list(config.plugins),
            "components": {k: spec(v) for k, v in sorted(config.components.items())},
            "evolvables": {k: spec(v) for k, v in sorted(config.evolvables.items())},
            "loop": config.loop.model_dump(mode="json"),
            "cells": config.cells.model_dump(mode="json") if getattr(config, "cells", None) else None}


__all__ = ["DerivedRecord", "EvidenceRecord", "RecordError", "RecordStore", "RunManifest", "sanitized_config"]
