"""谱系：记录每个快照从哪来、分数多少、为什么被接受或拒绝。被拒绝的候选也留档。

给了 path 就追加写入 JSONL，进程重启后能接着用（运行时靠它续跑）。
"""
from __future__ import annotations

from pathlib import Path

from openrsi import Lineage, LineageEntry, serve


class LineageStore(Lineage):
    name = "openrsi/lineage"
    version = "0.4.0"

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).resolve() if path else None  # 构造时定下绝对路径，之后工作目录变了也不影响
        self._entries: dict[str, LineageEntry] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = LineageEntry.model_validate_json(line)
                    self._entries.pop(e.id, None)  # 同一 id 以最后一次记录为准，位置也按最后一次
                    self._entries[e.id] = e

    def record(self, entry: LineageEntry) -> LineageEntry:
        self._entries.pop(entry.id, None)  # 重新记录的条目移到末尾，保持按时间排序
        self._entries[entry.id] = entry
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        return entry

    def log(self, target: str | None = None, status: str | None = None, limit: int | None = None) -> list[LineageEntry]:
        out = [e for e in self._entries.values()
               if (target is None or e.target == target) and (status is None or e.status == status)]
        return out[-limit:] if limit else out

    def best(self, metric: str = "holdout", target: str | None = None) -> LineageEntry | None:
        pool = [e for e in self.log(target) if e.status in ("root", "accepted") and metric in e.metrics]
        if not pool:
            return None
        # 分数相同取更早的（更简单的那个）
        return max(enumerate(pool), key=lambda p: (p[1].metrics[metric], -p[0]))[1]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-lineage")
    ap.add_argument("--path", help="JSONL 文件路径")
    args = ap.parse_args()
    serve(LineageStore(args.path))


__all__ = ["LineageStore", "main"]
