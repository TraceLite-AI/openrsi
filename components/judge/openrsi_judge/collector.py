"""从轨迹（和可选的工作目录）里收集证据。

证据类型：
- tool_call        agent 调用了哪个工具、参数是什么
- tool_result      工具返回了什么、是否报错
- test_result      从测试输出里解析出的通过/失败数
- protected_edit   agent 用编辑类工具改了受保护的路径（比如测试文件）
- file_hash        受保护文件当前的哈希，以及和基线比是否变了
- output           agent 的最终输出

只记录事实，不做判断；判断是判别器的事。
"""
from __future__ import annotations

import fnmatch
import hashlib
import re
from pathlib import Path
from typing import Any

from openrsi import EvidenceCollector, Evidence, Task, Trajectory

_SUMMARY = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped)", re.IGNORECASE)
_PATH_KEYS = ("path", "file", "file_path", "filename", "target_file")


def parse_test_summary(text: str) -> dict[str, Any] | None:
    """解析 pytest 风格的结尾汇总，如 "3 passed, 1 failed in 0.2s"。认不出就返回 None。"""
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    found = False
    for n, word in _SUMMARY.findall(text or ""):
        key = "errors" if word.lower().startswith("error") else word.lower()
        counts[key] += int(n)
        found = True
    if not found:
        return None
    counts["ok"] = counts["passed"] > 0 and counts["failed"] == 0 and counts["errors"] == 0
    return counts


class TrajectoryCollector(EvidenceCollector):
    """配置：
    test_tools  哪些工具名算"跑测试"（它们的输出会被解析成 test_result）
    edit_tools  哪些工具名算"改文件"（参数里的路径会和 protect 比对）
    protect     受保护路径的 glob，如 ["tests/**", "*.lock"]
    workspace   可选的工作目录；给了就对受保护文件算哈希，和 task.metadata["file_hashes"] 比较
    """

    name = "openrsi/trajectory-collector"
    version = "0.4.0"

    def __init__(self, *, test_tools: list[str] | None = None, edit_tools: list[str] | None = None,
                 protect: list[str] | None = None, workspace: str | Path | None = None):
        self.test_tools = set(test_tools or ["run_tests", "pytest", "test"])
        self.edit_tools = set(edit_tools or ["write_file", "edit_file", "str_replace", "create_file",
                                             "apply_patch", "str_replace_based_edit_tool"])
        self.protect = list(protect or [])
        self.workspace = Path(workspace) if workspace else None

    def _protected(self, path: str) -> bool:
        p = path.lstrip("./")
        return any(fnmatch.fnmatch(p, g) or fnmatch.fnmatch(p, g.rstrip("/*") + "/*") for g in self.protect)

    def collect(self, task: Task, trajectory: Trajectory, spec: dict | None = None) -> list[Evidence]:
        src = self.ref
        out: list[Evidence] = []
        names_by_call: dict[str, str] = {}

        for i, step in enumerate(trajectory.steps):
            where = f"trajectory:{trajectory.id}:step:{step.id or i}"
            if step.type == "tool_call":
                if step.call_id:
                    names_by_call[step.call_id] = step.name or ""
                out.append(Evidence.make("tool_call", {"name": step.name, "arguments": step.arguments or {}},
                                         where, src))
                if step.name in self.edit_tools and self.protect:
                    for key in _PATH_KEYS:
                        path = (step.arguments or {}).get(key)
                        if isinstance(path, str) and self._protected(path):
                            out.append(Evidence.make("protected_edit", {"path": path, "tool": step.name},
                                                     where, src))
            elif step.type == "tool_result":
                name = step.name or names_by_call.get(step.call_id or "", "")
                content = step.content if isinstance(step.content, str) else str(step.content)
                out.append(Evidence.make("tool_result", {"name": name, "content": step.content,
                                                         "is_error": bool(step.is_error)}, where, src))
                summary = parse_test_summary(content)
                if summary and (name in self.test_tools or "passed" in content or "failed" in content):
                    out.append(Evidence.make("test_result", {"tool": name, **summary}, where, src))

        out.extend(self._file_evidence(task))
        out.append(Evidence.make("output", {"value": trajectory.output}, f"trajectory:{trajectory.id}:output", src))

        include = (spec or {}).get("include")
        if include:
            out = [e for e in out if e.type in include]
        return out

    def _file_evidence(self, task: Task) -> list[Evidence]:
        if not self.workspace or not self.protect:
            return []
        baseline: dict[str, str] = dict(task.metadata.get("file_hashes", {}))
        current: dict[str, str] = {}
        for f in sorted(self.workspace.rglob("*")):
            rel = f.relative_to(self.workspace).as_posix()
            if f.is_file() and self._protected(rel):
                current[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
        out = []
        for path in sorted(set(current) | set(baseline)):
            now, base = current.get(path), baseline.get(path)
            changed = None if base is None else now != base
            out.append(Evidence.make("file_hash", {"path": path, "sha256": now, "baseline": base,
                                                   "changed": changed},
                                     f"workspace:{path}", self.ref))
        return out
