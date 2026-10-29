"""护栏：补丁生效之前检查它的内容。准入（gate）看分数，护栏看补丁本身。

默认规则（都可以在构造参数里调整）：

| 规则 | 级别 | 拦什么 |
|---|---|---|
| protected | block | 改动受保护的条目（按 id 的 glob，如 "system:*"） |
| too_many_ops / too_many_removals | block | 一次改动太多、删除太多 |
| text_too_long | block | 单个条目文字超长 |
| secret | block | 疑似 API key、私钥、密码 |
| eval_tampering | block | 教 agent 去改测试、改判别器、讨好评分 |
| code_syntax | block | skill 代码语法错误 |
| code_import | block | import 白名单之外的模块（os、subprocess、socket……） |
| code_call | block | eval、exec、open、__import__ 等危险调用 |
| code_dunder | block | 访问 __class__、__globals__ 这类双下划线属性（常见的沙箱逃逸手法） |
| injection | warn | 疑似提示注入（"ignore previous instructions"） |

AST 检查拦不住所有恶意代码；它是第一道防线，配合子进程限制和容器使用。
"""
from __future__ import annotations

import ast
import fnmatch
import re

from openrsi import EvolvableInfo, Guard, GuardFinding, GuardReport, Item, MemoryOp, Patch, serve

DEFAULT_ALLOWED_IMPORTS = ("math", "re", "json", "itertools", "functools", "collections", "statistics", "string",
                           "fractions", "decimal", "operator", "heapq", "bisect", "datetime", "typing", "dataclasses")
DANGEROUS_CALLS = ("eval", "exec", "compile", "open", "__import__", "input", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "memoryview")
SECRET_PATTERNS = (
    r"sk-ant-[A-Za-z0-9_\-]{10,}", r"\bsk-[A-Za-z0-9]{20,}", r"\bAKIA[0-9A-Z]{16}\b", r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"(?i)\b(password|passwd|secret)\s*[:=]\s*\S{6,}",
)
TAMPERING_PATTERNS = (
    r"(?i)\b(edit|modify|change|delete|rewrite|skip|disable)\b.{0,30}"
    r"\b(tests?|test files?|assertions?|rubric|judge|grader|evaluator)\b",
    r"(?i)\b(tell|convince|persuade|flatter)\b.{0,20}\b(the )?(judge|grader|evaluator)\b",
    r"(修改|改掉|删除|跳过|绕过).{0,10}(测试|断言|评分|判别器)",
)
INJECTION_PATTERNS = (r"(?i)ignore (all )?(previous|prior|above) instructions", r"(?i)disregard (the )?system prompt",
                      r"忽略(之前|以上|上面)的(所有)?(指令|要求)")


class PolicyGuard(Guard):
    name = "openrsi/policy-guard"
    version = "0.3.0"

    def __init__(self, protected: list[str] | None = None, max_ops: int = 20, max_removals: int = 5,
                 max_text_chars: int = 4000, allowed_imports: list[str] | None = None,
                 extra_secret_patterns: list[str] | None = None, check_tampering: bool = True):
        self.protected = list(protected or [])
        self.max_ops = max_ops
        self.max_removals = max_removals
        self.max_text_chars = max_text_chars
        self.allowed_imports = set(allowed_imports if allowed_imports is not None else DEFAULT_ALLOWED_IMPORTS)
        self.secret_patterns = [re.compile(p) for p in (*SECRET_PATTERNS, *(extra_secret_patterns or []))]
        self.tampering = [re.compile(p) for p in TAMPERING_PATTERNS] if check_tampering else []
        self.injection = [re.compile(p) for p in INJECTION_PATTERNS]

    def inspect(self, patch: Patch, target: EvolvableInfo | None = None) -> GuardReport:
        findings: list[GuardFinding] = []

        def block(rule, msg, i=None, item_id=None):
            findings.append(GuardFinding(rule=rule, severity="block", message=msg, op_index=i, item_id=item_id))

        def warn(rule, msg, i=None, item_id=None):
            findings.append(GuardFinding(rule=rule, severity="warn", message=msg, op_index=i, item_id=item_id))

        if len(patch.ops) > self.max_ops:
            block("too_many_ops", f"一次改动 {len(patch.ops)} 项，上限 {self.max_ops}")

        removals = 0
        for i, raw in enumerate(patch.ops):
            try:
                op = MemoryOp.model_validate(raw)
            except ValueError:
                continue  # 非条目型可进化体的 ops：只做文本类检查
            item_id = op.item.id if op.item else op.item_id
            if op.op == "remove":
                removals += 1
            if item_id and any(fnmatch.fnmatch(item_id, g) for g in self.protected):
                block("protected", f"条目 {item_id} 受保护，不允许 {op.op}", i, item_id)
            if op.item is not None:
                self._check_item(op.item, i, block, warn)

        if removals > self.max_removals:
            block("too_many_removals", f"一次删除 {removals} 项，上限 {self.max_removals}")

        return GuardReport(allowed=not any(f.severity == "block" for f in findings), findings=findings,
                           guard_ref=self.ref)

    def _check_item(self, item: Item, i: int, block, warn) -> None:
        text = item.text or ""
        blob = " ".join([text, *(str(v) for v in item.data.values() if isinstance(v, str))])
        if len(text) > self.max_text_chars:
            block("text_too_long", f"{len(text)} 字符，上限 {self.max_text_chars}", i, item.id)
        for p in self.secret_patterns:
            if p.search(blob):
                block("secret", "疑似包含密钥或密码", i, item.id)
                break
        for p in self.tampering:
            if p.search(text):
                block("eval_tampering", f"疑似教 agent 绕过评估：{p.search(text).group(0)!r}", i, item.id)
                break
        for p in self.injection:
            if p.search(blob):
                warn("injection", "疑似提示注入", i, item.id)
                break
        code = item.data.get("code")
        if isinstance(code, str):
            for rule, msg in self.check_code(code):
                block(rule, msg, i, item.id)

    def check_code(self, code: str) -> list[tuple[str, str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [("code_syntax", f"语法错误：{e.msg}（第 {e.lineno} 行）")]
        out = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mods:
                    if m.split(".")[0] not in self.allowed_imports:
                        out.append(("code_import", f"不允许 import {m}"))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_CALLS:
                out.append(("code_call", f"不允许调用 {node.func.id}()"))
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
                out.append(("code_dunder", f"不允许访问 .{node.attr}"))
            elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__loader__", "__spec__"):
                out.append(("code_dunder", f"不允许访问 {node.id}"))
        return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-guard")
    ap.add_argument("--protect", action="append", default=[], help="受保护条目 id 的 glob，可重复")
    args = ap.parse_args()
    serve(PolicyGuard(protected=args.protect))


__all__ = ["PolicyGuard", "main"]
