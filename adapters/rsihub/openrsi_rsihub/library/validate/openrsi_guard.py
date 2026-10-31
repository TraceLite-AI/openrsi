"""Validate 算子：用 OpenRSI 的 PolicyGuard 检查候选改动，再决定值不值得评估。

拦截：改动 evaluator/ 或 operators/（默认受保护，防止候选改评估器刷分）、写入密钥、
可选的 Python 代码检查（危险 import、语法错误）。提示注入只记警告。
"""
from __future__ import annotations

import sys

# install() 会把下面的路径换成安装 openrsi 的 site-packages，这样算子在 RSIHub 自己的 venv 里也能找到 OpenRSI。
_OPENRSI_PATHS: list[str] = []  # __OPENRSI_PATHS__
for _p in _OPENRSI_PATHS:
    if _p not in sys.path:
        sys.path.append(_p)

from pathlib import Path  # noqa: E402

from evolve.frozen import sdk  # noqa: E402
from evolve.frozen.config import Config, array, boolean, integer, string  # noqa: E402
from evolve.frozen.interfaces import OperatorContext, ValidateOperator, ValidateResult  # noqa: E402

CONFIG = Config({
    "protected_paths": array(string(), description="不许候选修改的路径（glob）；默认 evaluator/** 和 operators/**"),
    "check_code": boolean(default=False, description="对写入的 .py 文件做 AST 检查"),
    "max_removals": integer(default=50, minimum=0, description="一次最多删除多少个文件"),
    "compile_globs": array(string(), description="要求能编译的 Python 文件（glob），如 target/**/*.py"),
    "required_files": array(string(), description="改动之后必须还在的文件"),
    "strict": boolean(default=False, description="受控对照实验用：evaluator/**、operators/** 总受保护，默认做编译检查"),
})


class OpenRSIGuardValidate(ValidateOperator):
    def validate(self, checkout: Path, ctx: OperatorContext) -> ValidateResult:
        import json

        from openrsi_rsihub.core import guard_checkout

        accept, reason, findings = guard_checkout(
            checkout, protected_paths=ctx.config.get("protected_paths"),
            check_code=bool(ctx.config.get("check_code", False)),
            max_removals=int(ctx.config.get("max_removals", 50)),
            compile_globs=ctx.config.get("compile_globs"), required_files=ctx.config.get("required_files"),
            strict=bool(ctx.config.get("strict", False)))
        out = ctx.run_dir / "validate" / "openrsi_guard.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"accept": accept, "reason": reason, "findings": findings},
                                  ensure_ascii=False, indent=2) + "\n")
        return ValidateResult(accept=accept, reason=reason, artifacts=[str(out.relative_to(ctx.run_dir))])


if __name__ == "__main__":
    sdk.main(OpenRSIGuardValidate, config_schema=CONFIG)
