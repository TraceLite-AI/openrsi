"""Gate 算子：总分要提升，并且逐题不能回退（Self-Harness / AHE 式的不回退准入）。

逐题分数取自 task_vector，规则和 RSIHub 的 select/pareto 一致。任务集哈希不同时直接拒绝。
"""
from __future__ import annotations

import sys

# install() 会把下面的路径换成安装 openrsi 的 site-packages，这样算子在 RSIHub 自己的 venv 里也能找到 OpenRSI。
_OPENRSI_PATHS: list[str] = []  # __OPENRSI_PATHS__
for _p in _OPENRSI_PATHS:
    if _p not in sys.path:
        sys.path.append(_p)

from evolve.frozen import sdk  # noqa: E402
from evolve.frozen.config import Config, boolean, integer, number  # noqa: E402
from evolve.frozen.interfaces import GateOperator, GateResult, OperatorContext, Row  # noqa: E402

CONFIG = Config({
    "max_task_regressions": integer(default=0, minimum=0, description="最多允许几道题变差"),
    "task_tolerance": number(default=0.0, minimum=0, description="单题分数下降多少以内不算变差"),
    "min_gain": number(default=0.0, minimum=0, description="总分至少提升多少"),
    "allow_equal": boolean(default=False, description="总分持平也接受（仍要求不回退）"),
    "require_task_scores": boolean(default=False, description="没有可比的逐题分数时拒绝（默认退回只比总分）"),
})


class OpenRSINoRegressionGate(GateOperator):
    def decide(self, child: Row, parent: Row | None, ctx: OperatorContext) -> GateResult:
        from openrsi_rsihub.core import gate_decide

        accept, reason = gate_decide(
            child, parent, max_task_regressions=int(ctx.config.get("max_task_regressions", 0)),
            task_tolerance=float(ctx.config.get("task_tolerance", 0.0)),
            min_gain=float(ctx.config.get("min_gain", 0.0)), allow_equal=bool(ctx.config.get("allow_equal", False)),
            require_task_scores=bool(ctx.config.get("require_task_scores", False)))
        return GateResult(decision="accept" if accept else "reject", reason=reason)


if __name__ == "__main__":
    sdk.main(OpenRSINoRegressionGate, config_schema=CONFIG)
