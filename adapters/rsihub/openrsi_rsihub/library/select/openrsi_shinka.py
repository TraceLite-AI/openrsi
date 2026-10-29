"""Select 算子：ShinkaEvolve 式父代选择，兼顾高分和还没被充分探索的版本。

权重 = sigmoid(λ·(分数 − 中位数)/离散度) × 1/(1+已有子代数)。
"""
from __future__ import annotations

import sys

# install() 会把下面的路径换成安装 openrsi 的 site-packages，这样算子在 RSIHub 自己的 venv 里也能找到 OpenRSI。
_OPENRSI_PATHS: list[str] = []  # __OPENRSI_PATHS__
for _p in _OPENRSI_PATHS:
    if _p not in sys.path:
        sys.path.append(_p)

from evolve.frozen import sdk  # noqa: E402
from evolve.frozen.config import Config, number  # noqa: E402
from evolve.frozen.interfaces import ArchiveView, OperatorContext, SelectOperator, SelectResult  # noqa: E402

CONFIG = Config({
    "selection_lambda": number(default=10.0, minimum=0, description="越大越偏向高分父代"),
})


class OpenRSIShinkaSelect(SelectOperator):
    def pick(self, archive: ArchiveView, ctx: OperatorContext) -> SelectResult:
        from openrsi_rsihub.core import shinka_pick

        parents = archive.valid_parents()
        if not parents:
            raise SystemExit("openrsi_shinka: 没有可用父代")
        chosen = shinka_pick(parents, archive.rows(), ctx.rng, k=ctx.fan_out,
                             selection_lambda=float(ctx.config.get("selection_lambda", 10.0)))
        return SelectResult(parents=chosen)


if __name__ == "__main__":
    sdk.main(OpenRSIShinkaSelect, config_schema=CONFIG)
