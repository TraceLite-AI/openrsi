"""无法评分：判分器自己没能正常运行（不是 agent 做错了）。

约定：判别器把某个必需（required）的 rubric 项标成 result = "error"，就表示这道题这次无法评分，比如
测试脚本还没开始就因为环境问题退出（Terminal-Bench 里装不上测试要的 Python）、LLM 判别器调用失败。
运行时把这种题当作缺失：不算分、不参与准入比较、不交给提案器；一批里太多（loop.max_unscorable）就停下报错，
而不是把它们当成普通的 0 分——那样整个实验都在测环境故障。
"""
from __future__ import annotations

from anyrsi import Verdict


class UnscorableError(RuntimeError):
    """一批题里太多无法评分，评测环境多半坏了。"""


def unscorable(verdict: Verdict) -> str | None:
    """无法评分时返回原因，否则 None。"""
    for r in verdict.rubric:
        if r.required and r.result == "error":
            return r.note or r.id
    return None


__all__ = ["UnscorableError", "unscorable"]
