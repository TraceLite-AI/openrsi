"""权重测试用的 agent：同一道题多答几次时会随机答错，用来产生同题的对错配对。"""
from __future__ import annotations

import random

from secret_ops import OPS, RuleAgent


class NoisyRuleAgent(RuleAgent):
    """在 RuleAgent 的基础上，按固定种子随机把一部分答案改错（模拟采样温度 > 0）。"""

    name, version = "test/noisy-rule-agent", "0.1.0"

    def __init__(self, error_rate: float = 0.5, seed: int = 0):
        self.rng = random.Random(seed)
        self.error_rate = error_rate

    def run(self, task, context, snapshot_id=None):
        traj = super().run(task, context, snapshot_id)
        if self.rng.random() < self.error_rate:
            x = task.input
            wrong = next(f(x["a"], x["b"]) for f in OPS.values() if f(x["a"], x["b"]) != traj.output)
            traj.output = wrong
        return traj
