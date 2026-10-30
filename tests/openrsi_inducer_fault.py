"""测试用：第 N 次 propose 时让进程直接退出，模拟组件崩溃（N 写在 flag 文件里，0 表示不崩）。"""
import os
from pathlib import Path

from secret_ops import RuleInducer


class FlakyInducer(RuleInducer):
    name, version = "test/flaky-inducer", "0.4.0"

    def __init__(self, flag: str):
        self.flag = Path(flag)
        self.calls = 0

    def propose(self, request):
        self.calls += 1
        if self.calls == int(self.flag.read_text().strip() or 0):
            os._exit(3)
        return super().propose(request)
