"""benchmarks/noise/noise.py：同一版本跑多遍考试题，报告翻转、部分得分、rubric 项、标签。"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "noise"))
sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))
import group_toy as gt  # noqa: E402
import noise  # noqa: E402


def test_noise_report_counts_flips(tmp_path):
    cfg = gt.write_config(tmp_path / "t", "A", 0, holdout=20)
    rep = asyncio.run(noise.measure(cfg, 3))
    assert rep["tasks"] == 20 and len(rep["holdout_per_run"]) == 3
    assert all(len(v) == 3 for v in rep["per_task"].values())
    # 玩具 agent 按答对率掷硬币、不带种子：同一道题三遍结果会不一样
    flipped = {k for k, v in rep["per_task"].items() if len({x["passed"] for x in v}) > 1}
    assert rep["flipped"] == sorted(flipped) and flipped
    assert rep["rubric_items"] == ["correct"] and rep["partial_score"] == []
