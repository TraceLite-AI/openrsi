"""P5：RSIHub 适配器的严格模式（two-loop 设计 §8A）——逐个负向探针。

真实的 RSIHub 运行需要它自己的环境、模型和预算，这里不跑；算子脚本依赖 RSIHub 的 evolve.frozen，
也只能在 RSIHub 的 venv 里加载。这些测试直接调用算子背后的逻辑（core.py），也就是算子实际执行的代码。
"""
import subprocess
from pathlib import Path

import pytest

import openrsi_rsihub as rsihub


def row(genid, tasks, *, parent=None, h="H", score=None, incomplete=()):
    vec = {t: {"trials": [{"trial": 0, "status": "benchmark_complete", "reward": r}]} for t, r in tasks.items()}
    for t in incomplete:
        vec[t] = {"trials": [{"trial": 0, "status": "error", "reward": None}]}
    s = score if score is not None else (sum(tasks.values()) / len(tasks) if tasks else 0.0)
    return {"genid": genid, "parent": parent, "score": s, "task_set_hash": h,
            "task_vector": {"schema_version": 1, "tasks": vec}}


PARENT = row(0, {"a": 0.0, "b": 1.0, "c": 0.0})


def strict(child, parent=PARENT, **kw):
    return rsihub.gate_decide(child, parent, strict=True, **{"max_task_regressions": 0, **kw})


def test_strict_accepts_a_clean_improvement():
    ok, why = strict(row(1, {"a": 1.0, "b": 1.0, "c": 0.0}, parent=0))
    assert ok, why


@pytest.mark.parametrize("child,parent,match", [
    (row(1, {"a": 1.0, "b": 1.0}, parent=0, score=0.9), PARENT, "候选缺 \\['c'\\]"),           # 漏题
    (row(1, {"a": 1.0, "b": 1.0, "c": 0.0, "d": 1.0}, parent=0), PARENT, "多出 \\['d'\\]"),    # 多题
    (row(1, {"a": 1.0, "b": 1.0}, parent=0, incomplete=("c",), score=0.9), PARENT, "评估不完整 \\['c'\\]"),
    (row(1, {"a": 1.0, "b": 1.0, "c": 0.0}, parent=0, h=""), PARENT, "相同的任务集哈希"),
    (row(1, {"a": 1.0, "b": 1.0, "c": 0.0}, parent=0, h="X"), PARENT, "任务集不同"),
    (row(1, {"a": 1.0, "b": 1.0, "c": 0.0}, parent=0), {**PARENT, "_matched_from_evals": True}, "缺少可比的逐题分数"),
    ({**row(1, {}, parent=0, score=0.9), "task_vector": None}, PARENT, "缺少可比的逐题分数"),
])
def test_strict_rejects_what_the_default_would_let_through(child, parent, match):
    ok, why = strict(child, parent)
    assert not ok and __import__("re").search(match, why), why


def test_default_mode_still_falls_back_and_strict_needs_explicit_limit():
    matched = {**PARENT, "_matched_from_evals": True}
    ok, why = rsihub.gate_decide(row(1, {"a": 1.0, "b": 1.0, "c": 0.0}, parent=0), matched)
    assert ok and "只比总分" in why  # 默认行为不变
    ok, why = rsihub.gate_decide(row(1, {"a": 1.0, "b": 1.0, "c": 0.0}, parent=0), PARENT, strict=True,
                                 max_task_regressions=None)
    assert not ok and "显式设置" in why


def test_strict_still_enforces_no_regression():
    ok, why = strict(row(1, {"a": 1.0, "b": 0.0, "c": 1.0}, parent=0))  # 总分涨了，但 b 变差
    assert not ok and "1 道题变差" in why


# ---------------------------------------------------------------- 护栏

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    files = {"target/agent.py": "def run():\n    return 1\n", "target/prompt.md": "be careful\n",
             "evaluator/score.py": "def score(x):\n    return x\n", "operators/gate/no_reg.py": "ACCEPT = False\n"}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run([*git, "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "parent"], cwd=tmp_path, check=True)
    return tmp_path


def guard(ws, **kw):
    return rsihub.guard_checkout(ws, **{"strict": True, "required_files": ["target/agent.py"], **kw})


def test_strict_guard_accepts_a_normal_edit(workspace):
    (workspace / "target/prompt.md").write_text("be very careful\n", encoding="utf-8")
    ok, why, _ = guard(workspace)
    assert ok, why


@pytest.mark.parametrize("rel,text", [
    ("evaluator/score.py", "def score(x):\n    return 1.0\n"),   # 改评估器刷分
    ("operators/gate/no_reg.py", "ACCEPT = True\n"),             # 篡改 gate 算子
])
def test_strict_guard_protects_evaluator_and_operators_even_if_config_forgets(workspace, rel, text):
    (workspace / rel).write_text(text, encoding="utf-8")
    ok, why, _ = guard(workspace, protected_paths=["target/secret/**"])  # 配置里漏了 evaluator / operators
    assert not ok and rel in why
    loose, _, _ = rsihub.guard_checkout(workspace, protected_paths=["target/secret/**"])
    assert loose  # 非严格模式按配置来，会放过：这正是要严格模式的原因


def test_strict_guard_requires_files_and_compiling_code(workspace):
    (workspace / "target/agent.py").unlink()
    (workspace / "target/helper.py").write_text("x = 1\n", encoding="utf-8")
    ok, why, _ = guard(workspace)
    assert not ok and "必需文件缺失" in why
    (workspace / "target/agent.py").write_text("def run(:\n", encoding="utf-8")
    ok, why, _ = guard(workspace)
    assert not ok and "compile" in why and "target/agent.py" in why
