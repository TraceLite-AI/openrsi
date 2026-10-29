"""RSIHub 算子背后的纯逻辑，不依赖 RSIHub 本身，可以单独测试。

RSIHub 的约定（按 simple-agent-lab/RSIHub 源码整理）：
- 归档 archive.jsonl 每行一个候选：genid、parent、tag（gen/N）、score、valid_parent、task_vector、task_set_hash 等
- task_vector = {"schema_version": 1, "tasks": {task_id: {"trials": [{"status", "reward", ...}]}}}
  每道题的分数是全部 trial 奖励的均值，前提是每个 trial 都是 benchmark_complete 且奖励是数字
- validate / novelty 阶段，checkout 是以父代为 HEAD 的 git 工作区，候选改动尚未提交
- 工作区分 target/（被进化的 agent）、operators/、evaluator/ 三棵树
"""
from __future__ import annotations

import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable

from openrsi import GateRequest, LineageEntry, Patch

DEFAULT_PROTECTED = ["evaluator/**", "operators/**"]


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def task_scores(row: dict[str, Any]) -> dict[str, float]:
    """和 RSIHub 的 select/pareto 一致：只有全部 trial 都完成且有数字奖励的题才计分。"""
    vector = row.get("task_vector")
    if not isinstance(vector, dict) or not isinstance(vector.get("tasks"), dict):
        return {}
    out: dict[str, float] = {}
    for task_id, task in vector["tasks"].items():
        trials = task.get("trials") if isinstance(task, dict) else None
        if not trials:
            continue
        values = [float(t["reward"]) for t in trials
                  if t.get("status") == "benchmark_complete" and _num(t.get("reward"))]
        if len(values) == len(trials):
            out[str(task_id)] = sum(values) / len(values)
    return out


# ---------------------------------------------------------------- gate


def gate_decide(child: dict[str, Any], parent: dict[str, Any] | None, *, max_task_regressions: int | None = 0,
                task_tolerance: float = 0.0, min_gain: float = 0.0, allow_equal: bool = False,
                ) -> tuple[bool, str]:
    """用 OpenRSI 的 HoldoutGate 判定：总分提升，且逐题回退不超过上限（Self-Harness / AHE 的不回退准入）。"""
    from openrsi_gate import HoldoutGate

    if parent is None or not _num(parent.get("score")):
        return False, "父代没有分数"
    if not _num(child.get("score")):
        return False, "候选没有分数"
    ch, ph = child.get("task_set_hash"), parent.get("task_set_hash")
    if ch and ph and ch != ph:
        return False, f"任务集不同（{ch} vs {ph}），不能逐题比较"
    gate = HoldoutGate(min_gain=-1e-12 if allow_equal and min_gain == 0 else min_gain,
                       max_task_regressions=max_task_regressions, task_tolerance=task_tolerance)
    d = gate.check(GateRequest(candidate_id=str(child.get("genid")),
                               baseline={"holdout": float(parent["score"])},
                               candidate={"holdout": float(child["score"])},
                               baseline_tasks=task_scores(parent), candidate_tasks=task_scores(child)))
    return d.accepted, "; ".join(f"{c.name}: {c.detail}" for c in d.checks) or d.reason


# ---------------------------------------------------------------- select


def shinka_pick(parents: list[dict[str, Any]], rows: Iterable[dict[str, Any]], rng: random.Random, k: int = 1,
                selection_lambda: float = 10.0) -> list[str]:
    """ShinkaEvolve 的父代选择：sigmoid(λ·(分数−中位数)/离散度) × 1/(1+子代数)。"""
    scored = [p for p in parents if _num(p.get("score"))]
    if not scored:
        raise ValueError("没有带分数的可用父代")
    children: dict[str, int] = {}
    for r in rows:
        if r.get("parent") is not None:
            children[str(r["parent"])] = children.get(str(r["parent"]), 0) + 1
    scores = [float(p["score"]) for p in scored]
    med = statistics.median(scores)
    spread = statistics.median([abs(s - med) for s in scores]) or 1e-6
    ids = [str(p["genid"]) for p in scored]
    weights = []
    for p in scored:
        z = max(-60.0, min(60.0, selection_lambda * (float(p["score"]) - med) / spread))
        weights.append((1 / (1 + math.exp(-z))) / (1 + children.get(str(p["genid"]), 0)))
    return rng.choices(ids, weights=weights, k=max(1, k))


# ---------------------------------------------------------------- validate


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败：{r.stderr.strip()[-300:]}")
    return r.stdout


def checkout_ops(checkout: Path, max_bytes: int = 200_000) -> list[dict[str, Any]]:
    """把 checkout 相对 HEAD 的未提交改动（含未跟踪文件）转成 OpenRSI 的文件补丁操作。"""
    checkout = Path(checkout)
    ops: list[dict[str, Any]] = []
    fields = _git(checkout, "status", "--porcelain", "-z", "--untracked-files=all").split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if not entry:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:  # -z 格式下，重命名/复制的原路径是下一个字段
            old = fields[i]
            i += 1
            if "R" in code:
                ops.append({"op": "delete", "path": old})
        if "D" in code:
            ops.append({"op": "delete", "path": path})
            continue
        f = checkout / path
        if not f.is_file():
            continue
        data = f.read_bytes()[:max_bytes]
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            content = ""
        ops.append({"op": "write", "path": path, "content": content})
    return ops


def guard_checkout(checkout: Path, *, protected_paths: list[str] | None = None, check_code: bool = False,
                   max_removals: int = 50) -> tuple[bool, str, list[dict[str, Any]]]:
    """用 OpenRSI 的 PolicyGuard 检查候选改动：受保护路径、密钥、提示注入、可选的代码检查。"""
    from openrsi_guard import PolicyGuard

    ops = checkout_ops(checkout)
    if not ops:
        return False, "候选没有任何改动", []
    guard = PolicyGuard(protected_paths=DEFAULT_PROTECTED if protected_paths is None else protected_paths,
                        check_harness_code=check_code, max_ops=max(1000, len(ops)), max_removals=max_removals)
    report = guard.inspect(Patch(target="harness", ops=ops))
    findings = [f.model_dump(mode="json") for f in report.findings]
    blocks = [f for f in report.findings if f.severity == "block"]
    if blocks:
        return False, "; ".join(f"{f.rule}: {f.message}"
                                + (f"（{f.item_id}）" if f.item_id and f.item_id not in f.message else "")
                                for f in blocks)[:800], findings
    warns = [f.rule for f in report.findings if f.severity == "warn"]
    return True, f"{len(ops)} 个文件改动通过检查" + (f"（警告：{', '.join(warns)}）" if warns else ""), findings


# ---------------------------------------------------------------- archive → lineage


def import_archive(path: str | Path, target: str = "harness") -> list[LineageEntry]:
    """把 RSIHub 的 archive.jsonl 转成 OpenRSI 谱系条目，可以写进 LineageStore 统一查看。"""
    import json

    entries = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        genid = str(row["genid"])
        metrics = {"holdout": float(row["score"])} if _num(row.get("score")) else {}
        if _num(row.get("cost")):
            metrics["cost"] = float(row["cost"])
        if row.get("parent") is None:
            status = "root"
        else:
            status = "accepted" if row.get("valid_parent") is True or row.get("verdict") == "keep" else "rejected"
        entries.append(LineageEntry(
            id=f"gen{genid}", target=target,
            parent_ids=[f"gen{row['parent']}"] if row.get("parent") is not None else [],
            generation=int(genid) if genid.isdigit() else 0, status=status,
            metrics=metrics, task_scores=task_scores(row)))
    return entries
