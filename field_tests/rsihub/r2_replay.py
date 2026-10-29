"""R2（不花钱）：拿一个已经跑过的 RSIHub workspace，用 OpenRSI 的逻辑离线重放。

- 逐题分数：多少行能算出逐题分数，任务集哈希是否一致
- 准入：对每个有父代的候选，OpenRSI 的不回退准入怎么判，和 RSIHub 当时的判定对比
- 护栏：把每个候选相对父代的改动放进临时 worktree，用 PolicyGuard 检查；再人为改一下评估器，确认会被拦
- 选择：Shinka 选父代的分布
- 谱系：archive.jsonl 导入成 OpenRSI 谱系

用法：python field_tests/rsihub/r2_replay.py /path/to/RSIHub/runs/hill_climb-demo
"""
import argparse
import collections
import json
import random
import subprocess
import tempfile
from pathlib import Path

from openrsi_lineage import LineageStore
from openrsi_rsihub import gate_decide, guard_checkout, import_archive, shinka_pick, task_scores

OUT = Path(__file__).resolve().parent.parent / "out" / "rsihub"


def git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def tag_of(row: dict) -> str:
    return row.get("tag") or f"gen/{row['genid']}"


def guard_diff(ws: Path, parent_ref: str, child_ref: str, **kw) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "wt"
        git(ws, "worktree", "add", "--detach", str(wt), parent_ref)
        try:
            diff = git(ws, "diff", "--binary", parent_ref, child_ref)
            if diff.strip():
                subprocess.run(["git", "-C", str(wt), "apply", "--whitespace=nowarn", "-"], input=diff, text=True,
                               check=True, capture_output=True)
            ok, reason, _ = guard_checkout(wt, **kw)
            return ok, reason
        finally:
            git(ws, "worktree", "remove", "--force", str(wt), check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("--max-guard", type=int, default=50, help="最多检查多少个候选的改动")
    args = ap.parse_args()
    ws = Path(args.workspace).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ws / "archive.jsonl").read_text().splitlines() if x.strip()]
    by_id = {str(r["genid"]): r for r in rows}
    report: dict = {"workspace": str(ws), "rows": len(rows), "top_level_dirs": sorted(
        p.name for p in ws.iterdir() if p.is_dir() and not p.name.startswith("."))}

    # 逐题分数
    counts = [len(task_scores(r)) for r in rows]
    report["task_scores"] = {"rows_with_scores": sum(1 for c in counts if c), "tasks_per_row": counts,
                             "task_set_hashes": sorted({str(r.get("task_set_hash")) for r in rows})}

    # 准入对比
    gate_rows, agree = [], collections.Counter()
    for r in rows:
        if r.get("parent") is None or str(r["parent"]) not in by_id:
            continue
        parent = by_id[str(r["parent"])]
        ours, why = gate_decide(r, parent, max_task_regressions=0, allow_equal=True)
        theirs = r.get("verdict") == "keep" or r.get("valid_parent") is True
        agree[(ours, theirs)] += 1
        gate_rows.append({"genid": r["genid"], "parent": r["parent"], "score": r.get("score"),
                          "parent_score": parent.get("score"), "rsihub": r.get("verdict"), "openrsi": ours,
                          "reason": why})
    report["gate"] = {"pairs": {f"openrsi={a},rsihub={b}": n for (a, b), n in agree.items()}, "rows": gate_rows}

    # 护栏
    guard_rows = []
    for r in [r for r in rows if r.get("parent") is not None][: args.max_guard]:
        p = by_id.get(str(r["parent"]))
        if p is None:
            continue
        try:
            ok, reason = guard_diff(ws, tag_of(p), tag_of(r))
        except Exception as e:  # noqa: BLE001
            ok, reason = None, f"error: {e}"
        guard_rows.append({"genid": r["genid"], "allowed": ok, "reason": reason})
    report["guard"] = guard_rows

    # 人为篡改评估器：应当被拦
    root = next((r for r in rows if r.get("parent") is None), rows[0])
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "wt"
        git(ws, "worktree", "add", "--detach", str(wt), tag_of(root))
        try:
            ev = wt / "evaluator"
            target = next((f for f in sorted(ev.rglob("*")) if f.is_file()), None) if ev.is_dir() else None
            if target is None:
                ev.mkdir(exist_ok=True)
                target = ev / "openrsi_probe.py"
            with open(target, "a", encoding="utf-8") as f:
                f.write("\n# openrsi tamper probe: always pass\n")
            ok, reason, _ = guard_checkout(wt)
            report["tamper_probe"] = {"file": str(target.relative_to(wt)), "blocked": not ok, "reason": reason}
        finally:
            git(ws, "worktree", "remove", "--force", str(wt), check=False)

    # 选择
    parents = [r for r in rows if r.get("valid_parent") is True and isinstance(r.get("score"), (int, float))]
    if parents:
        picks = collections.Counter(shinka_pick(parents, rows, random.Random(0), k=1000))
        report["shinka_picks"] = {str(k): v for k, v in picks.most_common()}
        report["best_by_score"] = str(max(parents, key=lambda r: r["score"])["genid"])

    # 谱系
    lineage = LineageStore(OUT / "lineage.jsonl")
    for e in import_archive(ws / "archive.jsonl"):
        lineage.record(e)
    best = lineage.best()
    report["lineage_best"] = best.id if best else None

    (OUT / "r2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("gate", "guard")}, ensure_ascii=False, indent=2,
                     default=str))
    print("准入对比：", report["gate"]["pairs"])
    print("护栏：", collections.Counter(str(g["allowed"]) for g in guard_rows))
    ok = (report["task_scores"]["rows_with_scores"] > 0 and report["tamper_probe"]["blocked"]
          and all(g["allowed"] is not None for g in guard_rows))
    print("PASS R2" if ok else "FAIL R2")
    print("把 field_tests/out/rsihub/r2.json 发回来（准入不一致是预期的：我们要求逐题不回退，比 hillclimb 严）")


if __name__ == "__main__":
    main()
