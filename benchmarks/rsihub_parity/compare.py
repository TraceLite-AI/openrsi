"""汇总对照实验：每组的训练分、全量分、接受了几代、护栏拦了几次、逐题回退、花费，并附上 RSIHub 公布的数字。

用法：python compare.py /path/to/RSIHub [--arms baseline,openrsi]
输出 out/compare.md 和 out/compare.json。
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(ws: Path) -> list[dict]:
    f = ws / "archive.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def task_scores(row: dict) -> dict:
    from openrsi_rsihub import task_scores as ts

    return ts(row)


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def report_fields(rsihub: Path, ws: Path) -> dict:
    try:
        out = subprocess.run(["uv", "run", "--frozen", str(ws / "evolve"), "report", str(ws)], cwd=rsihub,
                             capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return dict(re.findall(r"^([\w.:-]+): (.*)$", out, re.M))


def full_score(name: str) -> tuple[float | None, int]:
    """full_eval.py 生成的全量评估 workspace 里第 0 代的分数和题数。"""
    meta = HERE / "out" / "full" / f"{name}.json"
    if not meta.exists():
        return None, 0
    ws = Path(json.loads(meta.read_text())["full_workspace"])
    r = next((r for r in rows(ws) if str(r.get("genid")) == "0"), None)
    return (num(r.get("score")), len(task_scores(r))) if r else (None, 0)


def arm_summary(rsihub: Path, arm: str) -> dict:
    ws = rsihub / "runs" / f"openrsi-parity-{arm}"
    rs = rows(ws)
    if not rs:
        return {"arm": arm, "workspace": str(ws), "error": "没有 archive.jsonl"}
    by_id = {str(r["genid"]): r for r in rs}
    seed = by_id.get("0", {})
    valid = [r for r in rs if r.get("valid_parent") is True and num(r.get("score")) is not None]
    best = max(valid, key=lambda r: r["score"]) if valid else None
    gens = []
    for r in rs:
        if str(r["genid"]) == "0":
            continue
        run = ws / "runs" / f"gen-{r['genid']}"
        validate = {}
        for f in (run / "validate" / "result.json", run / "validate" / "openrsi_guard.json"):
            if f.exists():
                validate = json.loads(f.read_text())
                break
        regress = None
        p = by_id.get(str(r.get("parent")))
        if p and task_scores(p) and task_scores(r):
            pt, ct = task_scores(p), task_scores(r)
            regress = sum(1 for t, v in pt.items() if t in ct and ct[t] < v)
        gens.append({"genid": r["genid"], "parent": r.get("parent"), "score": num(r.get("score")),
                     "verdict": r.get("verdict"), "valid_parent": r.get("valid_parent"), "reason": r.get("reason"),
                     "validate_accept": validate.get("accept"), "validate_reason": validate.get("reason"),
                     "tasks_regressed_vs_parent": regress, "cost": num(r.get("cost"))})
    anchors = [e for r in rs for e in (r.get("evals") or []) if isinstance(e, dict) and e.get("kind") == "anchor"]
    rep = report_fields(rsihub, ws)
    seed_full, n_seed = full_score(f"openrsi-parity-{arm}-full-gen0")
    if seed_full is None:  # 各组的原始 agent 相同，只评一次（默认在 baseline 组）
        seed_full, n_seed = full_score("openrsi-parity-baseline-full-gen0")
    best_full, n_best = full_score(f"openrsi-parity-{arm}-full-gen{best['genid']}") if best else (None, 0)
    return {
        "arm": arm, "workspace": str(ws), "generations": len(gens),
        "seed_train": num(seed.get("score")), "best_genid": best["genid"] if best else None,
        "best_train": num(best["score"]) if best else None,
        "seed_full": seed_full, "best_full": best_full, "full_tasks": [n_seed, n_best],
        "accepted": sum(1 for g in gens if g["valid_parent"] is True),
        "blocked_by_validate": sum(1 for g in gens if g["validate_accept"] is False),
        "anchor": [{"genid": e.get("genid"), "score": num(e.get("score"))} for e in anchors],
        "budget_spent_usd": rep.get("budget_spent_usd"), "report": rep, "per_generation": gens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rsihub")
    ap.add_argument("--arms", default="baseline,openrsi")
    args = ap.parse_args()
    rsihub = Path(args.rsihub).resolve()
    ref = json.loads((HERE / "reference.json").read_text())["terminal-bench-2"]["MiniSWE"]["Hyperagents"]
    arms = [arm_summary(rsihub, a) for a in args.arms.split(",")]
    lines = ["| 组 | 训练分 seed → best | 全量分 seed → best | 接受 / 代数 | 校验拦下 | 花费 USD |",
             "|---|---|---|---|---|---|",
             f"| RSIHub 公布（Hyperagents, MiniSWE） | {ref['train'][0]}% → {ref['train'][1]}% | "
             f"{ref['full'][0]}% → {ref['full'][1]}% | — | — | — |"]
    for a in arms:
        if "error" in a:
            lines.append(f"| {a['arm']} | {a['error']} | | | | |")
            continue
        lines.append(f"| {a['arm']} | {pct(a['seed_train'])} → {pct(a['best_train'])}（gen {a['best_genid']}） | "
                     f"{pct(a['seed_full'])} → {pct(a['best_full'])} | {a['accepted']} / {a['generations']} | "
                     f"{a['blocked_by_validate']} | {a['budget_spent_usd'] or '—'} |")
    lines += ["", "逐代：", ""]
    for a in arms:
        for g in a.get("per_generation", []):
            lines.append(f"- {a['arm']} gen {g['genid']} ← {g['parent']}：{pct(g['score'])}，"
                         f"{'入档' if g['valid_parent'] else '未入档'}，回退 {g['tasks_regressed_vs_parent']} 题；"
                         f"{(g['reason'] or '')[:160]}")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    (out / "compare.json").write_text(json.dumps(arms, ensure_ascii=False, indent=2, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
