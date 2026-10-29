"""在 Terminal-Bench 2 全部 89 题上评一个候选（RSIHub README 里的 "Full Benchmark Score"）。

做法：把候选在 gen/N 的 target/ 导出成一个独立 git 仓库，生成一个只评测、不进化的配方
（评测器换成全量 89 题，其余和对照配方相同），新建 workspace 只做第 0 代评估。

用法：
  python full_eval.py /path/to/RSIHub /path/to/workspace best      # 这组里训练分最高的候选
  python full_eval.py /path/to/RSIHub /path/to/workspace 0         # 原始 agent
  加 --dry-run 只生成配方、打印命令
"""
import argparse
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def sh(cmd: list[str], cwd: Path | None = None, **kw) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise SystemExit(f"命令失败：{' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r.stdout


def best_genid(ws: Path) -> str:
    rows = [json.loads(x) for x in (ws / "archive.jsonl").read_text().splitlines() if x.strip()]
    ok = [r for r in rows if r.get("valid_parent") is True and isinstance(r.get("score"), (int, float))]
    return str(max(ok, key=lambda r: (r["score"], -int(str(r["genid"])) if str(r["genid"]).isdigit() else 0))["genid"])


def export_target(ws: Path, genid: str, dest: Path) -> str:
    dest.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "t.tar"
        sh(["git", "-C", str(ws), "archive", "-o", str(tar), f"gen/{genid}", "target"])
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.name == "target":
                    continue
                m.name = m.name[len("target/"):]
                if hasattr(tarfile, "data_filter"):
                    t.extract(m, dest, filter="data")
                else:
                    t.extract(m, dest)
    for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=openrsi", "-c", "user.email=openrsi@localhost",
                                                 "commit", "-qm", f"export gen/{genid} from {ws.name}"]):
        sh(["git", "-C", str(dest), *args])
    return sh(["git", "-C", str(dest), "rev-parse", "HEAD"]).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rsihub")
    ap.add_argument("workspace")
    ap.add_argument("genid", help="代号，或 best")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rsihub, ws = Path(args.rsihub).resolve(), Path(args.workspace).resolve()
    genid = best_genid(ws) if args.genid == "best" else args.genid
    name = f"{ws.name}-full-gen{genid}"
    out = HERE / "out" / "full"
    repo = out / f"{name}-target"
    if repo.exists():
        raise SystemExit(f"{repo} 已存在")
    sha = export_target(ws, genid, repo)

    arm = ws.name.removeprefix("openrsi-parity-")
    source = rsihub / "recipes" / f"openrsi_parity_{arm}" / "evolve.yaml"
    recipe = yaml.safe_load((source if source.exists() else ws / "evolve.yaml").read_text())
    recipe["experiment"].update({"id": name, "max_generations": 0})
    recipe["target"].update({"seed": repo.as_uri(), "revision": sha})
    ev = recipe["evaluator"]
    for key in ("split", "anchor"):
        ev.pop(key, None)
    ev.update({"task_scope": "full", "evaluation_split": "train", "tasks_per_round": 89})
    recipe_path = out / f"{name}.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))

    full_ws = rsihub / "runs" / name
    asset = Path(os.environ.get("EVOLVE_ASSET_DIR", rsihub / ".evolve-assets" / "terminal-bench-2.0"))
    env_file = Path(os.environ.get("ENV_FILE", rsihub / ".env"))
    u = ["uv", "run", "--frozen", *(["--env-file", str(env_file)] if env_file.exists() else [])]
    cmds = [
        [*u, "evolve", "init", str(full_ws), "--recipe-path", str(recipe_path), "--dataset",
         str(asset / "raw" / "terminal-bench")],
        [*u, str(full_ws / "evolve"), "preflight", str(full_ws)],
        [*u, str(full_ws / "evolve"), "run", str(full_ws), "--max-generations", "0", "--children-per-gen", "1"],
        [*u, str(full_ws / "evolve"), "report", str(full_ws)],
    ]
    print(f"候选 gen/{genid}（{ws.name}）→ {repo}（{sha[:12]}）\n配方 {recipe_path}\nworkspace {full_ws}")
    for c in cmds:
        print("$", " ".join(c))
        if not args.dry_run:
            r = subprocess.run(c, cwd=rsihub)
            if r.returncode != 0:
                raise SystemExit(f"失败（退出码 {r.returncode}）。如果是 --max-generations 0 不被接受，"
                                 f"改用：{' '.join(u)} {full_ws}/evolve eval {full_ws} 0")
    (out / f"{name}.json").write_text(json.dumps({"source_workspace": str(ws), "genid": genid, "target_sha": sha,
                                                  "full_workspace": str(full_ws)}, indent=2))


if __name__ == "__main__":
    main()
