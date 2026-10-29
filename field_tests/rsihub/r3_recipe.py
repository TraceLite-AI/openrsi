"""R3（花 API 费用，需要 Docker 和 Terminal-Bench 数据）：在真实 RSIHub 配方里换上 OpenRSI 算子，跑 1～2 代。

从 RSIHub 的某个配方（默认 hill_climb）复制出 <配方>_openrsi：
  select   → openrsi_shinka
  gate     → openrsi_no_regression（逐题不回退，持平也接受，和 hillclimb 默认的 >= 对齐）
  validate → openrsi_guard（原配方有 validate 的会被替换，原来的写在报告里）

用法：
  python field_tests/rsihub/r3_recipe.py /path/to/RSIHub [--base hill_climb]
  然后按它打印的命令运行（先跑过 R1，并且按 RSIHub QUICKSTART 准备好数据和 .env）。
"""
import argparse
import shutil
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rsihub")
    ap.add_argument("--base", default="hill_climb")
    ap.add_argument("--generations", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.rsihub).resolve()
    src = root / "recipes" / args.base
    dst = root / "recipes" / f"{args.base}_openrsi"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    recipe = yaml.safe_load((dst / "evolve.yaml").read_text())
    ops = recipe["operators"]
    replaced = {k: ops.get(k) for k in ("select", "gate", "validate")}
    recipe["experiment"]["id"] = f"{recipe['experiment']['id']}_openrsi"
    ops["select"] = {"operator": "openrsi_shinka", "config": {"selection_lambda": 10.0}}
    ops["gate"] = {"operator": "openrsi_no_regression",
                   "config": {"max_task_regressions": 0, "allow_equal": True}}
    ops["validate"] = {"operator": "openrsi_guard", "timeout_s": 300, "config": {"check_code": False}}
    (dst / "evolve.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True))
    print(f"写好了 {dst / 'evolve.yaml'}；替换掉的原算子：{replaced}")
    ws = root / "runs" / f"{args.base}_openrsi-field"
    ds = "${EVOLVE_ASSET_DIR:-.evolve-assets/terminal-bench-2.0}/terminal-bench-2-30-v1"
    print(f"""
在 RSIHub 目录下运行（不要再执行 uv sync，它会删掉装进 RSIHub venv 的额外包；算子脚本里有后备路径）：

  cd {root}
  uv run --frozen evolve init {ws} --recipe-path {dst / 'evolve.yaml'} --dataset {ds}
  uv run --frozen {ws}/evolve preflight {ws}
  uv run --frozen {ws}/evolve run {ws} --max-generations {args.generations} --children-per-gen 1
  uv run --frozen {ws}/evolve status {ws}
  uv run --frozen {ws}/evolve verify {ws}

通过标准：
  - preflight 通过（RSIHub 接受了三个算子和它们的配置）
  - 每一代都有 runs/gen-N/validate/openrsi_guard.json 和 gate.json，gate.json 的 reason 里有 no_regression
  - archive.jsonl 里每一代都有 verdict；verify 通过
然后对这个 workspace 跑 R2：
  python field_tests/rsihub/r2_replay.py {ws}

失败时发回来：终端输出、{ws}/runs/ 下失败那一代的目录（至少 validate/、gate.json、各阶段的 stderr 日志）。
""")


if __name__ == "__main__":
    main()
