"""AnyRSI × RSIHub：把 AnyRSI 的组件装成 RSIHub 的算子。

RSIHub（simple-agent-lab/RSIHub）的算子是 library/<阶段>/<名字>.py 脚本，用 evolve.frozen.sdk.main 运行。
本包提供三个：

- validate/anyrsi_guard          PolicyGuard 检查候选改动（默认保护 evaluator/** 和 operators/**）
- gate/anyrsi_no_regression      总分提升且逐题不回退（HoldoutGate 的 no_regression）
- select/anyrsi_shinka           ShinkaEvolve 式父代选择

安装：anyrsi-rsihub install /path/to/RSIHub，然后在 RSIHub 的配方里引用这些算子。
最稳妥的做法是把 anyrsi-sdk、anyrsi-gate、anyrsi-guard、anyrsi-rsihub 装进 RSIHub 自己的 venv；
install 也会把当前环境的包路径写进脚本，作为找不到时的后备。

另外 import_archive 把 RSIHub 的 archive.jsonl 转成 AnyRSI 谱系，方便和其他运行放在一起看。

反方向（0.4.2）：harbor 模块把 RSIHub 的题库搬进 AnyRSI 的循环——RSIHubTaskEnv 按 RSIHub 的规则划分
Harbor 数据集，HarborAgent 每道题调一次 `harbor run`，HarborRewardJudge 按验证脚本的奖励打分。
这样可以在同一个循环里既改 agent 代码、又训权重（见 examples/rsihub_weights）。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .harbor import (HarborAgent, HarborRewardJudge, PARITY_SPLIT, RSIHubTaskEnv, assign_splits,
                     discover_task_names)
from .core import (DEFAULT_PROTECTED, checkout_ops, compile_check, gate_decide, guard_checkout, import_archive, shinka_pick,
                   task_scores)

LIBRARY = Path(__file__).parent / "library"
OPERATORS = {"validate": ["anyrsi_guard"], "gate": ["anyrsi_no_regression"], "select": ["anyrsi_shinka"]}
_MARKER = "_ANYRSI_PATHS: list[str] = []  # __ANYRSI_PATHS__"


def package_paths() -> list[str]:
    """AnyRSI 及其依赖所在的目录，写进算子脚本作为 sys.path 后备。"""
    out: list[str] = []
    for mod in ("anyrsi_rsihub", "anyrsi", "anyrsi_gate", "anyrsi_guard", "pydantic", "pydantic_core",
                "typing_extensions", "annotated_types", "mcp", "anyio", "sniffio", "httpx", "httpcore", "idna",
                "certifi", "h11", "typing_inspection"):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            continue
        origin = Path(spec.origin)
        root = origin.parent.parent if origin.name == "__init__.py" else origin.parent
        if str(root) not in out:
            out.append(str(root))
    return out


def render_operator(stage: str, name: str, paths: list[str] | None = None) -> str:
    text = (LIBRARY / stage / f"{name}.py").read_text(encoding="utf-8")
    if paths:
        text = text.replace(_MARKER, f"_ANYRSI_PATHS: list[str] = {paths!r}")
    return text


def install(rsihub_root: str | Path, *, embed_paths: bool = True, overwrite: bool = False) -> list[Path]:
    root = Path(rsihub_root).resolve()
    if not (root / "library").is_dir():
        raise FileNotFoundError(f"{root} 下没有 library/，不像是 RSIHub 仓库")
    paths = package_paths() if embed_paths else None
    written = []
    for stage, names in OPERATORS.items():
        for name in names:
            dest = root / "library" / stage / f"{name}.py"
            if dest.exists() and not overwrite:
                raise FileExistsError(f"{dest} 已存在；加 --overwrite 覆盖")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(render_operator(stage, name, paths), encoding="utf-8")
            written.append(dest)
    return written


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="anyrsi-rsihub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install", help="把 AnyRSI 算子复制进 RSIHub 的 library/")
    i.add_argument("rsihub_root")
    i.add_argument("--no-embed-paths", action="store_true", help="不写入当前环境的包路径")
    i.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("import-archive", help="把 archive.jsonl 写进 AnyRSI 谱系")
    a.add_argument("archive")
    a.add_argument("--lineage", required=True, help="LineageStore 的 JSONL 路径")
    a.add_argument("--target", default="harness")
    s = sub.add_parser("split", help="按 RSIHub 的规则划分 Harbor 数据集，打印 train / gate / sealed 各有哪些题")
    s.add_argument("dataset")
    s.add_argument("--ratios", default=None, help='JSON，如 {"train": 0.6, "gate": 0.2, "sealed": 0.2}；默认对齐实验的划分')
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "split":
        import json

        ratios = json.loads(args.ratios) if args.ratios else PARITY_SPLIT
        result = assign_splits(discover_task_names(Path(args.dataset)), ratios, args.seed)
        print(json.dumps({"counts": {k: len(v) for k, v in result.items()}, **result}, ensure_ascii=False, indent=2))
    elif args.cmd == "install":
        for p in install(args.rsihub_root, embed_paths=not args.no_embed_paths, overwrite=args.overwrite):
            print(p)
    else:
        from anyrsi_lineage import LineageStore

        store = LineageStore(args.lineage)
        entries = import_archive(args.archive, target=args.target)
        for e in entries:
            store.record(e)
        print(f"导入 {len(entries)} 条")


__all__ = ["DEFAULT_PROTECTED", "HarborAgent", "HarborRewardJudge", "LIBRARY", "PARITY_SPLIT", "RSIHubTaskEnv",
           "assign_splits", "discover_task_names", "OPERATORS", "checkout_ops", "compile_check", "gate_decide", "guard_checkout",
           "import_archive", "install", "main", "package_paths", "render_operator", "shinka_pick", "task_scores"]
