"""OpenRSI × RSIHub：把 OpenRSI 的组件装成 RSIHub 的算子。

RSIHub（simple-agent-lab/RSIHub）的算子是 library/<阶段>/<名字>.py 脚本，用 evolve.frozen.sdk.main 运行。
本包提供三个：

- validate/openrsi_guard          PolicyGuard 检查候选改动（默认保护 evaluator/** 和 operators/**）
- gate/openrsi_no_regression      总分提升且逐题不回退（HoldoutGate 的 no_regression）
- select/openrsi_shinka           ShinkaEvolve 式父代选择

安装：openrsi-rsihub install /path/to/RSIHub，然后在 RSIHub 的配方里引用这些算子。
最稳妥的做法是把 openrsi-sdk、openrsi-gate、openrsi-guard、openrsi-rsihub 装进 RSIHub 自己的 venv；
install 也会把当前环境的包路径写进脚本，作为找不到时的后备。

另外 import_archive 把 RSIHub 的 archive.jsonl 转成 OpenRSI 谱系，方便和其他运行放在一起看。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .core import (DEFAULT_PROTECTED, checkout_ops, gate_decide, guard_checkout, import_archive, shinka_pick,
                   task_scores)

LIBRARY = Path(__file__).parent / "library"
OPERATORS = {"validate": ["openrsi_guard"], "gate": ["openrsi_no_regression"], "select": ["openrsi_shinka"]}
_MARKER = "_OPENRSI_PATHS: list[str] = []  # __OPENRSI_PATHS__"


def package_paths() -> list[str]:
    """OpenRSI 及其依赖所在的目录，写进算子脚本作为 sys.path 后备。"""
    out: list[str] = []
    for mod in ("openrsi_rsihub", "openrsi", "openrsi_gate", "openrsi_guard", "pydantic", "pydantic_core",
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
        text = text.replace(_MARKER, f"_OPENRSI_PATHS: list[str] = {paths!r}")
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

    ap = argparse.ArgumentParser(prog="openrsi-rsihub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("install", help="把 OpenRSI 算子复制进 RSIHub 的 library/")
    i.add_argument("rsihub_root")
    i.add_argument("--no-embed-paths", action="store_true", help="不写入当前环境的包路径")
    i.add_argument("--overwrite", action="store_true")
    a = sub.add_parser("import-archive", help="把 archive.jsonl 写进 OpenRSI 谱系")
    a.add_argument("archive")
    a.add_argument("--lineage", required=True, help="LineageStore 的 JSONL 路径")
    a.add_argument("--target", default="harness")
    args = ap.parse_args()
    if args.cmd == "install":
        for p in install(args.rsihub_root, embed_paths=not args.no_embed_paths, overwrite=args.overwrite):
            print(p)
    else:
        from openrsi_lineage import LineageStore

        store = LineageStore(args.lineage)
        entries = import_archive(args.archive, target=args.target)
        for e in entries:
            store.record(e)
        print(f"导入 {len(entries)} 条")


__all__ = ["DEFAULT_PROTECTED", "LIBRARY", "OPERATORS", "checkout_ops", "gate_decide", "guard_checkout",
           "import_archive", "install", "main", "package_paths", "render_operator", "shinka_pick", "task_scores"]
