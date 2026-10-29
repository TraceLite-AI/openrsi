"""把任意 Python 组件类作为 MCP server 运行，不用为它写 main()。

    openrsi-serve openrsi_gate:HoldoutGate --with '{"min_gain": 0.01}'
    openrsi-serve my_pkg:MyJudge --path ./plugins --transport streamable-http --port 8765

openrsi.yaml 里把 python: 换成 command: 就能把一个位置拆到独立进程：
    gate: {command: [openrsi-serve, "openrsi_gate:HoldoutGate"]}
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from .server import serve


def load(target: str, kwargs: dict | None = None):
    module, _, name = target.partition(":")
    if not name:
        raise SystemExit(f"格式是 模块:类名，收到 {target!r}")
    return getattr(importlib.import_module(module), name)(**(kwargs or {}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="openrsi-serve", description=__doc__.split("\n")[0])
    ap.add_argument("target", help="模块:类名")
    ap.add_argument("--with", dest="with_", default="{}", help="构造参数（JSON 对象）")
    ap.add_argument("--path", action="append", default=[], help="加到 sys.path 的目录，可以给多次")
    ap.add_argument("--cwd", help="构造组件前切换到这个目录（相对路径参数以它为准）")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    for p in reversed(args.path):
        sys.path.insert(0, os.path.abspath(p))
    if args.cwd:
        os.chdir(args.cwd)
    component = load(args.target, json.loads(args.with_))
    if args.transport == "stdio":
        serve(component)
    else:
        serve(component, "streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
