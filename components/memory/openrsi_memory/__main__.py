"""以 MCP server 方式运行可进化 memory。

    python -m openrsi_memory --path ./memory-store
"""
from __future__ import annotations

import argparse

from openrsi import serve

from .memory import MemoryEvolvable


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="openrsi-memory")
    ap.add_argument("--path", help="持久化目录；不给则只在内存里")
    ap.add_argument("--target", default="memory", help="补丁里的 target 名")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    args = ap.parse_args(argv)
    serve(MemoryEvolvable(target=args.target, path=args.path), args.transport)


if __name__ == "__main__":
    main()
