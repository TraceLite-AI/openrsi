"""以 MCP server 方式运行取证 + 判别器。

    python -m openrsi_judge --rubric rubric.yaml --protect 'tests/**' [--workspace .]
"""
from __future__ import annotations

import argparse

from openrsi import serve

from .judge import EvidenceJudge


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="openrsi-judge")
    ap.add_argument("--rubric", required=True, help="rubric YAML 文件")
    ap.add_argument("--protect", action="append", default=[], help="受保护路径的 glob，可重复")
    ap.add_argument("--workspace", help="工作目录；给了就对受保护文件计算哈希")
    ap.add_argument("--test-tool", action="append", dest="test_tools", help="算作跑测试的工具名，可重复")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    args = ap.parse_args(argv)
    judge = EvidenceJudge(args.rubric, protect=args.protect, workspace=args.workspace, test_tools=args.test_tools)
    serve(judge, args.transport)


if __name__ == "__main__":
    main()
