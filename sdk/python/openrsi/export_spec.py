"""从 Python 模型导出协议的机器可读部分：spec/schemas/*.json 和 spec/tools.json。

    python -m openrsi.export_spec ../../spec

tools.json 是从真实 MCP server 上列出来的，所以和组件实际暴露的接口一定一致。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client

from .components import EvidenceCollector, Judge, Memory
from .protocol import TOOLS_BY_NAME
from .server import build_server
from .types import PROTOCOL_VERSION, SCHEMA_MODELS


class _AllKinds(EvidenceCollector, Judge, Memory):
    """只用来生成工具定义，方法不会被调用。"""

    name = "openrsi/spec"
    version = PROTOCOL_VERSION.split("/")[1]


async def _tool_defs() -> list[dict]:
    async with Client(build_server(_AllKinds())) as c:
        listed = await c.list_tools()
    out = []
    for t in sorted(listed.tools, key=lambda t: t.name):
        spec = TOOLS_BY_NAME[t.name]
        out.append({"name": t.name, "kind": spec.kind, "required": spec.required, "description": spec.description,
                    "input_schema": t.input_schema, "output_schema": t.output_schema})
    return out


def render() -> dict[str, str]:
    """返回 {相对路径: 文件内容}，测试用它检查仓库里的 spec 是否最新。"""
    files: dict[str, str] = {}
    for name, model in sorted(SCHEMA_MODELS.items()):
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema",
                  "$id": f"https://openrsi.dev/schemas/0.1/{name}.json", **model.model_json_schema()}
        files[f"schemas/{name}.json"] = json.dumps(schema, ensure_ascii=False, indent=2) + "\n"
    tools = {"protocol": PROTOCOL_VERSION, "tools": asyncio.run(_tool_defs())}
    files["tools.json"] = json.dumps(tools, ensure_ascii=False, indent=2) + "\n"
    return files


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "spec")
    for rel, text in render().items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    print(f"wrote {len(render())} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
