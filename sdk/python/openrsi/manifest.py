"""组件清单 rsi-component.yaml：组件是什么、怎么启动、实现了哪些工具。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from .protocol import KINDS, TOOLS_BY_NAME, required_tools
from .types import PROTOCOL_VERSION


class Runtime(BaseModel):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    command: list[str] = Field(default_factory=list)
    url: str | None = None


class Manifest(BaseModel):
    protocol: str = PROTOCOL_VERSION
    name: str
    version: str
    kinds: list[str]
    implements: list[str]
    capabilities: dict[str, Any] = Field(default_factory=dict)
    runtime: Runtime = Field(default_factory=Runtime)
    config_schema: str | None = None
    evolvable_surfaces: list[str] = Field(default_factory=list)
    description: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "Manifest":
        unknown_kinds = [k for k in self.kinds if k not in KINDS]
        if unknown_kinds:
            raise ValueError(f"未知的组件类型: {unknown_kinds}，可选: {list(KINDS)}")
        unknown_tools = [t for t in self.implements if t not in TOOLS_BY_NAME]
        if unknown_tools:
            raise ValueError(f"协议里没有这些工具: {unknown_tools}")
        for k in self.kinds:
            missing = required_tools(k) - set(self.implements)
            if missing:
                raise ValueError(f"类型 {k} 缺少必选工具: {sorted(missing)}")
        if not self.protocol.startswith("openrsi/"):
            raise ValueError(f"protocol 字段应形如 openrsi/0.1，实际是 {self.protocol}")
        return self


def load_manifest(path: str | Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
