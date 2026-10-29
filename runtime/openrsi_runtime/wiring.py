"""按配置把组件接起来。进程内组件和远程（MCP）组件被包成同样的 async 接口，主循环不区分它们。"""
from __future__ import annotations

import importlib
import os
import sys
from contextlib import AsyncExitStack, contextmanager
from pathlib import Path
from typing import Any

from mcp.client.stdio import StdioServerParameters

from openrsi import Component, connect

from .config import SLOT_KIND, ComponentSpec, RunConfig


@contextmanager
def _chdir(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class LocalProxy:
    """把同步的本地组件包成 async 接口，方法名不变。"""

    def __init__(self, obj: Component):
        self._obj = obj
        self.kinds = obj.kinds
        self.ref = obj.ref

    def __getattr__(self, name: str):
        method = getattr(self._obj, name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return method(*args, **kwargs)
            except NotImplementedError:
                if name == "export":  # 可选方法没实现
                    return None
                raise

        return call


def _import(ref: str) -> Any:
    module, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"python 组件要写成 module:Class，实际是 {ref!r}")
    return getattr(importlib.import_module(module), attr)


def slots_include_evolvables(slots: list[str], config: RunConfig) -> bool:
    """只连谱系（openrsi log）时不必启动可进化体。"""
    return set(slots) != {"lineage"}


class Wiring:
    """async with Wiring(config) as w: w.env / w.agent / ... 都是统一的 async 接口；没配置的可选位置是 None。"""

    def __init__(self, config: RunConfig, slots: list[str] | None = None):
        self.config = config
        self.slots = slots or list(config.components)
        self._stack = AsyncExitStack()
        self.components: dict[str, Any] = {}
        self.evolvables: dict[str, Any] = {}  # 名字 → 门面；包括 memory
        self.refs: dict[str, str] = {}

    async def __aenter__(self) -> "Wiring":
        base = self.config.base_dir
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        with _chdir(base):
            for mod in self.config.plugins:
                importlib.import_module(mod)
        try:
            wanted = set(self.slots)
            for slot in self.slots:
                spec = self.config.components.get(slot)
                if spec and spec.same_as:
                    wanted.add(spec.same_as)
            built: dict[str, Any] = {}
            for slot in [s for s in self.config.components if s in wanted]:
                spec = self.config.components[slot]
                if not spec.same_as:
                    built[slot] = await self._build(slot, spec)
            for slot in wanted:
                spec = self.config.components[slot]
                source = built[spec.same_as] if spec.same_as else built[slot]
                self.components[slot] = self._facade(slot, source)
            if slots_include_evolvables(self.slots, self.config):
                if "memory" in self.components:
                    self.evolvables["memory"] = self.components["memory"]
                for name, spec in self.config.evolvables.items():
                    source = await self._build(f"evolvables.{name}", spec)
                    self.evolvables[name] = self._evolvable_facade(name, source)
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def _build(self, slot: str, spec: ComponentSpec) -> Any:
        base = self.config.base_dir
        if spec.python:
            cls = _import(spec.python)
            with _chdir(base):
                obj = cls(**spec.with_)
            self.refs[slot] = f"{obj.ref} (python)"
            return LocalProxy(obj)
        if spec.command:
            params = StdioServerParameters(command=spec.command[0], args=spec.command[1:], cwd=str(base),
                                           env={**os.environ, **spec.env})
            rc = await self._stack.enter_async_context(connect(params))
        else:
            rc = await self._stack.enter_async_context(connect(spec.url))
        self.refs[slot] = f"{rc.info.name}:{rc.info.version} (mcp)" if rc.info else "(mcp)"
        return rc

    def _facade(self, slot: str, source: Any) -> Any:
        kind = SLOT_KIND[slot]
        if kind not in source.kinds:
            raise ValueError(f"{slot} 位置需要 {kind} 类型的组件，{self.refs.get(slot, '')} 只实现了 {source.kinds}")
        if isinstance(source, LocalProxy):
            return source
        return getattr(source, kind)  # RemoteComponent 上同名的门面

    def _evolvable_facade(self, name: str, source: Any) -> Any:
        if "evolvable" not in source.kinds:
            raise ValueError(f"evolvables.{name} 需要可进化体，{self.refs.get('evolvables.' + name, '')} "
                             f"只实现了 {source.kinds}")
        if isinstance(source, LocalProxy):
            return source
        for kind in ("skill", "memory", "evolvable"):  # 选能力最全的门面
            if kind in source.kinds:
                return getattr(source, kind)
        raise AssertionError("unreachable")

    def __getattr__(self, slot: str) -> Any:
        if slot in SLOT_KIND:
            return self.components.get(slot)
        raise AttributeError(slot)
