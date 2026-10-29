"""openrsi.yaml：声明每个位置用哪个组件、怎么启动、传什么配置。

    protocol: openrsi/0.2
    name: my-experiment
    plugins: [my_components]          # 相对本文件目录的 Python 模块
    components:
      env:      {python: my_components:MyEnv, with: {seed: 0}}
      memory:   {command: [openrsi-memory, --path, .openrsi/memory]}
      judge:    {url: http://judge.internal/mcp}
      evidence: {same_as: judge}      # 同一个组件同时担任两个位置
      ...
    loop:
      generations: 20
      retrieve_k: 5

组件来源三选一（外加 same_as）：python（进程内）、command（子进程，stdio MCP）、url（远程 MCP）。
相对路径一律相对 openrsi.yaml 所在目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REQUIRED_SLOTS = ("env", "agent", "judge", "proposer", "algorithm", "gate", "lineage")
OPTIONAL_SLOTS = ("memory", "evidence", "attribution", "guard")
SLOT_KIND = {s: s for s in (*REQUIRED_SLOTS, *OPTIONAL_SLOTS)}


class ComponentSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    python: str | None = Field(default=None, description="module:Class，进程内实例化")
    command: list[str] | None = Field(default=None, description="启动 stdio MCP server 的命令")
    url: str | None = Field(default=None, description="Streamable HTTP MCP 地址")
    same_as: str | None = Field(default=None, description="复用另一个位置的组件")
    with_: dict[str, Any] = Field(default_factory=dict, alias="with", description="python 组件的构造参数")
    env: dict[str, str] = Field(default_factory=dict, description="command 组件额外的环境变量")

    @model_validator(mode="after")
    def _one_source(self) -> "ComponentSpec":
        sources = [k for k in ("python", "command", "url", "same_as") if getattr(self, k)]
        if len(sources) != 1:
            raise ValueError(f"每个组件必须且只能指定 python / command / url / same_as 之一，现在是 {sources or '无'}")
        if self.with_ and not self.python:
            raise ValueError("with 只用于 python 组件；command 组件请把参数写进命令")
        return self


class LoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generations: int | None = Field(default=None, description="最多跑几代；不填则由进化算法决定何时停")
    retrieve_k: int = Field(default=5, description="每道题从 memory 检索几条给 agent")
    max_patches: int = Field(default=1, ge=1, description="每代最多评估几个候选补丁")
    train_limit: int | None = None
    holdout_limit: int | None = None
    metric: str = Field(default="holdout", description="谱系里选最佳快照用的指标；判别器参与进化时用 progress")
    anchor: str | None = Field(default=None, description="人工标注集（AnchorExample 的 JSONL），判别器参与进化时必需")
    judge_every: int | None = Field(default=None, ge=2,
                                    description="每隔几代轮到判别器进化；其余代只改 agent 侧。不填则判别器不进化")
    counterfactual_reruns: int = Field(default=0, ge=0,
                                       description="每代最多做几次反事实重跑（拿掉一条 memory 重跑同一道题），0 表示不做")


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: str = "openrsi/0.3"
    name: str
    plugins: list[str] = Field(default_factory=list)
    components: dict[str, ComponentSpec]
    evolvables: dict[str, ComponentSpec] = Field(
        default_factory=dict, description="memory 之外参与进化的对象（prompt、skill 等），名字随意")
    loop: LoopConfig = Field(default_factory=LoopConfig)
    base_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _slots(self) -> "RunConfig":
        unknown = sorted(set(self.components) - set(SLOT_KIND))
        if unknown:
            raise ValueError(f"未知的组件位置: {unknown}，可用: {sorted(SLOT_KIND)}")
        missing = [s for s in REQUIRED_SLOTS if s not in self.components]
        if missing:
            raise ValueError(f"缺少组件: {missing}")
        if "memory" not in self.components and not self.evolvables:
            raise ValueError("至少要有一个可进化体：components.memory 或 evolvables")
        for name, spec in self.evolvables.items():
            if spec.same_as and spec.same_as not in self.components:
                raise ValueError(f"evolvables.{name}.same_as 只能指向 components 里的位置（如 judge）")
        if self.loop.judge_every and not self.loop.anchor:
            raise ValueError("判别器参与进化（loop.judge_every）必须配置人工标注集 loop.anchor")
        for slot, spec in self.components.items():
            if spec.same_as and spec.same_as not in self.components:
                raise ValueError(f"{slot}.same_as 指向不存在的位置 {spec.same_as}")
            if spec.same_as and self.components[spec.same_as].same_as:
                raise ValueError(f"{slot}.same_as 不能指向另一个 same_as")
        return self


def load_anchor(config: RunConfig) -> list:
    """读人工标注集：每行一个 AnchorExample。"""
    from openrsi import AnchorExample

    if not config.loop.anchor:
        return []
    path = (config.base_dir / config.loop.anchor).resolve()
    return [AnchorExample.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate({**data, "base_dir": path.parent})
