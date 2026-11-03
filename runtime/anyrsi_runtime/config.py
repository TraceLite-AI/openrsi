"""anyrsi.yaml：声明每个位置用哪个组件、怎么启动、传什么配置。

    protocol: anyrsi/0.2
    name: my-experiment
    plugins: [my_components]          # 相对本文件目录的 Python 模块
    components:
      env:      {python: my_components:MyEnv, with: {seed: 0}}
      memory:   {command: [anyrsi-memory, --path, .anyrsi/memory]}
      judge:    {url: http://judge.internal/mcp}
      evidence: {same_as: judge}      # 同一个组件同时担任两个位置
      ...
    loop:
      generations: 20
      retrieve_k: 5

组件来源三选一（外加 same_as）：python（进程内）、command（子进程，stdio MCP）、url（远程 MCP）。
相对路径一律相对 anyrsi.yaml 所在目录。
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


class MomentumConfig(BaseModel):
    """方向动量（loop.group.momentum）：把组里测出来的方向交给算子（提案器 / self_propose 时的 agent）。
    考试上确认过的小改动记为"有用的方向"，测出来变差的记为"有害的方向"，连同它让哪些 rubric、哪类题（task.tags）
    变好变差；每过一代权重乘 decay。每代把权重最高的几条放进提案请求的 hints["momentum"]，算子沿好方向继续提、避开坏方向。
    每代有 explore 个改法不给方向（探索，也是对照：报告里比较有方向和没方向的改法哪个好）。"""

    model_config = ConfigDict(extra="forbid")

    decay: float = Field(default=0.7, gt=0, le=1, description="每过一代，旧方向的权重乘这个数")
    max_items: int = Field(default=6, ge=1, description="每次给算子看几条有用的方向（有害的方向同样最多这么多）")
    explore: int = Field(default=1, ge=0, description="每代几个改法不给方向（探索 / 对照）")
    negative: bool = Field(default=True, description="是否也告诉算子哪些方向变差了")


class GroupConfig(BaseModel):
    """组提案：一次出一组候选，在考试题上算组内优势（见 anyrsi_runtime.group）。写了 loop.group 才启用；
    启用后普通的解题侧代都按组来做（判别器代、元层代不变）。"""

    model_config = ConfigDict(extra="forbid")

    size: int = Field(default=4, ge=1, description="每组几个候选（G）")
    proposals: str = Field(default="samples", pattern="^(samples|directions)$",
                           description="samples：同一提案器多采样；directions：沿归因反复指出的不同对象各出一个")
    min_support: int = Field(default=2, ge=1, description="directions 模式下，一个对象至少在几道错题里被追责才算方向")
    strategy: str | dict[str, Any] = Field(
        default="parent_paired",
        description="组内优势策略：parent_paired / grpo / best_of_n / training_free，"
                    "或 {name: …, with: {…}}，或 {python: module:Class, with: {…}}")
    confirm_fraction: float = Field(default=0.5, ge=0.0, lt=1.0,
                                    description="考试题里留多少做确认半（准入只看确认半）；0 表示不分，挑和判用同一批题")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="parent_paired 的显著性水平（按组大小做 Bonferroni）")
    common_seed: bool = Field(default=True, description="同一代同一道题给所有版本同一个 rollout_seed（公共随机数）")
    experience_target: str | None = Field(default=None,
                                          description="training_free 策略写经验的可进化体（memory 之类）")
    atomize: bool = Field(default=False, description="把每个改法拆成一条条最小改动，分别在考试题上测（配合 merge 策略）")
    max_atoms: int = Field(default=16, ge=1, description="atomize 时一代最多测几条小改动")
    momentum: MomentumConfig | None = Field(default=None, description="把测出来的方向交给算子（方向动量）")


class ImproverFeedbackConfig(BaseModel):
    """把考试结果反馈给提案器：每代把"哪个改法在考试上好 / 差"写进提案器的策略说明（improver 类型的可进化体），
    提案器下次出改法时会读到（hints["improver"]）。组提案模式用组内优势，普通模式用准入结果。
    这些说明只影响提案器、不影响 agent，所以直接写入、跨代保留，不走准入。"""

    model_config = ConfigDict(extra="forbid")

    target: str = Field(description="写反馈的可进化体（类型必须是 improver）")
    max_notes: int = Field(default=12, ge=1, description="最多保留几条，超出时删掉最旧的")
    shuffle: bool = Field(default=False, description="对照实验用：把好 / 差随机打乱，验证起作用的是反馈内容而不是机制本身")


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
    train_samples: int = Field(default=1, ge=1,
                               description="每道练习题答几次（agent 要有随机性）；>1 时偏好训练（DPO）才有同题的对错配对")
    meta_every: int | None = Field(default=None, ge=2,
                                   description="每隔几代做一次元层进化（improver、curriculum 类型的可进化体）；不填则元层不进化")
    meta_min_gain: float = Field(default=0.0, ge=0.0,
                                 description="元层候选的下游提升至少要比当前版本多多少才接受")
    group: GroupConfig | None = Field(default=None, description="组提案：一组候选在考试题上比较，按策略选或写经验")
    self_propose: bool = Field(default=False,
                               description="agent 给自己出改法（递归改进）：解题侧的提案交给当前版本的 agent，"
                                           "不用 components.proposer（判别器、元层的提案仍用它）")
    improver_feedback: ImproverFeedbackConfig | None = Field(
        default=None, description="把考试结果写回提案器的策略说明，让提案器自己也改进（递归改进）")


class BindingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    python: str = Field(description="module:Class，ExecutionBinding 的子类，如 anyrsi_runtime.bindings:EnvVarBinding")
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")


class RecordsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="不可变记录库的目录（相对编排文件）")


class CellsConfig(BaseModel):
    """环境格模式（0.5 草案）。写了这一节才启用；不写时按 0.4 运行，即使 env 实现了 rsi_env_cells。"""

    model_config = ConfigDict(extra="forbid")

    train: list[str] | None = Field(default=None, description="train 用哪些格；不填用 env 声明的全部格")
    holdout: list[str] | None = Field(default=None, description="holdout 用哪些格；不填用 env 声明的全部格")
    expose_coords_to_agent: bool = Field(
        default=False, description="把格的坐标放进交给 agent 的 task.metadata.env_coords；默认不给，agent 自己探测")
    binding: BindingSpec | None = Field(
        default=None, description="执行绑定（严格模式）：每道题执行前准备并核对环境，agent 必须回报所用方案")

    @model_validator(mode="after")
    def _ids(self) -> "CellsConfig":
        for name in ("train", "holdout"):
            ids = getattr(self, name)
            if ids is not None and not ids:
                raise ValueError(f"cells.{name} 不能是空列表；不填表示用全部格")
            if ids and any("@" in i for i in ids):
                raise ValueError(f"cells.{name} 里的格 id 不能含 @")
            if ids and len(ids) != len(set(ids)):
                raise ValueError(f"cells.{name} 里有重复的格")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protocol: str = "anyrsi/0.4"
    name: str
    plugins: list[str] = Field(default_factory=list)
    components: dict[str, ComponentSpec]
    evolvables: dict[str, ComponentSpec] = Field(
        default_factory=dict, description="memory 之外参与进化的对象（prompt、skill 等），名字随意")
    loop: LoopConfig = Field(default_factory=LoopConfig)
    cells: CellsConfig | None = Field(default=None, description="环境格模式（0.5 草案），写了才启用")
    records: RecordsConfig | None = Field(
        default=None, description="不可变记录（P2）：原始题目/轨迹/证据/回执按内容寻址存档；也让续跑严格核对运行清单")
    base_dir: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _slots(self) -> "RunConfig":
        unknown = sorted(set(self.components) - set(SLOT_KIND))
        if unknown:
            raise ValueError(f"未知的组件位置: {unknown}，可用: {sorted(SLOT_KIND)}")
        missing = [s for s in REQUIRED_SLOTS if s not in self.components
                   and not (s == "proposer" and self.loop.self_propose)]
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
    from anyrsi import AnchorExample

    if not config.loop.anchor:
        return []
    path = (config.base_dir / config.loop.anchor).resolve()
    return [AnchorExample.model_validate_json(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate({**data, "base_dir": path.parent})
