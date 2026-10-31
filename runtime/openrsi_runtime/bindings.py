"""执行绑定（two-loop-evolution 设计 §4.2，P2）：把"环境格的声明"和"实际执行的环境"连起来。

    resolve(cell, fingerprint) -> BindingPlan      这个格怎么准备：环境变量、镜像、要跑哪些探测
    prepare(plan)              -> ExecutionReceipt 真的准备一次，在子进程里跑探测，记下观察到的值
    verify(plan, receipt, cell)-> BindingCheck     观察值和格声明的坐标逐项比对

只在编排文件的 cells 里配置了 binding 时启用（严格模式）。启用后运行时保证：
  - 每道题执行前都 prepare + verify；探测结果和坐标不符（环境漂移、宿主机不支持）就停止
  - 交给 agent 的题目带上 metadata.binding = {env, plan_hash}；agent 必须在轨迹的
    metadata.binding_plan_hash 里回报它用的方案，没回报说明它不支持绑定，同样停止
  - 回执作为 exec_receipt 证据进入 episode，并写进不可变记录

探测只能用预先登记的（PROBES），配置里写名字，不能写任意命令：probes 不是 shell。
回执证明的是"按这份方案准备出来的环境确实是这个样子"；agent 有没有真的照方案执行，
靠 agent 回报 plan_hash，官方的 HarnessAgent 会照做。进程内的 agent 无法被强制，严格模式下会被拒绝。
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from openrsi import EnvCell, content_hash
from openrsi.types import now_iso


class BindingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    fingerprint: str
    env: dict[str, str] = Field(default_factory=dict, description="agent 阶段要设的环境变量")
    image: str | None = Field(default=None, description="允许的镜像摘要（交给支持容器的执行器）")
    probes: dict[str, str] = Field(default_factory=dict, description="坐标键 → 探测名")

    @property
    def plan_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class ExecutionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    plan_hash: str
    stage: str = "agent"
    observed: dict[str, str] = Field(default_factory=dict, description="探测名 → 观察到的值")
    executor: str
    created_at: str = Field(default_factory=now_iso)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self.model_dump(mode="json", exclude={"created_at"}))


class BindingCheck(BaseModel):
    ok: bool
    mismatches: list[str] = Field(default_factory=list)
    unverified: list[str] = Field(default_factory=list, description="声明了但没有探测的坐标")


# ---------------------------------------------------------------- 预先登记的探测

def _py(code: str) -> Callable[[dict[str, str]], str]:
    def run(env: dict[str, str]) -> str:
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=30,
                             env={"PATH": os.environ.get("PATH", ""), **env})
        if out.returncode != 0:
            raise RuntimeError(f"探测失败：{out.stderr.strip()[-200:]}")
        return out.stdout.strip()
    return run


PROBES: dict[str, Callable[[dict[str, str]], str]] = {
    "python.preferred_encoding": _py("import locale; print(locale.getpreferredencoding(False))"),
    "python.stdout_encoding": _py("import sys; print(sys.stdout.encoding)"),
    "python.version": _py("import sys; print('%d.%d' % sys.version_info[:2])"),
    "platform.system": _py("import platform; print(platform.system().lower())"),
    "env.TZ": _py("import os; print(os.environ.get('TZ', ''))"),
    "env.LANG": _py("import os; print(os.environ.get('LANG', ''))"),
}


def _norm(value: str) -> str:
    v = value.strip().lower().replace("_", "-")
    return {"utf8": "utf-8", "ansi-x3.4-1968": "ascii", "us-ascii": "ascii", "darwin": "macos"}.get(v, v)


class ExecutionBinding:
    """绑定的基类。子类可以换成容器、虚拟机或远程执行器；运行时只依赖这三个方法。"""

    name = "openrsi/binding"

    def resolve(self, cell: EnvCell, fingerprint: str) -> BindingPlan:
        raise NotImplementedError

    def prepare(self, plan: BindingPlan) -> ExecutionReceipt:
        raise NotImplementedError

    def verify(self, plan: BindingPlan, receipt: ExecutionReceipt, cell: EnvCell) -> BindingCheck:
        mismatches, unverified = [], []
        if receipt.plan_hash != plan.plan_hash or receipt.cell_id != plan.cell_id:
            mismatches.append("回执不是这份方案的")
        for key, value in cell.coords.items():
            probe = plan.probes.get(key)
            if probe is None:
                unverified.append(key)
            elif probe not in receipt.observed:
                mismatches.append(f"{key}：探测 {probe} 没有结果")
            elif _norm(receipt.observed[probe]) != _norm(value):
                mismatches.append(f"{key}：声明 {value}，实际 {receipt.observed[probe]}")
        return BindingCheck(ok=not mismatches, mismatches=mismatches, unverified=unverified)


class EnvVarBinding(ExecutionBinding):
    """用环境变量准备每个格，在子进程里跑登记过的探测。

    with:
      cells:  {格 id: {env: {变量: 值}, image: 可选}}
      probes: {坐标键: 探测名}，如 {"E4.1:默认编码": "python.preferred_encoding"}；所有格共用
    """

    name = "openrsi/env-var-binding"

    def __init__(self, cells: dict[str, dict], probes: dict[str, str] | None = None):
        self.cells = cells
        self.probes = dict(probes or {})
        unknown = sorted(set(self.probes.values()) - set(PROBES))
        if unknown:
            raise ValueError(f"未登记的探测 {unknown}，可用：{sorted(PROBES)}")

    def resolve(self, cell: EnvCell, fingerprint: str) -> BindingPlan:
        spec = self.cells.get(cell.id)
        if spec is None:
            raise ValueError(f"绑定里没有格 {cell.id} 的准备方式")
        return BindingPlan(cell_id=cell.id, fingerprint=fingerprint, env=dict(spec.get("env") or {}),
                           image=spec.get("image"), probes={k: v for k, v in self.probes.items() if k in cell.coords})

    def prepare(self, plan: BindingPlan) -> ExecutionReceipt:
        observed = {name: PROBES[name](plan.env) for name in sorted(set(plan.probes.values()))}
        return ExecutionReceipt(cell_id=plan.cell_id, plan_hash=plan.plan_hash, observed=observed, executor=self.name)


__all__ = ["BindingCheck", "BindingPlan", "EnvVarBinding", "ExecutionBinding", "ExecutionReceipt", "PROBES"]
