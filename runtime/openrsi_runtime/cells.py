"""环境格模式（0.5 草案，spec/drafts/environment-cells.md）：只有编排文件写了 cells 才启用。

启用后由本模块保证：
  - env 必须实现 rsi_env_cells，配置里点名的格都必须是 env 声明过的
  - 运行开始时记下每个格的指纹，之后每次取题前都重新核对，变了就停止
  - 按格取题，每道题的 env_ref 必须等于所在格；同一 (题, 格) 不能出现两次
  - 逐题分数的键是 <task_id>@<cell_id>，所以题目 id 和格 id 都不能含 @
  - 钉住清单不是 digest 时给出警告：同一个 tag 两次拉到的可能不是同一个镜像
  - 轨迹 id 在取证之前加上 @<cell_id>（loop.py），同一道题在不同格里的轨迹和归因不会混
  - 每个 episode 多一条 env_cell 证据（格 id、坐标、指纹），归因器据此做跨格对比

只做声明和核对。题目真的在哪个环境里执行，是 agent / harness 的事（设计稿的 P2 执行绑定），这里不保证。
"""
from __future__ import annotations

from typing import Any, Callable

from openrsi import EnvCell, Evidence, Task

from .config import CellsConfig


class CellError(RuntimeError):
    """环境格模式下的声明或数据不一致，运行必须停止。"""


def task_key(task: Task) -> str:
    """逐题分数的键：环境格模式下是 <task_id>@<cell_id>，否则就是题目 id。"""
    return f"{task.id}@{task.env_ref}" if task.env_ref is not None else task.id


class CellRegistry:
    def __init__(self, cfg: CellsConfig, declared: list[EnvCell], log: Callable[[str], None] = lambda s: None):
        self.cfg = cfg
        self.cells = {c.id: c for c in declared}
        if len(self.cells) != len(declared):
            raise CellError(f"env 声明的格 id 重复：{[c.id for c in declared]}")
        bad = [i for i in self.cells if "@" in i]
        if bad:
            raise CellError(f"格 id 不能含 @：{bad}")
        self.splits = {"train": cfg.train or list(self.cells), "holdout": cfg.holdout or list(self.cells)}
        unknown = sorted({c for ids in self.splits.values() for c in ids} - set(self.cells))
        if unknown:
            raise CellError(f"配置里的格 {unknown} 不在 env 声明的格里：{sorted(self.cells)}")
        used = sorted({c for ids in self.splits.values() for c in ids})
        self.fingerprints = {c: self.cells[c].fingerprint() for c in used}
        for c in used:
            policy = self.cells[c].pins.get("pin_policy", "none")
            if policy != "digest":
                log(f"警告：环境格 {c} 的钉住方式是 {policy}，不是 digest，同样的声明不保证是同一个环境")

    @classmethod
    async def open(cls, env: Any, cfg: CellsConfig, log: Callable[[str], None] = lambda s: None) -> "CellRegistry":
        declared = await env.cells()
        if declared is None:
            raise CellError("配置了 cells，但 env 没有实现 rsi_env_cells，不能运行环境格模式")
        if not declared:
            raise CellError("配置了 cells，但 env 没有声明任何格")
        return cls(cfg, declared, log)

    async def verify(self, env: Any) -> None:
        """重新取一遍格的声明，和运行开始时的指纹比。"""
        now = {c.id: c for c in (await env.cells() or [])}
        for cid, fp in self.fingerprints.items():
            if cid not in now:
                raise CellError(f"环境格 {cid} 在运行中消失了")
            if now[cid].fingerprint() != fp:
                raise CellError(f"环境格 {cid} 的声明在运行中变了（指纹 {fp[:12]} → {now[cid].fingerprint()[:12]}）")

    def check_lineage(self, recorded: dict[str, str]) -> None:
        """续跑时，谱系里记的格必须和现在完全一样，否则新旧分数不可比。"""
        if recorded != self.fingerprints:
            raise CellError(f"谱系记录的环境格 {sorted(recorded)} 和现在的 {sorted(self.fingerprints)} 不一致，"
                            "或者同名格的指纹变了；不同环境下的分数不能接着比")

    async def tasks(self, env: Any, split: str, round: int, limit: int | None) -> list[Task]:
        """按配置的格依次取题；limit 是每个格的上限。"""
        await self.verify(env)
        out: list[Task] = []
        seen: set[str] = set()
        for cid in self.splits.get(split, list(self.fingerprints)):
            for t in await env.tasks(split, round, limit, cell=cid):
                if t.env_ref != cid:
                    raise CellError(f"按格 {cid} 取题，题目 {t.id} 的 env_ref 却是 {t.env_ref!r}")
                if "@" in t.id:
                    raise CellError(f"环境格模式下题目 id 不能含 @：{t.id}")
                key = task_key(t)
                if key in seen:
                    raise CellError(f"同一道题在同一个格里出现了两次：{key}")
                seen.add(key)
                out.append(t)
        return out

    def evidence(self, task: Task) -> Evidence:
        """这道题所在格的环境证据。由编排方记录，不是 agent 自报；agent 看不到它。"""
        cell = self.cells[task.env_ref]
        return Evidence.make("env_cell", {"cell": cell.id, "axes": cell.axes, "coords": dict(cell.coords),
                                          "fingerprint": self.fingerprints.get(cell.id, cell.fingerprint())},
                             f"runtime:cell:{cell.id}", "openrsi/runtime")

    def coords_for_agent(self, task: Task) -> dict[str, str] | None:
        if not self.cfg.expose_coords_to_agent or task.env_ref not in self.cells:
            return None
        return dict(self.cells[task.env_ref].coords)
