"""环境格测试用的 env：在 secret_ops 的环境外面声明两个格，同一批题在两个格里各出一份。

故障开关（构造参数）用来检查运行时会不会停下：
  drift_after  cells() 被调用这么多次以后，plain 格的钉住清单变掉
  bad_ref      按格取题时把 env_ref 写错
  duplicate    同一个格里同一道题出两次
  at_ids       题目 id 带 @
  variant_shift  variant 格里标准答案加 1：模拟"环境改变了正确答案"，agent 在这个格会做错
"""
from __future__ import annotations

from openrsi import EnvCell, Task

from secret_ops import RuleAgent, SecretOpsEnv

PLAIN = EnvCell(id="plain", coords={"E2.1:os": "linux", "E4.1:默认编码": "utf-8"},
                pins={"image": "base@sha256:aaaa", "pin_policy": "digest"})
VARIANT = EnvCell(id="variant", coords={"E2.1:os": "linux", "E4.1:默认编码": "cp1252"},
                  pins={"image": "base:latest", "pin_policy": "tag"})


class CellsEnv(SecretOpsEnv):
    name, version = "test/cells-env", "0.1.0"

    def __init__(self, seed: int = 0, drift_after: int | None = None, bad_ref: bool = False,
                 duplicate: bool = False, at_ids: bool = False, variant_shift: bool = False):
        super().__init__(seed=seed)
        self.drift_after, self.bad_ref, self.duplicate, self.at_ids = drift_after, bad_ref, duplicate, at_ids
        self.variant_shift = variant_shift
        self.cells_calls = 0

    def cells(self) -> list[EnvCell]:
        self.cells_calls += 1
        plain = PLAIN
        if self.drift_after is not None and self.cells_calls > self.drift_after:
            plain = PLAIN.model_copy(update={"pins": {"image": "base@sha256:bbbb", "pin_policy": "digest"}})
        return [plain, VARIANT]

    def tasks(self, split: str, round: int = 0, limit: int | None = None, cell: str | None = None) -> list[Task]:
        base = super().tasks(split, round, limit)
        if cell is None:
            return base
        ref = "variant" if self.bad_ref and cell == "plain" else cell
        out = [t.model_copy(update={"env_ref": ref, "id": f"{t.id}@x" if self.at_ids else t.id}) for t in base]
        if self.variant_shift and cell == "variant":
            out = [t.model_copy(update={"reference": {"expected": t.reference["expected"] + 1}}) for t in out]
        if self.duplicate and out:
            out.append(out[0])
        return out


class RecordingAgent(RuleAgent):
    """记下 agent 收到的题目 metadata，检查坐标有没有被交给 agent。"""

    seen: list[dict] = []

    def run(self, task, context, snapshot_id=None):
        RecordingAgent.seen.append(dict(task.metadata))
        assert task.reference is None
        return super().run(task, context, snapshot_id)


UTC = EnvCell(id="utc", coords={"E2.1:os": "linux", "E4.1:时区": "UTC"},
              pins={"image": "base@sha256:cccc", "pin_policy": "digest"})
SHANGHAI = EnvCell(id="shanghai", coords={"E2.1:os": "linux", "E4.1:时区": "Asia/Shanghai"},
                   pins={"image": "base@sha256:dddd", "pin_policy": "digest"})
CP1252 = EnvCell(id="cp1252", coords={"E4.1:默认编码": "cp1252"},
                 pins={"image": "base@sha256:eeee", "pin_policy": "digest"})


class TZCellsEnv(CellsEnv):
    """两个时区格；with_cp1252=True 时多一个声明 cp1252 编码的格（大多数宿主机做不出来，用来测漂移）。"""

    name, version = "test/tz-cells-env", "0.1.0"

    def __init__(self, seed: int = 0, with_cp1252: bool = False):
        super().__init__(seed=seed)
        self.with_cp1252 = with_cp1252

    def cells(self) -> list[EnvCell]:
        return [UTC, SHANGHAI, *([CP1252] if self.with_cp1252 else [])]


class BindingAwareAgent(RuleAgent):
    """进程内的测试 agent：照方案回报 plan_hash（真实执行器应当像 HarnessAgent 那样把环境变量加给子进程）。"""

    seen: list[dict] = []

    def run(self, task, context, snapshot_id=None):
        traj = super().run(task, context, snapshot_id)
        binding = task.metadata.get("binding")
        BindingAwareAgent.seen.append(binding)
        if binding:
            traj.metadata["binding_plan_hash"] = binding["plan_hash"]
        return traj
