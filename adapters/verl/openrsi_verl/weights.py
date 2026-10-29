"""模型权重作为可进化体：每个快照是一个 checkpoint，补丁是一次训练。

WeightsEvolvable 不自己训练，而是按模板调用外部训练命令（比如 verl 的 SFT 或 PPO/GRPO 入口），
训练完在输出目录得到新 checkpoint。回滚就是切换当前 checkpoint，旧的不删。

补丁格式：[{"op": "train", "method": "sft", "data": "train.parquet", "config": {...}}]
命令模板里可以用 {base_model}（父快照的模型路径）、{data}、{output_dir}、{method}、{config_json}。
method 为 "noop" 时不训练，只登记一个指向父模型的快照（一致性测试用）。

agent 怎么用：运行时把当前 checkpoint 作为一条 kind="weights" 的条目放进 agent 的上下文，
text 是模型路径，agent 据此加载模型或切换推理服务。

DistillProposer：从 train 里判别器判为通过的轨迹生成 SFT 数据，提出一次训练（拒绝采样式自我蒸馏）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from openrsi import (ComponentError, Diff, Evolvable, EvolvableInfo, Item, Patch, ProposalRequest, Proposer, Snapshot,
                     StaleSnapshotError)
from openrsi.types import canonical_json, content_hash, now_iso

from .data import episodes_to_sft_rows, write_rows


def _natural_key(p: Path) -> list:
    """global_step_10 排在 global_step_9 后面。"""
    import re

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]


class WeightsEvolvable(Evolvable):
    """with:
    base_model  初始模型路径或名字（根快照）
    path        checkpoint 登记和输出目录
    trainer     训练命令模板（字符串列表）；不给时只能用 dry_run
    dry_run     True 时不调用训练命令，只建一个标记目录（本地调通流程用）
    timeout     单次训练的超时（秒）
    model_glob  训练输出里模型所在的子路径（glob），比如 "global_step_*/huggingface"；匹配多个时取步数最大的。
                不给就把整个输出目录当作模型。verl 的 SFT / PPO 会把 checkpoint 放在 global_step_N 子目录里
    """

    name = "openrsi/weights"
    version = "0.4.0"

    def __init__(self, base_model: str, path: str | Path, target: str = "weights", trainer: list[str] | None = None,
                 dry_run: bool = False, timeout: float = 24 * 3600, model_glob: str | None = None):
        self.model_glob = model_glob
        self.base_model = base_model
        self.path = Path(path).resolve()
        self.target = target
        self.trainer = list(trainer or [])
        self.dry_run = dry_run
        self.timeout = timeout
        self.path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.path / "checkpoints.json"
        if self.registry_file.exists():
            data = json.loads(self.registry_file.read_text(encoding="utf-8"))
            self.ckpts: dict[str, dict[str, Any]] = data["checkpoints"]
            self.head: str = data["head"]
        else:
            self.ckpts = {"ckpt_root": {"model_path": base_model, "parent": None, "op": None, "created_at": now_iso()}}
            self.head = "ckpt_root"
            self._save()

    def _save(self) -> None:
        self.registry_file.write_text(json.dumps({"head": self.head, "checkpoints": self.ckpts}, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")

    def _snap(self, cid: str) -> Snapshot:
        c = self.ckpts[cid]
        return Snapshot(id=cid, target=self.target, parent_ids=[c["parent"]] if c["parent"] else [],
                        content_hash=content_hash({"model": c["model_path"], "op": c["op"]}),
                        created_at=c["created_at"], message=(c["op"] or {}).get("method"))

    def describe(self) -> EvolvableInfo:
        return EvolvableInfo(target=self.target, type="weights", head=self.head, surfaces=["weights"],
                             size=len(self.ckpts), example_ops=[{"op": "train", "method": "noop", "data": "",
                                                                 "config": {}}])

    def snapshot(self) -> Snapshot:
        return self._snap(self.head)

    def apply(self, patch: Patch) -> Snapshot:
        if patch.target != self.target:
            raise ValueError(f"补丁目标是 {patch.target}，本组件是 {self.target}")
        if patch.base_snapshot and patch.base_snapshot != self.head:
            raise StaleSnapshotError(f"补丁基于 {patch.base_snapshot}，当前是 {self.head}")
        if len(patch.ops) != 1 or patch.ops[0].get("op") != "train":
            raise ComponentError('权重补丁必须是且只能是一个 {"op": "train", ...}')
        op = {"method": "sft", "data": "", "config": {}, **patch.ops[0]}
        parent = self.head
        cid = "ckpt_" + hashlib.sha256(f"{parent}|{canonical_json(op)}".encode()).hexdigest()[:12]
        if cid not in self.ckpts:
            base = self.ckpts[parent]["model_path"]
            if op["method"] == "noop":
                model_path = base
            else:
                model_path = str(self._train(op, base, self.path / "checkpoints" / cid))
            self.ckpts[cid] = {"model_path": model_path, "parent": parent, "op": op, "created_at": now_iso()}
        self.head = cid
        self._save()
        return self._snap(cid)

    def _train(self, op: dict[str, Any], base_model: str, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / "DRY_RUN.json").write_text(json.dumps({"base_model": base_model, "op": op}, ensure_ascii=False))
            return out
        if not self.trainer:
            raise ComponentError("没有配置 trainer 命令；本地调试可以用 dry_run: true")
        fields = {"base_model": base_model, "data": op["data"], "output_dir": str(out), "method": op["method"],
                  "config_json": json.dumps(op.get("config") or {}, ensure_ascii=False)}
        argv = [part.format(**fields) for part in self.trainer]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise ComponentError(f"训练超时（{self.timeout}s）") from e
        (out.parent / f"{out.name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or [f"退出码 {proc.returncode}"])[-1]
            raise ComponentError(f"训练失败：{tail}")
        if not out.exists():
            raise ComponentError(f"训练命令成功退出，但没有生成 {out}")
        if self.model_glob:
            found = sorted(out.glob(self.model_glob), key=_natural_key)
            if not found:
                listing = ", ".join(sorted(p.name for p in out.iterdir())[:10]) or "空目录"
                raise ComponentError(f"训练输出里没有匹配 {self.model_glob} 的模型（{out} 下有：{listing}）")
            return found[-1]
        return out

    def rollback(self, snapshot_id: str) -> Snapshot:
        if snapshot_id not in self.ckpts:
            raise KeyError(f"没有快照 {snapshot_id}")
        self.head = snapshot_id
        self._save()
        return self._snap(snapshot_id)

    def _chain(self, cid: str) -> list[str]:
        out = []
        while cid:
            out.append(cid)
            cid = self.ckpts[cid]["parent"]
        return out

    def diff(self, from_id: str, to_id: str) -> Diff:
        a, b = self._chain(from_id), self._chain(to_id)
        if from_id in b:  # to 是 from 的后代：列出中间的训练
            steps = b[: b.index(from_id)][::-1]
            return Diff(from_id=from_id, to_id=to_id, ops=[self.ckpts[c]["op"] for c in steps])
        return Diff(from_id=from_id, to_id=to_id, ops=[{"op": "replace_model", "from": self.ckpts[a[0]]["model_path"],
                                                         "to": self.ckpts[b[0]]["model_path"]}])

    def export(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        cid = snapshot_id or self.head
        c = self.ckpts[cid]
        return [Item(id="model", kind="weights", text=c["model_path"],
                     data={"checkpoint": cid, "parent": c["parent"], "op": c["op"]}).model_dump(mode="json")]


class DistillProposer(Proposer):
    """with: data_dir（训练数据写到哪）、method（默认 sft）、min_examples、format（jsonl 或 parquet）、config"""

    name = "openrsi/distill-proposer"
    version = "0.4.0"

    def __init__(self, data_dir: str | Path, method: str = "sft", min_examples: int = 8, format: str = "jsonl",
                 config: dict[str, Any] | None = None):
        self.data_dir = Path(data_dir).resolve()
        self.method = method
        self.min_examples = min_examples
        self.format = format
        self.config = config or {}

    def propose(self, request: ProposalRequest) -> list[Patch]:
        target = next((t for t in (request.targets or [request.target]) if t.type == "weights"), None)
        if target is None:
            return []
        rows = episodes_to_sft_rows(request.episodes)
        if len(rows) < self.min_examples:
            return []
        digest = content_hash(rows)[:12]
        data = write_rows(rows, self.data_dir / f"sft-{request.parent_snapshot}-{digest}.{self.format}")
        op = {"op": "train", "method": self.method, "data": str(data), "config": self.config}
        return [Patch(target=target.target, base_snapshot=request.parent_snapshot, ops=[op], proposer_ref=self.ref,
                      rationale=f"{self.method} on {len(rows)} judge-verified train trajectories")]
