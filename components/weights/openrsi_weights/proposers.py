"""提案器：把判别器判过的 train 轨迹变成一次训练补丁。

- DistillProposer：判对的轨迹做 SFT（拒绝采样式自我蒸馏）
- PreferenceProposer：DPO（同一道题判对 × 判错配对）或 KTO（判对为好、判错为坏，不必配对），让做错的题也用上
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from openrsi import Patch, ProposalRequest, Proposer
from openrsi.types import content_hash

from .data import episodes_to_dpo_rows, episodes_to_kto_rows, episodes_to_sft_rows, write_rows


class DistillProposer(Proposer):
    """with: data_dir（训练数据写到哪）、method（默认 sft）、min_examples、format（jsonl 或 parquet）、config"""

    name = "openrsi/distill-proposer"
    version = "0.4.1"

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


class PreferenceProposer(Proposer):
    """with: data_dir、method（dpo 或 kto，默认 dpo）、min_examples、max_pairs_per_task（DPO）、config

    DPO 需要同一道题既有判对也有判错的回答：配合 loop.train_samples 让每道练习题答多次。
    KTO 不用配对，但要求本轮既有判对也有判错的样本（只有一种标签学不到区别）。"""

    name = "openrsi/preference-proposer"
    version = "0.4.1"

    def __init__(self, data_dir: str | Path, method: str = "dpo", min_examples: int = 4, max_pairs_per_task: int = 4,
                 config: dict[str, Any] | None = None):
        if method not in ("dpo", "kto"):
            raise ValueError(f"method 只能是 dpo 或 kto，收到 {method}")
        self.data_dir, self.method = Path(data_dir).resolve(), method
        self.min_examples, self.max_pairs_per_task = min_examples, max_pairs_per_task
        self.config = config or {}

    def propose(self, request: ProposalRequest) -> list[Patch]:
        target = next((t for t in (request.targets or [request.target]) if t.type == "weights"), None)
        if target is None:
            return []
        if self.method == "dpo":
            rows = episodes_to_dpo_rows(request.episodes, max_pairs_per_task=self.max_pairs_per_task)
        else:
            rows = episodes_to_kto_rows(request.episodes)
            if len({r["label"] for r in rows}) < 2:
                return []
        if len(rows) < self.min_examples:
            return []
        data = write_rows(rows, self.data_dir / f"{self.method}-{request.parent_snapshot}-{content_hash(rows)[:12]}.jsonl")
        wrong = sum(1 for e in request.episodes if not e.verdict.passed)
        op = {"op": "train", "method": self.method, "data": str(data), "config": self.config}
        return [Patch(target=target.target, base_snapshot=request.parent_snapshot, ops=[op], proposer_ref=self.ref,
                      rationale=f"{self.method} on {len(rows)} rows from judged train trajectories "
                                f"({wrong} wrong answers used)")]
