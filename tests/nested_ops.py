"""循环即 agent 的测试组件：外层的"题"是内层 secret_ops 的随机种子，外层改的是内层用哪个进化算法。"""
from __future__ import annotations

from openrsi import Environment, Item, Patch, Proposer, Task

INDEPENDENT = {"python": "openrsi_algorithm:IndependentAlgorithm", "with": {"max_generations": 100, "patience": 100}}
GREEDY = {"python": "openrsi_algorithm:GreedyAlgorithm", "with": {"max_generations": 100, "patience": 100}}


class InnerSeedsEnv(Environment):
    name, version = "test/inner-seeds", "0.1.0"

    def tasks(self, split, round=0, limit=None, cell=None):
        seeds = {"train": [0, 1], "holdout": [2, 3]}.get(split, [])
        out = [Task(id=f"{split}-seed{s}", split=split, tags=["inner"], input={"env_with": {"seed": s}}) for s in seeds]
        return out[:limit] if limit else out


class InnerConfigProposer(Proposer):
    """把内层算法从"每代从根出发"换成"继承改进"。"""

    name, version = "test/inner-config-proposer", "0.1.0"

    def propose(self, request):
        current = next((d for d in request.state or [] if d.get("id") == "algorithm"), None)
        if current is None or current["data"]["value"] == GREEDY:
            return []
        item = Item(id="algorithm", kind="loop_config", text="inner algorithm: greedy",
                    data={"path": "components.algorithm", "value": GREEDY})
        return [Patch(target=request.target.target, base_snapshot=request.parent_snapshot, proposer_ref=self.ref,
                      ops=[{"op": "replace", "item": item.model_dump(mode="json")}])]
