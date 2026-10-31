"""循环即 agent（runtime/openrsi_runtime/nesting.py）：外层把内层的一整轮运行当作一次做题。"""
import asyncio
from pathlib import Path

import pytest
import yaml

from openrsi import Item, Task
from openrsi_runtime import Runner, Wiring, load_config
from openrsi_runtime.nesting import GainJudge, LoopAgent

from nested_ops import INDEPENDENT

EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}


def inner_template(base: Path) -> Path:
    data = {"protocol": "openrsi/0.4", "name": "inner", "plugins": ["secret_ops"],
            "components": {
                "env": {"python": "secret_ops:SecretOpsEnv", "with": {"seed": 0}},
                "agent": {"python": "secret_ops:RuleAgent"}, "proposer": {"python": "secret_ops:RuleInducer"},
                "judge": {"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
                "evidence": {"same_as": "judge"},
                "attribution": {"python": "openrsi_attribution:RubricEvidenceAttributor"},
                "algorithm": INDEPENDENT,
                "gate": {"python": "openrsi_gate:HoldoutGate"},
                "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
                "memory": {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}}}
    base.mkdir(parents=True, exist_ok=True)
    (base / "inner.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return base / "inner.yaml"


def algo_item(value):
    return Item(id="algorithm", kind="loop_config", text="inner algorithm", data={"path": "components.algorithm",
                                                                                  "value": value})


def test_outer_loop_improves_how_the_inner_loop_improves(tmp_path):
    template = inner_template(tmp_path / "tpl")
    data = {"protocol": "openrsi/0.4", "name": "outer", "plugins": ["nested_ops"],
            "components": {
                "env": {"python": "nested_ops:InnerSeedsEnv"},
                "agent": {"python": "openrsi_runtime.nesting:LoopAgent",
                          "with": {"template": str(template), "generations": 4, "workdir": "state/nested"}},
                "judge": {"python": "openrsi_runtime.nesting:GainJudge"},
                "proposer": {"python": "nested_ops:InnerConfigProposer"},
                "algorithm": {"python": "openrsi_algorithm:GreedyAlgorithm", "with": {"max_generations": 1}},
                "gate": {"python": "openrsi_gate:HoldoutGate"},
                "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}}},
            "evolvables": {"inner_config": {"python": "openrsi_evolvables:PromptEvolvable", "with": {
                "target": "inner_config", "path": "state/inner_config",
                "initial": [{"id": "algorithm", "kind": "loop_config", "text": "inner algorithm: independent",
                             "data": {"path": "components.algorithm", "value": INDEPENDENT}}]}}}}
    (tmp_path / "outer.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(tmp_path / "outer.yaml")

    async def go():
        async with Wiring(cfg) as w:
            await Runner(w, cfg.loop).run(1)
            return await w.lineage.log()

    log = asyncio.run(go())
    root = next(e for e in log if e.status == "root")
    child = next(e for e in log if e.generation == 1)
    assert child.status == "accepted", child.decision
    assert child.metrics["holdout"] > root.metrics["holdout"] > 0  # 换成继承改进后，内层同预算下提升更多
    assert set(child.task_scores) == {"holdout-seed2", "holdout-seed3"}


def test_loop_agent_reports_inner_gain_from_fresh_state(tmp_path):
    agent = LoopAgent(template=str(inner_template(tmp_path)), generations=3, workdir=str(tmp_path / "w"))
    task = Task(id="t", input={"env_with": {"seed": 1}})
    a = agent.run(task, [algo_item(INDEPENDENT)], "s")
    b = agent.run(task, [algo_item(INDEPENDENT)], "s")
    assert a.output == b.output and a.output > 0  # 每次从全新状态开始，结果可复现
    assert a.metadata["inner"]["generations"] == 3 and a.metadata["overrides"] == ["components.algorithm"]
    v = GainJudge().evaluate(task, a)
    assert v.passed and v.score == pytest.approx(a.output) and v.collected[0].type == "inner_result"


@pytest.mark.parametrize("path", ["components.gate", "components.judge", "components.env", "loop.generations",
                                  "records", "cells"])
def test_fixed_points_cannot_be_overridden_by_the_outer_layer(tmp_path, path):
    agent = LoopAgent(template=str(inner_template(tmp_path)), generations=1, workdir=str(tmp_path / "w"))
    bad = Item(id="x", kind="loop_config", text="x", data={"path": path, "value": {"python": "m:C"}})
    traj = agent.run(Task(id="t", input={}), [bad], "s")
    assert traj.output is None and "不动点" in traj.metadata["error"]
    v = GainJudge().evaluate(Task(id="t"), traj)
    assert not v.passed and v.score == 0.0


def test_inner_budget_is_enforced(tmp_path):
    agent = LoopAgent(template=str(inner_template(tmp_path)), generations=10, max_agent_calls=60,
                      workdir=str(tmp_path / "w"))
    traj = agent.run(Task(id="t", input={}), [], "s")
    assert traj.metadata["inner"]["agent_calls"] == 60 and "预算" in traj.metadata["inner"]["stopped"]
