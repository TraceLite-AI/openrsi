"""环境、取证、归因怎么配合（设计文档"环境、取证、归因怎么配合"一节）：

- 环境格模式下每个 episode 多一条 env_cell 证据（编排方记录，agent 看不到）
- CellContrastAttributor：同一道题只在某些格失败 → 归到挂的格独有的环境坐标；全挂 → agent 能力
- HoldoutGate 的逐格不回退
- RoutedProposer：按归因把这一代交给对应的成员
"""
import asyncio
from pathlib import Path

import pytest
import yaml

from openrsi import (Attribution, AttributionTarget, Episode, Evidence, EvolvableInfo, GateRequest, Patch,
                     ProposalRequest, Proposer, Task, Trajectory, Verdict, build_server, connect, load_manifest)
from openrsi_attribution import CellContrastAttributor, RubricEvidenceAttributor
from openrsi_gate import HoldoutGate
from openrsi_proposer import RoutedProposer
from openrsi_runtime import CellRegistry, Runner, Wiring, load_config

from cells_env import PLAIN, VARIANT

ROOT = Path(__file__).resolve().parents[1]
EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}


def ep(task_id: str, cell: str | None, passed: bool, coords: dict | None = None) -> Episode:
    task = Task(id=task_id, split="train", env_ref=cell, input={"q": task_id})
    traj = Trajectory(id=f"traj-{task_id}@{cell}", task_id=task_id, output="x", steps=[])
    evidence = []
    if cell is not None and coords is not None:
        evidence.append(Evidence.make("env_cell", {"cell": cell, "coords": dict(coords), "fingerprint": "f"},
                                      f"runtime:cell:{cell}", "openrsi/runtime"))
    return Episode(task=task, trajectory=traj, evidence=evidence,
                   verdict=Verdict(task_id=task_id, trajectory_id=traj.id, passed=passed, score=float(passed),
                                   judge_ref="j"))


def env_targets(a: Attribution) -> list[AttributionTarget]:
    return [t for t in a.targets if t.kind == "env"]


# ---------------------------------------------------------------- 归因

def test_failure_in_one_cell_is_blamed_on_the_odd_coordinate():
    eps = [ep("t1", "plain", True, PLAIN.coords), ep("t1", "variant", False, VARIANT.coords)]
    plain, variant = CellContrastAttributor().explain(eps)
    assert not env_targets(plain)
    [t] = env_targets(variant)
    assert t.ref == "E4.1:默认编码=cp1252" and t.weight == -0.8  # E2.1:os 两格一样，不怪它
    assert "plain 通过" in t.note and "variant 失败" in t.note


def test_several_odd_coordinates_share_the_blame():
    odd = {"E2.1:os": "windows", "E4.1:默认编码": "cp1252"}
    eps = [ep("t1", "plain", True, PLAIN.coords), ep("t1", "win", False, odd)]
    targets = env_targets(CellContrastAttributor(env_weight=1.0).explain(eps)[1])
    assert {t.ref for t in targets} == {"E2.1:os=windows", "E4.1:默认编码=cp1252"}
    assert all(t.weight == -0.5 for t in targets)


def test_without_coordinates_blame_goes_to_the_cell():
    eps = [ep("t1", "plain", True), ep("t1", "variant", False)]
    [t] = env_targets(CellContrastAttributor().explain(eps)[1])
    assert t.ref == "cell:variant"


def test_tampered_env_evidence_is_ignored():
    bad = ep("t1", "variant", False, VARIANT.coords)
    bad.evidence[0].content["coords"]["E4.1:默认编码"] = "utf-8"  # 改了内容，哈希对不上
    [t] = env_targets(CellContrastAttributor().explain([ep("t1", "plain", True, PLAIN.coords), bad])[1])
    assert t.ref == "cell:variant"


def test_failing_everywhere_is_a_capability_problem():
    eps = [ep("t1", "plain", False, PLAIN.coords), ep("t1", "variant", False, VARIANT.coords)]
    for a in CellContrastAttributor().explain(eps):
        assert not env_targets(a)
        refs = [t.ref for t in a.targets if t.kind == "component"]
        assert refs.count("agent:capability") == 1  # 底层的 memory:missing 照旧保留


def test_same_as_base_attributor_without_contrast():
    eps = [ep("t1", None, False), ep("t2", "plain", False, PLAIN.coords), ep("t3", "plain", True, PLAIN.coords),
           ep("t3", "variant", True, VARIANT.coords)]
    assert CellContrastAttributor().explain(eps) == RubricEvidenceAttributor().explain(eps)


# ---------------------------------------------------------------- 运行时：环境证据 + 跨格归因

def config(tmp_path, **env_kw):
    data = {"protocol": "openrsi/0.4", "name": "contrast", "plugins": ["cells_env"], "cells": {},
            "components": {
                "env": {"python": "cells_env:CellsEnv", "with": env_kw},
                "agent": {"python": "secret_ops:RuleAgent"},
                "proposer": {"python": "secret_ops:RuleInducer"},
                "judge": {"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
                "evidence": {"same_as": "judge"},
                "attribution": {"python": "openrsi_attribution:CellContrastAttributor"},
                "algorithm": {"python": "openrsi_algorithm:GreedyAlgorithm", "with": {"max_generations": 3}},
                "gate": {"python": "openrsi_gate:HoldoutGate", "with": {"max_cell_regressions": 0}},
                "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
                "memory": {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}}}
    (tmp_path / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(tmp_path / "openrsi.yaml")


def test_runtime_records_env_evidence_and_contrast_finds_the_coordinate(tmp_path):
    cfg = config(tmp_path, variant_shift=True)

    async def go():
        async with Wiring(cfg) as w:
            r = Runner(w, cfg.loop)
            await r._load_targets()
            r.registry = await CellRegistry.open(w.env, cfg.cells)
            genome = {t: (await tgt.facade.snapshot()).id for t, tgt in r.targets.items()}
            eps = [await r.episode(t, genome) for t in await r.tasks("train", 1, None)]
            return eps, await w.attribution.explain(eps)

    eps, attrs = asyncio.run(go())
    for e in eps:
        [env] = [x for x in e.evidence if x.type == "env_cell"]
        assert env.verify() and env.content["cell"] == e.task.env_ref
        assert env.content["fingerprint"] == (PLAIN if e.task.env_ref == "plain" else VARIANT).fingerprint()
    variant_fails = [a for e, a in zip(eps, attrs) if e.task.env_ref == "variant" and not e.verdict.passed]
    plain_passes = {e.task.id for e in eps if e.task.env_ref == "plain" and e.verdict.passed}
    contrasted = [a for e, a in zip(eps, attrs)
                  if e.task.env_ref == "variant" and not e.verdict.passed and e.task.id in plain_passes]
    assert variant_fails and contrasted  # variant 格的标准答案变了：同一道题 plain 过、variant 挂
    for a in contrasted:
        assert [t.ref for t in env_targets(a)] == ["E4.1:默认编码=cp1252"]


def test_full_loop_runs_with_contrast_and_cell_gate(tmp_path):
    async def go():
        cfg = config(tmp_path)
        async with Wiring(cfg) as w:
            await Runner(w, cfg.loop).run(2)
            return await w.lineage.log()

    log = asyncio.run(go())
    decided = [e for e in log if e.decision]
    assert decided and all(any(c.name == "no_cell_regression" for c in e.decision.checks) for e in decided)


# ---------------------------------------------------------------- 准入：逐格不回退

def gate_request(base, cand):
    return GateRequest(candidate_id="c", baseline={"holdout": 0.5}, candidate={"holdout": 0.6},
                       baseline_tasks=base, candidate_tasks=cand)


def test_gate_rejects_a_cell_that_got_worse():
    base = {"a@plain": 0.0, "b@plain": 0.0, "a@variant": 1.0, "b@variant": 1.0}
    cand = {"a@plain": 1.0, "b@plain": 1.0, "a@variant": 1.0, "b@variant": 0.0}  # 总分涨了，variant 掉了
    d = HoldoutGate(max_cell_regressions=0).check(gate_request(base, cand))
    check = next(c for c in d.checks if c.name == "no_cell_regression")
    assert not d.accepted and not check.passed and "variant 1.000→0.500" in check.detail
    assert HoldoutGate(max_cell_regressions=0, cell_tolerance=0.5).check(gate_request(base, cand)).accepted
    assert HoldoutGate(max_cell_regressions=1).check(gate_request(base, cand)).accepted


def test_gate_cell_check_needs_matching_cells():
    gate = HoldoutGate(max_cell_regressions=0)
    assert not gate.check(gate_request({"a": 0.0}, {"a": 1.0})).accepted  # 不是环境格模式的分数
    assert not gate.check(gate_request({"a@x": 0.0}, {"a@y": 1.0})).accepted  # 格对不上
    assert HoldoutGate().check(gate_request({"a": 0.0}, {"a": 1.0})).accepted  # 不配置就不检查


# ---------------------------------------------------------------- 提案：按归因分流

class RecordingProposer(Proposer):
    name, version = "test/recording", "0.0.1"

    def __init__(self, label: str):
        self.label, self.calls = label, 0

    def propose(self, request):
        self.calls += 1
        return [Patch(target="memory", ops=[], rationale=f"from {self.label}")]


INFO = EvolvableInfo(target="memory", type="memory", head="s0")
MEMBERS = [{"name": n, "python": "test_cell_contrast:RecordingProposer", "with": {"label": n}}
           for n in ("prompt", "lesson", "skill")]
ROUTES = [{"match": "env", "to": "prompt"}, {"match": "component:memory:missing", "to": "lesson"},
          {"match": "step", "to": "skill"}]


def request(*targets, hints=None):
    return ProposalRequest(target=INFO, parent_snapshot="s0", hints=hints or {},
                           attributions=[Attribution(episode_id="e", method="m", targets=list(targets))])


def test_router_picks_the_member_with_most_blame():
    r = RoutedProposer(MEMBERS, ROUTES)
    [p] = r.propose(request(AttributionTarget(kind="env", ref="E4.1:默认编码=cp1252", weight=-0.8),
                            AttributionTarget(kind="component", ref="memory:missing", weight=-0.5)))
    assert p.proposer_ref.endswith("#prompt") and p.rationale.startswith("from prompt")
    assert "分流：env 责任 0.80" in p.rationale
    [p] = r.propose(request(AttributionTarget(kind="component", ref="memory:missing", weight=-0.5),
                            AttributionTarget(kind="item", ref="m1", weight=0.9)))  # 功劳不算责任
    assert p.proposer_ref.endswith("#lesson")
    assert r.members["skill"].calls == 0


def test_router_ties_go_to_the_earlier_route_and_hints_win():
    r = RoutedProposer(MEMBERS, ROUTES)
    tie = (AttributionTarget(kind="step", ref="s1", weight=-0.5), AttributionTarget(kind="env", ref="cell:x", weight=-0.5))
    assert r.propose(request(*tie))[0].proposer_ref.endswith("#prompt")
    hinted = r.propose(request(*tie, hints={"proposer": "skill"}))
    assert [p.proposer_ref.rsplit("#", 1)[1] for p in hinted] == ["skill"] and "分流" not in hinted[0].rationale


def test_router_falls_back_to_default_or_everyone():
    nothing = request(AttributionTarget(kind="rubric", ref="correct", weight=-1.0))
    [p] = RoutedProposer(MEMBERS, ROUTES, default="lesson").propose(nothing)
    assert p.proposer_ref.endswith("#lesson") and "默认成员" in p.rationale
    allp = RoutedProposer(MEMBERS, ROUTES).propose(nothing.model_copy(update={"max_patches": 3}))
    assert sorted(p.proposer_ref.rsplit("#", 1)[1] for p in allp) == ["lesson", "prompt", "skill"]


def test_router_config_is_checked():
    with pytest.raises(ValueError, match="路由"):
        RoutedProposer(MEMBERS, [{"match": "env", "to": "nobody"}])
    with pytest.raises(ValueError, match="default"):
        RoutedProposer(MEMBERS, ROUTES, default="nobody")


# ---------------------------------------------------------------- 清单

@pytest.mark.parametrize("manifest,factory", [
    ("attribution/rsi-component.cell-contrast.yaml", CellContrastAttributor),
    ("proposer/rsi-component.routed.yaml",
     lambda: RoutedProposer(["lesson"], [{"match": "component:memory:missing", "to": "lesson"}])),
])
def test_new_manifests_match(manifest, factory):
    async def served():
        async with connect(build_server(factory())) as c:
            return c.info

    info = asyncio.run(served())
    m = load_manifest(ROOT / "components" / manifest)
    assert (m.name, m.version) == (info.name, info.version)
    assert set(m.implements) == set(info.tools) and set(m.kinds) == set(info.kinds)
