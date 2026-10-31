"""元层进化：improver（提案器的策略）和 curriculum（练习题怎么挑）按同预算下的下游提升准入。"""
import asyncio
from pathlib import Path

import openrsi_llm
import pytest
import yaml

from openrsi import EvolvableInfo, Episode, ProposalRequest, Task, Trajectory, Verdict
from openrsi_evolvables import CurriculumEvolvable, ImproverEvolvable, apply_curriculum
from openrsi_llm import PromptRewriter, StrategyRewriter
from openrsi_proposer import CurriculumProposer
from openrsi_runtime import Runner, Wiring, load_config

from meta_ops import ContextRecordingAgent

EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
IMPROVER = {"python": "openrsi_evolvables:ImproverEvolvable",
            "with": {"path": "state/improver", "initial": [{"id": "max_rules", "text": "learn at most 1 rule per patch",
                                                            "data": {"max_rules": 1, "order": 0}}]}}


def config(base: Path, proposer: dict, evolvables: dict, *, meta_every=2, agent="meta_ops:ContextRecordingAgent",
           train_limit=None, meta_min_gain=0.0):
    data = {"protocol": "openrsi/0.4", "name": "meta", "plugins": ["meta_ops", "secret_ops"],
            "components": {
                "env": {"python": "secret_ops:SecretOpsEnv", "with": {"seed": 0}},
                "agent": {"python": agent}, "proposer": proposer,
                "judge": {"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
                "evidence": {"same_as": "judge"},
                "attribution": {"python": "openrsi_attribution:RubricEvidenceAttributor"},
                "algorithm": {"python": "openrsi_algorithm:GreedyAlgorithm",
                              "with": {"max_generations": 50, "patience": 50}},
                "gate": {"python": "openrsi_gate:HoldoutGate"},
                "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
                "memory": {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}},
            "evolvables": evolvables,
            "loop": {"meta_every": meta_every, "meta_min_gain": meta_min_gain,
                     **({"train_limit": train_limit} if train_limit else {})}}
    base.mkdir(parents=True, exist_ok=True)
    (base / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(base / "openrsi.yaml")


def run(cfg, generations):
    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop).run(generations)
            return reports, await w.lineage.log()

    return asyncio.run(go())


# ---------------------------------------------------------------- improver

def test_better_improver_strategy_is_accepted_with_the_improvement_it_produced(tmp_path):
    ContextRecordingAgent.seen_kinds = set()
    cfg = config(tmp_path, {"python": "meta_ops:StrategicInducer", "with": {"step": 1}}, {"improver": IMPROVER})
    reports, log = run(cfg, 2)
    meta = next(e for e in log if e.generation == 2)
    assert reports[1].note.startswith("元层进化代：improver")
    assert meta.status == "accepted", meta.decision
    assert meta.metrics["meta_gain_candidate"] > meta.metrics["meta_gain_base"] >= 0
    assert meta.patch.target == "improver"
    # 接受的版本同时带着新的策略和它产出的解题侧改进，所以考试分比父版本高、算法会从它出发
    parent = next(e for e in log if e.id == meta.parent_ids[0])
    assert meta.metrics["holdout"] > parent.metrics["holdout"]
    assert meta.snapshots["improver"] != parent.snapshots["improver"]
    assert meta.snapshots["memory"] != parent.snapshots["memory"]
    assert "improver" not in ContextRecordingAgent.seen_kinds  # agent 从来看不到元层内容


def test_improver_change_that_does_not_help_is_rejected(tmp_path):
    cfg = config(tmp_path, {"python": "meta_ops:StrategicInducer", "with": {"step": 0}}, {"improver": IMPROVER})
    _, log = run(cfg, 2)
    meta = next(e for e in log if e.generation == 2)
    assert meta.status == "rejected"
    assert meta.metrics["meta_gain_candidate"] == meta.metrics["meta_gain_base"]
    assert not next(c for c in meta.decision.checks if c.name == "meta_gain").passed


def test_meta_min_gain_raises_the_bar(tmp_path):
    cfg = config(tmp_path, {"python": "meta_ops:StrategicInducer", "with": {"step": 1}}, {"improver": IMPROVER},
                 meta_min_gain=1.0)
    _, log = run(cfg, 2)
    assert next(e for e in log if e.generation == 2).status == "rejected"


def test_solver_generations_never_touch_meta_targets_and_resume_works(tmp_path):
    cfg = config(tmp_path, {"python": "meta_ops:StrategicInducer", "with": {"step": 1}}, {"improver": IMPROVER},
                 meta_every=4)
    _, log = run(cfg, 3)  # 1、2、3 代都是普通代
    root = next(e for e in log if e.status == "root")
    assert all(e.snapshots["improver"] == root.snapshots["improver"] for e in log)
    _, log2 = run(cfg, 1)  # 续跑到第 4 代：元层代
    assert any(e.generation == 4 and e.patch and e.patch.target == "improver" for e in log2)


def test_without_meta_every_meta_targets_stay_frozen(tmp_path):
    cfg = config(tmp_path, {"python": "meta_ops:StrategicInducer", "with": {"step": 1}}, {"improver": IMPROVER},
                 meta_every=None)
    _, log = run(cfg, 3)
    assert len({e.snapshots["improver"] for e in log}) == 1


# ---------------------------------------------------------------- curriculum

def test_curriculum_changes_train_selection_only(tmp_path):
    proposer = {"python": "openrsi_proposer:CompositeProposer",
                "with": {"proposers": ["curriculum", {"name": "rules", "python": "meta_ops:SolverOnlyInducer"}]}}
    cfg = config(tmp_path, proposer, {"curriculum": {"python": "openrsi_evolvables:CurriculumEvolvable",
                                                     "with": {"path": "state/curriculum"}}}, train_limit=3)

    async def go():
        async with Wiring(cfg) as w:
            r = Runner(w, cfg.loop)
            reports = await r.run(4)
            log = await w.lineage.log()
            return r, reports, log

    r, reports, log = asyncio.run(go())
    metas = [e for e in log if e.patch and e.patch.target == "curriculum"]
    assert metas and all("meta_gain_candidate" in e.metrics for e in metas)
    assert all(e.metrics["n_holdout"] == 40 for e in log if "n_holdout" in e.metrics)  # 考试题始终是同一套
    assert "curriculum" not in ContextRecordingAgent.seen_kinds


def test_apply_curriculum_orders_drops_and_limits():
    tasks = [Task(id=f"t{i}", tags=[tag]) for i, tag in enumerate("aabbcc")]
    items = [{"data": {"tag": "a", "weight": 0}}, {"data": {"tag": "c", "weight": 3}}]
    assert [t.id for t in apply_curriculum(tasks, items, None)] == ["t4", "t5", "t2", "t3"]
    assert [t.id for t in apply_curriculum(tasks, items, 3)] == ["t4", "t5", "t2"]
    assert [t.id for t in apply_curriculum(tasks, [], 2)] == ["t0", "t1"]


def test_curriculum_evolvable_validates_weights(tmp_path):
    c = CurriculumEvolvable(path=tmp_path / "c", initial={"x": 1.0})
    from openrsi import ComponentError, Patch
    with pytest.raises(ComponentError, match="非负数"):
        c.apply(Patch(target="curriculum", ops=[{"op": "add", "item": {"id": "tag:y", "kind": "curriculum",
                                                                        "text": "y", "data": {"tag": "y", "weight": -1}}}]))


def _ep(tag, passed):
    task = Task(id=f"{tag}-{passed}", tags=[tag], split="train")
    traj = Trajectory(id=f"tr-{tag}-{passed}", task_id=task.id, steps=[])
    return Episode(task=task, trajectory=traj, verdict=Verdict(task_id=task.id, trajectory_id=traj.id, passed=passed,
                                                               score=float(passed), judge_ref="j"))


def test_curriculum_proposer_drops_mastered_and_focuses_failing_tags():
    info = EvolvableInfo(target="curriculum", type="curriculum", head="s")
    req = ProposalRequest(target=info, parent_snapshot="s", episodes=[_ep("a", True), _ep("a", True), _ep("b", False)])
    [p] = CurriculumProposer().propose(req)
    weights = {op["item"]["data"]["tag"]: op["item"]["data"]["weight"] for op in p.ops}
    assert weights == {"a": 0.0, "b": 2.0}
    other = req.model_copy(update={"target": EvolvableInfo(target="memory", type="memory", head="s")})
    assert CurriculumProposer().propose(other) == []


# ---------------------------------------------------------------- LLM 提案器读写策略

def _fake_chat(monkeypatch, reply):
    seen = []

    def chat(base_url, model, messages, **kw):
        seen.append(messages)
        return {"text": reply, "usage": {}, "finish_reason": "stop"}

    monkeypatch.setattr(openrsi_llm, "chat", chat)
    return seen


def test_prompt_rewriter_reads_improver_strategy(monkeypatch):
    seen = _fake_chat(monkeypatch, "Be precise.")
    info = EvolvableInfo(target="prompt", type="prompt", head="s")
    bad = _ep("a", False)
    req = ProposalRequest(target=info, targets=[info], parent_snapshot="s", episodes=[bad], state=[], states={})
    PromptRewriter("http://x/v1").propose(req)
    assert "Strategy notes" not in seen[-1][0]["content"]  # 没有策略时请求不变
    hinted = req.model_copy(update={"hints": {"improver": {"improver": [{"id": "s", "text": "Prefer short rules."}]}}})
    PromptRewriter("http://x/v1").propose(hinted)
    assert "Prefer short rules." in seen[-1][0]["content"]


def test_strategy_rewriter_only_touches_improver(monkeypatch):
    _fake_chat(monkeypatch, "Keep rules general.")
    imp = EvolvableInfo(target="improver", type="improver", head="s")
    req = ProposalRequest(target=imp, parent_snapshot="s", state=[], hints={"meta_history": [{"holdout": 0.5}]})
    [p] = StrategyRewriter("http://x/v1").propose(req)
    assert p.target == "improver" and p.ops[0]["item"]["text"] == "Keep rules general."
    ImproverEvolvable(path=None).apply(p.model_copy(update={"base_snapshot": None}))
    prompt = req.model_copy(update={"target": EvolvableInfo(target="prompt", type="prompt", head="s")})
    assert StrategyRewriter("http://x/v1").propose(prompt) == []


def test_new_manifests_match():
    from openrsi import build_server, connect, load_manifest

    root = Path(__file__).resolve().parents[1]
    for manifest, factory in (("evolvables/rsi-component.improver.yaml", lambda: ImproverEvolvable()),
                              ("evolvables/rsi-component.curriculum.yaml", lambda: CurriculumEvolvable())):
        async def served(factory=factory):
            async with connect(build_server(factory())) as c:
                return c.info

        info = asyncio.run(served())
        m = load_manifest(root / "components" / manifest)
        assert (m.name, m.version) == (info.name, info.version) and set(m.implements) == set(info.tools)


def test_ordinary_generations_do_not_offer_meta_targets_to_the_proposer(tmp_path):
    cfg = config(tmp_path, {"python": "meta_ops:SneakyProposer"}, {"improver": IMPROVER}, meta_every=None)
    _, log = run(cfg, 3)
    assert not [e for e in log if e.patch and e.patch.target == "improver"]
    assert len({e.snapshots["improver"] for e in log}) == 1


def test_curriculum_really_steers_which_train_tasks_are_picked(tmp_path):
    cfg = config(tmp_path, {"python": "meta_ops:SolverOnlyInducer"},
                 {"curriculum": {"python": "openrsi_evolvables:CurriculumEvolvable",
                                 "with": {"path": "state/curriculum"}}}, train_limit=3)

    async def go():
        async with Wiring(cfg) as w:
            r = Runner(w, cfg.loop)
            await r._load_targets()
            genome = {t: (await x.facade.snapshot()).id for t, x in r.targets.items()}
            plain = await r._train_tasks(1, genome)
            drop = plain[0].tags[0]
            from openrsi import Patch
            item = {"id": f"tag:{drop}", "kind": "curriculum", "text": "x", "data": {"tag": drop, "weight": 0}}
            snap = await r.targets["curriculum"].facade.apply(Patch(target="curriculum", ops=[{"op": "add", "item": item}]))
            steered = await r._train_tasks(1, {**genome, "curriculum": snap.id})
            return drop, plain, steered

    drop, plain, steered = asyncio.run(go())
    assert all(drop in t.tags for t in plain)  # 不加课程时前 3 题都是同一个符号
    assert len(steered) == 3 and not any(drop in t.tags for t in steered)
