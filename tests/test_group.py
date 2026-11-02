"""组提案（loop.group）：一组候选在考试题上比，四种策略、选择半 / 确认半、公共随机数、经验写入、主循环接线。"""
import asyncio
import math
import sys
from pathlib import Path

import pytest

from anyrsi import Item, Task
from anyrsi_runtime.group import (GRPO, BestOfN, GroupStrategy, Member, ParentPaired, TrainingFreeGRPO, lesson_patch,
                                  load_strategy, mean_z, rollout_seed, split_tasks, z_threshold)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))
import group_toy as gt  # noqa: E402

# ---------------------------------------------------------------- 工具函数


def test_split_is_deterministic_disjoint_and_sized():
    keys = [f"t{i}" for i in range(60)]
    sel, conf = split_tasks(keys, 0.5, "run1")
    assert len(sel) == 30 and len(conf) == 30 and not set(sel) & set(conf) and set(sel) | set(conf) == set(keys)
    assert split_tasks(list(reversed(keys)), 0.5, "run1") == (sel, conf)
    assert split_tasks(keys, 0.5, "run2") != (sel, conf)
    assert split_tasks(keys, 0.0, "x") == (sorted(keys), [])
    s2, c2 = split_tasks(["a", "b"], 0.9, "x")
    assert len(s2) == 1 and len(c2) == 1  # 两边至少各留一道


def test_rollout_seed_and_stats():
    assert rollout_seed("r", 3, "t1") == rollout_seed("r", 3, "t1") != rollout_seed("r", 4, "t1")
    assert mean_z([]) == (0.0, 0.0) and mean_z([0.5, 0.5]) == (0.5, math.inf)
    assert z_threshold(0.05, 4) == pytest.approx(2.2414, abs=1e-3)


# ---------------------------------------------------------------- 策略


def group(n_tasks=40, deltas=(0.0, 0.5, 0.0)):
    parent = {f"t{i}": float(i % 2) for i in range(n_tasks)}
    members = []
    for j, d in enumerate(deltas):
        scores = {t: min(1.0, max(0.0, v + (d if (int(t[1:]) % 4 < 2) else 0.0))) for t, v in parent.items()}
        members.append(Member(index=j, scores=scores, rationale=f"set s{j}=good", direction=f"item:s{j}"))
    return parent, members


def test_parent_paired_needs_significance_after_bonferroni():
    parent, members = group(deltas=(0.0, 1.0, -1.0))
    d = ParentPaired().decide(parent, members, 0.05)
    assert d.chosen == 1 and d.advantage[1] > 0 and d.z[1] > z_threshold(0.05, 3)
    parent, members = group(deltas=(0.0, 0.0, 0.0))
    assert ParentPaired().decide(parent, members, 0.05).chosen is None
    # 只在 1 道题上好：平均为正，但不显著
    parent = {f"t{i}": 0.0 for i in range(40)}
    lucky = Member(0, {**parent, "t0": 1.0})
    assert ParentPaired().decide(parent, [lucky], 0.05).chosen is None


def test_parent_paired_threshold_grows_with_group_size():
    """4 道题提升、36 道不变：z ≈ 2.08。单独一个候选够门槛（1.645），三个候选时门槛 2.13 就不够。"""
    parent = {f"t{i}": 0.0 for i in range(40)}
    good = Member(0, {**parent, **{f"t{i}": 1.0 for i in range(4)}})
    same = [Member(i, dict(parent)) for i in (1, 2)]
    assert 1.645 < mean_z([1.0] * 4 + [0.0] * 36)[1] < z_threshold(0.05, 3)
    assert ParentPaired().decide(parent, [good], 0.05).chosen == 0
    assert ParentPaired().decide(parent, [good, *same], 0.05).chosen is None


def test_training_free_advantage_is_relative_to_group_including_parent():
    parent = {"a": 0.0, "b": 0.0}
    members = [Member(0, {"a": 1.0, "b": 0.0}), Member(1, {"a": 1.0, "b": 1.0})]
    adv = TrainingFreeGRPO().advantages(parent, members)
    # 题 a：组 [0, 1, 1]，平均 2/3；题 b：组 [0, 0, 1]，平均 1/3
    assert adv[0] == pytest.approx({"a": 1 / 3, "b": -1 / 3}) and adv[1] == pytest.approx({"a": 1 / 3, "b": 2 / 3})


def test_grpo_advantage_formula_and_choice():
    parent = {"a": 0.0, "b": 1.0}
    members = [Member(0, {"a": 1.0, "b": 1.0}), Member(1, {"a": 0.0, "b": 0.0})]
    adv = GRPO(normalize=False).advantages(parent, members)
    # 题 a：组 [1, 0, 0]，平均 1/3；题 b：组 [1, 0, 1]，平均 2/3
    assert adv[0] == pytest.approx({"a": 2 / 3, "b": 1 / 3}) and adv[1] == pytest.approx({"a": -1 / 3, "b": -2 / 3})
    norm = GRPO().advantages(parent, members)
    assert norm[0]["a"] == pytest.approx((1 - 1 / 3) / (math.sqrt(2 / 9) + 1e-6))
    no_parent = GRPO(normalize=False, include_parent=False).advantages(parent, members)
    assert no_parent[0] == pytest.approx({"a": 0.5, "b": 0.5})
    assert GRPO().decide(parent, members, 0.05).chosen == 0
    assert GRPO().decide(parent, [Member(0, dict(parent))], 0.05).chosen is None  # 和父版本一样，不选


def test_best_of_n_picks_highest_raw_score_but_not_below_parent():
    parent, members = group(deltas=(0.3, 0.6, 0.1))
    assert BestOfN().decide(parent, members, 0.05).chosen == 1
    parent = {"a": 1.0, "b": 1.0}
    worse = [Member(0, {"a": 0.0, "b": 1.0}), Member(1, {"a": 0.0, "b": 0.0})]
    assert BestOfN().decide(parent, worse, 0.05).chosen is None


def test_training_free_writes_relative_lessons():
    parent, members = group(deltas=(0.0, 1.0, -1.0))
    d = TrainingFreeGRPO(max_lessons=2).decide(parent, members, 0.05)
    assert d.chosen is None and len(d.lessons) == 2
    texts = {x.text.split(" ")[0] for x in d.lessons}
    assert texts == {"Better", "Worse"}
    better = next(x for x in d.lessons if x.text.startswith("Better"))
    assert "set s1=good" in better.text and better.kind == "lesson" and better.data["direction"] == "item:s1"
    parent, members = group(deltas=(0.0, 0.0))
    assert TrainingFreeGRPO().decide(parent, members, 0.05).lessons == []


def custom_summarizer(parent, members, advantages):
    return [Item(id="custom", kind="lesson", text=f"{len(members)} members")]


class AlwaysFirst(GroupStrategy):
    name = "always_first"

    def advantages(self, parent, members):
        return [{t: 0.0 for t in parent} for _ in members]

    def decide(self, parent, members, alpha):
        from anyrsi_runtime.group import Decision
        return Decision(0, "always the first one", [0.0] * len(members), [0.0] * len(members))


def test_load_strategy_variants():
    assert isinstance(load_strategy("parent_paired"), ParentPaired)
    g = load_strategy({"name": "grpo", "with": {"normalize": False}})
    assert isinstance(g, GRPO) and g.normalize is False
    assert isinstance(load_strategy({"python": "test_group:AlwaysFirst"}), AlwaysFirst)
    tf = load_strategy({"name": "training_free", "with": {"summarizer": "test_group:custom_summarizer"}})
    parent, members = group()
    assert [x.id for x in tf.decide(parent, members, 0.05).lessons] == ["custom"]
    with pytest.raises(ValueError, match="未知"):
        load_strategy("nope")
    with pytest.raises(ValueError, match="不是组内优势策略"):
        load_strategy({"python": "anyrsi:Item", "with": {"id": "x"}})


def test_lesson_patch_adds_or_replaces():
    p = lesson_patch("exp", "snap", [Item(id="a", kind="lesson", text="x"), Item(id="b", kind="lesson", text="y")],
                     {"a"})
    assert [o["op"] for o in p.ops] == ["replace", "add"] and p.base_snapshot == "snap"


# ---------------------------------------------------------------- 主循环


def run_with_spies(tmp_path, arm, budget=1500, **kw):
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(gt.write_config(tmp_path / arm, arm, 0, **kw))
    seeds, gate_calls, proposal_targets = {}, [], []

    async def go():
        async with Wiring(cfg) as w:
            agent, gate, proposer = w.components["agent"], w.components["gate"], w.components["proposer"]

            class AgentSpy:
                kinds = agent.kinds

                async def run(self, task, context, snapshot_id=None):
                    if task.split == "holdout":
                        seeds.setdefault((task.id, task.metadata.get("rollout_seed")), set()).add(snapshot_id)
                    return await agent.run(task, context, snapshot_id)

                def __getattr__(self, n):
                    return getattr(agent, n)

            class GateSpy:
                kinds = gate.kinds

                async def check(self, req):
                    gate_calls.append(req)
                    return await gate.check(req)

                def __getattr__(self, n):
                    return getattr(gate, n)

            class ProposerSpy:
                kinds = proposer.kinds

                async def propose(self, req):
                    proposal_targets.append(req.target.target)
                    return await proposer.propose(req)

                def __getattr__(self, n):
                    return getattr(proposer, n)

            counter = gt._Counting(AgentSpy(), budget)
            w.components.update({"agent": counter, "gate": GateSpy(), "proposer": ProposerSpy()})
            runner = Runner(w, cfg.loop)
            reports = []
            runner.on_generation = reports.append
            try:
                await runner.run()
            except gt.BudgetExhausted:
                pass
            return runner, reports, await w.lineage.log()

    runner, reports, log = asyncio.run(go())
    return runner, reports, log, seeds, gate_calls, proposal_targets


@pytest.mark.parametrize("arm", ["paired", "grpo", "best_of_n"])
def test_group_selection_compares_on_holdout_and_confirms_on_the_other_half(tmp_path, arm):
    runner, reports, log, seeds, gate_calls, _ = run_with_spies(tmp_path, arm)
    groups = [r.group for r in reports if r.group]
    assert groups and all(g["size"] >= 1 for g in groups)
    sel, conf = set(runner.holdout_select), set(runner.holdout_confirm)
    assert sel and conf and not sel & conf
    # 准入只看确认半
    for req in gate_calls:
        assert set(req.candidate_tasks) == conf and set(req.baseline_tasks) == conf
    # 公共随机数：同一代同一道题，父版本和候选用的是同一个种子（同一个种子下出现了多个版本）
    multi = [v for (tid, seed), v in seeds.items() if seed is not None and len(v) > 1]
    assert multi, "同一个种子应当被多个版本用过"
    accepted = [e for e in log if e.status == "accepted"]
    if arm != "paired":
        assert accepted
    for e in accepted:
        assert "holdout_confirm" in e.metrics and any(c.name == "group" for c in e.decision.checks)


def test_training_free_writes_lessons_and_never_lets_the_proposer_touch_them(tmp_path):
    runner, reports, log, _, gate_calls, targets = run_with_spies(tmp_path, "training_free", budget=2500)
    assert targets and set(targets) == {"prompt"}  # 提案器只改 prompt，不碰经验库
    written = [r for r in reports if r.group and r.group["lessons"]]
    assert written
    accepted = [e for e in log if e.status == "accepted"]
    assert accepted and all(e.patch.target == "experience" for e in accepted)
    assert all(e.patch.ops[0]["item"]["kind"] == "lesson" for e in accepted)


def test_training_free_requires_an_experience_target(tmp_path):
    import yaml

    from anyrsi_runtime import Runner, Wiring, load_config

    path = gt.write_config(tmp_path / "bad", "training_free", 0)
    data = yaml.safe_load(path.read_text())
    del data["loop"]["group"]["experience_target"]
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)

    async def go():
        async with Wiring(cfg) as w:
            await Runner(w, cfg.loop).run(1)

    with pytest.raises(ValueError, match="experience_target"):
        asyncio.run(go())


def test_group_beats_nothing_on_the_toy(tmp_path):
    r = asyncio.run(gt.run_arm("best_of_n", tmp_path / "b", 0, 3000))
    assert r["true_final"] > r["true_start"] + 0.1 and r["harmful"] == 0


def test_chat_agent_passes_rollout_seed():
    from anyrsi_llm import seed_extra

    assert seed_extra(Task(id="t", metadata={"rollout_seed": 42})) == {"seed": 42}
    assert seed_extra(Task(id="t")) is None
    assert seed_extra(Task(id="t", metadata={"rollout_seed": True})) is None


def test_config_validation():
    from anyrsi_runtime.config import GroupConfig

    c = GroupConfig()
    assert (c.size, c.strategy, c.confirm_fraction, c.common_seed) == (4, "parent_paired", 0.5, True)
    with pytest.raises(ValueError):
        GroupConfig(proposals="whatever")
    with pytest.raises(ValueError):
        GroupConfig(confirm_fraction=1.0)
