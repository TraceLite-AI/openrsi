"""有方向的自我改进（loop.directed）：统计函数、一组的结论、动量、收敛判据、和主循环的接线。"""
import asyncio
import math
import random
import sys
from pathlib import Path

import pytest

from anyrsi import EvolvableInfo, ProposalRequest
from anyrsi_runtime.directed import (Version, analyze_group, betainc, chi2_sf, f_sf, flat, friedman, mean_z,
                                     nested_anova, pick_directions, update_momentum, z_threshold)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "directed_toy"))
import directed_toy as toy  # noqa: E402

# ---------------------------------------------------------------- 分布函数（对照统计表的值）


def test_distribution_functions_match_tables():
    assert chi2_sf(5.991, 2) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf(18.307, 10) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf(0.0, 3) == 1.0
    assert f_sf(4.256, 2, 9) == pytest.approx(0.05, abs=1e-4)
    assert f_sf(10.13, 1, 3) == pytest.approx(0.05, abs=1e-3)
    assert f_sf(1.0, 1, 1) == pytest.approx(0.5, abs=1e-9)
    assert f_sf(math.inf, 2, 3) == 0.0
    assert betainc(2, 3, 0.4) == pytest.approx(0.5248, abs=1e-4)
    assert z_threshold(0.05) == pytest.approx(1.6449, abs=1e-4)
    assert z_threshold(0.05, 10) == pytest.approx(2.5758, abs=1e-4)


def test_friedman_and_ties():
    stat, p = friedman([[1, 2, 3], [1, 3, 2], [1, 2, 3], [2, 1, 3]])
    assert stat == pytest.approx(4.5) and p == pytest.approx(chi2_sf(4.5, 2))
    assert friedman([[1, 1, 1], [0, 0, 0]]) == (0.0, 1.0)  # 全部并列的题没有信息
    _, p_strong = friedman([[0, 1]] * 20)
    assert p_strong < 1e-4


def test_friedman_tie_correction_matches_general_formula():
    from anyrsi_runtime.directed import _ranks

    blocks = [[0, 1, 1], [0, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0.5, 1, 1]]
    k, n = 3, len(blocks)
    ranks = [_ranks(b) for b in blocks]
    rj = [sum(r[j] for r in ranks) for j in range(k)]
    q = (k - 1) * sum((x - n * (k + 1) / 2) ** 2 for x in rj) / (
        sum(v * v for r in ranks for v in r) - n * k * (k + 1) ** 2 / 4)
    stat, p = friedman(blocks)
    assert stat == pytest.approx(q) and p == pytest.approx(chi2_sf(q, k - 1))


def test_nested_anova_separates_direction_from_lucky_edits():
    f, p = nested_anova({"a": [0.30, 0.32], "b": [0.0, 0.02]}, 40)
    assert p < 0.01 and f > 50
    f, p = nested_anova({"a": [0.40, -0.05], "b": [0.30, 0.0]}, 40)  # 同方向内差别比方向间大
    assert p > 0.3
    assert math.isnan(nested_anova({"a": [0.3], "b": [0.1]}, 40)[0])  # 每个方向只有 1 个修改
    assert math.isnan(nested_anova({"a": [0.3, 0.2]}, 40)[0])         # 只有 1 个方向
    assert nested_anova({"a": [0.3, 0.3], "b": [0.1, 0.1]}, 40) == (math.inf, 0.0)


def test_mean_z():
    assert mean_z([]) == (0.0, 0.0)
    assert mean_z([0.5, 0.5]) == (0.5, math.inf)
    mu, z = mean_z([1, 0, 1, 1])
    assert mu == 0.75 and z == pytest.approx(0.75 / (0.5 / 2))


# ---------------------------------------------------------------- 一组的结论


def versions(effects: dict[str, list[float]], n: int = 60, seed: int = 0, base: float = 0.4):
    """对照 + 每个方向若干修改。effects[方向] = 每个修改的真实提升；每题 0/1，按概率掷硬币。"""
    rng = random.Random(seed)
    tasks = [f"t{i}" for i in range(n)]
    control = Version(None, 0, {t: float(rng.random() < base) for t in tasks},
                      {t: {"correct": 0.5} for t in tasks})
    out = [Version(d, j, {t: float(rng.random() < base + e) for t in tasks}, {t: {"correct": 0.5 + e} for t in tasks})
           for d, es in effects.items() for j, e in enumerate(es)]
    return control, out


def test_clear_direction_is_established():
    control, vs = versions({"good": [0.45, 0.45], "meh": [0.0, 0.0], "bad": [-0.3, -0.3]}, n=80)
    g = analyze_group(control, vs, alpha=0.05, tests=3)
    assert g.winner == "good" and g.established, g.reason
    good = next(r for r in g.directions if r.direction == "good")
    assert good.significant and good.per_rubric["correct"] == pytest.approx(0.45)
    assert [r.direction for r in g.directions][0] == "good"
    assert len(good.diffs) == 80 and good.n == 80 and good.se > 0


def test_single_edit_per_direction_is_not_established():
    control, vs = versions({"good": [0.45], "bad": [-0.3]}, n=80)
    g = analyze_group(control, vs, alpha=0.05, tests=2)
    assert g.winner == "good" and not g.established and "只有 1 个修改" in g.reason


def test_direction_not_established_when_edits_disagree():
    # 同一方向里一个修改很好、一个没用，另一个方向两个都一般：方向之间的差别不大于方向内部
    control, vs = versions({"x": [0.5, 0.0], "y": [0.25, 0.25]}, n=80, seed=3)
    g = analyze_group(control, vs, alpha=0.05, tests=2)
    assert g.winner is not None and not g.established and "某次写法好" in g.reason


def test_no_difference_means_no_direction():
    control, vs = versions({"a": [0.0, 0.0], "b": [0.0, 0.0]}, n=40, seed=1)
    g = analyze_group(control, vs, alpha=0.05, tests=2)
    assert g.winner is None and not g.established


def test_evidence_accumulates_and_threshold_tightens():
    control, vs = versions({"a": [0.12]}, n=30, seed=5)
    alone = analyze_group(control, vs, alpha=0.05, tests=1)
    hist = {"a": [0.12 + (0.5 if i % 2 else -0.3) for i in range(200)]}  # 均值 0.22，强证据
    pooled = analyze_group(control, vs, alpha=0.05, tests=1, history=hist, looks={"a": 3})
    a0, a1 = alone.directions[0], pooled.directions[0]
    assert a1.n == 230 and a1.z > a0.z
    assert a1.z_crit > a0.z_crit  # 第 4 次检验：α / (4·5)
    assert a1.z_crit == pytest.approx(z_threshold(0.05, 1 * 4 * 5))


def test_inconsistent_task_groups_restrict_scope():
    tasks = [f"t{i}" for i in range(40)]
    control = Version(None, 0, {t: 0.5 for t in tasks})
    v = Version("d", 0, {t: (1.0 if i < 20 else 0.0) for i, t in enumerate(tasks)})
    groups = {t: ("easy" if i < 20 else "hard") for i, t in enumerate(tasks)}
    g = analyze_group(control, [v], alpha=0.05, tests=1, task_groups=groups)
    assert g.directions[0].scope == ["easy"]


def test_mismatched_tasks_refused():
    control = Version(None, 0, {"a": 1.0})
    with pytest.raises(ValueError, match="配对"):
        analyze_group(control, [Version("d", 0, {"b": 1.0})], alpha=0.05, tests=1)


def test_flat_needs_confidence_not_just_silence():
    control, vs = versions({"a": [0.0]}, n=20, seed=2)
    noisy = analyze_group(control, vs, alpha=0.05, tests=1)
    assert noisy.winner is None and not flat(noisy, 0.02)  # 只是证据不够
    tasks = [f"t{i}" for i in range(50)]
    same = analyze_group(Version(None, 0, {t: 0.5 for t in tasks}), [Version("a", 0, {t: 0.5 for t in tasks})],
                         alpha=0.05, tests=1)
    assert flat(same, 0.02)  # 50 道题上和对照完全一样：有把握没有效果
    control, vs = versions({"a": [0.45]}, n=60)
    assert not flat(analyze_group(control, vs, alpha=0.05, tests=1), 0.02)  # 有胜出者


def test_momentum_shrinks_weak_evidence():
    control, vs = versions({"good": [0.45, 0.45], "weak": [0.05, 0.05]}, n=80)
    g = analyze_group(control, vs, alpha=0.05, tests=2)
    m = update_momentum({"old": 0.3}, g, beta=0.5)
    good = next(r for r in g.directions if r.direction == "good")
    weak = next(r for r in g.directions if r.direction == "weak")
    assert m["old"] == 0.3  # 没被试过的方向不变
    assert m["good"] == pytest.approx(0.5 * good.advantage * (1 - 1 / good.z ** 2))
    assert abs(m["weak"]) < abs(0.5 * weak.advantage) + 1e-12


def test_pick_directions_orders_by_momentum_then_support_and_explores():
    cands = {"a": 5.0, "b": 1.0, "c": 3.0, "d": 0.0}
    rng = random.Random(0)
    assert pick_directions(cands, {"b": 0.4}, 2, 0, rng) == ["b", "a"]
    picked = pick_directions(cands, {}, 3, 1, random.Random(1))
    assert picked[:2] == ["a", "c"] and picked[2] in ("b", "d")
    thirds = {pick_directions(cands, {}, 3, 1, random.Random(i))[2] for i in range(30)}
    assert thirds == {"b", "d"}  # 探索名额不按排名，排在最后的方向也有机会


# ---------------------------------------------------------------- 主循环


def run_toy(tmp_path, group: str, budget: int, **kw):
    return asyncio.run(toy.run_group(group, tmp_path / group, seed=0, budget=budget, **kw))


def test_directed_loop_runs_and_records_group_statistics(tmp_path):
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(toy.write_config(tmp_path / "C", "C", 0, 30))
    seen_hints = []

    async def go():
        async with Wiring(cfg) as w:
            inner = w.components["proposer"]

            class Spy:
                kinds = inner.kinds

                async def propose(self, req):
                    seen_hints.append(req.hints)
                    return await inner.propose(req)

                def __getattr__(self, name):
                    return getattr(inner, name)

            w.components["proposer"] = Spy()
            counter = toy._Counting(w.components["agent"], 2500)
            w.components["agent"] = counter
            runner = Runner(w, cfg.loop)
            reports = []
            runner.on_generation = reports.append
            try:
                await runner.run()
            except toy.BudgetExhausted:
                pass
            return reports, await w.lineage.log()

    reports, log = asyncio.run(go())
    groups = [r.group for r in reports if r.group]
    assert groups, [r.note for r in reports]
    assert all(h.get("direction", {}).get("id", "").startswith("item:s") for h in seen_hints)
    assert {"friedman_p", "directions", "momentum", "tried"} <= set(groups[-1])
    accepted = [e for e in log if e.status == "accepted"]
    assert accepted, "玩具问题上 2500 次调用应当至少接受一个改动"
    e = accepted[0]
    assert "group_advantage" in e.metrics and e.patch.rationale.startswith("[方向 item:s")
    assert any(c.name == "group_comparison" for c in e.decision.checks)


def test_directed_beats_nothing_and_counts_control(tmp_path):
    r = run_toy(tmp_path, "C", 3000)
    assert r["true_final"] > r["true_start"]
    # 每组：对照 30 + 3 个方向 × 1 个修改 × 30 + 考试 30 ≈ 150 次调用，所以 3000 次里大约 20 组
    assert r["calls"] <= 3000


def test_convergence_when_no_direction_can_help(tmp_path):
    """方向只有两个无关的位置（权重 0），agent 用公共随机数：每一组都和对照完全一样 → 有把握到头了 → 停。"""
    import yaml

    from anyrsi_runtime import Runner, Wiring, load_config

    path = toy.write_config(tmp_path / "flat", "C", 0, 30)
    data = yaml.safe_load(path.read_text())
    data["loop"]["directed"].update({"directions": ["item:s10", "item:s11"], "from_attribution": False, "patience": 2,
                                     "explore": 0, "group_size": 2})
    data["components"]["proposer"]["with"].update({"q": 1.0, "collateral": 0.0})
    data["components"]["agent"]["with"]["common_noise"] = True
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)

    async def go():
        async with Wiring(cfg) as w:
            runner = Runner(w, cfg.loop)
            reports = await runner.run(20)
            return runner, reports, await w.lineage.log()

    runner, reports, log = asyncio.run(go())
    assert runner.converged and "局部收敛" in runner.converged
    assert len(reports) == 2 and all("有把握到头了" in r.note for r in reports)
    assert [e.status for e in log] == ["root"]  # 什么都没收


def test_noisy_irrelevant_edits_do_not_converge_early(tmp_path):
    """同样的无关方向，但 agent 每次重新掷硬币：只是证据不够，不能判成收敛。"""
    import yaml

    from anyrsi_runtime import Runner, Wiring, load_config

    path = toy.write_config(tmp_path / "noisy", "C", 0, 30)
    data = yaml.safe_load(path.read_text())
    data["loop"]["directed"].update({"directions": ["item:s10", "item:s11"], "from_attribution": False, "patience": 2,
                                     "explore": 0, "group_size": 2})
    data["components"]["proposer"]["with"].update({"q": 1.0, "collateral": 0.0})
    path.write_text(yaml.safe_dump(data))
    cfg = load_config(path)

    async def go():
        async with Wiring(cfg) as w:
            runner = Runner(w, cfg.loop)
            reports = await runner.run(3)
            return runner, reports

    runner, reports = asyncio.run(go())
    assert runner.converged is None and len(reports) == 3


def test_direction_note_reaches_llm_proposers():
    from anyrsi_llm import direction_note

    info = EvolvableInfo(target="prompt", type="prompt", head="h")
    req = ProposalRequest(target=info, parent_snapshot="p",
                          hints={"direction": {"id": "item:s1", "text": "always check units"}, "edit": 1, "step": 2})
    note = direction_note(req)
    assert "always check units" in note and "different variant" in note and "Size of the change: 2" in note
    assert direction_note(ProposalRequest(target=info, parent_snapshot="p")) == ""


def test_config_validation():
    from anyrsi_runtime.config import DirectedConfig

    assert DirectedConfig().edits_per_direction == 1 and DirectedConfig().submit == "best_edit"
    with pytest.raises(ValueError):
        DirectedConfig(submit="whatever")
    with pytest.raises(ValueError):
        DirectedConfig(alpha=1.5)


def test_direction_probe_common_random_numbers_find_the_right_direction():
    """方向识别测试的小规模版本：公共随机数 + 每方向 3 个修改，3 个方向里挑对最好那个的概率要明显高于随机（1/3）。"""
    import direction_probe as dp

    model = dp.Model(q=0.6, collateral=0.5)
    gains = model.true_gains(toy.START, draws=600)
    weak = dp.single_group(model, toy.START, gains, G=3, N=30, m=1, repeats=1, trials=150)
    strong = dp.single_group(model, toy.START, gains, G=3, N=30, m=3, repeats=1, trials=150, crn=True)
    assert strong["P(平均优势最高=最好)"] > 0.55
    assert strong["P(平均优势最高=最好)"] > weak["P(平均优势最高=最好)"] + 0.08
    assert 0.2 < strong["P(随机=最好)"] < 0.5
