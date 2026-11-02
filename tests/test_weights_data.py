"""权重的数据和提案器（不需要 torch）：SFT / DPO / KTO 数据、PreferenceProposer、每题多答（loop.train_samples）。"""
import asyncio
import json
from pathlib import Path

import pytest
import yaml

from anyrsi import EvolvableInfo, Episode, ProposalRequest, Task, Trajectory, Verdict
from anyrsi_runtime import Runner, Wiring, load_config
from anyrsi_weights import (DataLeakError, PreferenceProposer, episodes_to_dpo_rows, episodes_to_kto_rows,
                             episodes_to_sft_rows, read_rows)

EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
W = EvolvableInfo(target="weights", type="weights", head="ckpt_root")


def ep(task_id, output, passed, split="train", n=0):
    task = Task(id=task_id, input=f"q {task_id}", split=split)
    traj = Trajectory(id=f"tr-{task_id}-{n}", task_id=task_id, output=output, steps=[])
    return Episode(task=task, trajectory=traj, verdict=Verdict(passed=passed, score=float(passed), judge_ref="j"))


def test_dpo_pairs_right_and_wrong_answers_to_the_same_question():
    eps = [ep("a", "4", True), ep("a", "5", False, n=1), ep("a", "6", False, n=2), ep("b", "1", True)]
    rows = episodes_to_dpo_rows(eps)
    assert [(r["chosen"], r["rejected"]) for r in rows] == [("4", "5"), ("4", "6")]  # b 只有对的，配不出对
    assert rows[0]["prompt"] == [{"role": "user", "content": "q a"}]
    assert episodes_to_dpo_rows(eps, max_pairs_per_task=1) == rows[:1]
    assert episodes_to_dpo_rows([ep("a", "4", True), ep("a", "4", False, n=1)]) == []  # 同样的回答一对一错：判别有噪声


def test_kto_uses_every_answer_and_sft_only_the_right_ones():
    eps = [ep("a", "4", True), ep("a", "5", False, n=1)]
    assert [r["label"] for r in episodes_to_kto_rows(eps)] == [True, False]
    assert len(episodes_to_sft_rows(eps)) == 1


@pytest.mark.parametrize("fn", [episodes_to_sft_rows, episodes_to_dpo_rows, episodes_to_kto_rows])
def test_non_train_answers_never_become_training_data(fn):
    with pytest.raises(DataLeakError):
        fn([ep("h", "4", True, split="holdout"), ep("h", "5", False, split="holdout", n=1)])


def test_preference_proposer(tmp_path):
    eps = [ep("a", "4", True), ep("a", "5", False, n=1), ep("b", "1", True), ep("b", "2", False, n=1)]
    req = ProposalRequest(target=W, targets=[W], parent_snapshot="ckpt_root", episodes=eps)
    [p] = PreferenceProposer(tmp_path, method="dpo", min_examples=2).propose(req)
    op = p.ops[0]
    assert op["method"] == "dpo" and "2 wrong answers used" in p.rationale
    assert len(read_rows(op["data"])) == 2
    assert PreferenceProposer(tmp_path, method="dpo", min_examples=3).propose(req) == []
    [k] = PreferenceProposer(tmp_path, method="kto", min_examples=2).propose(req)
    assert {r["label"] for r in read_rows(k.ops[0]["data"])} == {True, False}
    only_right = req.model_copy(update={"episodes": [e for e in eps if e.verdict.passed]})
    assert PreferenceProposer(tmp_path, method="kto", min_examples=1).propose(only_right) == []  # 只有一种标签学不到区别
    other = req.model_copy(update={"targets": [EvolvableInfo(target="memory", type="memory", head="s")],
                                   "target": EvolvableInfo(target="memory", type="memory", head="s")})
    assert PreferenceProposer(tmp_path).propose(other) == []
    with pytest.raises(ValueError, match="dpo 或 kto"):
        PreferenceProposer(tmp_path, method="ppo")


def test_wrong_answers_reach_weights_training_through_the_loop(tmp_path):
    data = {"protocol": "anyrsi/0.4", "name": "w", "plugins": ["weights_ops", "secret_ops"],
            "components": {
                "env": {"python": "secret_ops:SecretOpsEnv", "with": {"seed": 0}},
                "agent": {"python": "weights_ops:NoisyRuleAgent", "with": {"error_rate": 0.5}},
                "proposer": {"python": "anyrsi_weights:PreferenceProposer",
                             "with": {"data_dir": "state/pref", "method": "dpo", "min_examples": 1}},
                "judge": {"python": "anyrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
                "evidence": {"same_as": "judge"},
                "algorithm": {"python": "anyrsi_algorithm:GreedyAlgorithm", "with": {"max_generations": 1}},
                "gate": {"python": "anyrsi_gate:HoldoutGate"},
                "lineage": {"python": "anyrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}}},
            "evolvables": {"weights": {"python": "anyrsi_weights:WeightsEvolvable",
                                       "with": {"base_model": "tiny", "path": "state/weights", "dry_run": True}}},
            "loop": {"train_samples": 4, "holdout_limit": 8}}
    (tmp_path / "anyrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(tmp_path / "anyrsi.yaml")

    async def go():
        async with Wiring(cfg) as w:
            r = Runner(w, cfg.loop)
            await r._load_targets()
            genome = {t: (await x.facade.snapshot()).id for t, x in r.targets.items()}
            eps = await r.train_episodes((await w.env.tasks("train", 1, 3)), genome)
            await Runner(w, cfg.loop).run(1)
            return eps, await w.lineage.log()

    eps, log = asyncio.run(go())
    assert len(eps) == 12 and len({e.trajectory.id for e in eps}) == 12  # 3 题 × 4 次，轨迹 id 不重复
    patched = [e for e in log if e.patch is not None]
    assert patched and patched[0].patch.ops[0]["method"] == "dpo"
    rows = read_rows(patched[0].patch.ops[0]["data"])
    assert rows and all(r["chosen"] != r["rejected"] for r in rows)
    ckpt = Path(tmp_path / "state" / "weights" / "checkpoints")
    marker = json.loads(next(ckpt.glob("*/DRY_RUN.json")).read_text())
    assert marker["op"]["method"] == "dpo"


def test_builtin_trainer_command(tmp_path):
    from anyrsi_weights import BUILTIN, WeightsEvolvable

    w = WeightsEvolvable("m", tmp_path, trainer="builtin")
    assert w.trainer == BUILTIN and "anyrsi_weights.train" in w.trainer
