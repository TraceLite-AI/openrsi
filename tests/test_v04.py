"""v0.4：前沿进化算法、逐题不回退准入、harness 进化、verl 适配、RSIHub 适配。"""
import asyncio
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from openrsi import (Episode, Patch, ProposalRequest, Proposer, Task, TellRequest, Trajectory, Verdict,
                     build_server, connect, load_manifest)
from openrsi.conformance import check
from openrsi_algorithm import (ArchiveAlgorithm, GreedyAlgorithm, IslandAlgorithm, MapElitesAlgorithm,
                               ParetoAlgorithm, ShinkaAlgorithm)
from openrsi_gate import HoldoutGate
from openrsi_guard import PolicyGuard
from openrsi_harness import HarnessAgent, HarnessEvolvable
from openrsi_judge import EvidenceJudge
from openrsi_runtime import Runner, Wiring, load_config
from openrsi.types import GateRequest

ROOT = Path(__file__).resolve().parents[1]
EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}


def tell(alg, gen, cid, score, parent=None, accepted=True, tasks=None, metrics=None, proposer=None):
    m = {"holdout": score, **(metrics or {})}
    return alg.tell(TellRequest(generation=gen, parent_id=parent, candidate_id=cid, accepted=accepted, metrics=m,
                                task_scores=tasks or {}, proposer=proposer))


def served_info(component):
    async def go():
        async with connect(build_server(component)) as c:
            return c.info

    return asyncio.run(go())


# ---------------------------------------------------------------- A. 算法

def test_pareto_keeps_specialists_and_drops_dominated():
    alg = ParetoAlgorithm(seed=1)
    tell(alg, 0, "root", 0.3, tasks={"a": 0.0, "b": 0.0, "c": 1.0})
    tell(alg, 1, "x", 0.5, "root", tasks={"a": 1.0, "b": 0.0, "c": 1.0})  # 支配 root
    tell(alg, 2, "y", 0.4, "root", tasks={"a": 0.0, "b": 1.0, "c": 0.0})  # 只在 b 上最好
    front = alg.frontier()
    assert set(front) == {"x", "y"}  # 平均分更低的 y 也留在前沿
    picks = {alg.ask().parent_id for _ in range(40)}
    assert picks == {"x", "y"}


def test_pareto_without_task_scores_falls_back_to_best():
    alg = ParetoAlgorithm()
    tell(alg, 0, "root", 0.1)
    tell(alg, 1, "x", 0.6, "root")
    assert alg.ask().parent_id == "x"


def test_shinka_screen_rejects_near_duplicates():
    alg = ShinkaAlgorithm(novelty_threshold=0.9)
    tell(alg, 0, "root", 0.0)
    p = Patch(target="memory", ops=[{"op": "add", "item": {"id": "rule:#", "text": "# means multiply a and b"}}])
    first = alg.screen(p, "root")
    assert first.accept and first.novelty == 1.0
    again = alg.screen(p, "root")
    assert not again.accept and again.novelty == 0.0 and "相似度" in again.reason
    other = Patch(target="memory", ops=[{"op": "add", "item": {"id": "rule:@", "text": "@ is subtraction, b - a"}}])
    assert alg.screen(other, "root").accept


def test_shinka_ucb_explores_then_exploits():
    alg = ShinkaAlgorithm(proposers=["lesson", "claude"], seed=0)
    tell(alg, 0, "root", 0.0)
    first, second = alg.ask().proposer, None
    tell(alg, 1, "c1", 0.5, "root", proposer=f"x/composite:0.4.0#{first}")
    second = alg.ask().proposer
    assert {first, second} == {"lesson", "claude"}  # 每个提案器先各试一次
    # 之后 first 总能刷新纪录、second 总失败（包括没有候选 id 的失败），UCB 应该越来越偏向 first
    for g in range(2, 60):
        arm = alg.ask().proposer
        if arm == first:
            tell(alg, g, f"c{g}", 0.5 + g / 100, "root", proposer=f"x#{arm}")
        else:
            tell(alg, g, None, 0.0, "root", accepted=False, proposer=f"x#{arm}")
    assert alg.arm_pulls[first] > 3 * alg.arm_pulls[second] and alg.arm_reward[second] == 0


def test_shinka_selection_prefers_high_scores_and_unexplored():
    alg = ShinkaAlgorithm(seed=3)
    tell(alg, 0, "root", 0.1)
    tell(alg, 1, "good", 0.9, "root")
    tell(alg, 2, "mid", 0.5, "root")
    counts = {"root": 0, "good": 0, "mid": 0}
    for _ in range(300):
        counts[alg._pick()] += 1
    assert counts["good"] > counts["mid"] > counts["root"]


def test_islands_evolve_separately_and_migrate():
    alg = IslandAlgorithm(islands=2, migrate_every=3)
    tell(alg, 0, "root", 0.0)
    a1 = alg.ask()  # 第 1 代 → 0 号岛
    tell(alg, 1, "i0", 0.5, a1.parent_id)
    a2 = alg.ask()  # 第 2 代 → 1 号岛，看不到 0 号岛的 i0
    assert a2.parent_id == "root"
    tell(alg, 2, "i1", 0.2, a2.parent_id)
    alg.ask()  # 第 3 代：迁移，每岛最好的复制到下一个岛
    assert "i0" in alg.members[1] and "i1" in alg.members[0]
    assert alg._best_in(1) == "i0"


def test_map_elites_keeps_one_elite_per_cell():
    alg = MapElitesAlgorithm(descriptors=[{"metric": "size", "bins": [5]}], seed=0)
    tell(alg, 0, "root", 0.1, metrics={"size": 1})
    tell(alg, 1, "small", 0.4, "root", metrics={"size": 2})
    tell(alg, 2, "big", 0.3, "root", metrics={"size": 9})
    tell(alg, 3, "small2", 0.2, "root", metrics={"size": 3})  # 同格更差，不替换
    assert alg.elites == {(0,): "small", (1,): "big"}
    assert {alg.ask().parent_id for _ in range(30)} == {"small", "big"}


@pytest.mark.parametrize("cls", [GreedyAlgorithm, ArchiveAlgorithm, ParetoAlgorithm, ShinkaAlgorithm,
                                 IslandAlgorithm, MapElitesAlgorithm])
def test_all_algorithms_pass_conformance(cls):
    report = asyncio.run(check(build_server(cls())))
    assert report.ok, report.text()


def test_screen_is_exposed_over_mcp_only_when_implemented():
    async def go(alg):
        async with connect(build_server(alg)) as c:
            await c.algorithm.tell(TellRequest(generation=0, candidate_id="root", accepted=True,
                                               metrics={"holdout": 0.0}))
            return c.info.tools, await c.algorithm.screen(Patch(target="t", ops=[{"op": "x"}]), "root")

    tools, res = asyncio.run(go(ShinkaAlgorithm()))
    assert "rsi_algorithm_screen" in tools and res.accept
    tools, res = asyncio.run(go(GreedyAlgorithm()))
    assert "rsi_algorithm_screen" not in tools and res is None


# ---------------------------------------------------------------- A. 准入：逐题不回退

def test_gate_no_regression():
    gate = HoldoutGate(max_task_regressions=0)
    base = {"a": 1.0, "b": 0.0, "c": 1.0}
    better_but_breaks_c = gate.check(GateRequest(candidate_id="x", baseline={"holdout": 0.66},
                                                 candidate={"holdout": 0.67}, baseline_tasks=base,
                                                 candidate_tasks={"a": 1.0, "b": 1.0, "c": 0.0}))
    assert not better_but_breaks_c.accepted and "c" in better_but_breaks_c.checks[-1].detail
    clean = gate.check(GateRequest(candidate_id="y", baseline={"holdout": 0.66}, candidate={"holdout": 1.0},
                                   baseline_tasks=base, candidate_tasks={"a": 1.0, "b": 1.0, "c": 1.0}))
    assert clean.accepted
    missing = gate.check(GateRequest(candidate_id="z", baseline={"holdout": 0.0}, candidate={"holdout": 1.0}))
    assert not missing.accepted and "缺少逐题分数" in missing.checks[-1].detail
    assert HoldoutGate(max_task_regressions=1).check(GateRequest(
        candidate_id="x", baseline={"holdout": 0.66}, candidate={"holdout": 0.67}, baseline_tasks=base,
        candidate_tasks={"a": 1.0, "b": 1.0, "c": 0.0})).accepted


# ---------------------------------------------------------------- A. 运行时：逐题分数、筛查、提案器提示

class DupProposer(Proposer):
    """每次给出同一个补丁两遍，第二遍应该在评估前被 Shinka 的新颖度筛查拦下。"""

    name, version = "test/dup", "0.4.0"

    def propose(self, request: ProposalRequest) -> list[Patch]:
        assert request.hints.get("proposer") in (None, "only")
        n = len(request.state or [])
        op = {"op": "add", "item": {"id": f"note:{n}", "text": f"note number {n} about the secret operators"}}
        p = Patch(target=request.target.target, base_snapshot=request.parent_snapshot, ops=[op],
                  proposer_ref=self.ref + "#only")
        return [p, p]


def secret_config(tmp_path, **components):
    data = yaml.safe_load((ROOT / "examples/secret_ops/openrsi.yaml").read_text(encoding="utf-8"))
    data["components"]["memory"] = {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}
    data["components"]["lineage"]["with"]["path"] = "state/lineage.jsonl"
    data["components"].update(components)
    data["loop"]["max_patches"] = 2
    path = tmp_path / "openrsi.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(path)


def run(cfg, generations=None):
    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop).run(generations)
            return reports, await w.lineage.log()

    return asyncio.run(go())


def test_runtime_screens_duplicates_and_records_task_scores(tmp_path):
    cfg = secret_config(tmp_path, proposer={"python": "test_v04:DupProposer"},
                        algorithm={"python": "openrsi_algorithm:ShinkaAlgorithm",
                                   "with": {"proposers": ["only"], "max_generations": 2, "patience": 5}})
    reports, log = run(cfg)
    first = reports[0]
    assert len(first.candidates) == 2
    assert "评估前筛掉" in first.candidates[1].reason and first.candidates[1].id is None
    root = next(e for e in log if e.status == "root")
    assert root.task_scores and all(t.startswith("h") or t for t in root.task_scores)
    assert all(0.0 <= v <= 1.0 for v in root.task_scores.values())
    assert "rubric:correct" in root.metrics


def test_runtime_pareto_with_no_regression_gate(tmp_path):
    cfg = secret_config(tmp_path, algorithm={"python": "openrsi_algorithm:ParetoAlgorithm",
                                             "with": {"max_generations": 12, "patience": 6, "target_score": 1.0}},
                        gate={"python": "openrsi_gate:HoldoutGate", "with": {"max_task_regressions": 0}})
    _, log = run(cfg)
    accepted = [e for e in log if e.status == "accepted"]
    assert accepted and max(e.metrics["holdout"] for e in accepted) == 1.0
    by_id = {e.id: e for e in log}
    for e in accepted:  # 每个被接受的版本逐题都不比父代差
        parent = by_id[e.parent_ids[0]]
        assert all(e.task_scores.get(t, 0.0) >= s for t, s in parent.task_scores.items())


# ---------------------------------------------------------------- B. harness

def make_harness(tmp_path, **kw):
    tpl = tmp_path / "tpl"
    (tpl / "agent").mkdir(parents=True)
    (tpl / "eval").mkdir()
    (tpl / "agent" / "main.py").write_text(
        "import json, sys\nreq = json.load(sys.stdin)\n"
        "print(json.dumps({'output': req['task']['input'], 'steps': []}))\n")
    (tpl / "agent" / "notes.md").write_text("v1\n")
    (tpl / "eval" / "score.py").write_text("SCORE = 0\n")
    return HarnessEvolvable(tmp_path / "repo", template=tpl, mutable=["agent/**"], protected=["eval/**"], **kw)


def test_harness_snapshots_are_git_commits(tmp_path):
    h = make_harness(tmp_path)
    root = h.snapshot()
    s1 = h.apply(Patch(target="harness", base_snapshot=root.id,
                       ops=[{"op": "write", "path": "agent/notes.md", "content": "v2\n"},
                            {"op": "write", "path": "agent/new.py", "content": "X = 1\n"}]))
    assert s1.parent_ids == [root.id] and len(s1.id) == 12
    assert (h.repo / "agent" / "new.py").exists()
    d = h.diff(root.id, s1.id)
    assert {o["path"] for o in d.ops} == {"agent/notes.md", "agent/new.py"}
    exported = {i["id"]: i["text"] for i in h.export()}
    assert exported["agent/notes.md"] == "v2\n" and "eval/score.py" not in exported
    h.rollback(root.id)
    assert (h.repo / "agent" / "notes.md").read_text() == "v1\n" and not (h.repo / "agent" / "new.py").exists()
    s2 = h.apply(Patch(target="harness", ops=[{"op": "delete", "path": "agent/notes.md"}]))
    assert s2.parent_ids == [root.id]  # 从回滚后的版本分叉
    h.rollback(s1.id)
    assert (h.repo / "agent" / "new.py").exists()


def test_harness_rejects_out_of_scope_and_is_atomic(tmp_path):
    h = make_harness(tmp_path)
    head = h.snapshot().id
    for ops, msg in [
        ([{"op": "write", "path": "eval/score.py", "content": "SCORE = 1\n"}], "受保护"),
        ([{"op": "write", "path": "README.md", "content": "x"}], "不在可改范围"),
        ([{"op": "write", "path": "../escape.py", "content": "x"}], None),
        ([{"op": "write", "path": "agent/ok.py", "content": "x"}, {"op": "delete", "path": "agent/missing.py"}],
         "不存在"),
        ([{"op": "rename", "path": "agent/main.py"}], "不支持"),
    ]:
        with pytest.raises(Exception, match=msg):
            h.apply(Patch(target="harness", ops=ops))
        assert h.snapshot().id == head and not (h.repo / "agent" / "ok.py").exists()
    with pytest.raises(KeyError):
        h.apply(Patch(target="harness", base_snapshot="0" * 12, ops=[]))


def test_harness_unified_diff_op(tmp_path):
    h = make_harness(tmp_path)
    diff = ("--- a/agent/notes.md\n+++ b/agent/notes.md\n@@ -1 +1,2 @@\n v1\n+more\n")
    h.apply(Patch(target="harness", ops=[{"op": "diff", "unified": diff}]))
    assert (h.repo / "agent" / "notes.md").read_text() == "v1\nmore\n"
    bad = "--- a/eval/score.py\n+++ b/eval/score.py\n@@ -1 +1 @@\n-SCORE = 0\n+SCORE = 1\n"
    with pytest.raises(Exception, match="受保护"):
        h.apply(Patch(target="harness", ops=[{"op": "diff", "unified": bad}]))


def test_harness_conformance_and_manifest(tmp_path):
    report = asyncio.run(check(build_server(make_harness(tmp_path))))
    assert report.ok, report.text()
    manifest = load_manifest(ROOT / "components/harness/rsi-component.yaml")
    info = served_info(make_harness(tmp_path / "b"))
    assert (manifest.name, manifest.version) == (info.name, info.version)


def test_harness_agent_runs_subprocess(tmp_path):
    h = make_harness(tmp_path)
    agent = HarnessAgent(h.repo, command=["python", "-m", "agent.main"])
    t = agent.run(Task(id="t1", input="hello", split="train"), [], h.snapshot().id)
    assert t.output == "hello" and t.task_id == "t1" and "duration_s" in t.metadata
    assert not h._dirty() and not list(h.repo.rglob("__pycache__"))
    broken = HarnessAgent(h.repo, command=["python", "-c", "import sys; sys.exit('boom')"])
    t = broken.run(Task(id="t2", input="x", split="train"), [])
    assert t.output is None and "boom" in t.metadata["error"]
    slow = HarnessAgent(h.repo, command=["python", "-c", "import time; time.sleep(5)"], timeout=0.5)
    assert "超时" in slow.run(Task(id="t3", input="x", split="train"), []).metadata["error"]


def test_guard_file_ops():
    g = PolicyGuard(protected_paths=["eval/**"], check_harness_code=True)
    r = g.inspect(Patch(target="harness", ops=[{"op": "write", "path": "eval/score.py", "content": "x = 1\n"}]))
    assert not r.allowed and r.findings[0].rule == "protected_path"
    diff = "--- a/eval/x.py\n+++ b/eval/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert not g.inspect(Patch(target="harness", ops=[{"op": "diff", "unified": diff}])).allowed
    secret = g.inspect(Patch(target="harness", ops=[{"op": "write", "path": "agent/k.txt",
                                                     "content": "key = sk-ant-api03-abcdefghijklmnopqrstuvwxyz"}]))
    assert not secret.allowed and "secret" in {f.rule for f in secret.findings}
    code = g.inspect(Patch(target="harness", ops=[{"op": "write", "path": "agent/a.py",
                                                   "content": "import socket\n"}]))
    assert not code.allowed and "code_import" in {f.rule for f in code.findings}
    ok = g.inspect(Patch(target="harness", ops=[{"op": "write", "path": "agent/a.py", "content": "X = 1\n"}]))
    assert ok.allowed


def test_harness_example_blocks_eval_tampering_and_solves(tmp_path):
    src = ROOT / "examples/harness_ops"
    dst = tmp_path / "harness_ops"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".openrsi", "__pycache__"))
    cfg = load_config(dst / "openrsi.yaml")
    reports, log = run(cfg)
    blocked = [e for e in log if e.id.startswith("blocked_")]
    assert len(blocked) == 1 and "protected_path" in blocked[0].decision.reason
    assert max(e.metrics.get("holdout", 0.0) for e in log) == 1.0
    repo = dst / ".openrsi" / "harness"
    assert (repo / "eval" / "README.md").exists() and not (repo / "eval" / "score.py").exists()
    commits = subprocess.run(["git", "-C", str(repo), "log", "--all", "--oneline"], capture_output=True,
                             text=True).stdout.splitlines()
    assert len(commits) >= 2


# ---------------------------------------------------------------- C. verl

from openrsi_verl import (DataLeakError, DistillProposer, WeightsEvolvable, episodes_to_sft_rows,  # noqa: E402
                          make_compute_score, reference_from, tasks_to_rl_rows, write_rows)
from openrsi_verl import reward as verl_reward  # noqa: E402


def episode(task_id, split, output, expected, passed):
    task = Task(id=task_id, input={"q": task_id}, split=split, reference={"expected": expected})
    traj = Trajectory(id=f"tr-{task_id}", task_id=task_id, output=output, steps=[])
    v = Verdict(task_id=task_id, trajectory_id=traj.id, passed=passed, score=1.0 if passed else 0.0,
                judge_ref="j")
    return Episode(task=task, trajectory=traj, verdict=v)


def test_verl_reward_function_uses_judge():
    fn = make_compute_score(EvidenceJudge(EQ))
    assert fn("ops", "42", "42") == 1.0 and fn("ops", "41", "42") == 0.0
    assert fn("ops", "42", json.dumps({"expected": "42"}), {"index": 3}) == 1.0
    detail = make_compute_score(EvidenceJudge(EQ), return_dict=True)("ops", "41", "42")
    assert detail == {"score": 0.0, "passed": 0.0, "rubric/correct": 0.0}
    assert reference_from("[1, 2]") == {"expected": "[1, 2]"} and reference_from({"a": 1}) == {"a": 1}


def test_verl_module_level_reward_reads_env(tmp_path, monkeypatch):
    spec = tmp_path / "judge.yaml"
    spec.write_text(yaml.safe_dump({"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}}))
    monkeypatch.setattr(verl_reward, "_CACHED", None)
    monkeypatch.setenv("OPENRSI_JUDGE", str(spec))
    assert verl_reward.compute_score("ops", "7", "7") == 1.0
    monkeypatch.setattr(verl_reward, "_CACHED", None)
    monkeypatch.delenv("OPENRSI_JUDGE")
    with pytest.raises(RuntimeError, match="OPENRSI_JUDGE"):
        verl_reward.compute_score("ops", "7", "7")


def test_verl_rl_rows_hide_reference_and_refuse_holdout():
    rows = tasks_to_rl_rows([Task(id="t1", input="1 # 2", split="train", reference={"expected": 2})], "ops")
    assert rows[0]["prompt"] == [{"role": "user", "content": "1 # 2"}]
    assert rows[0]["reward_model"]["ground_truth"] == 2  # 只有 expected 时直接放值（verl 自带奖励要这个格式）
    multi = tasks_to_rl_rows([Task(id="t2", input="x", split="train", reference={"expected": 1, "unit": "m"})], "ops")
    assert json.loads(multi[0]["reward_model"]["ground_truth"]) == {"expected": 1, "unit": "m"}
    fn = make_compute_score(EvidenceJudge(EQ))
    assert fn("ops", "2", rows[0]["reward_model"]["ground_truth"]) == 1.0
    assert "expected" not in json.dumps(rows[0]["prompt"]) and "reference" not in rows[0]["extra_info"]
    with pytest.raises(DataLeakError):
        tasks_to_rl_rows([Task(id="h1", input="x", split="holdout", reference={"expected": 1})], "ops")
    val = tasks_to_rl_rows([Task(id="h1", input="x", split="holdout", reference={"expected": 1})], "ops",
                           for_validation=True)
    assert val[0]["extra_info"]["split"] == "holdout"
    with pytest.raises(DataLeakError):
        tasks_to_rl_rows([Task(id="t1", input="x", split="train", reference={"expected": 1})], "ops",
                         for_validation=True)


def test_verl_sft_rows_only_passed_train(tmp_path):
    eps = [episode("a", "train", "1", "1", True), episode("b", "train", "0", "1", False)]
    rows = episodes_to_sft_rows(eps)
    assert len(rows) == 1 and rows[0]["messages"][-1] == {"role": "assistant", "content": "1"}
    with pytest.raises(DataLeakError):
        episodes_to_sft_rows([episode("h", "holdout", "1", "1", True)])
    out = write_rows(rows, tmp_path / "d" / "sft.jsonl")
    assert json.loads(out.read_text().splitlines()[0])["extra_info"]["task_id"] == "a"


def test_weights_evolvable_with_fake_trainer(tmp_path):
    trainer = tmp_path / "train.py"
    trainer.write_text("import json, pathlib, sys\nout = pathlib.Path(sys.argv[2]); out.mkdir(parents=True)\n"
                       "(out / 'args.json').write_text(json.dumps(sys.argv[1:]))\n")
    w = WeightsEvolvable("base-model", tmp_path / "w",
                         trainer=[sys.executable, str(trainer), "{base_model}", "{output_dir}", "{data}",
                                  "{config_json}"])
    root = w.snapshot()
    s1 = w.apply(Patch(target="weights", base_snapshot=root.id,
                       ops=[{"op": "train", "method": "sft", "data": "d.jsonl", "config": {"lr": 1e-5}}]))
    args = json.loads((Path(w.export()[0]["text"]) / "args.json").read_text())
    assert args[0] == "base-model" and args[2] == "d.jsonl" and json.loads(args[3]) == {"lr": 1e-5}
    s2 = w.apply(Patch(target="weights", ops=[{"op": "train", "method": "sft", "data": "e.jsonl"}]))
    assert json.loads((Path(w.export()[0]["text"]) / "args.json").read_text())[0] == w.ckpts[s1.id]["model_path"]
    assert [o["data"] for o in w.diff(root.id, s2.id).ops] == ["d.jsonl", "e.jsonl"]
    w.rollback(root.id)
    assert w.export()[0]["text"] == "base-model"
    reloaded = WeightsEvolvable("base-model", tmp_path / "w")
    assert reloaded.snapshot().id == root.id and s2.id in reloaded.ckpts
    with pytest.raises(Exception, match="StaleSnapshot|基于"):
        w.apply(Patch(target="weights", base_snapshot=s2.id, ops=[{"op": "train", "method": "noop"}]))
    failing = WeightsEvolvable("m", tmp_path / "f", trainer=[sys.executable, "-c", "import sys; sys.exit('oom')"])
    with pytest.raises(Exception, match="oom"):
        failing.apply(Patch(target="weights", ops=[{"op": "train", "data": "x"}]))
    assert failing.snapshot().id == "ckpt_root"


def test_weights_model_glob_picks_latest_step(tmp_path):
    trainer = tmp_path / "train.py"
    trainer.write_text("import pathlib, sys\nout = pathlib.Path(sys.argv[1])\n"
                       "for n in (2, 10, 9):\n    (out / f'global_step_{n}' / 'huggingface').mkdir(parents=True)\n")
    w = WeightsEvolvable("base", tmp_path / "w", trainer=[sys.executable, str(trainer), "{output_dir}"],
                         model_glob="global_step_*/huggingface")
    w.apply(Patch(target="weights", ops=[{"op": "train", "data": "a"}]))
    assert w.export()[0]["text"].endswith("global_step_10/huggingface")
    bad = WeightsEvolvable("base", tmp_path / "b", trainer=[sys.executable, str(trainer), "{output_dir}"],
                           model_glob="actor/*")
    with pytest.raises(Exception, match="没有匹配"):
        bad.apply(Patch(target="weights", ops=[{"op": "train", "data": "a"}]))


def test_weights_conformance_and_manifest(tmp_path):
    report = asyncio.run(check(build_server(WeightsEvolvable("m", tmp_path / "w", dry_run=True))))
    assert report.ok, report.text()
    manifest = load_manifest(ROOT / "adapters/verl/rsi-component.yaml")
    info = served_info(WeightsEvolvable("m", tmp_path / "w2", dry_run=True))
    assert (manifest.name, manifest.version) == (info.name, info.version)
    assert set(manifest.implements) == set(info.tools)


def test_distill_proposer_trains_on_verified_trajectories(tmp_path):
    w = WeightsEvolvable("m", tmp_path / "w", dry_run=True)
    p = DistillProposer(tmp_path / "data", min_examples=2)
    info = w.describe()
    eps = [episode(str(i), "train", "1", "1", i < 3) for i in range(5)]
    req = ProposalRequest(target=info, targets=[info], parent_snapshot=w.snapshot().id, episodes=eps,
                          attributions=[], max_patches=1)
    [patch] = p.propose(req)
    data = Path(patch.ops[0]["data"])
    assert len(data.read_text().splitlines()) == 3
    snap = w.apply(patch)
    assert (Path(w.ckpts[snap.id]["model_path"]) / "DRY_RUN.json").exists()
    assert DistillProposer(tmp_path / "data", min_examples=10).propose(req) == []


# ---------------------------------------------------------------- D. RSIHub

import openrsi_rsihub as rsihub  # noqa: E402


def row(genid, score, tasks, parent=None, valid=True, h="H"):
    return {"genid": genid, "parent": parent, "tag": f"gen/{genid}", "score": score, "valid_parent": valid,
            "task_set_hash": h, "task_vector": {"schema_version": 1, "tasks": {
                t: {"trials": [{"trial": 0, "status": "benchmark_complete", "reward": r}]} for t, r in tasks.items()}}}


def test_rsihub_task_scores_follow_their_rule():
    r = row(1, 0.5, {"a": 1.0, "b": 0.0})
    r["task_vector"]["tasks"]["c"] = {"trials": [{"status": "benchmark_complete", "reward": 1.0},
                                                 {"status": "error", "reward": None}]}
    r["task_vector"]["tasks"]["d"] = {"trials": [{"status": "benchmark_complete", "reward": 1},
                                                 {"status": "benchmark_complete", "reward": 0}]}
    assert rsihub.task_scores(r) == {"a": 1.0, "b": 0.0, "d": 0.5}
    assert rsihub.task_scores({"task_vector": None}) == {}


def test_rsihub_gate():
    parent = row(0, 0.5, {"a": 1.0, "b": 0.0})
    ok, why = rsihub.gate_decide(row(1, 1.0, {"a": 1.0, "b": 1.0}, 0), parent)
    assert ok and "no_regression" in why
    ok, why = rsihub.gate_decide(row(2, 0.6, {"a": 0.0, "b": 1.0, "c": 1.0}, 0), parent)
    assert not ok and "1 道题变差" in why
    assert rsihub.gate_decide(row(2, 0.6, {"a": 0.0, "b": 1.0}, 0), parent, max_task_regressions=1)[0]
    assert not rsihub.gate_decide(row(3, 0.5, {"a": 1.0, "b": 0.0}, 0), parent)[0]
    assert rsihub.gate_decide(row(3, 0.5, {"a": 1.0, "b": 0.0}, 0), parent, allow_equal=True)[0]
    assert "任务集不同" in rsihub.gate_decide(row(4, 1.0, {"a": 1.0}, 0, h="X"), parent)[1]
    assert not rsihub.gate_decide(row(5, 1.0, {}, 0), None)[0]


def test_rsihub_shinka_select():
    parents = [row(0, 0.1, {}), row(1, 0.9, {}, 0), row(2, 0.5, {}, 0)]
    rows = parents + [row(3, 0.0, {}, 1, valid=False)] * 5  # 1 号已经有 5 个子代
    picks = rsihub.shinka_pick(parents, rows, random.Random(0), k=400)
    assert picks.count("2") > picks.count("0")
    no_kids = rsihub.shinka_pick(parents, parents, random.Random(0), k=400)
    assert no_kids.count("1") > picks.count("1")  # 子代多的父代被选中的机会下降
    with pytest.raises(ValueError):
        rsihub.shinka_pick([{"genid": 1, "score": None}], [], random.Random(0))


def git_checkout(tmp_path):
    repo = tmp_path / "checkout"
    for d in ("target", "evaluator", "operators"):
        (repo / d).mkdir(parents=True)
        (repo / d / "a.py").write_text("X = 0\n")
    (repo / "target" / "old.md").write_text("old\n")
    for args in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"], ["add", "-A"],
                 ["commit", "-qm", "parent"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    return repo


def test_rsihub_guard_validate(tmp_path):
    repo = git_checkout(tmp_path)
    assert rsihub.guard_checkout(repo)[:2] == (False, "候选没有任何改动")
    (repo / "target" / "a.py").write_text("X = 1\n")
    (repo / "target" / "new dir").mkdir()
    (repo / "target" / "new dir" / "b.py").write_text("Y = 2\n")
    subprocess.run(["git", "-C", str(repo), "mv", "target/old.md", "target/renamed.md"], check=True)
    ops = rsihub.checkout_ops(repo)
    assert {(o["op"], o["path"]) for o in ops} == {("write", "target/a.py"), ("write", "target/new dir/b.py"),
                                                   ("delete", "target/old.md"), ("write", "target/renamed.md")}
    ok, reason, _ = rsihub.guard_checkout(repo)
    assert ok and "通过" in reason
    (repo / "evaluator" / "a.py").write_text("X = 1  # always pass\n")
    ok, reason, findings = rsihub.guard_checkout(repo)
    assert not ok and "protected_path" in reason and "evaluator/a.py" in reason
    assert rsihub.guard_checkout(repo, protected_paths=[])[0]
    (repo / "evaluator" / "a.py").write_text("X = 0\n")
    (repo / "target" / "a.py").write_text("import socket\n")
    assert not rsihub.guard_checkout(repo, check_code=True)[0]


def test_rsihub_import_archive(tmp_path):
    archive = tmp_path / "archive.jsonl"
    rows = [row(0, 0.2, {"a": 0.0}), row(1, 0.8, {"a": 1.0}, 0), {**row(2, 0.1, {"a": 0.0}, 1, valid=False),
                                                                     "verdict": "discard"}]
    archive.write_text("".join(json.dumps(r) + "\n" for r in rows))
    entries = rsihub.import_archive(archive)
    assert [(e.id, e.status, e.parent_ids) for e in entries] == [
        ("gen0", "root", []), ("gen1", "accepted", ["gen0"]), ("gen2", "rejected", ["gen1"])]
    assert entries[1].task_scores == {"a": 1.0} and entries[1].metrics["holdout"] == 0.8


def test_rsihub_install_operator_scripts(tmp_path):
    root = tmp_path / "RSIHub"
    with pytest.raises(FileNotFoundError):
        rsihub.install(root)
    (root / "library" / "gate").mkdir(parents=True)
    written = rsihub.install(root)
    assert sorted(p.relative_to(root).as_posix() for p in written) == [
        "library/gate/openrsi_no_regression.py", "library/select/openrsi_shinka.py",
        "library/validate/openrsi_guard.py"]
    text = written[0].read_text()
    assert "__OPENRSI_PATHS__" not in text and "_OPENRSI_PATHS: list[str] = [" in text
    for p in written:  # 脚本语法正确，并且按 RSIHub 约定在 __main__ 里调用 sdk.main
        compile(p.read_text(), str(p), "exec")
        assert "sdk.main(" in p.read_text() and "config_schema=CONFIG" in p.read_text()
    with pytest.raises(FileExistsError):
        rsihub.install(root)
    rsihub.install(root, overwrite=True, embed_paths=False)
    assert "__OPENRSI_PATHS__" in written[0].read_text()


def test_rsihub_operator_classes_against_stub_interfaces(tmp_path, monkeypatch):
    """用最小的 evolve.frozen 替身加载算子脚本，确认类定义和调用约定对得上（不是真实 RSIHub）。"""
    import types
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class GateResult:
        decision: str
        reason: str

    @dataclass(frozen=True)
    class ValidateResult:
        accept: bool
        reason: str
        artifacts: list

    @dataclass(frozen=True)
    class SelectResult:
        parents: list

    interfaces = types.SimpleNamespace(GateOperator=object, ValidateOperator=object, SelectOperator=object,
                                       GateResult=GateResult, ValidateResult=ValidateResult,
                                       SelectResult=SelectResult, OperatorContext=object, ArchiveView=object,
                                       Row=dict)
    config = types.SimpleNamespace(Config=lambda fields: fields, **{
        k: (lambda *a, **kw: kw) for k in ("array", "boolean", "integer", "number", "string")})
    frozen = types.SimpleNamespace(sdk=types.SimpleNamespace(main=lambda *a, **kw: None))
    for name, mod in {"evolve": types.SimpleNamespace(), "evolve.frozen": frozen,
                      "evolve.frozen.interfaces": interfaces, "evolve.frozen.config": config}.items():
        monkeypatch.setitem(sys.modules, name, mod)

    def load(stage, name):
        ns: dict = {"__name__": "op"}
        exec(compile(rsihub.render_operator(stage, name), name, "exec"), ns)
        return ns

    ctx = types.SimpleNamespace(config={}, run_dir=tmp_path / "run", rng=random.Random(0), fan_out=2)
    gate = load("gate", "openrsi_no_regression")["OpenRSINoRegressionGate"]()
    res = gate.decide(row(1, 1.0, {"a": 1.0}, 0), row(0, 0.0, {"a": 0.0}), ctx)
    assert res.decision == "accept"

    sel = load("select", "openrsi_shinka")["OpenRSIShinkaSelect"]()
    archive = types.SimpleNamespace(valid_parents=lambda: [row(0, 0.2, {}), row(1, 0.8, {}, 0)],
                                    rows=lambda: [row(0, 0.2, {}), row(1, 0.8, {}, 0)])
    assert len(sel.pick(archive, ctx).parents) == 2

    repo = git_checkout(tmp_path)
    (repo / "operators" / "a.py").write_text("X = 9\n")
    val = load("validate", "openrsi_guard")["OpenRSIGuardValidate"]()
    out = val.validate(repo, ctx)
    assert not out.accept and out.artifacts == ["validate/openrsi_guard.json"]
    assert json.loads((ctx.run_dir / "validate" / "openrsi_guard.json").read_text())["findings"]

