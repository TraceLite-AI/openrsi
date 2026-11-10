"""跑前体检（判分器真的在判分吗）和挑考试题（哪些题有提升空间），用假 Harbor，不需要 Docker。"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "terminal_preflight"))
sys.path.insert(0, str(ROOT / "benchmarks" / "noise"))
sys.path.insert(0, str(ROOT / "tests"))
import headroom  # noqa: E402
import preflight  # noqa: E402
from test_rsihub_harbor import FAKE, make_dataset, make_repo  # noqa: E402


def write_config(tmp_path: Path, broken=(), gate=None, answer="n") -> Path:
    ds = make_dataset(tmp_path / "ds", n=12)
    for name in broken:
        (ds / name / "tests" / "broken_verifier").write_text("")
    repo = make_repo(tmp_path / "template", answer=answer)
    env = {"dataset": str(ds), "split": {"train": 0.5, "gate": 0.25, "sealed": 0.25}, "seed": 0}
    if gate is not None:
        env["gate"] = list(gate)
    data = {"protocol": "anyrsi/0.4", "name": "preflight-test", "components": {
        "env": {"python": "anyrsi_rsihub:RSIHubTaskEnv", "with": env},
        "agent": {"python": "anyrsi_rsihub:HarborAgent", "with": {
            "repo": str(tmp_path / "harness"), "agent_args": ["--agent-import-path", "echo.agent:EchoAgent"],
            "command": [sys.executable, FAKE, "run", "-p", "{dataset}", "--include-task-name", "{task}", "--job-name",
                        "{job}", "--jobs-dir", "{jobs_dir}"], "workdir": str(tmp_path / "jobs"), "timeout": 60}},
        "judge": {"python": "anyrsi_rsihub:HarborRewardJudge"},
        "proposer": {"python": "anyrsi_llm:CodeRewriter", "with": {"base_url": "http://127.0.0.1:9/v1"}},
        "algorithm": {"python": "anyrsi_algorithm:GreedyAlgorithm", "with": {"max_generations": 1}},
        "gate": {"python": "anyrsi_gate:HoldoutGate"},
        "lineage": {"python": "anyrsi_lineage:LineageStore", "with": {"path": str(tmp_path / "lineage.jsonl")}}},
        "evolvables": {"harness": {"python": "anyrsi_harness:HarnessEvolvable", "with": {
            "repo": str(tmp_path / "harness"), "template": str(repo), "mutable": ["echo/**"]}}},
        "loop": {"max_patches": 1}}
    path = tmp_path / "anyrsi.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def splits_of(tmp_path):
    from anyrsi_rsihub import RSIHubTaskEnv
    return RSIHubTaskEnv(str(tmp_path / "ds"), split={"train": 0.5, "gate": 0.25, "sealed": 0.25}).splits


def test_preflight_passes_on_a_working_verifier_and_flags_a_broken_one(tmp_path, capsys):
    cfg = write_config(tmp_path / "ok")
    out = tmp_path / "ok.json"
    assert preflight.main([str(cfg), "--per-split", "2", "--out", str(out)]) == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 12 and all(r["ok"] and r["tests_started"] for r in rows)
    assert {(r["probe"], r["passed"]) for r in rows} == {("oracle", True), ("nop", False)}

    sp = splits_of(tmp_path / "ok")
    bad_task = sp["gate"][0]
    cfg = write_config(tmp_path / "bad", broken=[bad_task])
    out = tmp_path / "bad.json"
    assert preflight.main([str(cfg), "--splits", "holdout", "--per-split", "3", "--out", str(out)]) == 1
    rows = json.loads(out.read_text())
    flagged = {r["task"] for r in rows if not r["ok"]}
    assert flagged == {bad_task} and all("判分器没跑到测试" in p for r in rows for p in r["problems"])


def test_headroom_picks_gate_tasks_with_room_and_never_sealed(tmp_path, capsys):
    report = {"splits": {"a": "train", "b": "train", "c": "holdout", "d": "train", "e": "holdout", "s": "final"},
              "per_task": {
                  "a": [{"passed": True}] * 3,                                          # saturated
                  "b": [{"passed": True}, {"passed": False}, {"passed": True}],         # flaky 2/3
                  "c": [{"passed": False}, {"passed": True}, {"passed": False}],        # flaky 1/3
                  "d": [{"passed": False}] * 3,                                         # failing
                  "e": [{"passed": False}, {"unscorable": "read-only"}, {"passed": False}],  # broken
                  "s": [{"passed": False}] * 3}}                                        # sealed: never
    classes = headroom.classify(report)
    assert {t: c["kind"] for t, c in classes.items()} == {"a": "saturated", "b": "flaky", "c": "flaky",
                                                          "d": "failing", "e": "broken"}
    assert headroom.pick(classes, 10) == ["c", "b", "d"]
    assert headroom.pick(classes, 2) == ["c", "b"] and headroom.pick(classes, 10, include_failing=False) == ["c", "b"]
    path = tmp_path / "noise.json"
    path.write_text(json.dumps(report))
    sys.argv = ["headroom", str(path), "--size", "5", "--min-size", "4"]
    headroom.main()
    out = capsys.readouterr().out
    assert 'gate: ["c", "b", "d"]' in out and "太少" in out and "e" in out


def test_noise_then_headroom_then_gate_on_the_fake_harbor(tmp_path):
    """测噪声（练习题 + 考试题）→ 挑考试题 → 用指定考试题跑，封存题不变。"""
    import asyncio

    import noise
    from anyrsi_rsihub import RSIHubTaskEnv

    make_dataset(tmp_path / "probe" / "ds", n=12)  # 同样的题名，划分一样
    sp = splits_of(tmp_path / "probe")
    broken = [sp["train"][0], sp["gate"][0]]
    cfg = write_config(tmp_path / "run", broken=broken)
    rep = asyncio.run(noise.measure(cfg, 2, split="all"))
    assert rep["tasks"] == len(sp["train"]) + len(sp["gate"]) and sorted(rep["unscorable"]) == sorted(broken)
    assert not set(rep["per_task"]) & set(sp["sealed"])
    classes = headroom.classify(rep)
    assert {t for t, c in classes.items() if c["kind"] == "broken"} == set(broken)
    # echo agent 全对：没有提升空间；换一个全错的 agent，所有能评分的题都有空间
    assert headroom.pick(classes, 10) == []
    cfg2 = write_config(tmp_path / "wrong", broken=broken, answer='"wrong"')
    picked = headroom.pick(headroom.classify(asyncio.run(noise.measure(cfg2, 1, split="all"))), 4)
    assert len(picked) == 4 and not set(picked) & set(broken)
    env = RSIHubTaskEnv(str(tmp_path / "wrong" / "ds"), split={"train": 0.5, "gate": 0.25, "sealed": 0.25},
                        gate=picked)
    assert sorted(t.id for t in env.tasks("holdout")) == sorted(picked)
    assert env.splits["sealed"] == sp["sealed"]
    assert {t.id for t in env.tasks("train")} == (set(sp["train"]) | set(sp["gate"])) - set(picked)


def test_gate_cannot_take_sealed_or_unknown_tasks(tmp_path):
    import pytest

    from anyrsi import ComponentError
    from anyrsi_rsihub import RSIHubTaskEnv

    ds = make_dataset(tmp_path / "ds")
    sealed = RSIHubTaskEnv(str(ds)).splits["sealed"][0]
    with pytest.raises(ComponentError, match="封存题"):
        RSIHubTaskEnv(str(ds), gate=[sealed])
    with pytest.raises(ComponentError, match="不在数据集"):
        RSIHubTaskEnv(str(ds), gate=["nope"])


def test_preflight_flags_a_verifier_that_passes_an_empty_answer(tmp_path):
    cfg = write_config(tmp_path / "loose")
    task = splits_of(tmp_path / "loose")["gate"][0]
    (tmp_path / "loose" / "ds" / task / "tests" / "expected.txt").write_text("\n")  # 什么都不交也算对
    out = tmp_path / "loose.json"
    assert preflight.main([str(cfg), "--splits", "holdout", "--per-split", "3", "--out", str(out)]) == 1
    rows = json.loads(out.read_text())
    bad = [r for r in rows if not r["ok"]]
    assert [(r["task"], r["probe"]) for r in bad] == [(task, "nop")]
    assert "什么都不做却通过了" in bad[0]["problems"][0]


def test_handpatch_analysis_separates_real_gains_from_noise():
    import handpatch

    base = {f"t{i}": [0.0, 0.0, 0.0] for i in range(10)}
    good = handpatch.analyse(base, {t: [1.0, 1.0, 1.0] if t < "t6" else [0.0] * 3 for t in base})
    assert good["verdict"] == "测出提升" and good["diff"] == 0.6 and len(good["better"]) == 6 and good["ci95"][0] > 0
    bad = handpatch.analyse({t: [1.0] * 3 for t in base}, {t: [0.0] * 3 if t < "t5" else [1.0] * 3 for t in base})
    assert bad["verdict"] == "测出变差" and len(bad["worse"]) == 5
    # 一道题变好、一道题变差：测不出差别；起点自己不稳定的题单独列出
    noisy_base = {**base, "t0": [1.0, 0.0, 1.0]}
    none = handpatch.analyse(noisy_base, {**base, "t0": [0.0] * 3, "t1": [1.0, 1.0, 0.0]})
    assert none["verdict"] == "测不出差别" and none["baseline_flaky"] == ["t0"]
    assert none["better"] == ["t1"] and none["worse"] == ["t0"] and none["same"] == 8 and none["min_detectable"] > 0
    # 10 道里只有 1 道变好：平均是正的，但区间包含 0，不能算测出提升
    one = handpatch.analyse(base, {**base, "t3": [1.0, 1.0, 1.0]})
    assert one["diff"] > 0 and one["ci95"][0] <= 0 and one["verdict"] == "测不出差别"
    single = handpatch.analyse({"dna": [0, 1, 1]}, {"dna": [1, 0, 0]})
    assert single["ci95"] is None and single["min_detectable"] is None and single["diff"] < 0
    missing = handpatch.analyse({"a": [None, 1, 0], "b": [None, 1]}, {"a": [1, 1, None], "b": [0, None]})
    assert missing["paired_counts"] == {"a": 1, "b": 0} and missing["diff"] == 0
    # 只比两边都有分的题
    part = handpatch.analyse({"a": [0.0], "b": [0.0], "c": []}, {"a": [1.0], "c": [1.0]})
    assert part["tasks"] == 1 and handpatch.analyse({"a": []}, {"a": [1.0]})["tasks"] == 0


def test_handpatch_runs_a_diff_against_the_fake_harbor_and_restores_the_parent(tmp_path):
    import subprocess

    import handpatch

    make_dataset(tmp_path / "probe" / "ds", n=12)
    sp = splits_of(tmp_path / "probe")
    cfg = write_config(tmp_path / "run", broken=[sp["gate"][0]], answer='"wrong"')
    # 手写补丁：把总是答错的 agent 改对（harness 仓库里的 unified diff）
    repo = tmp_path / "run" / "template"
    old = (repo / "echo" / "agent.py").read_text()
    (tmp_path / "new").mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    (repo / "echo" / "agent.py").write_text(old.replace('answer = "wrong"', "answer = n"))
    diff = subprocess.run(["git", "diff"], cwd=repo, check=True, capture_output=True, text=True).stdout
    (repo / "echo" / "agent.py").write_text(old)
    shutil = __import__("shutil")
    shutil.rmtree(repo / ".git")
    (tmp_path / "fix.diff").write_text(diff)
    out = tmp_path / "hp.json"
    assert handpatch.main([str(cfg), "--diff", str(tmp_path / "fix.diff"), "--repeats", "3", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "测出提升" and rep["baseline"] == 0.0 and rep["patched"] == 1.0
    assert rep["tasks"] == len(sp["gate"]) - 1 and rep["unscorable"]["baseline"] == [sp["gate"][0]]
    assert all(len(v) == 3 for v in rep["per_task"]["patched"].values())
    # 最后一遍跑的是打了补丁的版本；结束后工作区切回了起点版本
    assert 'answer = "wrong"' in (tmp_path / "run" / "harness" / "echo" / "agent.py").read_text()
    # 提示类的可进化体用 --patch；没有这个可进化体时报错
    (tmp_path / "p.json").write_text(json.dumps({"target": "nope", "ops": [{"op": "write", "path": "x", "content": "y"}]}))
    import pytest
    with pytest.raises(SystemExit, match="没有可进化体"):
        handpatch.main([str(cfg), "--patch", str(tmp_path / "p.json"), "--out", str(out)])
