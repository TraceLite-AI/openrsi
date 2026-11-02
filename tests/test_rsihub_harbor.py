"""RSIHub 的题库进 AnyRSI 的循环：划分、出题、调 Harbor、按验证奖励打分、改代码的提案器，以及改代码 + 训权重的全流程。

Harbor 用 examples/rsihub_weights/fake_harbor.py 代替（同样的参数和结果目录布局），不需要 Docker。
"""
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from anyrsi import Episode, EvolvableInfo, Item, ProposalRequest, Step, Task, Trajectory, Verdict
from anyrsi_rsihub import (PARITY_SPLIT, HarborAgent, HarborRewardJudge, RSIHubTaskEnv, assign_splits,
                            discover_task_names)
from anyrsi_rsihub.harbor import atif_steps, find_trial, read_reward, task_digest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "rsihub_weights"
FAKE = str(EX / "fake_harbor.py")


def make_dataset(root: Path, n: int = 12) -> Path:
    for i in range(n):
        d = root / f"t{i:02d}"
        (d / "tests").mkdir(parents=True)
        (d / "task.toml").write_text('version = "1.0"\n')
        (d / "instruction.md").write_text(f"Echo the number {i}\n")
        (d / "tests" / "expected.txt").write_text(f"{i}\n")
    (root / "not-a-task").mkdir()
    return root


AGENT_SRC = '''
import re
class EchoAgent:
    def solve(self, instruction):
        n = re.findall(r"\\d+", instruction)[0]
        answer = {answer}
        return answer, [{{"step_id": 1, "source": "user", "message": instruction}},
                        {{"step_id": 2, "source": "agent", "message": answer,
                          "tool_calls": [{{"tool_call_id": "c1", "function_name": "echo", "arguments": '{{"n": 1}}'}}],
                          "observation": {{"results": [{{"source_call_id": "c1", "content": n}}]}}}}]
'''


def make_repo(root: Path, answer: str = "n") -> Path:
    (root / "echo").mkdir(parents=True, exist_ok=True)
    (root / "echo" / "__init__.py").write_text("")
    (root / "echo" / "agent.py").write_text(AGENT_SRC.format(answer=answer))
    return root


def harbor_agent(tmp_path: Path, repo: Path, **kw) -> HarborAgent:
    return HarborAgent(agent_args=["--agent-import-path", "echo.agent:EchoAgent"], repo=str(repo),
                       command=[sys.executable, FAKE, "run", "-p", "{dataset}", "--include-task-name", "{task}",
                                "--job-name", "{job}", "--jobs-dir", "{jobs_dir}"],
                       workdir=str(tmp_path / "jobs"), timeout=60, **kw)


# ---------------------------------------------------------------- 划分


def test_split_counts_match_the_rsihub_parity_setup():
    names = [f"task-{i:03d}" for i in range(89)]
    s = assign_splits(names, PARITY_SPLIT, 0)
    assert {k: len(v) for k, v in s.items()} == {"train": 50, "gate": 19, "sealed": 20}
    assert sorted(s["train"] + s["gate"] + s["sealed"]) == names
    assert all(v == sorted(v) for v in s.values())


def test_split_is_ordered_by_seeded_hash_and_depends_on_seed():
    import hashlib

    names = [f"n{i}" for i in range(30)]
    s = assign_splits(names, {"train": 0.5, "gate": 0.25, "sealed": 0.25}, 3)
    ordered = sorted(names, key=lambda n: hashlib.sha256(f"3\0{n}".encode()).hexdigest())
    assert set(s["train"]) == set(ordered[:15]) and set(s["sealed"]) == set(ordered[-7:])
    assert assign_splits(names, {"train": 0.5, "gate": 0.25, "sealed": 0.25}, 4) != s
    assert assign_splits(list(reversed(names)), {"train": 0.5, "gate": 0.25, "sealed": 0.25}, 3) == s


def test_split_remainder_goes_to_largest_fraction_then_train_first():
    s = assign_splits([f"n{i}" for i in range(10)], {"train": 1, "gate": 1, "sealed": 1}, 0)
    assert [len(s[k]) for k in ("train", "gate", "sealed")] == [4, 3, 3]
    s = assign_splits([f"n{i}" for i in range(10)], {"train": 0.62, "gate": 0.19, "sealed": 0.19}, 0)
    assert [len(s[k]) for k in ("train", "gate", "sealed")] == [6, 2, 2]  # 6.2/1.9/1.9 → 余 2 给 gate、sealed
    with pytest.raises(ValueError):
        assign_splits(["a"], {"train": 1, "holdout": 1, "sealed": 1}, 0)
    with pytest.raises(ValueError):
        assign_splits(["a"], {"train": 0, "gate": 0, "sealed": 0}, 0)


# ---------------------------------------------------------------- 环境


def test_task_env_maps_rsihub_splits_and_rotates_train(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    assert discover_task_names(ds) == [f"t{i:02d}" for i in range(12)]
    env = RSIHubTaskEnv(str(ds), split={"train": 0.5, "gate": 0.25, "sealed": 0.25}, train_per_round=3)
    train0, train1 = env.tasks("train", 0), env.tasks("train", 1)
    holdout, final = env.tasks("holdout"), env.tasks("final")
    assert len(train0) == 3 and {t.id for t in train0} <= set(env.splits["train"])
    assert {t.id for t in train0} != {t.id for t in train1}  # 按代轮换
    assert [t.id for t in holdout] == env.splits["gate"] and [t.id for t in final] == env.splits["sealed"]
    assert {t.split for t in holdout} == {"holdout"} and {t.split for t in final} == {"final"}
    assert env.tasks("anchor") == []
    t = holdout[0]
    assert t.input["harbor_task"] == t.id and t.input["instruction"].startswith("Echo")
    assert t.metadata["dataset"] == str(ds) and t.metadata["rsihub_split"] == "gate"
    assert t.reference is None  # 答案在验证脚本里，不在题目上
    with pytest.raises(Exception, match="环境格"):
        env.tasks("holdout", cell="x")


def test_task_env_include_digest_and_errors(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    env = RSIHubTaskEnv(str(ds), include=["t00", "t01", "t02", "t03"], split={"train": 1, "gate": 1, "sealed": 2})
    assert sorted(sum(env.splits.values(), [])) == ["t00", "t01", "t02", "t03"]
    before = task_digest(ds / "t00")
    (ds / "t00" / "tests" / "expected.txt").write_text("changed\n")
    assert task_digest(ds / "t00") != before
    with pytest.raises(Exception, match="不在数据集"):
        RSIHubTaskEnv(str(ds), include=["nope"])
    with pytest.raises(Exception, match="找不到"):
        RSIHubTaskEnv(str(tmp_path / "missing"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(Exception, match="没有带 task.toml"):
        RSIHubTaskEnv(str(tmp_path / "empty"))


# ---------------------------------------------------------------- 结果解析


def test_atif_steps_and_reward():
    steps, last = atif_steps({"steps": [
        {"step_id": 1, "source": "system", "message": "sys"},
        {"step_id": 2, "source": "user", "message": [{"type": "text", "text": "do it"}]},
        {"step_id": 3, "source": "agent", "message": "ok",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "bash", "arguments": '{"cmd": "ls"}'},
                        {"tool_call_id": "c2", "function_name": "x", "arguments": "not json"}],
         "observation": {"results": [{"source_call_id": "c1", "content": "a.txt"}]}},
        {"step_id": 4, "source": "agent", "message": "done"}, "junk"]})
    kinds = [(s.type, s.role or s.name) for s in steps]
    assert kinds == [("message", "system"), ("message", "user"), ("message", "assistant"), ("tool_call", "bash"),
                     ("tool_call", "x"), ("tool_result", None), ("message", "assistant")]
    assert steps[1].content == "do it" and steps[3].arguments == {"cmd": "ls"} and steps[4].arguments == {"raw": "not json"}
    assert steps[5].call_id == "c1" and last == "done"
    assert read_reward({"verifier_result": {"rewards": {"reward": 1}}}) == 1.0
    for bad in ({}, {"verifier_result": {"rewards": {"reward": True}}}, {"verifier_result": {"rewards": {"reward": "1"}}},
                {"verifier_result": {"rewards": {"reward": float("nan")}}}):
        assert read_reward(bad) is None


# ---------------------------------------------------------------- agent + judge


def test_harbor_agent_runs_evolved_code_and_judge_reads_verifier(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    repo = make_repo(tmp_path / "repo")
    env = RSIHubTaskEnv(str(ds))
    agent, judge = harbor_agent(tmp_path, repo), HarborRewardJudge()
    task = env.tasks("holdout")[0]
    traj = agent.run(task, [Item(id="note", kind="lesson", text="x")], "snap1")
    h = traj.metadata["harbor"]
    assert h["reward"] == 1.0 and h["task_digest"] == task.metadata["task_digest"] and traj.used_items == ["note"]
    assert [s.type for s in traj.steps] == ["message", "message", "tool_call", "tool_result"]
    assert traj.output == str(int(task.id[1:]))
    v = judge.evaluate(task, traj, [])
    assert v.passed and v.score == 1.0 and v.rubric[0].result == "pass"
    assert v.collected[0].content["reward"] == 1.0 and v.collected[0].verify()

    # 改了仓库里的代码，Harbor 用的就是改过的代码
    make_repo(repo, answer='"wrong"')
    traj2 = agent.run(task, [], "snap2")
    v2 = judge.evaluate(task, traj2, [])
    assert traj2.metadata["harbor"]["reward"] == 0.0 and not v2.passed and v2.score == 0.0
    assert "0" in v2.rubric[0].note


def test_judge_rejects_tampered_or_mismatched_results(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    agent, judge = harbor_agent(tmp_path, make_repo(tmp_path / "repo", answer='"wrong"')), HarborRewardJudge()
    task = RSIHubTaskEnv(str(ds)).tasks("holdout")[0]
    traj = agent.run(task, [], "s")
    path = Path(traj.metadata["harbor"]["result_path"])
    data = json.loads(path.read_text())
    data["verifier_result"]["rewards"]["reward"] = 1.0  # 有人把分数改成满分
    path.write_text(json.dumps(data))
    v = judge.evaluate(task, traj, [])
    assert not v.passed and "被改过" in v.notes[0]

    # agent 转述的分数不算数：judge 只看文件
    traj.metadata["harbor"]["reward"] = 1.0
    traj.metadata["harbor"]["result_sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    other = RSIHubTaskEnv(str(ds)).tasks("final")[0]
    v = judge.evaluate(other, traj, [])
    assert not v.passed and "不是" in v.notes[0]

    path.unlink()
    assert "不在了" in judge.evaluate(task, traj, []).notes[0]
    assert not judge.evaluate(task, Trajectory(id="x", task_id=task.id, metadata={"error": "boom"}), []).passed


def test_agent_exception_and_missing_harbor(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    repo = make_repo(tmp_path / "repo", answer="1 / 0")
    task = RSIHubTaskEnv(str(ds)).tasks("holdout")[0]
    traj = harbor_agent(tmp_path, repo).run(task, [], "s")
    assert traj.metadata["harbor"]["reward"] is None and traj.metadata["harbor"]["exception_type"] == "ZeroDivisionError"
    assert traj.steps[-1].is_error
    v = HarborRewardJudge().evaluate(task, traj, [])
    assert not v.passed and "ZeroDivisionError" in v.notes[0]

    missing = HarborAgent(agent_args=[], command=["definitely-not-harbor-xyz", "run"], workdir=str(tmp_path / "j"))
    t = missing.run(task, [], "s")
    assert "找不到命令" in t.metadata["error"]
    no_ds = harbor_agent(tmp_path, repo).run(Task(id="t00", input={"harbor_task": "t00"}), [], "s")
    assert "metadata.dataset" in no_ds.metadata["error"]
    silent = HarborAgent(agent_args=[], command=[sys.executable, "-c", "pass"], workdir=str(tmp_path / "j2"))
    assert "没有产出" in silent.run(task, [], "s").metadata["error"]
    assert find_trial(tmp_path / "nowhere", "t00") is None


def test_threshold_and_partial_reward(tmp_path):
    ds = make_dataset(tmp_path / "ds")
    task = RSIHubTaskEnv(str(ds)).tasks("holdout")[0]
    traj = harbor_agent(tmp_path, make_repo(tmp_path / "repo")).run(task, [], "s")
    path = Path(traj.metadata["harbor"]["result_path"])
    data = json.loads(path.read_text())
    data["verifier_result"]["rewards"]["reward"] = 0.5
    path.write_text(json.dumps(data))
    traj.metadata["harbor"]["result_sha256"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert HarborRewardJudge().evaluate(task, traj, []).score == 0.5
    assert not HarborRewardJudge().evaluate(task, traj, []).passed
    assert HarborRewardJudge(pass_threshold=0.5).evaluate(task, traj, []).passed


# ---------------------------------------------------------------- 改代码的提案器


def code_request(files: dict[str, str], surfaces=("agent/**",), passed=False, ttype="harness") -> ProposalRequest:
    task = Task(id="t1", input={"instruction": "Compute the product of: 3, 9, 7, 2, 5, 8, 4"})
    traj = Trajectory(id="tr1", task_id="t1", steps=[Step(type="tool_call", name="bash", arguments={"cmd": "ls"}),
                                                    Step(type="observation", content="boom", is_error=True)])
    ep = Episode(task=task, trajectory=traj, verdict=Verdict(score=0.0, passed=passed, judge_ref="j"))
    info = EvolvableInfo(target="harness", type=ttype, head="h", surfaces=list(surfaces))
    return ProposalRequest(target=info, parent_snapshot="p", targets=[info], episodes=[ep],
                           states={"harness": [Item(id=p, kind="file", text=t).model_dump(mode="json")
                                               for p, t in files.items()]})


def test_code_rewriter_writes_editable_files(monkeypatch):
    import anyrsi_llm

    seen = {}

    def fake_chat(base_url, model, messages, **kw):
        seen["user"] = messages[-1]["content"]
        seen["system"] = messages[0]["content"]
        return {"text": "Here you go:\n```json\n" + json.dumps({"rationale": "add a tool", "files": [
            {"path": "agent/main.py", "content": "print('new')\n"},
            {"path": "eval/check.py", "content": "hacked"},           # 不在可改范围
            {"path": "agent/same.py", "content": "same\n"},           # 没变
            {"path": "agent/extra.py", "content": "x = 1\n"}]}) + "\n```"}

    monkeypatch.setattr(anyrsi_llm, "chat", fake_chat)
    req = code_request({"agent/main.py": "print('old')\n", "agent/same.py": "same\n", "agent/big.py": "x" * 30000})
    [patch] = anyrsi_llm.CodeRewriter("http://x/v1").propose(req)
    assert patch.target == "harness" and patch.ops == [{"op": "write", "path": "agent/main.py", "content": "print('new')\n"},
                                                       {"op": "write", "path": "agent/extra.py", "content": "x = 1\n"}]
    assert "add a tool" in patch.rationale and "1 train failures" in patch.rationale
    assert "=== agent/main.py ===" in seen["user"] and "agent/big.py" not in seen["user"]  # 太大的文件不给看
    assert "[call bash]" in seen["user"] and "boom" in seen["user"]
    assert seen["system"].startswith("You improve the source code")


def test_code_rewriter_refuses_leaks_and_bad_replies(monkeypatch):
    import anyrsi_llm

    reply = {}
    monkeypatch.setattr(anyrsi_llm, "chat", lambda *a, **k: {"text": reply["text"]})
    files = {"agent/main.py": "old\n"}
    rewriter = anyrsi_llm.CodeRewriter("http://x/v1")
    reply["text"] = json.dumps({"files": [{"path": "agent/main.py",
                                           "content": "if task == 'Compute the product of: 3, 9, 7, 2, 5, 8, 4': ..."}]})
    assert rewriter.propose(code_request(files)) == []  # 把题目原文写进代码 = 背答案
    for bad in ("not json at all", json.dumps({"files": "nope"}), json.dumps({"rationale": "no change", "files": []}),
                json.dumps({"files": [{"path": "agent/main.py", "content": 3}]})):
        reply["text"] = bad
        assert rewriter.propose(code_request(files)) == []
    reply["text"] = json.dumps({"files": [{"path": "agent/main.py", "content": "new\n"}]})
    assert rewriter.propose(code_request(files, passed=True)) == []   # 没有错题
    assert rewriter.propose(code_request(files, ttype="prompt")) == []  # 不是 harness
    assert len(rewriter.propose(code_request(files))) == 1

    def down(*a, **k):
        raise anyrsi_llm.ComponentError("down")

    monkeypatch.setattr(anyrsi_llm, "chat", down)
    assert rewriter.propose(code_request(files)) == []


# ---------------------------------------------------------------- 清单


@pytest.mark.parametrize("manifest,factory", [
    ("adapters/rsihub/rsi-component.task-env.yaml", lambda d: RSIHubTaskEnv(str(d))),
    ("adapters/rsihub/rsi-component.harbor-agent.yaml", lambda d: HarborAgent(agent_args=["--agent", "oracle"])),
    ("adapters/rsihub/rsi-component.reward-judge.yaml", lambda d: HarborRewardJudge()),
    ("components/llm/rsi-component.code-rewriter.yaml", lambda d: __import__("anyrsi_llm").CodeRewriter("http://x/v1")),
])
def test_manifests_match(manifest, factory, tmp_path):
    from anyrsi import build_server, connect, load_manifest

    ds = make_dataset(tmp_path / "ds", n=3)

    async def served():
        async with connect(build_server(factory(ds))) as c:
            return c.info

    info = asyncio.run(served())
    m = load_manifest(ROOT / manifest)
    assert (m.name, m.version) == (info.name, info.version)
    assert set(m.implements) == set(info.tools) and set(m.kinds) == set(info.kinds)


# ---------------------------------------------------------------- 全流程


def test_code_and_weights_evolve_in_one_loop(tmp_path):
    from anyrsi_runtime import Runner, Wiring, load_config

    work = tmp_path / "rsihub_weights"
    shutil.copytree(EX, work, ignore=shutil.ignore_patterns(".anyrsi", "__pycache__", "toy_tasks"))
    subprocess.run([sys.executable, "make_toy_dataset.py", "toy_tasks"], cwd=work, check=True, capture_output=True)
    port = _free_port()
    text = (work / "anyrsi.mock.yaml").read_text().replace("8197", str(port))
    (work / "anyrsi.mock.yaml").write_text(text)
    cfg = load_config(work / "anyrsi.mock.yaml")

    async def go():
        async with Wiring(cfg) as w:
            await Runner(w, cfg.loop).run()
            return await w.lineage.log(), w.components["agent"]

    import os

    old = Path.cwd()
    os.chdir(work)  # 命令模板里的 fake_harbor.py、mock_llm.py 是相对路径
    try:
        log, agent = asyncio.run(go())
    finally:
        os.chdir(old)
    agent._obj.server.stop()
    accepted = [e for e in log if e.status == "accepted"]
    targets = {e.patch.target for e in accepted if e.patch}
    assert targets == {"harness", "weights"}, [(e.patch.target, e.metrics) for e in accepted if e.patch]
    root = next(e for e in log if e.status == "root")
    best = max(accepted, key=lambda e: e.metrics["holdout"])
    assert best.metrics["holdout"] > root.metrics["holdout"] + 0.5
    assert best.snapshots["weights"] != "ckpt_root" and best.snapshots["harness"] != log[0].snapshots["harness"]
    weights = next(e for e in accepted if e.patch and e.patch.target == "weights")
    assert "wrong answers used" in weights.patch.rationale  # 错题也进了训练数据
    rows = [json.loads(x) for f in (work / ".anyrsi" / "kto").glob("*.jsonl") for x in f.read_text().splitlines()]
    assert {r["label"] for r in rows} == {True, False}
    holdout_ids = set(RSIHubTaskEnv(str(work / "toy_tasks"), split={"train": 0.5, "gate": 0.25, "sealed": 0.25}).splits["gate"])
    used = {m["content"].strip() for r in rows for m in r["messages"] if m["role"] == "user"}
    holdout_text = {(work / "toy_tasks" / n / "instruction.md").read_text().strip() for n in holdout_ids}
    assert not used & holdout_text  # 考试题没有进训练数据


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
