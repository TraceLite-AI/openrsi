"""小模型全流程（用假推理服务和假训练在 CPU 上跑）：prompt 进化、权重进化、换权重重启服务、判别器抽取答案。"""
import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from openrsi import Episode, Item, ProposalRequest, Step, Task, Trajectory, Verdict
from openrsi_judge import EvidenceJudge
from openrsi_llm import ChatAgent, PromptRewriter
from openrsi_proposer import CompositeProposer
from openrsi_runtime import Runner, Wiring, load_config
from openrsi_verl import episode_messages

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "small_model"
GSM = {"items": [{"id": "correct", "check": "output_equals", "required": True,
                  "params": {"extract": r"#### (\-?[0-9\.\,]+)"}}]}


def test_judge_extracts_answer():
    j = EvidenceJudge(GSM)
    task = Task(id="t", input={"question": "q"}, split="train", reference={"expected": "1234"})

    def verdict(text):
        traj = Trajectory(id="tr", task_id="t", output=text, steps=[])
        return j.evaluate(task, traj, j.collect(task, traj))

    assert verdict("work\n#### 1,234").passed
    assert verdict("#### 1\nno wait\n#### 1234").passed  # 取最后一个
    assert not verdict("#### $1234").passed  # 和 verl 的 gsm8k 正则一样，$ 写在数字前不认
    assert not verdict("The answer is 1234.").passed
    assert "没有匹配" in verdict("The answer is 1234.").rubric[0].note


def test_composite_accepts_any_proposer_class(tmp_path):
    p = CompositeProposer([{"name": "weights", "python": "openrsi_verl:DistillProposer",
                            "with": {"data_dir": str(tmp_path), "min_examples": 1}}, "lesson"])
    assert set(p.members) == {"weights", "lesson"}
    with pytest.raises(ValueError, match="未知"):
        CompositeProposer(["nope"])
    with pytest.raises(ValueError, match="重复"):
        CompositeProposer([{"name": "a", "python": "openrsi_proposer:LessonProposer"}] * 2)


def test_episode_messages_uses_agent_conversation():
    task = Task(id="t", input={"question": "2+2?"}, split="train", reference={"expected": "4"})
    traj = Trajectory(id="tr", task_id="t", output="#### 4", steps=[
        Step(id="s0", type="message", role="system", content="Be brief."),
        Step(id="s1", type="message", role="user", content="2+2?"),
        Step(id="s2", type="message", role="assistant", content="#### 4")])
    ep = Episode(task=task, trajectory=traj, verdict=Verdict(task_id="t", trajectory_id="tr", passed=True,
                                                             score=1.0, judge_ref="j"))
    assert episode_messages(ep) == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"},
                                    {"role": "assistant", "content": "#### 4"}]


def test_chat_agent_reports_unreachable_endpoint():
    agent = ChatAgent(base_url="http://127.0.0.1:9/v1", timeout=2)
    t = agent.run(Task(id="t", input="x", split="train"), [])
    assert t.output is None and "连不上" in t.metadata["error"]
    with pytest.raises(ValueError):
        ChatAgent()


def test_managed_server_restarts_on_new_weights(tmp_path):
    model = tmp_path / "m1"
    model.mkdir()
    (model / "skill.json").write_text(json.dumps({"level": 1.0}))
    agent = ChatAgent(server={"command": [sys.executable, str(EX / "mock_llm.py"), "--model", "{model}", "--port",
                                          "{port}"], "port": _free_port(), "startup_timeout": 20})
    try:
        task = Task(id="t", input={"question": "What is 2 plus 3?"}, split="train")
        ctx = [Item(id="instructions", kind="prompt", text="End with #### <number>."),
               Item(id="model", kind="weights", text=str(model))]
        assert agent.run(task, ctx).output.endswith("#### 5")
        assert agent.run(task, ctx).metadata["model"] == str(model) and agent.server.restarts == 1
        agent.run(task, [ctx[0], Item(id="model", kind="weights", text=str(tmp_path / "m2"))])
        assert agent.server.restarts == 2
    finally:
        agent.server.stop()


def test_prompt_rewriter_patch(tmp_path):
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(EX / "mock_llm.py"), "--model", "x", "--port", str(port)])
    try:
        _wait(port)
        from openrsi_evolvables import PromptEvolvable

        prompt = PromptEvolvable()
        info = prompt.describe()
        task = Task(id="t", input={"question": "1 + 2?"}, split="train", reference={"expected": "3"})
        bad = Episode(task=task, trajectory=Trajectory(id="tr", task_id="t", output="The answer is 3.", steps=[]),
                      verdict=Verdict(task_id="t", trajectory_id="tr", passed=False, score=0.0, judge_ref="j"))
        req = ProposalRequest(target=info, targets=[info], parent_snapshot=prompt.snapshot().id, episodes=[bad],
                              attributions=[], max_patches=1, state=prompt.export(), states={"prompt": prompt.export()})
        [patch] = PromptRewriter(f"http://127.0.0.1:{port}/v1").propose(req)
        prompt.apply(patch)
        assert "####" in prompt.render()
        req2 = req.model_copy(update={"state": prompt.export(), "states": {"prompt": prompt.export()},
                                      "parent_snapshot": prompt.snapshot().id})
        assert PromptRewriter(f"http://127.0.0.1:{port}/v1").propose(req2) == []  # 改写结果没变就不提补丁
    finally:
        proc.terminate()


def test_full_pipeline_with_mock_model(tmp_path):
    work = tmp_path / "small_model"
    shutil.copytree(EX, work, ignore=shutil.ignore_patterns(".openrsi", "__pycache__", "toy_data"))
    subprocess.run([sys.executable, "make_toy_data.py", "toy_data"], cwd=work, check=True, capture_output=True)
    cfg_text = (work / "openrsi.mock.yaml").read_text().replace("port: 8199", f"port: {_free_port()}")
    port = int(cfg_text.split("port: ")[1].split()[0])
    cfg_text = cfg_text.replace("127.0.0.1:8199", f"127.0.0.1:{port}")
    (work / "openrsi.mock.yaml").write_text(cfg_text)
    cfg = load_config(work / "openrsi.mock.yaml")

    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop).run()
            return reports, await w.lineage.log(), w.components["agent"]

    reports, log, agent = asyncio.run(go())
    agent._obj.server.stop()
    accepted = [e for e in log if e.status == "accepted"]
    kinds = {e.patch.proposer_ref.rsplit("#", 1)[-1] for e in accepted if e.patch}
    assert kinds == {"prompt", "weights"}, kinds  # 两种进化都有被接受的
    assert max(e.metrics["holdout"] for e in log) >= 0.9
    root = next(e for e in log if e.status == "root")
    assert root.metrics["holdout"] == 0.0  # 起点：答案格式不对，判别器一题都抽不到
    best = max(accepted, key=lambda e: e.metrics["holdout"])
    assert set(best.snapshots) == {"prompt", "weights"} and best.snapshots["weights"] != "ckpt_root"
    assert len(list((work / ".openrsi" / "server-logs").glob("server-*.log"))) >= 2  # 换过权重，重启过服务
    sft = next((work / ".openrsi" / "sft").glob("*.jsonl"))
    first = json.loads(sft.read_text().splitlines()[0])
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
    assert "####" in first["messages"][0]["content"]  # 训练数据带着进化后的系统提示


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(port: int) -> None:
    import socket
    import time

    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("mock server 没有启动")


@pytest.mark.parametrize("manifest,component", [
    ("rsi-component.yaml", lambda: ChatAgent(base_url="http://127.0.0.1:9/v1")),
    ("rsi-component.rewriter.yaml", lambda: PromptRewriter(base_url="http://127.0.0.1:9/v1")),
])
def test_llm_manifests_match(manifest, component):
    from openrsi import build_server, connect, load_manifest

    async def served():
        async with connect(build_server(component())) as c:
            return c.info

    info = asyncio.run(served())
    m = load_manifest(ROOT / "components" / "llm" / manifest)
    assert (m.name, m.version) == (info.name, info.version)
    assert set(m.implements) == set(info.tools) and set(m.kinds) == set(info.kinds)
