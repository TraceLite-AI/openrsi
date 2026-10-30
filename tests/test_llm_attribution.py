"""PromptRewriter 读归因（two-loop-evolution 设计的 P0）：归因要真的出现在发给 LLM 的请求里。"""
import asyncio
import sys
from pathlib import Path

import openrsi_llm
import pytest
import yaml

from openrsi import (Attribution, AttributionTarget, EvolvableInfo, Episode, ProposalRequest, RubricResult, Task,
                     Trajectory, Verdict, build_server, connect)
from openrsi_llm import PromptRewriter
from openrsi_runtime import Runner, Wiring, load_config

ROOT = Path(__file__).resolve().parents[1]
INFO = EvolvableInfo(target="prompt", type="prompt", head="s0")
TRAIN_SENTINEL = "TRAIN-ATTRIBUTION-7f3a"
HOLDOUT_SENTINEL = "HOLDOUT-ATTRIBUTION-91c2"


def episode(i: int, passed: bool = False) -> Episode:
    task = Task(id=f"t{i}", split="train", input={"question": f"question number {i}"})
    traj = Trajectory(id=f"traj-{i}", task_id=task.id, output=f"answer {i}", steps=[])
    rubric = [] if passed else [RubricResult(id="correct", result="fail", note="wrong value")]
    return Episode(task=task, trajectory=traj,
                   verdict=Verdict(task_id=task.id, trajectory_id=traj.id, passed=passed, score=float(passed),
                                   rubric=rubric, judge_ref="j"))


def attribution(episode_id: str, note: str, weights=(-0.8,), rationale=None) -> Attribution:
    return Attribution(episode_id=episode_id, method="rubric-evidence", rationale=rationale,
                       targets=[AttributionTarget(kind="step", ref=f"s{k}", weight=w, note=f"{note} {k}")
                                for k, w in enumerate(weights)])


def request(episodes, attributions) -> ProposalRequest:
    return ProposalRequest(target=INFO, targets=[INFO], parent_snapshot="s0", episodes=episodes,
                           attributions=attributions, state=[], states={"prompt": []})


@pytest.fixture
def calls(monkeypatch):
    """替换 openrsi_llm.chat，记下每次请求的 messages，返回一段新的系统提示。"""
    seen = []

    def fake_chat(base_url, model, messages, **kw):
        seen.append(messages)
        return {"text": "Think step by step and state the final value.", "usage": {}, "finish_reason": "stop"}

    monkeypatch.setattr(openrsi_llm, "chat", fake_chat)
    return seen


def user_text(calls) -> str:
    return calls[-1][-1]["content"]


def test_attribution_reaches_request_payload(calls):
    eps = [episode(0), episode(1)]
    attrs = [attribution("traj-0", TRAIN_SENTINEL), attribution("holdout-traj-9", HOLDOUT_SENTINEL)]
    [patch] = PromptRewriter("http://x/v1").propose(request(eps, attrs))
    text = user_text(calls)
    assert TRAIN_SENTINEL in text and "Attribution notes:" in text and openrsi_llm.ATTRIBUTION_HEADER in text
    assert HOLDOUT_SENTINEL not in text  # 不属于本轮 train 的归因必须丢掉
    # 归因放在对应错题下面，不放到别的题下面
    block0, block1 = text.split("---")
    assert TRAIN_SENTINEL in block0 and TRAIN_SENTINEL not in block1
    assert "attributions used 1:" in patch.rationale and "unknown_episode=1" in patch.rationale


def test_no_attribution_keeps_old_payload(calls):
    eps = [episode(0), episode(1)]
    [p1] = PromptRewriter("http://x/v1").propose(request(eps, []))
    [p2] = PromptRewriter("http://x/v1", max_attribution_lines=0).propose(
        request(eps, [attribution("traj-0", TRAIN_SENTINEL)]))
    assert calls[0] == calls[1]
    assert "Attribution" not in user_text(calls)
    assert p1.rationale == p2.rationale == "rewrite system prompt from 2 train failures"


def test_no_failures_still_no_patch(calls):
    assert PromptRewriter("http://x/v1").propose(request([episode(0, passed=True)],
                                                         [attribution("traj-0", TRAIN_SENTINEL)])) == []
    assert calls == []


def test_passing_train_episode_attribution_is_not_shown(calls):
    eps = [episode(0), episode(1, passed=True)]
    [patch] = PromptRewriter("http://x/v1").propose(request(eps, [attribution("traj-1", TRAIN_SENTINEL)]))
    assert TRAIN_SENTINEL not in user_text(calls)
    assert "not_shown=1" in patch.rationale and "attributions used" not in patch.rationale


def test_truncation_is_stable_and_ordered(calls):
    eps = [episode(0)]
    attrs = [attribution("traj-0", "note", weights=(0.5, -0.9, -0.1, 0.2, -0.4))]
    rewriter = PromptRewriter("http://x/v1", max_attribution_lines=3)
    [p1] = rewriter.propose(request(eps, attrs))
    [p2] = rewriter.propose(request(eps, attrs))
    assert calls[0] == calls[1] and p1.rationale == p2.rationale  # 同样的输入，同样的截断
    notes = [line for line in user_text(calls).splitlines() if line.startswith("- [")]
    assert [n.split("weight=")[1][:5] for n in notes] == ["-0.90", "-0.40", "-0.10"]  # 责任最大的先放
    assert "truncated=1" in p1.rationale


def test_long_notes_are_clipped_and_char_budget_holds(calls):
    attrs = [attribution("traj-0", "x" * 5000, weights=(-0.5, -0.4, -0.3))]
    [patch] = PromptRewriter("http://x/v1", max_attribution_chars=300).propose(request([episode(0)], attrs))
    notes = [line for line in user_text(calls).splitlines() if line.startswith("- [")]
    assert notes and all(len(n) < 300 for n in notes) and sum(len(n) - 2 for n in notes) <= 300
    assert "truncated=1" in patch.rationale


def test_empty_attribution_is_counted(calls):
    attrs = [Attribution(episode_id="traj-0", method="m", targets=[])]
    [patch] = PromptRewriter("http://x/v1").propose(request([episode(0)], attrs))
    assert "Attribution notes" not in user_text(calls) and "empty=1" in patch.rationale


def test_attribution_survives_mcp_transport(calls):
    """经过 MCP 序列化以后，归因仍然到达 LLM 请求。"""
    req = request([episode(0)], [attribution("traj-0", TRAIN_SENTINEL)])

    async def go():
        async with connect(build_server(PromptRewriter("http://x/v1"))) as c:
            return await c.proposer.propose(req)

    [patch] = asyncio.run(go())
    assert TRAIN_SENTINEL in user_text(calls) and "attributions used 1" in patch.rationale


def test_runner_feeds_real_attributions_to_rewriter(tmp_path, calls, monkeypatch):
    """完整循环：官方归因器的输出经运行时进入 PromptRewriter 的请求，补丁说明记下用了哪些归因。"""
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "secret_ops"))
    data = yaml.safe_load((ROOT / "examples" / "secret_ops" / "openrsi.yaml").read_text(encoding="utf-8"))
    data["components"]["memory"] = {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}
    data["components"]["lineage"]["with"]["path"] = "state/lineage.jsonl"
    data["components"]["proposer"] = {"python": "openrsi_llm:PromptRewriter", "with": {"base_url": "http://x/v1"}}
    data["evolvables"] = {"prompt": {"python": "openrsi_evolvables:PromptEvolvable", "with": {"path": "state/prompt"}}}
    path = tmp_path / "openrsi.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(path)

    async def go():
        async with Wiring(cfg) as w:
            await Runner(w, cfg.loop).run(1)
            return await w.lineage.log()

    log = asyncio.run(go())
    assert calls, "PromptRewriter 没有被调用"
    text = user_text(calls)
    assert "Attribution notes:" in text and "[rubric" in text
    assert "holdout" not in text.lower()  # holdout 的题和归因都不能进提案
    patched = [e for e in log if e.patch is not None]
    assert patched and "attributions used" in patched[0].patch.rationale


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
