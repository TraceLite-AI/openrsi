"""agent 给自己出改法（loop.self_propose）：出题、解析、主循环接线、玩具里的递归。"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from anyrsi import EvolvableInfo, ProposalRequest
from anyrsi_runtime.self_propose import KIND, build_task, parse_patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "group_toy"))
import group_toy as gt  # noqa: E402


def _request(**kw):
    info = EvolvableInfo(target="prompt", type="prompt", head="h",
                         example_ops=[{"op": "replace", "item": {"id": "x", "text": "y"}}])
    state = [{"id": "s1", "text": "good"}]
    return ProposalRequest(target=info, targets=[info], parent_snapshot="p", state=state,
                           states={"prompt": state}, **kw)


def test_build_task_is_a_train_task_with_targets_and_seed():
    t = build_task(_request(hints={"direction": "d", "secret": 1}), gen=3, index=7, seed=42)
    assert t.split == "train" and KIND in t.tags and t.id == "self-improve-g3-7"
    assert t.metadata["rollout_seed"] == 42 and t.metadata["generation"] == 3
    assert t.input["kind"] == KIND
    assert t.input["targets"][0]["target"] == "prompt" and t.input["targets"][0]["current"] == [{"id": "s1", "text": "good"}]
    assert t.input["hints"] == {"direction": "d"}  # 只放出改法用得上的提示
    assert "example_ops" in t.input["instruction"] or "replace" in t.input["instruction"]


def test_parse_patch_accepts_json_inside_text_and_rejects_bad_output():
    req = _request()
    ok = 'I will fix it: {"target": "prompt", "ops": [{"op": "replace", "item": {"id": "s1", "text": "bad"}}], ' \
         '"rationale": "flip s1"} done'
    p = parse_patch(ok, req, "agent")
    assert p and p.target == "prompt" and p.base_snapshot == "p" and p.proposer_ref == "self:agent"
    assert p.ops[0]["item"]["text"] == "bad" and p.rationale == "flip s1"
    assert parse_patch({"ops": [{"op": "remove", "id": "s1"}]}, req, "a").target == "prompt"  # 缺 target 用默认
    for bad in ["no json here", '{"target": "prompt", "ops": []}', '{"target": "judge", "ops": [{"op": "x"}]}',
                '{"target": "prompt", "ops": ["text"]}', None, 3]:
        assert parse_patch(bad, req, "a") is None


def test_config_drops_the_proposer_requirement_only_with_self_propose(tmp_path):
    import yaml

    from anyrsi_runtime import load_config

    path = gt.write_config(tmp_path / "a", "A:self", 0)
    assert load_config(path).loop.self_propose
    data = yaml.safe_load(path.read_text())
    data["loop"]["self_propose"] = False
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match="proposer"):
        load_config(path)


@pytest.mark.parametrize("arm", ["A", "group"])
def test_runner_asks_the_agent_itself_for_edits(tmp_path, arm):
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(gt.write_config(tmp_path / arm, f"{arm}:self", 0))
    asked: list = []

    async def go():
        async with Wiring(cfg) as w:
            assert "proposer" not in w.components
            agent = w.components["agent"]

            class Spy:
                kinds = agent.kinds

                async def run(self, task, context, snapshot_id=None):
                    if KIND in task.tags:
                        asked.append((task, snapshot_id))
                    return await agent.run(task, context, snapshot_id)

                def __getattr__(self, n):
                    return getattr(agent, n)

            counter = gt._Counting(Spy(), 2500)
            w.components["agent"] = counter
            runner = Runner(w, cfg.loop)
            try:
                await runner.run()
            except gt.BudgetExhausted:
                pass
            return runner, counter, await w.lineage.log()

    runner, counter, log = asyncio.run(go())
    assert asked, "改法应当由 agent 自己出"
    assert all(t.split == "train" for t, _ in asked)
    assert counter.calls >= 2500  # 出改法的调用也算进预算
    seeds = [t.metadata["rollout_seed"] for t, _ in asked]
    assert len(set(seeds)) == len(seeds)
    accepted = [e for e in log if e.status == "accepted"]
    assert accepted and all(e.patch.proposer_ref.startswith("self:") for e in accepted)
    # 出改法的是当时的父版本：接受新版本后，出题时用的快照也跟着变（递归）
    assert len({sid for _, sid in asked}) > 1


def test_meta_generation_still_needs_a_proposer(tmp_path):
    from anyrsi_runtime import Runner, Wiring, load_config

    cfg = load_config(gt.write_config(tmp_path / "m", "A:self", 0))

    async def go():
        async with Wiring(cfg) as w:
            return await Runner(w, cfg.loop)._propose(_request(), {}, solver=False)

    with pytest.raises(ValueError, match="proposer"):
        asyncio.run(go())


def test_toy_agent_writes_better_edits_as_it_gets_better():
    """玩具的设定：agent 越强，给自己出的改法越准（同一个被怪罪的位置，改对的比例更高）。"""
    agent = gt.ToyAgent(seed=0)
    att = [gt.Attribution(episode_id="e", method="t", targets=[gt.AttributionTarget(kind="item", ref="s0", weight=-1)])]

    def fix_rate(values):
        state = [{"id": k, "kind": "prompt", "text": v} for k, v in values.items()]
        req = _request(attributions=att).model_copy(update={"state": state, "states": {"prompt": state}})
        items = [gt.Item(id=k, kind="prompt", text=v) for k, v in values.items()]
        hits = 0
        for i in range(400):
            out = agent.run(build_task(req, gen=0, index=i, seed=i), items)
            out = asyncio.run(out) if asyncio.iscoroutine(out) else out
            ops = {o["item"]["id"]: o["item"]["text"] for o in json.loads(out.output)["ops"]}
            hits += ops.get("s0") == gt.BEST["s0"]
        return hits / 400

    weak = {**gt.WORST}
    strong = {**gt.BEST, "s0": gt.WORST["s0"]}
    assert gt.capability(strong) > 0.7 > 0.1 > gt.capability(weak)
    assert fix_rate(strong) > fix_rate(weak) + 0.3
