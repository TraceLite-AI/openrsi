"""运行时：配置校验、完整循环、续跑、数据隔离、进程内与 MCP 子进程混用。"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openrsi import Agent, Environment, Proposer, build_server
from openrsi.conformance import check
from openrsi_runtime import IsolationError, Runner, RunConfig, Wiring, load_config

import secret_ops

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "secret_ops" / "openrsi.yaml"


def config(tmp_path, memory_command=False, **overrides):
    """示例配置的副本，状态写到临时目录；默认 memory 也在进程内。"""
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    if memory_command:
        pythonpath = os.pathsep.join(str(ROOT / p) for p in ("sdk/python", "components/memory"))
        data["components"]["memory"] = {"command": [sys.executable, "-m", "openrsi_memory", "--path", "state/memory"],
                                        "env": {"PYTHONPATH": pythonpath}}
    else:
        data["components"]["memory"] = {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}
    data["components"]["lineage"]["with"]["path"] = "state/lineage.jsonl"
    for slot, spec in overrides.items():
        data["components"][slot] = spec
    path = tmp_path / "openrsi.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(path)


def run(cfg, generations=None):
    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop).run(generations)
            return reports, await w.lineage.log(), await w.lineage.best()

    return asyncio.run(go())


def test_config_validation():
    base = {"name": "x", "components": {s: {"python": "m:C"} for s in
                                        ("env", "agent", "judge", "memory", "proposer", "algorithm", "gate", "lineage")}}
    RunConfig.model_validate(base)
    with pytest.raises(ValidationError, match="缺少组件"):
        RunConfig.model_validate({**base, "components": {"env": {"python": "m:C"}}})
    bad = {**base, "components": {**base["components"], "judge": {"python": "m:C", "url": "http://x"}}}
    with pytest.raises(ValidationError, match="只能指定"):
        RunConfig.model_validate(bad)
    chain = {**base, "components": {**base["components"], "evidence": {"same_as": "attribution"},
                                    "attribution": {"same_as": "judge"}}}
    with pytest.raises(ValidationError, match="same_as"):
        RunConfig.model_validate(chain)
    with pytest.raises(ValidationError, match="未知的组件位置"):
        RunConfig.model_validate({**base, "components": {**base["components"], "oracle": {"python": "m:C"}}})


def test_full_loop_reaches_target(tmp_path):
    reports, log, best = run(config(tmp_path))
    assert best.metrics["holdout"] == 1.0
    assert log[0].status == "root" and log[0].metrics["holdout"] == 0.0
    accepted = [e for e in log if e.status == "accepted"]
    for e in accepted:  # 每个被接受的候选都严格好于它的父快照
        parent = next(p for p in log if p.id == e.parent_ids[0])
        assert e.metrics["holdout"] > parent.metrics["holdout"]
        assert e.decision.accepted and e.patch is not None
    assert (tmp_path / "state" / "lineage.jsonl").exists()


def test_resume_continues_from_lineage(tmp_path):
    cfg = config(tmp_path)
    first, log1, _ = run(cfg, generations=2)
    second, log2, best = run(cfg)
    assert second[0].generation == max(e.generation for e in log1) + 1
    assert second[0].parent_id == log1[-1].id  # 从上次最好的快照接着跑
    assert best.metrics["holdout"] == 1.0 and len(log2) > len(log1)


class SpyAgent(secret_ops.RuleAgent):
    seen_reference = []

    def run(self, task, context, snapshot_id=None):
        SpyAgent.seen_reference.append(task.reference)
        return super().run(task, context, snapshot_id)


class SpyProposer(secret_ops.RuleInducer):
    splits = []

    def propose(self, request):
        SpyProposer.splits.extend(ep.task.split for ep in request.episodes)
        return super().propose(request)


def test_isolation_agent_never_sees_reference_and_proposer_only_train(tmp_path):
    sys.modules["spies"] = sys.modules[__name__]
    cfg = config(tmp_path, agent={"python": f"{__name__}:SpyAgent"}, proposer={"python": f"{__name__}:SpyProposer"})
    run(cfg, generations=3)
    assert SpyAgent.seen_reference and all(r is None for r in SpyAgent.seen_reference)
    assert SpyProposer.splits and set(SpyProposer.splits) == {"train"}


class LeakyEnv(secret_ops.SecretOpsEnv):
    """把 holdout 题混进 train。"""

    def tasks(self, split, round=0, limit=None):
        if split == "train":
            return super().tasks("train", round, limit) + super().tasks("holdout", 0, 1)
        return super().tasks(split, round, limit)


def test_env_mixing_splits_is_rejected(tmp_path):
    cfg = config(tmp_path, env={"python": f"{__name__}:LeakyEnv"})
    with pytest.raises(IsolationError, match="train"):
        run(cfg, generations=1)


def test_mixed_local_and_mcp_subprocess(tmp_path):
    cfg = config(tmp_path, memory_command=True)
    reports, log, best = run(cfg)
    assert best.metrics["holdout"] == 1.0
    assert (tmp_path / "state" / "memory" / "HEAD").read_text() == best.id  # 子进程里的 memory 切到了最佳快照


def test_wrong_component_kind_is_reported(tmp_path):
    cfg = config(tmp_path, gate={"python": "openrsi_lineage:LineageStore"})
    with pytest.raises(ValueError, match="gate 位置需要 gate"):
        run(cfg, generations=1)


def test_cli_run_and_log(tmp_path, capsys):
    from openrsi_runtime.cli import main
    cfg = config(tmp_path)
    path = str(cfg.base_dir / "openrsi.yaml")
    assert main(["run", path, "--generations", "2"]) == 0
    assert main(["log", path]) == 0
    out = capsys.readouterr().out
    assert "root" in out and "accepted" in out and "最佳快照" in out


@pytest.mark.parametrize("comp", [secret_ops.SecretOpsEnv(), secret_ops.RuleAgent(), secret_ops.RuleInducer()],
                         ids=["env", "agent", "proposer"])
def test_example_components_conform(comp):
    assert isinstance(comp, (Environment, Agent, Proposer))
    sample = secret_ops.SecretOpsEnv().tasks("train", 1)[0]  # 领域相关的组件用本领域的题目检查
    report = asyncio.run(check(build_server(comp), sample_task=sample))
    assert report.ok, report.text()


def test_resume_detects_lost_memory(tmp_path):
    import shutil
    cfg = config(tmp_path)
    run(cfg, generations=2)
    shutil.rmtree(tmp_path / "state" / "memory")
    with pytest.raises(RuntimeError, match="memory 里没有它"):
        run(cfg, generations=1)
