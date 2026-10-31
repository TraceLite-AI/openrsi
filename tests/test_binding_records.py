"""P2：执行绑定、不可变记录、运行清单（spec/drafts/two-loop-evolution.md §4.1–4.3）。"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from openrsi import Item, Task
from openrsi_harness import HarnessAgent
from openrsi_runtime import CellError, Runner, Wiring, load_config
from openrsi_runtime.bindings import PROBES, BindingPlan, EnvVarBinding
from openrsi_runtime.records import RecordError, RecordStore

from cells_env import SHANGHAI, UTC, BindingAwareAgent

EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
BINDING = {"python": "openrsi_runtime.bindings:EnvVarBinding",
           "with": {"cells": {"utc": {"env": {"TZ": "UTC"}}, "shanghai": {"env": {"TZ": "Asia/Shanghai"}},
                              "cp1252": {"env": {"LC_ALL": "C", "LANG": "C"}}},
                    "probes": {"E4.1:时区": "env.TZ", "E4.1:默认编码": "python.preferred_encoding"}}}


def config(base: Path, *, binding=True, records=True, agent="cells_env:BindingAwareAgent", env_kw=None, gate=None,
           cells=None):
    data = {"protocol": "openrsi/0.4", "name": "p2", "plugins": ["cells_env"],
            "cells": {**(cells or {}), **({"binding": BINDING} if binding else {})},
            "components": {
                "env": {"python": "cells_env:TZCellsEnv", "with": env_kw or {}},
                "agent": {"python": agent},
                "proposer": {"python": "secret_ops:RuleInducer"},
                "judge": {"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
                "evidence": {"same_as": "judge"},
                "attribution": {"python": "openrsi_attribution:CellContrastAttributor"},
                "algorithm": {"python": "openrsi_algorithm:GreedyAlgorithm", "with": {"max_generations": 6}},
                "gate": {"python": "openrsi_gate:HoldoutGate", "with": gate or {}},
                "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
                "memory": {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}}},
            "loop": {"holdout_limit": 6, "train_limit": 6}}
    if records:
        data["records"] = {"path": "state/records"}
    base.mkdir(parents=True, exist_ok=True)
    (base / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(base / "openrsi.yaml")


def run(cfg, generations=1, logs=None):
    async def go():
        async with Wiring(cfg) as w:
            await Runner(w, cfg.loop, log=(logs.append if logs is not None else lambda s: None)).run(generations)
            return await w.lineage.log()

    return asyncio.run(go())


# ---------------------------------------------------------------- 绑定

def test_probes_really_run_in_a_subprocess():
    assert PROBES["env.TZ"]({"TZ": "Asia/Shanghai"}) == "Asia/Shanghai"
    assert PROBES["env.TZ"]({}) == ""


def test_binding_verifies_declared_coordinates():
    b = EnvVarBinding(**BINDING["with"])
    plan = b.resolve(SHANGHAI, SHANGHAI.fingerprint())
    receipt = b.prepare(plan)
    check = b.verify(plan, receipt, SHANGHAI)
    assert check.ok and receipt.observed == {"env.TZ": "Asia/Shanghai"} and check.unverified == ["E2.1:os"]
    wrong = b.verify(plan, b.prepare(b.resolve(UTC, UTC.fingerprint())), SHANGHAI)  # 拿别的格的回执
    assert not wrong.ok and "回执不是这份方案的" in wrong.mismatches
    with pytest.raises(ValueError, match="未登记的探测"):
        EnvVarBinding(cells={}, probes={"E2.1:os": "rm -rf /"})


def test_bound_run_records_receipts_and_hands_agent_the_plan(tmp_path):
    BindingAwareAgent.seen = []
    log = run(config(tmp_path, records=False))
    assert log and all(b and set(b) == {"env", "plan_hash", "image"} for b in BindingAwareAgent.seen)
    assert {b["env"]["TZ"] for b in BindingAwareAgent.seen} == {"UTC", "Asia/Shanghai"}

    async def one_episode():
        async with Wiring(config(tmp_path / "ep", records=False)) as w:
            r = Runner(w, w.config.loop)
            await r._load_targets()
            from openrsi_runtime import CellRegistry
            r.registry = await CellRegistry.open(w.env, w.config.cells)
            r.binding = EnvVarBinding(**BINDING["with"])
            genome = {t: (await tgt.facade.snapshot()).id for t, tgt in r.targets.items()}
            [task] = [t for t in await r.tasks("holdout", 0, 1) if t.env_ref == "shanghai"]
            return await r.episode(task, genome)

    ep = asyncio.run(one_episode())
    [rc] = [e for e in ep.evidence if e.type == "exec_receipt"]
    assert rc.verify() and rc.content["observed"] == {"env.TZ": "Asia/Shanghai"} and rc.content["cell_id"] == "shanghai"


def test_environment_the_host_cannot_produce_stops_the_run(tmp_path):
    # 声明 cp1252，但宿主机按 LC_ALL=C 准备出来的仍是 utf-8：声明和实际不符
    with pytest.raises(CellError, match="实际执行环境和声明不符.*cp1252"):
        run(config(tmp_path, records=False, env_kw={"with_cp1252": True}))


def test_agent_that_ignores_the_plan_is_refused(tmp_path):
    with pytest.raises(CellError, match="不支持执行绑定"):
        run(config(tmp_path, records=False, agent="secret_ops:RuleAgent"))


def test_harness_agent_applies_the_plan_in_its_process(tmp_path):
    repo = tmp_path / "harness"
    repo.mkdir()
    (repo / "agent.py").write_text(
        "import json, os, sys\nreq = json.load(sys.stdin)\n"
        "print(json.dumps({'output': os.environ.get('TZ', ''), 'steps': []}))\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    plan = BindingPlan(cell_id="shanghai", fingerprint="f", env={"TZ": "Asia/Shanghai"})
    task = Task(id="t", input="q", metadata={"binding": {"env": plan.env, "plan_hash": plan.plan_hash, "image": None}})
    traj = HarnessAgent(repo=repo, command=[sys.executable, "agent.py"]).run(task, [Item(id="i", text="x")], "s")
    assert traj.output == "Asia/Shanghai" and traj.metadata["binding_plan_hash"] == plan.plan_hash
    plain = HarnessAgent(repo=repo, command=[sys.executable, "agent.py"]).run(Task(id="t", input="q"), [], "s")
    assert plain.output == "" and "binding_plan_hash" not in plain.metadata


# ---------------------------------------------------------------- 不可变记录

def test_records_are_append_only_and_tamper_evident(tmp_path):
    run(config(tmp_path), generations=2)
    store = RecordStore(tmp_path / "state" / "records")
    kinds = {r["kind"] for r in store.entries()}
    assert {"manifest", "task", "trajectory", "evidence", "receipt", "evidence_record", "verdict",
            "attribution", "derived_record"} <= kinds
    assert store.verify() == []
    for row in store.entries("derived_record"):  # 每条派生记录都指回一条原始记录
        d = store.get(row["hash"])
        assert d["what"] in ("verdict", "attribution")
        for src in d["source_hashes"]:
            assert store.get(src)["kind"] == "evidence_record"
    rec = store.get(store.entries("evidence_record")[0]["hash"])
    assert rec["execution_receipt_hash"] and rec["cell_fingerprint"]
    for h in [rec["raw_trajectory_hash"], *rec["raw_artifact_hashes"]]:
        store.get(h)

    victim = store.entries("trajectory")[0]["hash"]
    f = store._file(victim)
    data = json.loads(f.read_text(encoding="utf-8"))
    data["output"] = "forged"
    f.write_text(json.dumps(data), encoding="utf-8")
    assert store.verify() == [victim]
    with pytest.raises(RecordError, match="被篡改"):
        store.get(victim)
    store._file(store.put({"x": 1}, "x")).write_text('{"x":2}', encoding="utf-8")
    with pytest.raises(RecordError, match="内容不同"):  # 库里的文件被改过，再写同一个对象会被发现
        store.put({"x": 1}, "x")


def test_same_content_is_stored_once(tmp_path):
    store = RecordStore(tmp_path)
    assert store.put({"a": 1}, "x") == store.put({"a": 1}, "x")
    assert len(store.entries()) == 1
    with pytest.raises(RecordError, match="缺少记录"):
        store.get("0" * 64)


# ---------------------------------------------------------------- 运行清单

def test_resume_requires_same_manifest_in_strict_mode(tmp_path):
    log = run(config(tmp_path))
    root = next(e for e in log if e.status == "root")
    assert root.manifest and len(root.manifest) == 64
    run(config(tmp_path))  # 同样的配置：接着跑
    with pytest.raises(RecordError, match="运行清单变了"):
        run(config(tmp_path, gate={"min_gain": 0.5}))


def test_resume_with_changed_manifest_only_warns_without_records(tmp_path):
    run(config(tmp_path, binding=False, records=False))
    logs = []
    run(config(tmp_path, binding=False, records=False, gate={"min_gain": 0.5}), logs=logs)
    assert any("运行清单变了" in s for s in logs)


def test_manifest_ignores_credentials_and_transport(tmp_path):
    from openrsi_runtime.records import sanitized_config
    a = sanitized_config(config(tmp_path / "a"))
    cfg_b = config(tmp_path / "b")
    cfg_b.components["gate"] = cfg_b.components["gate"].model_copy(
        update={"python": None, "command": ["openrsi-gate"], "env": {"API_KEY": "secret"}, "with_": {}})
    b = sanitized_config(cfg_b)
    assert "secret" not in json.dumps(b) and b["components"]["gate"]["env"] == ["API_KEY"]
    assert a["components"]["judge"] == b["components"]["judge"]
