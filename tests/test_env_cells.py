"""环境格模式（spec/drafts/environment-cells.md，two-loop-evolution 设计的 P1）。

写了 cells 才启用；启用后每个格单独打分、指纹不能变、env_ref 必须对、续跑时格必须一致。
不写 cells 时和 0.4 完全一样。
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from openrsi import ToolCallError, build_server, connect
from openrsi.conformance import check
from openrsi_runtime import CellError, CellsConfig, Runner, Wiring, load_config

from cells_env import PLAIN, VARIANT, CellsEnv, RecordingAgent
from secret_ops import SecretOpsEnv

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
PYTHONPATH = os.pathsep.join(
    [str(ROOT / p) for p in (ROOT / "pytest.ini").read_text().split("pythonpath =", 1)[1].split("\n", 1)[0].split()]
    + [str(TESTS)])
EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
N_HOLDOUT = len(SecretOpsEnv().tasks("holdout"))


def components(env=None, agent=None):
    return {
        "env": env or {"python": "cells_env:CellsEnv"},
        "agent": agent or {"python": "secret_ops:RuleAgent"},
        "proposer": {"python": "secret_ops:RuleInducer"},
        "judge": {"python": "openrsi_judge:EvidenceJudge", "with": {"rubric": EQ}},
        "evidence": {"same_as": "judge"},
        "attribution": {"python": "openrsi_attribution:RubricEvidenceAttributor"},
        "algorithm": {"python": "openrsi_algorithm:GreedyAlgorithm",
                      "with": {"max_generations": 12, "patience": 6, "target_score": 1.0}},
        "gate": {"python": "openrsi_gate:HoldoutGate"},
        "lineage": {"python": "openrsi_lineage:LineageStore", "with": {"path": "state/lineage.jsonl"}},
        "memory": {"python": "openrsi_memory:MemoryEvolvable", "with": {"path": "state/memory"}},
    }


def write_config(base: Path, comps=None, cells=None, **loop):
    data = {"protocol": "openrsi/0.4", "name": "cells", "plugins": ["cells_env"],
            "components": comps or components(), "loop": {"retrieve_k": 5, **loop}}
    if cells is not None:
        data["cells"] = cells
    base.mkdir(parents=True, exist_ok=True)
    (base / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(base / "openrsi.yaml")


def run(cfg, generations=None, logs=None):
    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop, log=(logs.append if logs is not None else lambda s: None)).run(
                generations)
            return reports, await w.lineage.log()

    return asyncio.run(go())


def stable(log):
    """谱系里与时间无关的部分。"""
    return [(e.id, e.status, tuple(e.parent_ids), e.generation, tuple(sorted(e.metrics.items())),
             tuple(sorted(e.task_scores.items())), tuple(sorted(e.env_cells.items()))) for e in log]


# ---------------------------------------------------------------- 正常路径

def test_same_task_scored_once_per_cell(tmp_path):
    logs = []
    _, log = run(write_config(tmp_path, cells={}), generations=2, logs=logs)
    root = next(e for e in log if e.status == "root")
    assert len(root.task_scores) == 2 * N_HOLDOUT  # 同一批题在两个格里各打一次分
    assert {k.rsplit("@", 1)[1] for k in root.task_scores} == {"plain", "variant"}
    assert root.metrics["n_holdout"] == 2 * N_HOLDOUT
    assert root.env_cells == {"plain": PLAIN.fingerprint(), "variant": VARIANT.fingerprint()}
    assert all(e.env_cells == root.env_cells for e in log)
    for e in log:  # 父子比较用的是同一批 (题, 格)
        assert set(e.task_scores) in (set(), set(root.task_scores))
    assert any("variant" in s and "tag" in s for s in logs)  # 非 digest 的钉住方式要警告
    assert not any("plain" in s and "警告" in s for s in logs)


def test_train_and_holdout_cells_can_differ(tmp_path):
    _, log = run(write_config(tmp_path, cells={"train": ["plain"], "holdout": ["variant"]}), generations=1)
    root = next(e for e in log if e.status == "root")
    assert len(root.task_scores) == N_HOLDOUT and all(k.endswith("@variant") for k in root.task_scores)
    assert set(root.env_cells) == {"plain", "variant"}


def test_legacy_config_unchanged_even_if_env_has_cells(tmp_path):
    _, a = run(write_config(tmp_path / "cells_env"), generations=3)
    comps = components(env={"python": "secret_ops:SecretOpsEnv", "with": {"seed": 0}})
    _, b = run(write_config(tmp_path / "plain_env", comps), generations=3)
    assert stable(a) == stable(b)  # 不写 cells：和普通 env 跑出完全一样的谱系
    assert all(not e.env_cells and "@" not in "".join(e.task_scores) for e in a)


def test_coords_hidden_from_agent_by_default(tmp_path):
    comps = components(agent={"python": "cells_env:RecordingAgent"})
    RecordingAgent.seen = []
    run(write_config(tmp_path / "hidden", comps, cells={}), generations=1)
    assert RecordingAgent.seen and all("env_coords" not in m for m in RecordingAgent.seen)
    RecordingAgent.seen = []
    run(write_config(tmp_path / "shown", comps, cells={"expose_coords_to_agent": True}), generations=1)
    coords = [m["env_coords"] for m in RecordingAgent.seen]
    assert PLAIN.coords in coords and VARIANT.coords in coords
    assert not [m for m in RecordingAgent.seen if set(m) - {"env_coords"}]  # 只多了坐标，没带出别的


def test_trajectory_ids_distinguish_cells(tmp_path):
    cfg = write_config(tmp_path, cells={})

    async def go():
        async with Wiring(cfg) as w:
            r = Runner(w, cfg.loop)
            await r._load_targets()
            from openrsi_runtime import CellRegistry
            r.registry = await CellRegistry.open(w.env, cfg.cells)
            tasks = await r.tasks("train", 1, 2)
            genome = {t: (await tgt.facade.snapshot()).id for t, tgt in r.targets.items()}
            return [await r.episode(t, genome) for t in tasks]

    eps = asyncio.run(go())
    ids = [e.trajectory.id for e in eps]
    assert len(ids) == len(set(ids)) == 4 and all(i.endswith(("@plain", "@variant")) for i in ids)


# ---------------------------------------------------------------- 必须停下的情况

@pytest.mark.parametrize("env,cells,match", [
    ({"python": "secret_ops:SecretOpsEnv", "with": {"seed": 0}}, {}, "没有实现 rsi_env_cells"),
    (None, {"holdout": ["nope"]}, "不在 env 声明的格里"),
    ({"python": "cells_env:CellsEnv", "with": {"bad_ref": True}}, {}, "env_ref"),
    ({"python": "cells_env:CellsEnv", "with": {"duplicate": True}}, {}, "出现了两次"),
    ({"python": "cells_env:CellsEnv", "with": {"at_ids": True}}, {}, "不能含 @"),
])
def test_bad_cells_stop_the_run(tmp_path, env, cells, match):
    with pytest.raises(CellError, match=match):
        run(write_config(tmp_path, components(env=env), cells=cells), generations=1)


def test_fingerprint_drift_mid_run_stops(tmp_path):
    env = {"python": "cells_env:CellsEnv", "with": {"drift_after": 3}}
    with pytest.raises(CellError, match="变了"):
        run(write_config(tmp_path, components(env=env), cells={}), generations=5)


def test_resume_requires_same_cells(tmp_path):
    run(write_config(tmp_path, cells={}), generations=1)
    _, log = run(write_config(tmp_path, cells={}), generations=1)  # 同样的格：接着跑
    assert max(e.generation for e in log) >= 2
    drifted = components(env={"python": "cells_env:CellsEnv", "with": {"drift_after": 0}})
    with pytest.raises(CellError, match="不一致"):
        run(write_config(tmp_path, drifted, cells={}), generations=1)
    with pytest.raises(CellError, match="不一致"):  # 换了格的组合也不行
        run(write_config(tmp_path, cells={"holdout": ["plain"], "train": ["plain"]}), generations=1)
    with pytest.raises(CellError, match="没有写 cells"):  # 环境格模式的谱系不能用普通模式续跑
        run(write_config(tmp_path), generations=1)


def test_cells_config_validation():
    CellsConfig.model_validate({"train": ["a"], "holdout": ["a", "b"]})
    for bad, match in [({"train": []}, "空列表"), ({"holdout": ["a@b"]}, "@"), ({"train": ["a", "a"]}, "重复"),
                       ({"oops": 1}, "Extra")]:
        with pytest.raises(ValidationError, match=match):
            CellsConfig.model_validate(bad)


# ---------------------------------------------------------------- SDK 与传输

def test_server_exposes_cells_and_rejects_unknown_cell():
    async def go():
        async with connect(build_server(CellsEnv())) as c:
            assert "rsi_env_cells" in c.tools and "rsi_env_cells" in c.info.tools
            cells = await c.env.cells()
            tasks = await c.env.tasks("holdout", 0, 2, cell="variant")
            with pytest.raises(ToolCallError, match="未知的环境格"):
                await c.env.tasks("holdout", 0, None, cell="nope")
            legacy = await c.env.tasks("holdout", 0, 2)
        async with connect(build_server(SecretOpsEnv())) as c:
            assert "rsi_env_cells" not in c.tools and await c.env.cells() is None
            with pytest.raises(ToolCallError, match="没有声明环境格"):
                await c.env.tasks("holdout", 0, None, cell="plain")
            old = await c.env.tasks("holdout", 0, 2)
        return cells, tasks, legacy, old

    cells, tasks, legacy, old = asyncio.run(go())
    assert [c.id for c in cells] == ["plain", "variant"] and cells[0].fingerprint() == PLAIN.fingerprint()
    assert [t.env_ref for t in tasks] == ["variant", "variant"]
    assert all(t.env_ref is None for t in legacy + old)


def test_conformance_checks_cells():
    rep = asyncio.run(check(build_server(CellsEnv())))
    names = {r.name: r.ok for r in rep.results}
    assert names["环境格声明合法"] and names["按格取题：env_ref 等于格 id"] and names["未知的格报错"], rep.summary()
    bad = asyncio.run(check(build_server(CellsEnv(bad_ref=True))))
    assert not {r.name: r.ok for r in bad.results}["按格取题：env_ref 等于格 id"]
    plain = asyncio.run(check(build_server(SecretOpsEnv())))
    assert not [r for r in plain.results if "环境格" in r.name]  # 没声明格的 env 不做这些检查


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _launch(kw, base, http_port=None):
    cmd = [sys.executable, "-m", "openrsi.launch", "cells_env:CellsEnv", "--with", json.dumps(kw),
           "--path", str(TESTS), "--path", str(ROOT / "examples" / "secret_ops"), "--cwd", str(base)]
    if http_port is None:
        return {"command": cmd, "env": {"PYTHONPATH": PYTHONPATH}}
    return cmd + ["--transport", "streamable-http", "--port", str(http_port)]


def test_same_lineage_over_three_transports(tmp_path):
    _, a = run(write_config(tmp_path / "inproc", cells={}), generations=3)

    b_dir = tmp_path / "stdio"
    b_dir.mkdir()
    _, b = run(write_config(b_dir, components(env=_launch({}, b_dir)), cells={}), generations=3)

    c_dir = tmp_path / "http"
    c_dir.mkdir()
    port = _free_port()
    proc = subprocess.Popen(_launch({}, c_dir, port), env={**os.environ, "PYTHONPATH": PYTHONPATH},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        deadline = time.time() + 30
        while True:
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read().decode()[-2000:])
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)
        _, c = run(write_config(c_dir, components(env={"url": f"http://127.0.0.1:{port}/mcp"}), cells={}),
                   generations=3)
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    assert stable(a) == stable(b) == stable(c)
    assert all(e.env_cells for e in a)
