"""组件化设计本身的测试：任意组合、换传输方式结果不变、跨语言、故障与续跑、向前兼容。

跨语言部分需要 Node.js 和 tests/xlang 下的 npm 依赖（npm ci --prefix tests/xlang），缺了就跳过。
"""
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from openrsi import ComponentUnavailableError, connect
from openrsi.conformance import check
from openrsi_algorithm import ALGORITHMS
from openrsi_runtime import Runner, Wiring, load_config

ROOT = Path(__file__).resolve().parents[1]
XLANG = ROOT / "tests" / "xlang"
PYTHONPATH = os.pathsep.join(
    [str(ROOT / p) for p in (ROOT / "pytest.ini").read_text().split("pythonpath =", 1)[1].split("\n", 1)[0].split()]
    + [str(ROOT / "tests")])
EQ = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}

# secret_ops 示例的每个位置，进程内写法
SLOTS = {
    "env": ("secret_ops:SecretOpsEnv", {"seed": 0}),
    "agent": ("secret_ops:RuleAgent", {}),
    "proposer": ("secret_ops:RuleInducer", {}),
    "judge": ("openrsi_judge:EvidenceJudge", {"rubric": EQ}),
    "attribution": ("openrsi_attribution:RubricEvidenceAttributor", {}),
    "algorithm": ("openrsi_algorithm:GreedyAlgorithm", {"max_generations": 12, "patience": 6, "target_score": 1.0}),
    "gate": ("openrsi_gate:HoldoutGate", {}),
    "lineage": ("openrsi_lineage:LineageStore", {"path": "state/lineage.jsonl"}),
    "memory": ("openrsi_memory:MemoryEvolvable", {"path": "state/memory"}),
}


def python_spec(target, kw):
    return {"python": target, "with": kw}


def stdio_spec(target, kw, base):
    return {"command": [sys.executable, "-m", "openrsi.launch", target, "--with", json.dumps(kw),
                        "--path", str(ROOT / "examples" / "secret_ops"), "--cwd", str(base)],
            "env": {"PYTHONPATH": PYTHONPATH}}


def write_config(base: Path, components: dict, loop: dict | None = None):
    data = {"protocol": "openrsi/0.4", "name": "composition", "plugins": ["secret_ops"],
            "components": {**components, "evidence": {"same_as": "judge"}},
            "loop": {"retrieve_k": 5, "max_patches": 1, **(loop or {})}}
    base.mkdir(parents=True, exist_ok=True)
    (base / "openrsi.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(base / "openrsi.yaml")


def run(cfg, generations=None):
    async def go():
        async with Wiring(cfg) as w:
            reports = await Runner(w, cfg.loop).run(generations)
            return reports, await w.lineage.log()

    return asyncio.run(go())


def fingerprint(log):
    """谱系里与时间无关的部分：两次运行应当完全一样。"""
    return [(e.id, e.status, tuple(e.parent_ids), e.generation, tuple(sorted(e.metrics.items())),
             tuple(sorted(e.task_scores.items()))) for e in log]


def check_invariants(log):
    by_id = {e.id: e for e in log}
    roots = [e for e in log if e.status == "root"]
    assert len(roots) == 1 and len(by_id) == len(log)
    for e in log:
        for p in e.parent_ids:
            assert p in by_id and by_id[p].status in ("root", "accepted")
        if e.status == "accepted":
            assert e.metrics["holdout"] >= by_id[e.parent_ids[0]].metrics["holdout"]


# ---------------------------------------------------------------- 1. 任意组合

GATES = {"holdout": {}, "no_regression": {"max_task_regressions": 0}}


@pytest.mark.parametrize("gate", sorted(GATES))
@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_composition_matrix(tmp_path, algorithm, gate):
    comps = {slot: python_spec(*spec) for slot, spec in SLOTS.items()}
    cls = ALGORITHMS[algorithm]
    comps["algorithm"] = python_spec(f"openrsi_algorithm:{cls.__name__}",
                                     {"max_generations": 12, "patience": 6, "target_score": 1.0})
    comps["gate"] = python_spec("openrsi_gate:HoldoutGate", GATES[gate])
    comps["guard"] = python_spec("openrsi_guard:PolicyGuard", {})
    _, log = run(write_config(tmp_path, comps))
    check_invariants(log)
    best = max(e.metrics["holdout"] for e in log if e.status in ("root", "accepted"))
    root = next(e for e in log if e.status == "root")
    assert best > root.metrics["holdout"], f"{algorithm} + {gate} 没有任何提升"


# ---------------------------------------------------------------- 2. 换传输方式，结果不变


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_http(target, kw, base):
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "openrsi.launch", target, "--with", json.dumps(kw), "--path",
                             str(ROOT / "examples" / "secret_ops"), "--cwd", str(base), "--transport",
                             "streamable-http", "--port", str(port)],
                            env={**os.environ, "PYTHONPATH": PYTHONPATH}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read().decode()[-2000:])
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc, f"http://127.0.0.1:{port}/mcp"
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("HTTP server 没有启动")


def test_same_lineage_in_process_subprocess_and_http(tmp_path):
    a = tmp_path / "inproc"
    _, log_a = run(write_config(a, {s: python_spec(*spec) for s, spec in SLOTS.items()}))

    b = tmp_path / "stdio"
    b.mkdir()
    _, log_b = run(write_config(b, {s: stdio_spec(*spec, b) for s, spec in SLOTS.items()}))

    c = tmp_path / "http"
    c.mkdir()
    procs, comps = [], {s: python_spec(*spec) for s, spec in SLOTS.items()}
    try:
        for slot in ("judge", "gate", "algorithm", "attribution"):
            proc, url = start_http(*SLOTS[slot], c)
            procs.append(proc)
            comps[slot] = {"url": url}
        _, log_c = run(write_config(c, comps))
    finally:
        for p in procs:
            p.terminate()
            p.wait(timeout=10)

    check_invariants(log_a)
    assert fingerprint(log_a) == fingerprint(log_b) == fingerprint(log_c)
    assert max(e.metrics["holdout"] for e in log_a) == 1.0


# ---------------------------------------------------------------- 3. 跨语言

def node_ready() -> bool:
    return shutil.which("node") is not None and (XLANG / "node_modules" / "@modelcontextprotocol" / "sdk").exists()


xlang = pytest.mark.skipif(not node_ready(), reason="需要 node 和 npm ci --prefix tests/xlang")


@xlang
def test_node_gate_passes_conformance_and_runs_in_loop(tmp_path):
    report = asyncio.run(check(["node", str(XLANG / "gate.mjs")]))
    assert report.ok, report.text()

    comps = {s: python_spec(*spec) for s, spec in SLOTS.items()}
    _, log_py = run(write_config(tmp_path / "py", comps))
    comps["gate"] = {"command": ["node", str(XLANG / "gate.mjs")]}
    _, log_js = run(write_config(tmp_path / "js", comps))
    assert fingerprint(log_js) == fingerprint(log_py)  # 和 Python 的 HoldoutGate 决定完全一样

    async def info():
        async with connect(["node", str(XLANG / "gate.mjs")]) as c:
            return c.info

    assert asyncio.run(info()).name == "example/node-gate"


@xlang
def test_node_client_uses_python_judge():
    out = subprocess.run(["node", str(XLANG / "client.mjs"), sys.executable, "-m", "openrsi.launch",
                          "openrsi_judge:EvidenceJudge", "--with", json.dumps({"rubric": EQ})],
                         capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONPATH": PYTHONPATH})
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["kinds"] == ["evidence", "judge"]
    assert result["right"]["passed"] is True and result["wrong"]["passed"] is False


# ---------------------------------------------------------------- 4. 故障与续跑

def test_crashed_subprocess_component_then_resume(tmp_path):
    flag = tmp_path / "crash-at"
    flag.write_text("2")  # 第 2 次调用 propose 时进程退出
    comps = {s: python_spec(*spec) for s, spec in SLOTS.items()}
    comps["proposer"] = {"command": [sys.executable, "-m", "openrsi.launch", "openrsi_inducer_fault:FlakyInducer",
                                     "--with", json.dumps({"flag": str(flag)}), "--path",
                                     str(ROOT / "examples" / "secret_ops")],
                         "env": {"PYTHONPATH": PYTHONPATH}}
    cfg = write_config(tmp_path, comps)
    with pytest.raises(ComponentUnavailableError, match="test/flaky-inducer.*rsi_proposer_propose"):
        run(cfg)

    lineage = tmp_path / "state" / "lineage.jsonl"
    before = [json.loads(x) for x in lineage.read_text().splitlines() if x.strip()]
    assert before and any(e["status"] == "root" for e in before)

    flag.write_text("0")  # 修好了
    _, log = run(cfg)
    check_invariants(log)
    assert sum(1 for e in log if e.status == "root") == 1  # 没有重新登记根版本
    assert [e.id for e in log][: len(before)] == [e["id"] for e in before]  # 旧记录原样保留
    assert max(e.metrics["holdout"] for e in log) == 1.0


# ---------------------------------------------------------------- 5. 向前兼容


def test_every_schema_accepts_unknown_fields():
    """新版本加的字段，旧组件收到时不能报错：所有数据格式都允许额外字段。"""
    for f in sorted((ROOT / "spec" / "schemas").glob("*.json")):
        schema = json.loads(f.read_text())
        assert schema.get("additionalProperties") is True, f.name


def test_component_without_optional_tools_still_works(tmp_path):
    """0.3 时代的算法（没有 screen、不回应提案器提示）在 0.4 运行时里照常工作。"""
    comps = {s: stdio_spec(*spec, tmp_path) for s, spec in SLOTS.items() if s in ("algorithm",)}
    comps.update({s: python_spec(*spec) for s, spec in SLOTS.items() if s not in comps})
    _, log = run(write_config(tmp_path, comps))
    check_invariants(log)
    assert max(e.metrics["holdout"] for e in log) == 1.0
