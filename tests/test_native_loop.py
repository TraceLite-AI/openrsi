"""P3：原生循环 A/B/C 对照实验（benchmarks/native_loop）。"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "native_loop"))

import native_loop as nl  # noqa: E402

BUDGET = 400


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    out = tmp_path_factory.mktemp("native")
    return out, asyncio.run(nl.experiment(out, [0], BUDGET))


def lineage(out: Path, group: str) -> list[dict]:
    path = out / "seed0" / group / "state" / "lineage.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def test_report_is_labelled_and_complete(report):
    out, rep = report
    assert rep["label"] == nl.SYNTHETIC and "synthetic smoke" in (out / "report.md").read_text(encoding="utf-8")
    groups = {r["group"]: r for r in rep["runs"]}
    assert set(groups) == {"A", "B", "C"}
    for r in groups.values():
        assert r["search_agent_calls"] <= BUDGET  # 硬上限，不超
        assert len(r["final_tasks"]) == 40 and r["sealed_agent_calls"] == 40  # 封存考试相同、另计
    assert {p["pair"] for p in rep["paired"]} == {"B-A", "C-A", "C-B"}


def test_group_a_does_not_search(report):
    out, rep = report
    a = next(r for r in rep["runs"] if r["group"] == "A")
    assert a["generations"] == 0 and a["accepted"] == a["rejected"] == 0
    assert [e["status"] for e in lineage(out, "A")] == ["root"]


def test_group_b_always_starts_from_root(report):
    out, _ = report
    log = lineage(out, "B")
    root = next(e["id"] for e in log if e["status"] == "root")
    children = [e for e in log if e["status"] != "root"]
    assert children and all(e["parent_ids"] == [root] for e in children)


def test_group_c_inherits_accepted_improvements(report):
    out, rep = report
    log = lineage(out, "C")
    root = next(e["id"] for e in log if e["status"] == "root")
    accepted = {e["id"] for e in log if e["status"] == "accepted"}
    assert any(e["parent_ids"] and e["parent_ids"][0] in accepted for e in log if e["status"] != "root")
    c = next(r for r in rep["runs"] if r["group"] == "C")
    assert c["locked_version"] != root and c["final_pass"] > 0


def test_budget_runs_out_mid_generation_and_is_reported(tmp_path):
    nl.SealedEnv.requested = []
    r = asyncio.run(nl.run_group("C", tmp_path, 0, 60))  # 根版本 40 次 + 一代的一部分
    assert r["incomplete_generation"] and r["search_agent_calls"] == 60 and "预算" in r["stopped"]


def test_sealed_split_requested_during_search_invalidates_the_run(tmp_path):
    nl.SealedEnv.requested = ["final"]
    with pytest.raises(RuntimeError, match="封存考试"):
        asyncio.run(nl.run_group("A", tmp_path, 0, BUDGET))
    nl.SealedEnv.requested = []


def test_paired_counts_fixed_and_broken():
    assert nl.paired({"a": 0, "b": 1, "c": 0}, {"a": 1, "b": 0, "c": 1}) == {"fixed": 2, "broke": 1,
                                                                               "mean_diff": 0.3333}


def test_same_seed_same_result(tmp_path, report):
    _, rep = report
    again = asyncio.run(nl.experiment(tmp_path, [0], BUDGET))

    def strip(r):
        return [{k: v for k, v in x.items() if k != "seconds"} for x in r["runs"]]

    assert strip(again) == strip(rep)
