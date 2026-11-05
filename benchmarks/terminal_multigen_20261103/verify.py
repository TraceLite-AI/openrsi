"""Recompute the published descriptive audit without model calls or external packages.

Integrity success does NOT mean the experiment measured method quality reliably.
Original rewards/statuses are kept; verifier startup flags are a separate audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def clean(receipt, task):
    value = receipt["tasks"][task]
    return value["status"] == "benchmark_complete" and not value["exception"]


def passed(receipt):
    return sum((v["reward"] or 0) >= 1 for v in receipt["tasks"].values())


def paired(a, b):
    keys = sorted(k for k in set(a["tasks"]) & set(b["tasks"]) if clean(a, k) and clean(b, k))
    return {
        "native_clean_pairs": len(keys),
        "changed": {k: [a["tasks"][k]["reward"], b["tasks"][k]["reward"]]
                    for k in keys if a["tasks"][k]["reward"] != b["tasks"][k]["reward"]},
    }


def analyze(root):
    data = read(root, "records.json")
    files = data["files"]
    receipts = {k.removeprefix("receipts/"): v for k, v in files.items() if k.startswith("receipts/")}
    audits = read(root, "evidence/verifier-audit.json")
    assert len(audits) == len({(x["receipt"], x["task"]) for x in audits})
    for audit in audits:
        recorded = receipts[audit["receipt"]]["tasks"][audit["task"]]
        assert (audit["reward"], audit["recorded_status"], audit["exception"]) == (
            recorded["reward"], recorded["status"], recorded["exception"])
        if audit["readonly_python_failure"]:
            assert any("Read-only file system" in s and "/opt/evolve/uv/python/" in s for s in audit["excerpt"])
    assert sum(len(r["tasks"]) for r in receipts.values()) == len(audits)
    splits = files["splits.json"]
    assert [len(splits[x]) for x in ("train", "gate", "sealed")] == [50, 19, 20]
    assert len(set(sum(splits.values(), []))) == 89

    root_runs = [(n, r) for n, r in receipts.items()
                 if n == "initial-gate.json" or n.endswith("-gate.json") or n.endswith("confirm-parent.json")]
    initial = {n: r for n, r in receipts.items() if "-candidate-" in n}
    stable_fail = [t for t in splits["gate"] if all(r["tasks"][t]["reward"] == 0 for _, r in root_runs)]
    rescued = [t for t in stable_fail if any(r["tasks"][t]["reward"] == 1 for r in initial.values())]
    variable = [t for t in splits["gate"] if len({r["tasks"][t]["reward"] for _, r in root_runs}) > 1]
    epochs = []
    for gen in range(1, 6):
        before = receipts["initial-gate.json" if gen == 1 else f"g{gen}-p0-gate.json"]
        confirm = receipts[f"g{gen}-p0-confirm-parent.json"]
        candidates = {n.split("-candidate-")[1].removesuffix(".json"): {"passed": passed(r), **paired(before, r)}
                      for n, r in initial.items() if n.startswith(f"g{gen}-")}
        checks = {n.split("-confirm-")[1].removesuffix(".json"): {"passed": passed(r), **paired(confirm, r)}
                  for n, r in receipts.items()
                  if n.startswith(f"g{gen}-p0-confirm-") and not n.endswith("parent.json")}
        epochs.append({"generation": gen, "parent_initial_passed": passed(before),
                       "parent_confirmation_passed": passed(confirm), "initial": candidates, "confirmation": checks})

    failure = [x for x in audits if x["readonly_python_failure"]]
    usage = [u for r in receipts.values() for u in r["usage"].values()]
    final = files["final-results.json"]["results"]
    native_parent = final["root"]
    assert all(final[k] == native_parent for k in ("whole", "best_atom", "merge"))
    for role in ("train", "gate", "sealed"):
        r = receipts[f"final-{role}-root.json"]
        assert set(r["tasks"]) == set(splits[role])
        assert passed(r) == native_parent[role]["passed"]
    assert sum(native_parent[k]["passed"] for k in ("train", "gate", "sealed")) / 89 == native_parent["all89"]

    activation = read(root, "evidence/prompt-activation.json")
    assert all(x["traces"] == x["instance_template_matches"] == x["rendered_user_contains_all_additions"]
               for x in activation)
    proof = read(root, "evidence/runtime-proof.json")
    assert all(proof["code_pin_match"].values())
    source_diff_hash = {}
    for sid in [x["genid"] for x in data["archive"] if x["genid"] != "0"]:
        patch = root / "patches" / f"{sid}.diff"
        assert patch.is_file()
        source_diff_hash[sid] = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert all(source_diff_hash[a["genid"]] == a["diff_sha256"] for a in activation)

    pool_sizes = [len(v["reasons"]) for k, v in data["pool"].items() if k.endswith("outcome.json")]
    lineages = files["frozen-winners.json"]["lineages"]
    return {
        "integrity": "passed",
        "valid_for_method_comparison": False,
        "reason": "Verifier startup failure silently recorded as ordinary zero rewards.",
        "generations": 5,
        "accepted": {arm: sum(x.get("accepted", False) for x in rows) for arm, rows in lineages.items()},
        "final_versions": files["frozen-winners.json"]["versions"],
        "candidate_versions": len(source_diff_hash),
        "changed_file_set": sorted({p for row in data["archive"] for p in row["mutated"]}),
        "prompt_activation_traces_checked": sum(x["traces"] for x in activation),
        "physical_trials": len(audits),
        "verifier_readonly_failures": len(failure),
        "readonly_marked_ordinary_zero": sum(x["recorded_status"] == "benchmark_complete" and x["reward"] == 0
                                           for x in failure),
        "readonly_with_pytest_started": sum(x["pytest_started"] for x in failure),
        "readonly_by_phase": dict(Counter("final" if x["receipt"].startswith("final")
                                         else "train" if x["receipt"].endswith("train.json") else "gate"
                                         for x in failure)),
        "readonly_final_by_split": {split: sum(x["receipt"] == f"final-{split}-root.json" for x in failure)
                                   for split in ("train", "gate", "sealed")},
        "native_exception_codes": dict(Counter(x["exception"] for x in audits if x["exception"])),
        "selection_parent_runs": len(root_runs),
        "selection_parent_passed": [passed(r) for _, r in root_runs],
        "stable_failed_gate_tasks": stable_fail,
        "stable_failed_gate_tasks_rescued": rescued,
        "varying_gate_tasks": {t: {
            "parent_successes": sum(r["tasks"][t]["reward"] == 1 for _, r in root_runs),
            "parent_attempts": len(root_runs),
            "initial_candidate_successes": sum(r["tasks"][t]["reward"] == 1 for r in initial.values()),
            "initial_candidate_attempts": len(initial),
        } for t in variable},
        "generated_unique_atoms_per_generation": pool_sizes,
        "explicitly_scored_singletons": 10,
        "epochs": epochs,
        "raw_final": native_parent,
        "native_recorded_model_calls": sum(u["api_calls"] for u in usage),
        "native_estimated_target_usd": round(sum(u["estimated_usd"] for u in usage), 8),
        "estimated_target_usd_on_readonly_trials": round(sum(
            receipts[x["receipt"]]["usage"][x["task"]]["estimated_usd"] or 0 for x in failure), 8),
        "proposer_http_attempts": sum(v["search_used"] for k, v in data["pool"].items() if k.endswith("budget.json")),
        "proposer_reported_tokens": sum(v["reported_tokens"]["total_tokens"]
                                       for k, v in data["pool"].items() if k.endswith("budget.json")),
    }


def verify_manifest(root):
    for name, expected in read(root, "MANIFEST.json")["exported_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-summary", action="store_true", help="Regenerate derived audit-summary.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    result = analyze(root)
    if args.write_summary:
        (root / "audit-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    else:
        verify_manifest(root)
        assert read(root, "audit-summary.json") == result
        print(json.dumps({k: result[k] for k in ("integrity", "valid_for_method_comparison",
                                               "physical_trials", "verifier_readonly_failures",
                                               "candidate_versions", "accepted")}, ensure_ascii=False, indent=2))
