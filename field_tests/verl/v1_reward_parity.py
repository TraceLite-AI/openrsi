"""V1（CPU）：OpenRSI 判别器奖励和 verl 自带 gsm8k 奖励逐条对比，并用 verl 自己的加载函数加载我们的奖励。

用法：python field_tests/verl/v1_reward_parity.py --data ~/data/gsm8k/test.parquet [--n 500]
数据用 verl 仓库的 examples/data_preprocess/gsm8k.py 生成。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "out" / "verl"


def cases(gt: str):
    yield "correct", f"Let's compute. ... so the answer is {gt}.\n#### {gt}"
    yield "wrong", f"Step by step ...\n#### {int(float(gt)) + 1 if gt.lstrip('-').replace('.', '').isdigit() else gt + '0'}"
    yield "no_marker", f"The answer is {gt}."
    yield "two_markers", f"#### 0\nActually, correcting myself:\n#### {gt}"
    if gt.isdigit() and len(gt) >= 4:
        yield "commas", f"#### {int(gt):,}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    import gsm8k_reward

    try:
        from verl.utils.reward_score import gsm8k as verl_gsm8k
        report["verl_version"] = __import__("verl").__version__
    except Exception as e:  # noqa: BLE001
        verl_gsm8k = None
        report["verl_import_error"] = repr(e)

    # 1. 用 verl 自己的加载逻辑加载我们的奖励函数
    try:
        from omegaconf import OmegaConf
        from verl.trainer.ppo.reward import get_custom_reward_fn

        cfg = OmegaConf.create({"custom_reward_function": {"path": str(HERE / "gsm8k_reward.py"),
                                                           "name": "compute_score"}})
        loaded = get_custom_reward_fn(cfg)
        report["verl_loader"] = "ok" if loaded is not None else "returned None"
    except Exception as e:  # noqa: BLE001
        loaded = None
        report["verl_loader"] = f"error: {e!r}"

    # 2. 逐条对比
    df = pd.read_parquet(args.data).head(args.n)
    rows, mismatches, elapsed = 0, [], 0.0
    by_case: dict[str, list[int]] = {}
    for _, r in df.iterrows():
        gt = str(r["reward_model"]["ground_truth"])
        extra = dict(r["extra_info"]) if isinstance(r.get("extra_info"), dict) else {}
        for name, sol in cases(gt):
            t0 = time.perf_counter()
            ours = gsm8k_reward.compute_score(r["data_source"], sol, gt, extra)
            elapsed += time.perf_counter() - t0
            if loaded is not None:
                assert loaded(r["data_source"], sol, gt, extra) == ours
            ref = verl_gsm8k.compute_score(sol, gt) if verl_gsm8k else None
            rows += 1
            stat = by_case.setdefault(name, [0, 0])
            stat[0] += 1
            if ref is not None and float(ref) != float(ours):
                stat[1] += 1
                if len(mismatches) < 20:
                    mismatches.append({"case": name, "ground_truth": gt, "solution": sol, "ours": ours, "verl": ref})
    report.update({"rows": rows, "by_case": {k: {"n": n, "mismatch": m} for k, (n, m) in by_case.items()},
                   "mismatches": mismatches, "ms_per_call": round(1000 * elapsed / max(rows, 1), 3)})
    (OUT / "v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = report["verl_loader"] == "ok" and verl_gsm8k is not None and not mismatches
    print("PASS V1" if ok else "FAIL V1（把 field_tests/out/verl/v1.json 发回来）")


if __name__ == "__main__":
    main()
