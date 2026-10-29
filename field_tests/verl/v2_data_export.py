"""V2（CPU）：用 OpenRSI 导出 verl 的 RL 数据，检查格式能被 verl 的数据集类读进去，并检查防泄漏。

把 verl 的 gsm8k parquet 转成 OpenRSI Task（train / test 分别标成 train / holdout），再用 tasks_to_rl_rows 导出。
导出的 train.parquet / test.parquet 给 V3 的 GRPO 运行用。

用法：python field_tests/verl/v2_data_export.py --src ~/data/gsm8k --tokenizer Qwen/Qwen2.5-0.5B-Instruct
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from openrsi import Task
from openrsi_verl import DataLeakError, tasks_to_rl_rows, write_rows

OUT = Path(__file__).resolve().parent.parent / "out" / "verl"


def to_tasks(df: pd.DataFrame, split: str) -> list[Task]:
    tasks = []
    for i, r in df.iterrows():
        prompt = r["prompt"][0]["content"] if len(r["prompt"]) else ""
        tasks.append(Task(id=f"{split}-{i}", input=prompt, split=split,
                          reference={"expected": str(r["reward_model"]["ground_truth"])}))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="verl gsm8k 预处理输出目录（有 train.parquet、test.parquet）")
    ap.add_argument("--tokenizer", help="可选：用 verl 的 RLHFDataset 读一遍导出的数据")
    ap.add_argument("--n-test", type=int, default=500)
    args = ap.parse_args()
    src = Path(args.src).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    train_src, test_src = pd.read_parquet(src / "train.parquet"), pd.read_parquet(src / "test.parquet")
    train, test = to_tasks(train_src, "train"), to_tasks(test_src.head(args.n_test), "holdout")

    # 防泄漏：holdout 不能导出成训练数据
    try:
        tasks_to_rl_rows(test[:1], "openai/gsm8k")
        report["leak_guard"] = "FAIL: holdout 被导出了"
    except DataLeakError:
        report["leak_guard"] = "ok"

    rows = tasks_to_rl_rows(train, "openai/gsm8k", ability="math")
    val_rows = tasks_to_rl_rows(test, "openai/gsm8k", ability="math", for_validation=True)
    try:
        tasks_to_rl_rows(train[:1], "openai/gsm8k", for_validation=True)
        report["val_guard"] = "FAIL: train 题进了验证集"
    except DataLeakError:
        report["val_guard"] = "ok"
    write_rows(rows, OUT / "data" / "train.parquet")
    write_rows(val_rows, OUT / "data" / "test.parquet")
    ours = pd.read_parquet(OUT / "data" / "train.parquet")
    report["columns_ours"] = sorted(ours.columns)
    report["columns_verl"] = sorted(train_src.columns)
    report["missing_vs_verl"] = sorted(set(train_src.columns) - set(ours.columns))
    report["ground_truth_same"] = bool((ours["reward_model"].map(lambda x: x["ground_truth"]).values
                                        == train_src["reward_model"].map(lambda x: str(x["ground_truth"])).values).all())
    report["prompt_same"] = bool((ours["prompt"].map(lambda x: x[0]["content"]).values
                                  == train_src["prompt"].map(lambda x: x[0]["content"]).values).all())
    report["reference_not_in_prompt"] = all("expected" not in json.dumps(list(p), default=str) for p in ours["prompt"])

    if args.tokenizer:
        try:
            from omegaconf import OmegaConf
            from transformers import AutoTokenizer
            from verl.utils.dataset.rl_dataset import RLHFDataset

            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            cfg = OmegaConf.create({"prompt_key": "prompt", "max_prompt_length": 512, "truncation": "error",
                                    "filter_overlong_prompts": False, "return_raw_chat": False})
            ds = RLHFDataset(data_files=str(OUT / "data" / "train.parquet"), tokenizer=tok, config=cfg)
            item = ds[0]
            report["rlhf_dataset"] = {"len": len(ds), "keys": sorted(item.keys())}
        except Exception as e:  # noqa: BLE001
            report["rlhf_dataset"] = f"error: {e!r}"

    (OUT / "v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    ok = (report["leak_guard"] == "ok" and report["val_guard"] == "ok" and not report["missing_vs_verl"]
          and report["ground_truth_same"]
          and report["prompt_same"] and not str(report.get("rlhf_dataset", "")).startswith("error"))
    print(f"数据写到 {OUT / 'data'}")
    print("PASS V2" if ok else "FAIL V2（把 field_tests/out/verl/v2.json 发回来）")


if __name__ == "__main__":
    main()
