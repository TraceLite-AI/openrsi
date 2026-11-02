"""本地调通用的假训练：读 KTO 数据（答对的标 true、答错的标 false），产出一个 level 提高 0.3 的"模型"目录（见 mock_llm.py）。"""
import argparse
import json
from pathlib import Path

from mock_llm import level_of

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()
rows = [json.loads(x) for x in Path(args.data).read_text().split("\n") if x.strip()]
assert rows and all({"messages", "label"} <= set(r) for r in rows), "KTO 数据格式不对"
assert {r["label"] for r in rows} == {True, False}, "KTO 数据要同时有答对和答错的样本"
out = Path(args.output)
out.mkdir(parents=True)
level = min(1.0, level_of(args.model) + 0.3)
info = {"level": level, "base": args.model, "rows": len(rows), "wrong": sum(not r["label"] for r in rows)}
(out / "skill.json").write_text(json.dumps(info))
print(f"trained on {len(rows)} KTO rows, level {level}")
