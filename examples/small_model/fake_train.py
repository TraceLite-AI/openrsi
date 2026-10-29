"""本地调通流程用的假训练：读 SFT 数据，产出一个 level 提高 0.25 的"模型"目录（见 mock_llm.py）。"""
import argparse
import json
from pathlib import Path

from mock_llm import level_of

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()
rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
assert rows and all(r["messages"][-1]["role"] == "assistant" for r in rows), "SFT 数据格式不对"
assert all(r["messages"][0]["role"] == "system" for r in rows), "SFT 数据里缺系统提示"
out = Path(args.output)
out.mkdir(parents=True)
level = min(1.0, level_of(args.model) + 0.25)
(out / "skill.json").write_text(json.dumps({"level": level, "base": args.model, "examples": len(rows)}))
print(f"trained on {len(rows)} examples, level {level}")
