"""生成本地调通用的玩具数据（加法题），格式和 GSM8KEnv 读的 jsonl 一样。"""
import json
import random
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "toy_data")
out.mkdir(parents=True, exist_ok=True)
for name, n, seed in (("train", 300, 0), ("test", 160, 1)):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        rows.append({"question": f"Tom has {a} apples and buys {b} more. How many apples does he have?",
                     "answer": f"{a} + {b} = {a + b}\n#### {a + b}"})
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(out)
