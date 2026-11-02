"""生成一个 Harbor 格式的玩具题库（每道题一个带 task.toml 的目录），给 openrsi.mock.yaml 用。

两种题：
  sum      "Compute the sum of: 3, 9, 4"     模型自己算，答对的概率取决于权重（mock_llm.py 里的 level）
  product  "Compute the product of: 3, 9"    模型永远算错；agent 代码里加上乘法工具才能做对（要改代码）

所以要两样都改：改代码解决 product，训权重提高 sum 的正确率。

    python make_toy_dataset.py toy_tasks [--n 60]
"""
import argparse
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    rng = random.Random(0)
    out = Path(args.out)
    seen = set()
    for i in range(args.n):
        kind = "product" if i % 3 == 0 else "sum"
        nums = [rng.randint(2, 19) for _ in range(2 if kind == "product" else 3)]
        while (kind, tuple(nums)) in seen:  # 题目不重复，否则同一道题可能同时出现在练习和考试里
            nums = [rng.randint(2, 19) for _ in range(len(nums))]
        seen.add((kind, tuple(nums)))
        answer = 1 if kind == "product" else 0
        for x in nums:
            answer = answer * x if kind == "product" else answer + x
        d = out / f"toy-{i:03d}"
        (d / "tests").mkdir(parents=True, exist_ok=True)
        (d / "task.toml").write_text(f'version = "1.0"\n\n[metadata]\nkind = "{kind}"\n', encoding="utf-8")
        (d / "instruction.md").write_text(f"Compute the {kind} of: {', '.join(map(str, nums))}\n", encoding="utf-8")
        (d / "tests" / "expected.txt").write_text(f"{answer}\n", encoding="utf-8")
    print(f"{args.n} tasks → {out}")


if __name__ == "__main__":
    main()
