"""harness 的入口：从 stdin 读题，用 strategy.solve 解题，把轨迹写到 stdout。

HarnessAgent 在仓库根目录用 `python -m agent.main` 启动它。
"""
import json
import sys

from agent.strategy import solve


def main() -> None:
    data = json.load(sys.stdin)
    x = data["task"]["input"]
    value = solve(x["sym"], x["a"], x["b"])
    print(json.dumps({
        "output": value,
        "steps": [
            {"id": "s0", "type": "tool_call", "name": "solve", "call_id": "c",
             "arguments": {"sym": x["sym"], "a": x["a"], "b": x["b"]}},
            {"id": "s1", "type": "tool_result", "name": "solve", "call_id": "c", "content": str(value)},
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
