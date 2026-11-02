"""本地调通用的假推理服务（OpenAI 兼容接口），不需要 GPU。

"模型"是一个目录，里面的 skill.json 记着 level（目录不存在时按 0.3）：
- 做 sum 题：每道题有一个固定难度 q（题目哈希），q < level 就答对。结果固定，重复评估分数不会乱跳
- 做 product 题：总是答错（把乘法当加法）。只有 agent 代码自己会乘法才能做对
- 收到改代码的请求（CodeRewriter）：返回一份带乘法工具的 agent/toy_agent.py

fake_train.py 训练一次把 level 提高 0.3。
"""
import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IMPROVED = (Path(__file__).parent / "improved_agent.py").read_text(encoding="utf-8")


def level_of(model: str) -> float:
    f = Path(model) / "skill.json"
    return json.loads(f.read_text())["level"] if f.exists() else 0.3


def make_handler(model: str):
    level = level_of(model)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": "policy", "root": model}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = next((m["content"] for m in req["messages"] if m["role"] == "system"), "")
            user = req["messages"][-1]["content"]
            if system.startswith("You improve the source code"):
                text = json.dumps({"rationale": "add a multiplication tool for product tasks",
                                   "files": [{"path": "agent/toy_agent.py", "content": IMPROVED}]})
            else:
                nums = [int(x) for x in re.findall(r"-?\d+", user)]
                q = int(hashlib.sha256(user.encode()).hexdigest(), 16) % 1000 / 1000
                right = q < level
                total = sum(nums)
                text = str(total if "sum" in user and right else total + 1)
            self._send({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": len(system + user) // 4, "completion_tokens": len(text) // 4}})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.model)).serve_forever()


if __name__ == "__main__":
    main()
