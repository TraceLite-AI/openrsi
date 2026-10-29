"""本地调通流程用的假推理服务（OpenAI 兼容接口），不需要 GPU，也不下载模型。

"模型"是一个目录，里面的 skill.json 记着它答对的概率 level（目录不存在时按 0.4）。
- 系统提示里提到 "####"，答案就按 "#### 数字" 格式输出，否则只说 "The answer is N"（判别器抽不到答案）
- 收到改写系统提示的请求时，返回一个要求按 #### 输出的系统提示
fake_train.py 训练一次会把 level 提高 0.25。这样整条流程（prompt 进化、权重进化、换模型重启服务）都能在 CPU 上走一遍。
"""
import argparse
import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def level_of(model: str) -> float:
    f = Path(model) / "skill.json"
    return json.loads(f.read_text())["level"] if f.exists() else 0.4


def answer(question: str) -> int:
    nums = [int(x) for x in re.findall(r"-?\d+", question)]
    return sum(nums)


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
            user = next((m["content"] for m in req["messages"] if m["role"] == "user"), "")
            if system.startswith("You improve the system prompt"):
                text = "Solve the problem step by step. Put the final answer on the last line as '#### <number>'."
            else:
                ans = answer(user)
                knows = int(hashlib.sha256(user.encode()).hexdigest(), 16) % 100 < round(level * 100)
                ans = ans if knows else ans + 1
                text = f"Adding up the numbers.\n#### {ans}" if "####" in system else f"The answer is {ans}."
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
