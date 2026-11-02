"""被进化的玩具 agent：把题目原样交给模型，模型说什么就答什么。"""
import json
import os
import urllib.request


def ask(text: str) -> str:
    req = urllib.request.Request(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps({"model": "policy", "messages": [{"role": "user", "content": text}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


class ToyAgent:
    def solve(self, instruction: str):
        answer = ask(instruction)
        steps = [{"step_id": 1, "source": "user", "message": instruction},
                 {"step_id": 2, "source": "agent", "message": answer}]
        return answer, steps
