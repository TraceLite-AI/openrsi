"""被进化的玩具 agent：乘法交给工具算，其余交给模型。"""
import json
import math
import os
import re
import urllib.request


def ask(text: str) -> str:
    req = urllib.request.Request(os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions",
                                 data=json.dumps({"model": "policy", "messages": [{"role": "user", "content": text}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


class ToyAgent:
    def solve(self, instruction: str):
        steps = [{"step_id": 1, "source": "user", "message": instruction}]
        if "product" in instruction:
            nums = [int(x) for x in re.findall(r"-?\d+", instruction)]
            answer = str(math.prod(nums))
            steps.append({"step_id": 2, "source": "agent", "message": answer,
                          "tool_calls": [{"tool_call_id": "c1", "function_name": "multiply", "arguments": {"numbers": nums}}],
                          "observation": {"results": [{"source_call_id": "c1", "content": answer}]}})
            return answer, steps
        answer = ask(instruction)
        steps.append({"step_id": 2, "source": "agent", "message": answer})
        return answer, steps
