"""用 transformers 起一个最小的 OpenAI 兼容推理服务（没有 vLLM 时用，比如 CPU 上调通流程）。

  python hf_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8100

只实现 /v1/models 和 /v1/chat/completions（贪心或按 temperature 采样），请求串行处理。速度远不如 vLLM，
只适合小模型、少量题目。
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--served-model-name", default="policy")
    ap.add_argument("--threads", type=int, default=0, help="torch 线程数，0 表示默认")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.threads:
        torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"data": [{"id": args.served_model_name, "root": args.model}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(device)
            temp = float(req.get("temperature") or 0.0)
            kw = {"do_sample": True, "temperature": temp, "top_p": 0.95} if temp > 0 else {"do_sample": False}
            t0 = time.time()
            with lock, torch.no_grad():
                out = model.generate(**ids, max_new_tokens=int(req.get("max_tokens") or 256),
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
            new = out[0][ids["input_ids"].shape[1]:]
            content = tok.decode(new, skip_special_tokens=True)
            finish = "length" if len(new) >= int(req.get("max_tokens") or 256) else "stop"
            self._send({"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
                        "usage": {"prompt_tokens": ids["input_ids"].shape[1], "completion_tokens": len(new),
                                  "seconds": round(time.time() - t0, 2)}})

    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
