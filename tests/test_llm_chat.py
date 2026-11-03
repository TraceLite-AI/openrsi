"""anyrsi_llm.chat：遇到"参数不支持"的 400 时改写参数重试，并按模型记住。"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import anyrsi_llm
from anyrsi import ComponentError
from anyrsi_llm import chat


def serve(reject):
    """reject(body) -> 错误信息或 None。记录每次收到的请求体。"""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            err = reject(body)
            if err:
                data, code = json.dumps({"error": {"message": err}}).encode(), 400
            else:
                data, code = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {}}).encode(), 200
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", seen


def reasoning_model(body):
    if "max_tokens" in body:
        return "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."
    if "temperature" in body and body["temperature"] != 1:
        return "Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported."
    return None


def test_rewrites_unsupported_params_and_remembers_per_model():
    anyrsi_llm._QUIRKS.clear()
    srv, url, seen = serve(reasoning_model)
    try:
        out = chat(url, "m", [{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=50, extra={"seed": 3})
        assert out["text"] == "ok"
        last = seen[-1]
        assert last["max_completion_tokens"] == 50 and "max_tokens" not in last and "temperature" not in last
        assert last["seed"] == 3
        n = len(seen)
        chat(url, "m", [{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=50)
        assert len(seen) == n + 1  # 记住了，第二次一次就成功
        chat(url, "other", [{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=50)
        assert len(seen) == n + 4  # 别的模型不沿用
    finally:
        srv.shutdown()


def test_normal_models_get_params_unchanged_and_other_errors_raise():
    anyrsi_llm._QUIRKS.clear()
    srv, url, seen = serve(lambda b: None)
    try:
        chat(url, "m", [{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=9)
        assert seen[-1]["max_tokens"] == 9 and seen[-1]["temperature"] == 0.2
        chat(url, "m", [{"role": "user", "content": "hi"}], temperature=None, max_tokens=9)
        assert "temperature" not in seen[-1]
    finally:
        srv.shutdown()
    srv, url, seen = serve(lambda b: "context length exceeded")
    try:
        with pytest.raises(ComponentError, match="400"):
            chat(url, "m", [{"role": "user", "content": "hi"}])
        assert len(seen) == 1
    finally:
        srv.shutdown()


def test_transient_network_errors_are_retried(monkeypatch):
    import urllib.request

    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()

    def flaky(req, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionResetError(104, "Connection reset by peer")
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(anyrsi_llm.time, "sleep", lambda s: None)
    assert chat("http://x/v1", "m", [{"role": "user", "content": "hi"}])["text"] == "ok" and len(calls) == 3

    calls.clear()

    def down(req, timeout=None):
        calls.append(1)
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    with pytest.raises(ComponentError, match="连不上"):
        chat("http://x/v1", "m", [{"role": "user", "content": "hi"}])
    assert len(calls) == 3


def test_quota_and_auth_errors_stop_the_run():
    from anyrsi_llm import ChatAgent, LLMFatalError, PatchProposer

    anyrsi_llm._QUIRKS.clear()
    srv, url, seen = serve(lambda b: "You have no credits remaining. Add credits to continue using the API")
    try:
        with pytest.raises(LLMFatalError):
            chat(url, "m", [{"role": "user", "content": "hi"}])
        from anyrsi import EvolvableInfo, ProposalRequest, Task
        with pytest.raises(LLMFatalError):  # 执行体和提案器都不吞掉它
            ChatAgent(base_url=url, model="m").run(Task(id="t", split="train", input={"question": "q"}), [])
        info = EvolvableInfo(target="prompt", type="prompt", head="h")
        with pytest.raises(LLMFatalError):
            PatchProposer(base_url=url, model="m").propose(ProposalRequest(target=info, parent_snapshot="p", state=[]))
    finally:
        srv.shutdown()
