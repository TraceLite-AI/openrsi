"""接真实 LLM 的组件：任何 OpenAI 兼容接口（vLLM、SGLang、Ollama、各家 API）。

- ChatAgent      执行体。prompt 可进化体的段落拼成系统提示，memory 条目作为参考附在后面；
                 上下文里有权重条目（kind=weights）时，按其中的模型路径启动或重启推理服务。
- PromptRewriter 提案器。把答错的 train 题交给 LLM，让它改写系统提示，生成 prompt 可进化体的补丁。

只用标准库发 HTTP 请求，不依赖 openai 包。
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openrsi import Agent, ComponentError, Item, Patch, ProposalRequest, Proposer, Step, Task, Trajectory

__version__ = "0.4.0"


def chat(base_url: str, model: str, messages: list[dict[str, str]], *, temperature: float = 0.0,
         max_tokens: int = 512, timeout: float = 300.0, api_key: str | None = None,
         extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """调用 /chat/completions，返回 {"text", "usage", "finish_reason"}。"""
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            **(extra or {})}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise ComponentError(f"LLM 接口返回 {e.code}：{e.read()[:500]!r}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ComponentError(f"连不上 LLM 接口 {base_url}：{e}") from e
    choice = data["choices"][0]
    return {"text": choice["message"].get("content") or "", "usage": data.get("usage", {}),
            "finish_reason": choice.get("finish_reason")}


class ManagedServer:
    """按模型路径启动推理服务（比如 vLLM），模型变了就重启。

    command 是命令模板，可用 {model}、{port}，例如
      [python, -m, vllm.entrypoints.openai.api_server, --model, "{model}", --served-model-name, policy,
       --port, "{port}", --gpu-memory-utilization, "0.35"]
    """

    def __init__(self, command: list[str], port: int, startup_timeout: float = 600.0, log_dir: str | None = None):
        self.command, self.port, self.startup_timeout = list(command), port, startup_timeout
        self.cwd = Path.cwd()  # 命令里的相对路径以构造时的目录（openrsi.yaml 所在目录）为准
        self.log_dir = Path(log_dir).resolve() if log_dir else None
        self.proc: subprocess.Popen | None = None
        self.model: str | None = None
        self.restarts = 0
        atexit.register(self.stop)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def ensure(self, model: str) -> None:
        if self.proc is not None and self.proc.poll() is None and self.model == model:
            return
        self.stop()
        argv = [part.format(model=model, port=self.port) for part in self.command]
        if argv and argv[0] in ("python", "python3"):  # 和当前运行时用同一个解释器
            argv[0] = sys.executable
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / f"server-{self.restarts}.log", "w") if self.log_dir else subprocess.DEVNULL  # noqa: SIM115
        self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=self.cwd)
        self.model, self.restarts = model, self.restarts + 1
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise ComponentError(f"推理服务启动失败（退出码 {self.proc.returncode}）：{' '.join(argv)}")
            try:
                with urllib.request.urlopen(self.base_url + "/models", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        self.stop()
        raise ComponentError(f"推理服务 {self.startup_timeout}s 内没有就绪：{' '.join(argv)}")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None


def render_prompt(items: list[Item]) -> str:
    prompt = sorted((i for i in items if i.kind == "prompt"), key=lambda i: (i.data.get("order", 0), i.id))
    return "\n\n".join(i.text for i in prompt)


def task_text(task: Task) -> str:
    x = task.input
    if isinstance(x, dict):
        return str(x.get("question") or x.get("prompt") or json.dumps(x, ensure_ascii=False))
    return str(x)


class ChatAgent(Agent):
    """with:
    base_url / model    OpenAI 兼容接口；配置了 server 时 base_url 可省略
    server              可选，{command: [...], port: 8100, startup_timeout: 600, log_dir: ...}
    default_system      没有 prompt 段落时用的系统提示
    temperature / max_tokens / timeout / api_key_env
    """

    name = "openrsi/chat-agent"
    version = __version__

    def __init__(self, base_url: str | None = None, model: str = "policy", server: dict[str, Any] | None = None,
                 default_system: str = "You are a helpful assistant.", temperature: float = 0.0,
                 max_tokens: int = 512, timeout: float = 300.0, api_key_env: str | None = None,
                 memory_header: str = "Notes that may help:"):
        self.server = ManagedServer(**server) if server else None
        if not base_url and not self.server:
            raise ValueError("ChatAgent 需要 base_url 或 server")
        self.base_url = base_url or self.server.base_url  # type: ignore[union-attr]
        self.model, self.default_system = model, default_system
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.memory_header = memory_header

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        weights = next((i for i in context if i.kind == "weights"), None)
        if weights is not None and self.server is not None:
            self.server.ensure(weights.text)
        system = render_prompt(context) or self.default_system
        notes = [i for i in context if i.kind not in ("prompt", "weights")]
        if notes:
            system += "\n\n" + self.memory_header + "\n" + "\n".join(f"- {i.text}" for i in notes)
        question = task_text(task)
        start = time.monotonic()
        try:
            out = chat(self.base_url, self.model, [{"role": "system", "content": system},
                                                   {"role": "user", "content": question}],
                       temperature=self.temperature, max_tokens=self.max_tokens, timeout=self.timeout,
                       api_key=self.api_key)
        except ComponentError as e:
            return Trajectory(id=f"traj-{task.id}-{snapshot_id}-error", task_id=task.id, snapshot_id=snapshot_id,
                              output=None, steps=[Step(id="s0", type="observation", content=str(e), is_error=True)],
                              metadata={"error": str(e)})
        meta = {"duration_s": round(time.monotonic() - start, 3), "usage": out["usage"],
                "finish_reason": out["finish_reason"], "model": weights.text if weights else self.model}
        return Trajectory(id=f"traj-{task.id}-{snapshot_id}", task_id=task.id, snapshot_id=snapshot_id,
                          output=out["text"], used_items=[i.id for i in context if i.kind != "weights"],
                          steps=[Step(id="s0", type="message", role="system", content=system),
                                 Step(id="s1", type="message", role="user", content=question),
                                 Step(id="s2", type="message", role="assistant", content=out["text"])],
                          metadata=meta)


REWRITE_SYSTEM = (
    "You improve the system prompt of an AI assistant. You will see the current system prompt and some "
    "problems the assistant got wrong, with its answers and why they were judged wrong. Write a better system "
    "prompt that fixes these failures in general, without mentioning specific problems or answers. "
    "Reply with the new system prompt only.")


class PromptRewriter(Proposer):
    """with: base_url、model（可以就是被进化的那个小模型，也可以换成更强的）、item_id（默认 instructions）、
    max_examples（给 LLM 看几道错题）、max_chars、temperature、api_key_env"""

    name = "openrsi/prompt-rewriter"
    version = __version__

    def __init__(self, base_url: str, model: str = "policy", item_id: str = "instructions", max_examples: int = 6,
                 max_chars: int = 2000, temperature: float = 0.7, max_tokens: int = 600, timeout: float = 300.0,
                 api_key_env: str | None = None):
        self.base_url, self.model, self.item_id = base_url, model, item_id
        self.max_examples, self.max_chars = max_examples, max_chars
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        self.api_key = os.environ.get(api_key_env) if api_key_env else None

    def propose(self, request: ProposalRequest) -> list[Patch]:
        target = next((t for t in (request.targets or [request.target]) if t.type == "prompt"), None)
        if target is None:
            return []
        state = request.states.get(target.target, request.state) if request.states else request.state
        items = {d["id"]: Item.model_validate(d) for d in state or [] if isinstance(d, dict) and "id" in d}
        current = render_prompt(list(items.values()))
        failed = [ep for ep in request.episodes if not ep.verdict.passed][: self.max_examples]
        if not failed:
            return []
        examples = []
        for ep in failed:
            why = "; ".join(r.note or r.id for r in ep.verdict.rubric if r.result == "fail") or "judged wrong"
            answer = str(ep.trajectory.output or "")[-800:]
            examples.append(f"Problem: {task_text(ep.task)}\nAssistant answer: {answer}\nWhy wrong: {why}")
        user = f"Current system prompt:\n{current or '(empty)'}\n\nFailures:\n\n" + "\n\n---\n\n".join(examples)
        try:
            out = chat(self.base_url, self.model, [{"role": "system", "content": REWRITE_SYSTEM},
                                                   {"role": "user", "content": user}],
                       temperature=self.temperature, max_tokens=self.max_tokens, timeout=self.timeout,
                       api_key=self.api_key)
        except ComponentError:
            return []
        text = out["text"].strip().strip("`").strip()
        if not text or len(text) > self.max_chars or text == current:
            return []
        op = "replace" if self.item_id in items else "add"
        item = Item(id=self.item_id, kind="prompt", text=text, data={"order": 0})
        return [Patch(target=target.target, base_snapshot=request.parent_snapshot,
                      ops=[{"op": op, "item": item.model_dump(mode="json")}], proposer_ref=self.ref,
                      rationale=f"rewrite system prompt from {len(failed)} train failures")]


__all__ = ["ChatAgent", "ManagedServer", "PromptRewriter", "chat", "render_prompt", "task_text"]
