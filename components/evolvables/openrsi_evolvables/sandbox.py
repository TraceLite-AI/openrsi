"""在独立子进程里运行 skill 代码。

限制：独立的 Python 进程（-I，不读环境变量和用户 site-packages）、空环境变量、临时工作目录、
CPU 时间和内存上限、墙钟超时、结果必须能 JSON 序列化、stdout 截断。

这不是安全沙箱：子进程仍然能访问文件系统和网络。不可信的代码应当先过护栏（openrsi-guard），
并且整个组件跑在容器或虚拟机里。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from typing import Any

from openrsi import SkillResult

_RUNNER = r'''
import contextlib, io, json, sys
limits = json.loads(sys.argv[1])
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu"], limits["cpu"]))
    resource.setrlimit(resource.RLIMIT_AS, (limits["mem"], limits["mem"]))
except Exception:
    pass  # 不支持的平台（如 Windows）只靠超时
payload = json.loads(sys.stdin.read())
buf = io.StringIO()
out = {}
try:
    ns = {"__name__": "openrsi_skill"}
    with contextlib.redirect_stdout(buf):
        exec(compile(payload["code"], "<skill>", "exec"), ns)
        fn = ns.get(payload["entry"])
        if not callable(fn):
            raise NameError("skill 没有定义入口函数 " + payload["entry"])
        result = fn(**payload["arguments"])
    json.dumps(result)
    out = {"ok": True, "result": result}
except BaseException as e:
    out = {"ok": False, "error": type(e).__name__ + ": " + str(e)}
out["stdout"] = buf.getvalue()[-4000:]
sys.__stdout__.write(json.dumps(out))
'''


def run_skill(code: str, arguments: dict[str, Any] | None = None, *, entry: str = "run", timeout: float = 5.0,
              cpu_seconds: int = 5, memory_mb: int = 512) -> SkillResult:
    limits = json.dumps({"cpu": cpu_seconds, "mem": memory_mb * 1024 * 1024})
    payload = json.dumps({"code": code, "entry": entry, "arguments": arguments or {}})
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="openrsi-skill-") as cwd:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER, limits], input=payload, capture_output=True,
                                  text=True, timeout=timeout, env={}, cwd=cwd)
        except subprocess.TimeoutExpired:
            return SkillResult(ok=False, error=f"超时（{timeout}s）", duration_ms=(time.monotonic() - start) * 1000)
    ms = (time.monotonic() - start) * 1000
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # 被 CPU / 内存上限杀掉时没有输出
        if proc.returncode in (-9, -24):  # SIGKILL / SIGXCPU：超出 CPU 时间上限
            return SkillResult(ok=False, error=f"超出 CPU 时间上限（{cpu_seconds}s）", duration_ms=ms)
        err = (proc.stderr.strip().splitlines() or [f"进程退出码 {proc.returncode}"])[-1]
        return SkillResult(ok=False, error=f"skill 进程异常结束：{err}", duration_ms=ms)
    return SkillResult(ok=data["ok"], result=data.get("result"), error=data.get("error"),
                       stdout=data.get("stdout", ""), duration_ms=ms)
