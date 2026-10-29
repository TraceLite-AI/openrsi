"""OpenRSI 参考运行时：按 openrsi.yaml 接通组件，跑自我改进循环。组件可以在进程内，也可以是任何 MCP server。"""
from .config import ComponentSpec, LoopConfig, RunConfig, load_config
from .loop import Candidate, GenerationReport, IsolationError, Runner
from .wiring import LocalProxy, Wiring

__all__ = ["Candidate", "ComponentSpec", "GenerationReport", "IsolationError", "LocalProxy", "LoopConfig",
           "RunConfig", "Runner", "Wiring", "load_config"]
