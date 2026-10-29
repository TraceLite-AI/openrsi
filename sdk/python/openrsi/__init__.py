"""OpenRSI Python SDK：协议数据格式、组件基类、MCP 服务端与客户端、一致性测试。"""
from .client import RemoteComponent, ToolCallError, connect
from .components import Component, ComponentError, EvidenceCollector, Evolvable, Judge, Memory, StaleSnapshotError
from .convert import from_messages
from .manifest import Manifest, load_manifest
from .server import build_server, serve
from .types import (
    PROTOCOL_VERSION, AnchorExample, Attribution, CalibrationReport, ComponentInfo, Diff, Evidence,
    EvolvableInfo, GateDecision, Item, MemoryOp, Patch, RubricResult, Snapshot, Step, Task, Trajectory,
    Verdict, VerifyResult, content_hash,
)

__version__ = "0.1.0"

__all__ = [
    "PROTOCOL_VERSION", "AnchorExample", "Attribution", "CalibrationReport", "Component", "ComponentError", "ComponentInfo",
    "Diff", "Evidence", "EvidenceCollector", "Evolvable", "EvolvableInfo", "GateDecision", "Item", "Judge",
    "Manifest", "Memory", "MemoryOp", "Patch", "RemoteComponent", "RubricResult", "Snapshot",
    "StaleSnapshotError", "Step", "Task", "ToolCallError", "Trajectory", "Verdict", "VerifyResult",
    "build_server", "connect", "content_hash", "from_messages", "load_manifest", "serve",
]
