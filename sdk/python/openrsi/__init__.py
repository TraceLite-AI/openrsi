"""OpenRSI Python SDK：协议数据格式、组件基类、MCP 服务端与客户端、一致性测试。"""
from .client import RemoteComponent, ToolCallError, connect
from .components import (
    Agent, Algorithm, Attributor, Component, ComponentError, Environment, EvidenceCollector, Evolvable, Gate,
    Guard, Judge, Lineage, Memory, Proposer, Skills, StaleSnapshotError,
)
from .itemstore import ItemStore
from .convert import from_messages
from .manifest import Manifest, load_manifest
from .server import build_server, serve
from .types import (
    PROTOCOL_VERSION, AlgorithmStatus, AnchorExample, AskResult, Attribution, AttributionTarget,
    CalibrationReport, ComponentInfo, Diff, Episode, Evidence, EvolvableInfo, GateCheck, GateDecision,
    GateRequest, GuardFinding, GuardReport, Item, LineageEntry, MemoryOp, Patch, ProposalRequest, RubricResult,
    ScreenResult, SkillResult, Snapshot, Step, Task, TellRequest, Trajectory, Verdict, VerifyResult, content_hash,
)

__version__ = "0.4.0"

__all__ = [
    "Agent", "Algorithm", "AlgorithmStatus", "AnchorExample", "AskResult", "Attribution", "AttributionTarget",
    "Attributor", "CalibrationReport", "Component", "ComponentError", "ComponentInfo", "Diff", "Environment",
    "Episode", "Evidence", "EvidenceCollector", "Evolvable", "EvolvableInfo", "Gate", "GateCheck",
    "GateDecision", "GateRequest", "Guard", "GuardFinding", "GuardReport", "Item", "ItemStore", "Judge",
    "Lineage", "LineageEntry", "Manifest", "Memory", "MemoryOp", "PROTOCOL_VERSION", "Patch",
    "ProposalRequest", "Proposer", "RemoteComponent", "RubricResult", "ScreenResult", "SkillResult", "Skills", "Snapshot",
    "StaleSnapshotError", "Step", "Task", "TellRequest", "ToolCallError", "Trajectory", "Verdict",
    "VerifyResult", "build_server", "connect", "content_hash", "from_messages", "load_manifest", "serve",
]
