"""兼容旧的导入路径：实现在 anyrsi.self_improve。"""
from anyrsi.self_improve import INSTRUCTION, KIND, build_task, parse_patch

__all__ = ["INSTRUCTION", "KIND", "build_task", "parse_patch"]
