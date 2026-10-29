"""把任务和通过准入的轨迹导出成 verl 的训练数据。

- RL（PPO / GRPO）：每行 {data_source, prompt, ability, reward_model: {style, ground_truth}, extra_info}，
  标准答案只放在 reward_model.ground_truth 里，不进 prompt。
- SFT：每行 {"messages": [...]}，只取 train 划分里判别器判为通过的轨迹（拒绝采样式的自我蒸馏）。
  holdout / anchor / final 的轨迹一律拒绝导出，避免评估集泄漏进训练。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from openrsi import Episode, Task


class DataLeakError(ValueError):
    """试图把非 train 划分的数据导出成训练数据。"""


def default_prompt(task: Task) -> str:
    x = task.input
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def tasks_to_rl_rows(tasks: Iterable[Task], data_source: str, *, ability: str = "agent",
                     prompt_fn: Callable[[Task], str] = default_prompt,
                     for_validation: bool = False) -> list[dict[str, Any]]:
    """for_validation=True 时导出 verl 的验证集（data.val_files）：只接受非 train 的题，反过来防止训练题混进评估。"""
    rows = []
    for t in tasks:
        if for_validation and t.split == "train":
            raise DataLeakError(f"{t.id} 属于 train，不能放进验证集")
        if not for_validation and t.split != "train":
            raise DataLeakError(f"{t.id} 属于 {t.split}，不能导出成训练数据")
        ref = t.reference
        if isinstance(ref, dict) and set(ref) == {"expected"}:
            ground_truth = ref["expected"]  # 只有标准答案时直接放值，和 verl 自带奖励（如 gsm8k）的格式一致
        elif isinstance(ref, dict):
            ground_truth = json.dumps(ref, ensure_ascii=False, sort_keys=True)
        else:
            ground_truth = ref
        rows.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_fn(t)}],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {"split": t.split, "index": t.id, "input": t.input, "tags": list(t.tags)},
        })
    return rows


def episode_messages(ep: Episode, prompt_fn: Callable[[Task], str] = default_prompt) -> list[dict[str, Any]]:
    """轨迹转成对话：题目作为 user，工具调用和结果按顺序展开，最终输出作为最后一条 assistant。"""
    msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt_fn(ep.task)}]
    for s in ep.trajectory.steps:
        if s.type == "message" and s.role in ("assistant", "user", "system"):
            msgs.append({"role": s.role, "content": s.content if isinstance(s.content, str) else json.dumps(s.content)})
        elif s.type == "tool_call":
            msgs.append({"role": "assistant", "content": json.dumps({"tool": s.name, "arguments": s.arguments or {}},
                                                                    ensure_ascii=False)})
        elif s.type == "tool_result":
            msgs.append({"role": "tool", "content": s.content if isinstance(s.content, str) else json.dumps(s.content)})
    out = ep.trajectory.output
    final = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    if not (msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == final):
        msgs.append({"role": "assistant", "content": final})
    return msgs


def episodes_to_sft_rows(episodes: Iterable[Episode], *, min_score: float = 1.0,
                         prompt_fn: Callable[[Task], str] = default_prompt) -> list[dict[str, Any]]:
    rows = []
    for ep in episodes:
        if ep.task.split != "train":
            raise DataLeakError(f"{ep.task.id} 属于 {ep.task.split}，不能导出成训练数据")
        if ep.verdict.passed and ep.verdict.score >= min_score:
            rows.append({"messages": episode_messages(ep, prompt_fn),
                         "extra_info": {"task_id": ep.task.id, "score": ep.verdict.score,
                                        "judge": ep.verdict.judge_ref}})
    return rows


def write_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """按扩展名写 .jsonl 或 .parquet（parquet 需要 pandas 和 pyarrow）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path)
    else:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path
