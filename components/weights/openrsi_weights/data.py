"""把判别器判过的 train 轨迹变成训练数据。只接受 train 划分：holdout / anchor / final 一律拒绝，防止评估集泄漏进训练。

- SFT：每行 {"messages": [...]}，只取判为通过的轨迹（拒绝采样式自我蒸馏）
- DPO：同一道题，判对的回答当 chosen、判错的当 rejected：{"prompt": [...], "chosen": str, "rejected": str}
- KTO：每条轨迹一行，判对的 label=true、判错的 label=false：{"messages": [...], "label": bool}

DPO / KTO 让做错的题也能用来训练权重。DPO 需要同一道题既有对也有错，所以通常要配合 loop.train_samples
让每道练习题答多次（agent 的采样要有随机性）。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from openrsi import Episode, Task


class DataLeakError(ValueError):
    """试图把非 train 划分的数据导出成训练数据。"""


def default_prompt(task: Task) -> str:
    x = task.input
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def _text(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def episode_messages(ep: Episode, prompt_fn: Callable[[Task], str] = default_prompt) -> list[dict[str, Any]]:
    """轨迹转成对话。agent 自己记下了 user 消息（比如 ChatAgent）就原样用轨迹里的对话（含系统提示）；
    否则用 prompt_fn(task) 作为 user。工具调用和结果按顺序展开，最终输出作为最后一条 assistant。"""
    steps = ep.trajectory.steps
    own = any(s.type == "message" and s.role == "user" for s in steps)
    msgs: list[dict[str, Any]] = [] if own else [{"role": "user", "content": prompt_fn(ep.task)}]
    for s in steps:
        if s.type == "message" and s.role in ("assistant", "user", "system"):
            msgs.append({"role": s.role, "content": _text(s.content)})
        elif s.type == "tool_call":
            msgs.append({"role": "assistant", "content": json.dumps({"tool": s.name, "arguments": s.arguments or {}},
                                                                    ensure_ascii=False)})
        elif s.type == "tool_result":
            msgs.append({"role": "tool", "content": _text(s.content)})
    final = _text(ep.trajectory.output)
    if not (msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == final):
        msgs.append({"role": "assistant", "content": final})
    return msgs


def _check_train(ep: Episode) -> None:
    if ep.task.split != "train":
        raise DataLeakError(f"{ep.task.id} 属于 {ep.task.split}，不能导出成训练数据")


def episodes_to_sft_rows(episodes: Iterable[Episode], *, min_score: float = 1.0,
                         prompt_fn: Callable[[Task], str] = default_prompt) -> list[dict[str, Any]]:
    rows = []
    for ep in episodes:
        _check_train(ep)
        if ep.verdict.passed and ep.verdict.score >= min_score:
            rows.append({"messages": episode_messages(ep, prompt_fn),
                         "extra_info": {"task_id": ep.task.id, "score": ep.verdict.score,
                                        "judge": ep.verdict.judge_ref}})
    return rows


def episodes_to_dpo_rows(episodes: Iterable[Episode], *, max_pairs_per_task: int = 4,
                         prompt_fn: Callable[[Task], str] = default_prompt) -> list[dict[str, Any]]:
    """同一道题里判对 × 判错配对；prompt 取判对那条的对话前缀（两者题面和系统提示相同）。"""
    by_task: dict[str, dict[bool, list[Episode]]] = defaultdict(lambda: {True: [], False: []})
    for ep in episodes:
        _check_train(ep)
        by_task[ep.task.id][bool(ep.verdict.passed)].append(ep)
    rows = []
    for task_id in sorted(by_task):
        good, bad = by_task[task_id][True], by_task[task_id][False]
        pairs = [(g, b) for g in good for b in bad][:max_pairs_per_task]
        for g, b in pairs:
            gm, bm = episode_messages(g, prompt_fn), episode_messages(b, prompt_fn)
            if gm[-1]["content"] == bm[-1]["content"]:
                continue  # 回答一字不差却一对一错，说明判别有噪声，不拿来训练
            rows.append({"prompt": gm[:-1], "chosen": gm[-1]["content"], "rejected": bm[-1]["content"],
                         "extra_info": {"task_id": task_id, "chosen": g.trajectory.id, "rejected": b.trajectory.id}})
    return rows


def episodes_to_kto_rows(episodes: Iterable[Episode], *,
                         prompt_fn: Callable[[Task], str] = default_prompt) -> list[dict[str, Any]]:
    rows = []
    for ep in episodes:
        _check_train(ep)
        rows.append({"messages": episode_messages(ep, prompt_fn), "label": bool(ep.verdict.passed),
                     "extra_info": {"task_id": ep.task.id, "score": ep.verdict.score}})
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


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]
