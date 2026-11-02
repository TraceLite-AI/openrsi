"""把任务和通过准入的轨迹导出成 verl 的训练数据。

- RL（PPO / GRPO）：每行 {data_source, prompt, ability, reward_model: {style, ground_truth}, extra_info}，
  标准答案只放在 reward_model.ground_truth 里，不进 prompt。
- SFT / DPO / KTO 的数据转换在 anyrsi_weights.data（这里重新导出 SFT 部分，旧代码照常可用）。
  holdout / anchor / final 的轨迹一律拒绝导出，避免评估集泄漏进训练。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Iterable

from anyrsi import Task


from anyrsi_weights.data import (DataLeakError, default_prompt, episode_messages, episodes_to_sft_rows,  # noqa: F401
                                  write_rows)


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
