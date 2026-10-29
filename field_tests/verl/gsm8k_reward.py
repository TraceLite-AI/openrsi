"""GSM8K 奖励：用 OpenRSI 判别器打分，答案抽取规则和 verl 自带的 gsm8k 奖励一致（取最后一个 "#### 数字"）。

verl 配置：
    custom_reward_function.path=<本文件>
    custom_reward_function.name=compute_score
"""
import re

from openrsi_judge import EvidenceJudge
from openrsi_verl import make_compute_score
from openrsi_verl.reward import default_trajectory

RUBRIC = {"items": [{"id": "correct", "check": "output_equals", "required": True}]}
_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract(solution: str) -> str | None:
    found = _ANSWER.findall(solution)
    return found[-1].replace(",", "").replace("$", "") if found else None


def _trajectory(task, solution, extra):
    return default_trajectory(task, solution, extra).model_copy(update={"output": extract(solution)})


compute_score = make_compute_score(EvidenceJudge(RUBRIC), trajectory_from=_trajectory)
compute_score_dict = make_compute_score(EvidenceJudge(RUBRIC), trajectory_from=_trajectory, return_dict=True)
