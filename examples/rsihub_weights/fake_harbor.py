"""本地调通用的假 Harbor：接受 `harbor run` 的一小部分参数，产出和真 Harbor 一样布局的结果目录。

    python fake_harbor.py run -p <数据集> --include-task-name <题> --job-name <job> --jobs-dir <目录> \
        --agent-import-path <模块:类> [--ae KEY=VALUE ...] [其余参数忽略]

agent 类从 PYTHONPATH 导入（HarborAgent 会把被进化的 harness 仓库放在最前面），
用 agent.solve(instruction) 做题，返回 (答案, ATIF steps)。
验证：答案和 tests/expected.txt 一致记 1，否则 0。agent 抛异常时 reward 为空，记下 exception_info。
写出 <jobs-dir>/<job>/<题>__<随机>/result.json 和 agent/trajectory.json。
"""
import argparse
import importlib
import json
import os
import sys
import traceback
import uuid
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("-p", dest="dataset", required=True)
    ap.add_argument("--include-task-name", action="append", default=[])
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--jobs-dir", required=True)
    ap.add_argument("--agent-import-path", required=True)
    ap.add_argument("--ae", action="append", default=[])
    args, _ = ap.parse_known_args(argv)
    for kv in args.ae:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    module, _, cls = args.agent_import_path.partition(":")
    agent = getattr(importlib.import_module(module), cls)()
    for name in args.include_task_name:
        task = Path(args.dataset) / name
        trial = Path(args.jobs_dir) / args.job_name / f"{name}__{uuid.uuid4().hex[:7]}"
        (trial / "agent").mkdir(parents=True)
        instruction = (task / "instruction.md").read_text(encoding="utf-8")
        result = {"task_name": name, "trial_name": trial.name, "agent_result": {"n_input_tokens": len(instruction)}}
        try:
            answer, steps = agent.solve(instruction)
            expected = (task / "tests" / "expected.txt").read_text(encoding="utf-8").strip()
            result["verifier_result"] = {"rewards": {"reward": 1.0 if str(answer).strip() == expected else 0.0}}
        except Exception as e:  # noqa: BLE001 - 和真 Harbor 一样，agent 出错记进结果而不是让整次运行失败
            steps = [{"step_id": 1, "source": "user", "message": instruction}]
            result["exception_info"] = {"exception_type": type(e).__name__, "exception_message": str(e),
                                        "exception_traceback": traceback.format_exc()[-2000:]}
        (trial / "agent" / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.4", "steps": steps}),
                                                        encoding="utf-8")
        (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
