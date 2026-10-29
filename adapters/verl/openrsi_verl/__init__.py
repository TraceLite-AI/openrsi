"""OpenRSI × verl：判别器当奖励函数、训练数据导出、模型权重作为可进化体。

    from openrsi_verl import make_compute_score, tasks_to_rl_rows, WeightsEvolvable, DistillProposer

verl 的训练命令示例（参数名以你安装的 verl 版本为准）：

    WeightsEvolvable(
        base_model="Qwen/Qwen3-8B", path=".openrsi/weights",
        trainer=["torchrun", "--nproc_per_node=8", "-m", "verl.trainer.fsdp_sft_trainer",
                 "data.train_files={data}", "model.partial_pretrain={base_model}",
                 "trainer.default_local_dir={output_dir}"])
"""
from .data import DataLeakError, episode_messages, episodes_to_sft_rows, tasks_to_rl_rows, write_rows
from .reward import compute_score, load_judge, make_compute_score, reference_from
from .weights import DistillProposer, WeightsEvolvable


def reward_path() -> str:
    """verl 配置 custom_reward_function.path 要填的文件路径。"""
    from . import reward

    return reward.__file__


def main() -> None:
    import argparse

    from openrsi import serve

    ap = argparse.ArgumentParser(prog="openrsi-verl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reward-path", help="打印给 verl custom_reward_function.path 用的文件路径")
    w = sub.add_parser("weights", help="把 WeightsEvolvable 作为 MCP server 运行")
    w.add_argument("--base-model", required=True)
    w.add_argument("--path", required=True)
    w.add_argument("--dry-run", action="store_true")
    w.add_argument("--model-glob", help='训练输出里模型的子路径，如 "global_step_*/huggingface"')
    w.add_argument("--trainer", nargs=argparse.REMAINDER, help="训练命令模板，放在最后")
    args = ap.parse_args()
    if args.cmd == "reward-path":
        print(reward_path())
    else:
        serve(WeightsEvolvable(args.base_model, args.path, trainer=args.trainer, dry_run=args.dry_run,
                                model_glob=args.model_glob))


__all__ = ["DataLeakError", "DistillProposer", "WeightsEvolvable", "compute_score", "episode_messages",
           "episodes_to_sft_rows", "load_judge", "main", "make_compute_score", "reference_from", "reward_path",
           "tasks_to_rl_rows", "write_rows"]
