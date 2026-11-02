"""OpenRSI 的模型权重：checkpoint 树作为可进化体、判别器判过的训练数据、自带训练器、可替换的外部训练命令。

    from openrsi_weights import WeightsEvolvable, DistillProposer, PreferenceProposer

    WeightsEvolvable(base_model="Qwen/Qwen2.5-0.5B-Instruct", path=".openrsi/weights", trainer="builtin")

自带训练器（只依赖 torch 和 transformers，LoRA 需要 peft）：python -m openrsi_weights.train --help
换成 verl、LLaMA-Factory 等外部框架时，把 trainer 写成命令模板即可，见 evolvable.py。
"""
from .data import (DataLeakError, default_prompt, episode_messages, episodes_to_dpo_rows, episodes_to_kto_rows,
                   episodes_to_sft_rows, read_rows, write_rows)
from .evolvable import BUILTIN, WeightsEvolvable
from .proposers import DistillProposer, PreferenceProposer


def main() -> None:
    import argparse

    from openrsi import serve

    ap = argparse.ArgumentParser(prog="openrsi-weights")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model-glob", help='训练输出里模型的子路径，如 "global_step_*/huggingface"')
    ap.add_argument("--trainer", nargs=argparse.REMAINDER, help='"builtin" 或训练命令模板，放在最后')
    args = ap.parse_args()
    trainer = "builtin" if args.trainer == ["builtin"] else args.trainer
    serve(WeightsEvolvable(args.base_model, args.path, trainer=trainer, dry_run=args.dry_run,
                           model_glob=args.model_glob))


__all__ = ["BUILTIN", "DataLeakError", "DistillProposer", "PreferenceProposer", "WeightsEvolvable", "default_prompt",
           "episode_messages", "episodes_to_dpo_rows", "episodes_to_kto_rows", "episodes_to_sft_rows", "main",
           "read_rows", "write_rows"]
