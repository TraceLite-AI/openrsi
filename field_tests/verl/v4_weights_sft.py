"""V4（GPU）：模型权重作为可进化体，连续两次 verl SFT（第二次从第一次的输出继续训）。

DistillProposer 从"判别器判为通过"的 train 轨迹生成 SFT 数据（这里用 gsm8k 训练集的标准解答模拟通过的轨迹），
WeightsEvolvable 调用 verl 的 SFT 入口训练、登记 checkpoint，然后回滚、再分叉。

用法：
  python field_tests/verl/v4_weights_sft.py --src ~/data/gsm8k --model Qwen/Qwen2.5-0.5B-Instruct --gpus 1
  训练入口和参数名不对时，用 --trainer-json 传完整命令模板（JSON 列表，可用 {base_model} {data} {output_dir}）。
  --model-glob 指定训练输出里 HF 模型所在子路径（默认 global_step_*/huggingface）；跑完看 layout 再调整。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from openrsi import Episode, ProposalRequest, Task, Trajectory, Verdict
from openrsi_verl import DistillProposer, WeightsEvolvable

OUT = Path(__file__).resolve().parent.parent / "out" / "verl" / "v4"


def default_trainer(gpus: int) -> list[str]:
    return ["torchrun", "--standalone", "--nnodes=1", f"--nproc_per_node={gpus}", "-m",
            "verl.trainer.fsdp_sft_trainer",
            "data.train_files={data}", "data.val_files={data}",
            "data.multiturn.enable=true", "data.multiturn.messages_key=messages",
            "data.micro_batch_size_per_gpu=4", "data.max_length=1024",
            "model.partial_pretrain={base_model}",
            "trainer.default_local_dir={output_dir}", "trainer.project_name=openrsi-field",
            "trainer.experiment_name=v4", "trainer.total_epochs=1", 'trainer.logger=["console"]']


def episodes(df: pd.DataFrame, start: int, n: int) -> list[Episode]:
    eps = []
    for i, r in df.iloc[start:start + n].iterrows():
        q = r["prompt"][0]["content"]
        answer = r["extra_info"].get("answer") or f"#### {r['reward_model']['ground_truth']}"
        task = Task(id=f"train-{i}", input=q, split="train", reference={"expected": r["reward_model"]["ground_truth"]})
        traj = Trajectory(id=f"tr-{i}", task_id=task.id, output=answer, steps=[])
        v = Verdict(task_id=task.id, trajectory_id=traj.id, passed=True, score=1.0, judge_ref="gsm8k")
        eps.append(Episode(task=task, trajectory=traj, verdict=v))
    return eps


def layout(p: Path, depth: int = 3) -> list[str]:
    return sorted(str(x.relative_to(p)) for x in p.rglob("*") if len(x.relative_to(p).parts) <= depth)[:80]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--n", type=int, default=256, help="每次训练用多少条")
    ap.add_argument("--trainer-json")
    ap.add_argument("--model-glob", default="global_step_*/huggingface")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(Path(args.src).expanduser() / "train.parquet")
    trainer = json.loads(args.trainer_json) if args.trainer_json else default_trainer(args.gpus)
    w = WeightsEvolvable(args.model, OUT / "weights", trainer=trainer, model_glob=args.model_glob or None)
    proposer = DistillProposer(OUT / "data", min_examples=8, format="parquet")
    report: dict = {"trainer": trainer, "steps": []}
    root = w.snapshot().id

    def train_once(start: int) -> None:
        info = w.describe()
        req = ProposalRequest(target=info, targets=[info], parent_snapshot=w.snapshot().id,
                              episodes=episodes(df, start, args.n), attributions=[], max_patches=1)
        [patch] = proposer.propose(req)
        step = {"parent": w.snapshot().id, "parent_model": w.export()[0]["text"], "data": patch.ops[0]["data"]}
        try:
            snap = w.apply(patch)
            model = Path(w.export()[0]["text"])
            step.update(ok=True, snapshot=snap.id, model=str(model),
                        has_config=(model / "config.json").exists(),
                        has_weights=any(model.glob("*.safetensors")) or any(model.glob("*.bin")),
                        layout=layout(model.parent.parent if args.model_glob else model))
        except Exception as e:  # noqa: BLE001
            ckpt = OUT / "weights" / "checkpoints"
            step.update(ok=False, error=str(e), layout=layout(ckpt) if ckpt.exists() else [])
        report["steps"].append(step)
        print(json.dumps(step, ensure_ascii=False, indent=2))

    train_once(0)
    if report["steps"][-1]["ok"]:
        train_once(args.n)  # 第二次从第一次的模型继续训
    if all(s["ok"] for s in report["steps"]) and len(report["steps"]) == 2:
        w.rollback(root)
        report["after_rollback"] = w.export()[0]["text"]
        report["lineage"] = {cid: {"parent": c["parent"], "model": c["model_path"]} for cid, c in w.ckpts.items()}
        try:  # 检查训练出的模型能用 transformers 加载配置
            from transformers import AutoConfig

            AutoConfig.from_pretrained(report["steps"][-1]["model"])
            report["transformers_load_config"] = "ok"
        except Exception as e:  # noqa: BLE001
            report["transformers_load_config"] = f"error: {e!r}"
    (OUT.parent / "v4.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    s = report["steps"]
    ok = (len(s) == 2 and all(x["ok"] and x["has_config"] and x["has_weights"] for x in s)
          and s[1]["parent_model"] == s[0]["model"] and report.get("after_rollback") == args.model)
    print("PASS V4" if ok else "FAIL V4（把 field_tests/out/verl/v4.json 和 out/verl/v4/weights/checkpoints/*.log 发回来）")


if __name__ == "__main__":
    main()
