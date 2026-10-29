"""最小的 SFT 训练脚本（只依赖 torch 和 transformers），给 WeightsEvolvable 当训练命令用。

读 DistillProposer 写出的 jsonl（每行 {"messages": [...]}），只在最后一条 assistant 消息上计算损失，
训练完把模型和 tokenizer 存到 --output。小模型单卡就能跑；大规模训练换成 verl 的 SFT 入口。

  python train_sft.py --model Qwen/Qwen2.5-0.5B-Instruct --data sft.jsonl --output out/ [--epochs 2 --lr 1e-5]
"""
import argparse
import json
import math
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.gradient_checkpointing_enable()

    examples = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        msgs = json.loads(line)["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            continue
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = tok(full, add_special_tokens=False)["input_ids"][: args.max_len]
        labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
        if any(x != -100 for x in labels):
            examples.append((ids, labels))
    if not examples:
        raise SystemExit("没有可训练的样本")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = args.epochs * math.ceil(len(examples) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)))
    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch):
            batch = examples[i:i + args.batch]
            n = max(len(x) for x, _ in batch)
            ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in batch], device=device)
            lab = torch.tensor([y + [-100] * (n - len(y)) for _, y in batch], device=device)
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in batch], device=device)
            loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            print(json.dumps({"epoch": epoch, "step": step, "of": steps, "loss": round(loss.item(), 4)}), flush=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "openrsi_train.json").write_text(json.dumps({"base": args.model, "examples": len(examples),
                                                        "epochs": args.epochs, "lr": args.lr}))


if __name__ == "__main__":
    main()
