"""自带训练器：只依赖 torch 和 transformers（LoRA 需要 peft），给 WeightsEvolvable 的 trainer="builtin" 用。

    python -m anyrsi_weights.train --method sft|dpo|kto --model <模型> --data <rows.jsonl> --output <目录> \
        [--config '{"epochs": 2, "lr": 1e-5, "beta": 0.1, "lora": {"r": 8}}']

数据格式见 anyrsi_weights.data：
  sft  {"messages": [...]}                         只在最后一条 assistant 消息上算损失
  dpo  {"prompt": [...], "chosen": str, "rejected": str}
       损失 −log σ(β·[(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))])
  kto  {"messages": [...], "label": bool}
       r = log π(y) − log π_ref(y)，z = max(0, 本批 r 的均值)（参考点，不回传梯度）
       好样本损失 1 − σ(β(r − z))，坏样本损失 1 − σ(β(z − r))。这是 KTO 的简化版：参考点用批内均值估计

config：epochs（默认 2）、lr（1e-5）、batch（4）、max_len（1024）、seed（0）、beta（0.1）、
        lora（null 或 {r, alpha, dropout, target_modules}）、merge（LoRA 训完是否合并成完整模型，默认 true）
参考模型（DPO / KTO）：用了 LoRA 时就是关掉 adapter 的同一个模型，否则复制一份冻结的原模型。
训完在 --output 存模型、tokenizer 和 anyrsi_train.json（方法、样本数、每轮损失、配置）。
这是单卡的最小实现；大模型、多卡、在线强化学习请换成 verl 等外部框架。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

DEFAULTS = {"epochs": 2, "lr": 1e-5, "batch": 4, "max_len": 1024, "seed": 0, "beta": 0.1, "lora": None, "merge": True}


def render(tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """有对话模板就用模板；没有（比如测试用的小模型）就用一个简单固定格式。"""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)
    return text + ("<|assistant|>\n" if add_generation_prompt else "")


def encode(tokenizer: Any, prompt: list[dict[str, Any]], completion: str, max_len: int) -> tuple[list[int], int]:
    """返回 (token ids, 补全部分从第几个 token 开始)。超长时从左边截断提示，保留补全。"""
    prefix = tokenizer(render(tokenizer, prompt, True), add_special_tokens=False)["input_ids"]
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    comp = tokenizer(completion, add_special_tokens=False)["input_ids"] + eos
    comp = comp[:max_len]
    prefix = prefix[-(max_len - len(comp)):] if len(prefix) + len(comp) > max_len else prefix
    return prefix + comp, len(prefix)


def seq_logprob(model: Any, ids: list[int], start: int) -> Any:
    """补全部分每个 token 的对数概率之和。"""
    import torch

    x = torch.tensor([ids], device=model.device)
    logits = model(x).logits[0, :-1].float()
    targets = x[0, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0]
    return logp[max(start, 1) - 1:].sum()


def _examples(method: str, rows: list[dict[str, Any]], tok: Any, max_len: int) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if method == "sft":
            msgs = r["messages"]
            ids, start = encode(tok, msgs[:-1], msgs[-1]["content"], max_len)
            out.append({"ids": ids, "start": start})
        elif method == "dpo":
            c, cs = encode(tok, r["prompt"], r["chosen"], max_len)
            rj, rs = encode(tok, r["prompt"], r["rejected"], max_len)
            out.append({"chosen": (c, cs), "rejected": (rj, rs)})
        elif method == "kto":
            msgs = r["messages"]
            ids, start = encode(tok, msgs[:-1], msgs[-1]["content"], max_len)
            out.append({"ids": ids, "start": start, "label": bool(r["label"])})
        else:
            raise ValueError(f"不支持的方法 {method}；自带训练器支持 sft、dpo、kto")
    return out


def train(method: str, model_path: str, data: str, output: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data import read_rows

    cfg = {**DEFAULTS, **(config or {})}
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    rows = read_rows(data)
    if not rows:
        raise ValueError(f"{data} 里没有数据")
    examples = _examples(method, rows, tok, int(cfg["max_len"]))

    ref = None
    lora = cfg.get("lora")
    if lora:
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(r=int(lora.get("r", 8)), lora_alpha=int(lora.get("alpha", 16)),
                        lora_dropout=float(lora.get("dropout", 0.0)), target_modules=lora.get("target_modules"),
                        task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
    elif method in ("dpo", "kto"):
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    def ref_logprob(ids: list[int], start: int) -> Any:
        with torch.no_grad():
            if ref is not None:
                return seq_logprob(ref, ids, start)
            ctx = model.disable_adapter() if lora else contextlib.nullcontext()
            with ctx:
                return seq_logprob(model, ids, start)

    beta = float(cfg["beta"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]))
    history = []
    model.train()
    for _ in range(int(cfg["epochs"])):
        order = list(range(len(examples)))
        random.shuffle(order)
        total, n = 0.0, 0
        for b in range(0, len(order), int(cfg["batch"])):
            batch = [examples[i] for i in order[b:b + int(cfg["batch"])]]
            if method == "sft":
                losses = []
                for ex in batch:
                    k = len(ex["ids"]) - ex["start"]
                    losses.append(-seq_logprob(model, ex["ids"], ex["start"]) / max(k, 1))
                loss = torch.stack(losses).mean()
            elif method == "dpo":
                margins = []
                for ex in batch:
                    pc = seq_logprob(model, *ex["chosen"]) - ref_logprob(*ex["chosen"])
                    pr = seq_logprob(model, *ex["rejected"]) - ref_logprob(*ex["rejected"])
                    margins.append(beta * (pc - pr))
                loss = -F.logsigmoid(torch.stack(margins)).mean()
            else:
                ratios = torch.stack([seq_logprob(model, ex["ids"], ex["start"]) - ref_logprob(ex["ids"], ex["start"])
                                      for ex in batch])
                z = ratios.mean().detach().clamp(min=0)
                labels = torch.tensor([ex["label"] for ex in batch], device=ratios.device)
                good = 1 - torch.sigmoid(beta * (ratios - z))
                bad = 1 - torch.sigmoid(beta * (z - ratios))
                loss = torch.where(labels, good, bad).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            n += len(batch)
        history.append(total / max(n, 1))
        if not math.isfinite(history[-1]):
            raise RuntimeError(f"损失变成了 {history[-1]}，训练发散")

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if lora and cfg.get("merge", True):
        model = model.merge_and_unload()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    report = {"method": method, "examples": len(examples), "rows": len(rows), "loss_per_epoch": history,
              "config": cfg, "base_model": model_path, "device": device, "lora_merged": bool(lora and cfg.get("merge", True))}
    (out / "anyrsi_train.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m anyrsi_weights.train", description="AnyRSI 自带训练器")
    ap.add_argument("--method", required=True, choices=["sft", "dpo", "kto"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="{}", help="JSON：epochs、lr、batch、max_len、seed、beta、lora、merge")
    args = ap.parse_args(argv)
    report = train(args.method, args.model, args.data, args.output, json.loads(args.config or "{}"))
    print(json.dumps({k: report[k] for k in ("method", "examples", "loss_per_epoch")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
