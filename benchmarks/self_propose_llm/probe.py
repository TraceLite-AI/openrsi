"""真实 LLM 上测递归改进的关键前提：agent 越强，它给自己出的改法是否越好？

任务：秘密运算符（examples/secret_ops）。8 个符号各代表一种未知的二元运算，agent 是一个 LLM，系统提示里写了规则就照规则算，
没写就按 a+b。被进化的是系统提示。

做法：造出强弱不同的 agent 版本（系统提示里写对 0 / 3 / 6 条规则），对每个版本：
  1. 用它做练习题，收集错题，生成同一道"改进你自己"的题（anyrsi_runtime.self_propose.build_task，和 loop.self_propose 完全一样）。
  2. 同一道题分别交给三方回答（同一个模型、同样的采样种子，只有系统提示不同）：
       self     当前版本的 agent 自己（系统提示 = 当前版本的提示）    ← loop.self_propose
       fixed    起点版本的 agent（系统提示 = 起点提示）              ← 水平固定在起点的提案器
       neutral  普通助手（系统提示 = "You are a helpful assistant."）← 常见的独立提案器
  3. 每个回答用 parse_patch 解析成补丁，应用后在考试题上测分数，减去应用前的分数，就是这个改法的真实效果。

如果"agent 越强改法越好"成立，self − fixed 应当随版本变强而变大。

    python -m llama_cpp.server --model qwen2.5-3b-instruct-q4_k_m.gguf --port 8100 --model_alias policy &
    python benchmarks/self_propose_llm/probe.py --base-url http://localhost:8100/v1 --samples 6
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import json
import random
import re
import sys
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "secret_ops"))
from secret_ops import OPS, SYMBOLS  # noqa: E402

from anyrsi import Episode, EvolvableInfo, ProposalRequest, RubricResult, Task, Trajectory, Verdict  # noqa: E402
from anyrsi_llm import chat, task_text  # noqa: E402
from anyrsi_runtime.self_propose import build_task, parse_patch  # noqa: E402

BASE = ("You compute expressions that use special symbols. Each symbol stands for an operation on two integers a and b. "
        "If the rule for a symbol is not given below, assume it means a + b.")
NEUTRAL = "You are a helpful assistant."
FORMAT = "Reply with one line in exactly this form: <arithmetic expression with numbers only> = <integer>"
RULE_TEXT = {"add": "a + b", "sub": "a - b", "mul": "a * b", "mul+1": "a * b + 1", "add+1": "a + b + 1",
             "2a+b": "2 * a + b", "a+2b": "a + 2 * b", "max": "max(a, b)", "min": "min(a, b)",
             "a*a+b": "a * a + b", "a-b-1": "a - b - 1", "absdiff": "|a - b|"}


# 难版本：运算更复杂，两三个例子能拟合很多公式，补规则时要真的推理、验证
HARD = {
    "a*b-a": lambda a, b: a * b - a, "(a+b)%7": lambda a, b: (a + b) % 7, "a*a-b": lambda a, b: a * a - b,
    "(a+1)*(b-1)": lambda a, b: (a + 1) * (b - 1), "a*b-(a+b)": lambda a, b: a * b - (a + b),
    "2*max-min": lambda a, b: 2 * max(a, b) - min(a, b), "a+b*b": lambda a, b: a + b * b,
    "a>b?a-b:a*b": lambda a, b: a - b if a > b else a * b, "(a*b)%10": lambda a, b: (a * b) % 10,
    "a*(b+2)": lambda a, b: a * (b + 2),
}
HARD_TEXT = {"a*b-a": "a * b - a", "(a+b)%7": "(a + b) mod 7", "a*a-b": "a * a - b",
             "(a+1)*(b-1)": "(a + 1) * (b - 1)", "a*b-(a+b)": "a * b - (a + b)",
             "2*max-min": "2 * max(a, b) - min(a, b)", "a+b*b": "a + b * b",
             "a>b?a-b:a*b": "a - b if a > b, otherwise a * b", "(a*b)%10": "(a * b) mod 10",
             "a*(b+2)": "a * (b + 2)"}


class Bench:
    def __init__(self, base_url: str, model: str = "policy", api_key: str | None = None, seed: int = 0,
                 holdout_per_symbol: int = 2, threads: int = 8, log: Path | None = None, hard: bool = False):
        self.base_url, self.model, self.api_key, self.threads = base_url, model, api_key, threads
        self.ops, self.text = (HARD, HARD_TEXT) if hard else (OPS, RULE_TEXT)
        self.train_per_symbol = 3 if hard else 2
        rng = random.Random(seed)
        # 去掉 add：否则"没写规则就按 a+b"会让一个符号白送
        ops = [o for o in sorted(self.ops) if o != "add"]
        self.secret = dict(zip(SYMBOLS, rng.sample(ops, len(SYMBOLS))))
        self.order = rng.sample(SYMBOLS, len(SYMBOLS))  # 版本变强时按这个顺序写进规则
        hr = random.Random(f"{seed}-holdout")
        self.holdout = [self.task("holdout", f"{s}{i}", s, hr.randint(2, 9), hr.randint(2, 9))
                        for s in SYMBOLS for i in range(holdout_per_symbol)]
        self.cache: dict[str, float] = {}
        self.calls = 0
        self.log = log.open("a") if log else None

    def task(self, split: str, key: str, sym: str, a: int, b: int) -> Task:
        return Task(id=f"{split}-{key}", split=split, tags=[sym], input={"question": f"Compute: {a} {sym} {b}\n{FORMAT}"},
                    reference={"expected": self.ops[self.secret[sym]](a, b), "sym": sym, "a": a, "b": b})

    def rule(self, sym: str) -> dict[str, Any]:
        return {"id": f"rule-{sym}", "kind": "prompt", "text": f"a {sym} b = {self.text[self.secret[sym]]}",
                "order": 1}

    def version(self, k: int) -> list[dict[str, Any]]:
        return [{"id": "base", "kind": "prompt", "text": BASE, "order": 0}] + [self.rule(s) for s in self.order[:k]]

    @staticmethod
    def render(state: list[dict[str, Any]]) -> str:
        return "\n\n".join(d["text"] for d in sorted(state, key=lambda d: (d.get("order", 0), d["id"])) if d.get("text"))

    def ask(self, system: str, user: str, *, max_tokens: int = 2000, temperature: float | None = None,
            seed: int | None = None) -> str:
        self.calls += 1
        for attempt in range(4):
            try:
                out = chat(self.base_url, self.model,
                           [{"role": "system", "content": system}, {"role": "user", "content": user}],
                           temperature=temperature, max_tokens=max_tokens, timeout=600, api_key=self.api_key,
                           extra={"seed": seed} if seed is not None else None)
                return out["text"]
            except Exception:  # noqa: BLE001  限流、网络抖动：退避重试
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        return ""

    @staticmethod
    def number(text: str) -> int | None:
        line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
        m = re.findall(r"=\s*(-?\d+)\s*$", line)
        return int(m[-1]) if m else None

    def run(self, state: list[dict[str, Any]], tasks: list[Task]) -> list[Episode]:
        system, eps = self.render(state), []
        with ThreadPoolExecutor(self.threads) as pool:
            outs = list(pool.map(lambda t: self.ask(system, task_text(t)), tasks))
        for t, out in zip(tasks, outs):
            got, want = self.number(out), t.reference["expected"]
            ok = got == want
            note = None if ok else f"expected {want}, got {out.strip()!r}"
            eps.append(Episode(task=t, trajectory=Trajectory(id=f"traj-{t.id}", task_id=t.id, output=out),
                               verdict=Verdict(score=float(ok), passed=ok, judge_ref="exact",
                                               rubric=[RubricResult(id="correct", result="pass" if ok else "fail",
                                                                    note=note)])))
        return eps

    def score(self, state: list[dict[str, Any]]) -> float:
        key = self.render(state)
        if key not in self.cache:
            self.cache[key] = fmean(e.verdict.score for e in self.run(state, self.holdout))
        return self.cache[key]

    def train(self, round_: int, per_symbol: int | None = None) -> list[Task]:
        per_symbol = per_symbol or self.train_per_symbol
        rng = random.Random(f"train-{round_}")
        return [self.task("train", f"{round_}.{s}{i}", s, rng.randint(2, 9), rng.randint(2, 9))
                for s in SYMBOLS for i in range(per_symbol)]


def apply(state: list[dict[str, Any]], ops: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """PromptEvolvable 的补丁语义：add / replace / remove，条目一律当作提示段落。"""
    items = {d["id"]: dict(d) for d in state}
    for i, op in enumerate(ops):
        item = op.get("item") if isinstance(op.get("item"), dict) else {}
        kind = op.get("op")
        if kind in ("add", "replace"):
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                return None
            iid = str(item.get("id") or f"new-{i}")
            if kind == "replace" and iid not in items:
                return None
            items[iid] = {"id": iid, "kind": "prompt", "text": text,
                          "order": items.get(iid, {}).get("order", item.get("order", 1))}
        elif kind == "remove":
            iid = op.get("id") or item.get("id")
            if iid not in items:
                return None
            del items[iid]
        else:
            return None
    return list(items.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--model", default="gpt-5.4-mini")  # 先用便宜的模型
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 3, 6])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", default="probe_results.jsonl")
    ap.add_argument("--hard", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    bench = Bench(args.base_url, args.model, os.environ.get(args.api_key_env), hard=args.hard)
    info = EvolvableInfo(target="prompt", type="prompt", head="v", surfaces=["system_prompt"],
                         example_ops=[{"op": "add", "item": {"id": "rule-example", "kind": "prompt",
                                                             "text": "a ◇ b = a * b"}},
                                      {"op": "replace", "item": {"id": "base", "kind": "prompt", "text": "..."}},
                                      {"op": "remove", "id": "rule-example"}])
    start_state = bench.version(0)
    print("secret:", {s: bench.secret[s] for s in bench.order}, flush=True)
    for k in args.levels:
        state = bench.version(k)
        t0 = time.monotonic()
        before = bench.score(state)
        episodes = bench.run(state, bench.train(k))
        req = ProposalRequest(target=info, targets=[info], parent_snapshot=f"v{k}", state=state,
                              states={"prompt": state}, episodes=episodes, max_patches=1)
        task = build_task(req, gen=k, index=0)
        user = task_text(task)
        print(f"\n== level {k}: holdout {before:.3f}, train failures "
              f"{sum(not e.verdict.passed for e in episodes)}/{len(episodes)}, prompt chars {len(user)}", flush=True)
        systems = {"self": bench.render(state), "fixed": bench.render(start_state), "neutral": NEUTRAL}
        for i in range(args.samples):
            seed = 1000 * k + i
            for who, system in systems.items():
                t1 = time.monotonic()
                reply = bench.ask(system, user, max_tokens=4000, temperature=args.temperature, seed=seed)
                patch = parse_patch(reply, req, who)
                new = apply(state, patch.ops) if patch else None
                after = bench.score(new) if new is not None else None
                gain = (after - before) if after is not None else 0.0
                row = {"level": k, "sample": i, "who": who, "before": before, "parsed": patch is not None,
                       "applied": new is not None, "after": after, "gain": gain, "reply": reply,
                       "secs": round(time.monotonic() - t1, 1)}
                with out.open("a") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {who:7s} s{i} parsed={row['parsed']!s:5} gain={gain:+.3f} ({row['secs']}s)", flush=True)
        print(f"  level {k} took {time.monotonic() - t0:.0f}s, calls so far {bench.calls}", flush=True)
    summarize(out)


def summarize(path: Path) -> None:
    rows = [json.loads(line) for line in path.open()]
    print("\nlevel  who      n  parsed  mean gain  good(>0)")
    for k in sorted({r["level"] for r in rows}):
        for who in ("self", "fixed", "neutral"):
            rs = [r for r in rows if r["level"] == k and r["who"] == who]
            if rs:
                g = [r["gain"] for r in rs]
                print(f"{k:5d}  {who:7s} {len(rs):2d}  {fmean(r['parsed'] for r in rs):.2f}    "
                      f"{fmean(g):+.3f}±{pstdev(g) / max(1, len(g) - 1) ** .5:.3f}  {fmean(x > 0 for x in g):.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(Path(sys.argv[2]))
    else:
        main()
