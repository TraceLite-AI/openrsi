"""RSIHub 的题库当 AnyRSI 的环境：同一批 Harbor 题、同样的划分规则，由 AnyRSI 的循环来改代码和权重。

RSIHub 的评测器是 Harbor：每道题是数据集目录下一个带 task.toml 的子目录，`harbor run` 在容器里让 agent 做题，
再由题目自带的验证脚本打分，结果写在 <jobs>/<job>/<trial>/result.json（verifier_result.rewards.reward），
agent 的过程写在同一目录的 agent/trajectory.json（ATIF 格式：steps[].source / message / tool_calls / observation）。

本模块提供三个组件：

  RSIHubTaskEnv      读 Harbor 数据集，按 RSIHub 的规则（splits._assign）划分 train / gate / sealed，
                     映射成 AnyRSI 的 train / holdout / final。划分和 RSIHub 配方里同样的比例、种子完全一致
  HarborAgent        对一道题调用一次 `harbor run`，把 result.json 和 agent/trajectory.json 转成 AnyRSI 轨迹。
                     被进化的 agent 代码（harness 仓库）通过 PYTHONPATH 交给 Harbor；上下文里有权重条目时，
                     先用新 checkpoint 启动或重启推理服务（和 ChatAgent 一样），命令模板里可用 {model}、{base_url}
  HarborRewardJudge  按验证脚本的奖励打分。它自己重新读 result.json 并核对哈希，不信 agent 转述的分数

和 RSIHub 原样跑的区别：RSIHub 一次评一整批题、由它的 driver 决定改什么；这里每道题单独调一次 Harbor，
改什么由 AnyRSI 的提案器和算法决定，所以可以把"改代码"和"改权重"放进同一个循环。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from anyrsi import (Agent, ComponentError, Environment, Evidence, Item, Judge, RubricResult, Step, Task, Trajectory,
                     Verdict)

SPLIT_NAMES = ("train", "gate", "sealed")
TO_ANYRSI = {"train": "train", "gate": "holdout", "sealed": "final"}
# RSIHub 对齐实验（benchmarks/rsihub_parity）用的划分：Terminal-Bench 2 的 89 题分成 50 / 19 / 20
PARITY_SPLIT = {"train": 0.5618, "gate": 0.2135, "sealed": 0.2247}


# ---------------------------------------------------------------- 划分（照抄 RSIHub splits._assign）


def assign_splits(names: list[str], ratios: dict[str, float], seed: int = 0) -> dict[str, list[str]]:
    """按 sha256(f"{seed}\\0{name}") 排序，按比例取整，余数给小数部分最大的划分（同分按 train、gate、sealed 的顺序）。"""
    if sorted(ratios) != sorted(SPLIT_NAMES):
        raise ValueError(f"划分比例要正好给 {', '.join(SPLIT_NAMES)}，收到 {sorted(ratios)}")
    if any(not isinstance(v, (int, float)) or v < 0 for v in ratios.values()) or not sum(ratios.values()):
        raise ValueError(f"划分比例必须是非负数且不全为 0：{ratios}")
    total = float(sum(ratios.values()))
    ratios = {k: float(v) / total for k, v in ratios.items()}
    ordered = sorted(names, key=lambda n: hashlib.sha256(f"{seed}\0{n}".encode()).hexdigest())
    raw = {k: len(ordered) * ratios[k] for k in SPLIT_NAMES}
    counts = {k: math.floor(raw[k]) for k in SPLIT_NAMES}
    remainder = len(ordered) - sum(counts.values())
    for k in sorted(SPLIT_NAMES, key=lambda k: (-(raw[k] - counts[k]), SPLIT_NAMES.index(k)))[:remainder]:
        counts[k] += 1
    out, offset = {}, 0
    for k in SPLIT_NAMES:
        out[k] = sorted(ordered[offset:offset + counts[k]])
        offset += counts[k]
    return out


def discover_task_names(dataset: Path) -> list[str]:
    """数据集目录下带 task.toml 的子目录名（和 RSIHub 一样；目录本身就是一道题时返回它自己）。"""
    if (dataset / "task.toml").is_file():
        return [dataset.name]
    return sorted(p.name for p in dataset.iterdir() if p.is_dir() and (p / "task.toml").is_file())


def task_digest(task_dir: Path) -> str:
    """题目目录内容的哈希（相对路径 + 文件内容）。用来确认几次运行用的是同一份题，不和 RSIHub 的摘要通用。"""
    h = hashlib.sha256()
    for p in sorted(task_dir.rglob("*"), key=lambda x: x.relative_to(task_dir).as_posix()):
        if p.is_file() and not p.is_symlink():
            h.update(p.relative_to(task_dir).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


class RSIHubTaskEnv(Environment):
    """with:
    dataset          Harbor 数据集目录（RSIHub 的 setup 脚本下载到 .evolve-assets/<名字>/raw/<名字>）
    split            {train, gate, sealed} 的比例，默认和 RSIHub 对齐实验一样（89 题 → 50 / 19 / 20）
    seed             划分种子，默认 0
    train_per_round  每代练习多少道 train 题（按代号轮换），不给就每代全部
    include          可选：只用这些题（名字列表），在划分之前筛选
    """

    name = "anyrsi/rsihub-task-env"
    version = "0.4.2"

    def __init__(self, dataset: str, split: dict[str, float] | None = None, seed: int = 0,
                 train_per_round: int | None = None, include: list[str] | None = None):
        self.dataset = Path(dataset).expanduser().resolve()
        if not self.dataset.is_dir():
            raise ComponentError(f"找不到 Harbor 数据集目录 {self.dataset}")
        names = discover_task_names(self.dataset)
        if include is not None:
            missing = sorted(set(include) - set(names))
            if missing:
                raise ComponentError(f"include 里的题不在数据集中：{missing[:5]}")
            names = [n for n in names if n in set(include)]
        if not names:
            raise ComponentError(f"{self.dataset} 下没有带 task.toml 的题目")
        self.seed, self.train_per_round = seed, train_per_round
        self.ratios = dict(split or PARITY_SPLIT)
        self.splits = assign_splits(names, self.ratios, seed)
        self.root = self.dataset if not (self.dataset / "task.toml").is_file() else self.dataset.parent

    def _task(self, name: str, rsihub_split: str) -> Task:
        task_dir = self.root / name
        instruction = task_dir / "instruction.md"
        return Task(id=name, split=TO_ANYRSI[rsihub_split], tags=[f"rsihub:{rsihub_split}"],
                    input={"harbor_task": name,
                           "instruction": instruction.read_text(encoding="utf-8") if instruction.is_file() else None},
                    metadata={"dataset": str(self.root), "task_digest": task_digest(task_dir),
                              "rsihub_split": rsihub_split, "split_seed": self.seed})

    def tasks(self, split: str, round: int = 0, limit: int | None = None, cell: str | None = None) -> list[Task]:
        if cell is not None:
            raise ComponentError("RSIHubTaskEnv 没有环境格")
        by_anyrsi = {v: k for k, v in TO_ANYRSI.items()}
        if split not in by_anyrsi:
            return []  # anchor：RSIHub 的题库没有人工标注集
        names = list(self.splits[by_anyrsi[split]])
        if split == "train" and self.train_per_round:
            names = sorted(names, key=lambda n: hashlib.sha256(f"{self.seed}\0{round}\0{n}".encode()).hexdigest())
            names = sorted(names[: self.train_per_round])
        if limit is not None:
            names = names[:limit]
        return [self._task(n, by_anyrsi[split]) for n in names]


# ---------------------------------------------------------------- Harbor 结果解析


def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):  # 多段消息
        return "\n".join(filter(None, (_text(i.get("text") if isinstance(i, dict) else i) for i in x)))
    return json.dumps(x, ensure_ascii=False)


def atif_steps(trajectory: dict[str, Any]) -> tuple[list[Step], str]:
    """ATIF 的 steps → AnyRSI 的 Step 列表，外加 agent 的最后一条消息（作为轨迹输出）。"""
    steps: list[Step] = []
    last = ""
    for i, s in enumerate(trajectory.get("steps") or []):
        if not isinstance(s, dict):
            continue
        source = s.get("source")
        role = {"agent": "assistant", "user": "user", "system": "system"}.get(str(source))
        message = _text(s.get("message"))
        if role and message:
            steps.append(Step(id=f"s{i}", type="message", role=role, content=message))
            if role == "assistant":
                last = message
        for j, call in enumerate(s.get("tool_calls") or []):
            if isinstance(call, dict):
                args = call.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                steps.append(Step(id=f"s{i}c{j}", type="tool_call", name=str(call.get("function_name") or call.get("name")
                                                                                 or "unknown"),
                                  call_id=str(call.get("tool_call_id") or f"s{i}c{j}"),
                                  arguments=args if isinstance(args, dict) else {"value": args}))
        obs = s.get("observation")
        for j, r in enumerate((obs.get("results") if isinstance(obs, dict) else None) or []):
            if isinstance(r, dict):
                steps.append(Step(id=f"s{i}r{j}", type="tool_result", call_id=str(r.get("source_call_id") or ""),
                                  content=_text(r.get("content"))))
    return steps, last


def read_reward(result: dict[str, Any]) -> float | None:
    """和 RSIHub harbor_artifacts._reward 一样：verifier_result.rewards.reward，必须是有限数字。"""
    reward = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        return None
    return float(reward)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_trial(jobs_dir: Path, task_name: str) -> Path | None:
    """jobs 目录里这道题的 result.json（只取 task_name 对得上的；有多个时取最后写的）。"""
    found = []
    for path in jobs_dir.rglob("result.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("task_name") == task_name and data.get("trial_name"):
            found.append(path)
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


# ---------------------------------------------------------------- agent


DEFAULT_COMMAND = ["harbor", "run", "-p", "{dataset}", "--include-task-name", "{task}", "--job-name", "{job}",
                   "--jobs-dir", "{jobs_dir}", "--n-attempts", "1", "-n", "1", "-y", "-q"]


class HarborAgent(Agent):
    """with:
    agent_args   选 agent 的参数，比如 ["--agent-import-path", "my_agent:Agent"]（从 repo 里导入，被进化的代码）
                 或 ["--agent", "mini-swe-agent", "--model", "openai/policy"]
    repo         被进化的 agent 代码目录（HarnessEvolvable 的仓库）；加到 Harbor 进程的 PYTHONPATH 最前面
    command      Harbor 命令模板，默认 DEFAULT_COMMAND；可用 {dataset} {task} {job} {jobs_dir} {repo} {model} {base_url}
    extra_args   追加在命令后面的参数（同样可用占位符），比如 ["--ae", "OPENAI_BASE_URL={base_url}"]
    env          传给 Harbor 的环境变量名（比如 OPENAI_API_KEY），其余不传（PATH、HOME、DOCKER_HOST 总是传）
    workdir      Harbor 的 jobs 目录放哪（默认 .anyrsi/harbor-jobs）
    timeout      每道题的超时（秒）
    server       可选：本地推理服务（和 ChatAgent 的 server 一样：command、port、startup_timeout、log_dir）。
                 上下文里有权重条目时用它的模型路径启动，权重变了就重启；没有权重条目时用 model
    model        没有权重条目时 {model} 的值
    """

    name = "anyrsi/harbor-agent"
    version = "0.4.2"

    def __init__(self, agent_args: list[str], repo: str | None = None, command: list[str] | None = None,
                 extra_args: list[str] | None = None, env: list[str] | None = None,
                 workdir: str = ".anyrsi/harbor-jobs", timeout: float = 3600.0,
                 server: dict[str, Any] | None = None, model: str = "policy"):
        self.agent_args = list(agent_args)
        self.repo = Path(repo).resolve() if repo else None
        self.command = list(command or DEFAULT_COMMAND)
        self.extra_args = list(extra_args or [])
        self.env_names = list(env or [])
        self.workdir = Path(workdir).resolve()
        self.cwd = Path.cwd()  # 命令里的相对路径以构造时的目录（anyrsi.yaml 所在目录）为准
        self.timeout, self.model = timeout, model
        self.server = None
        if server:
            from anyrsi_llm import ManagedServer  # 需要 anyrsi-llm

            self.server = ManagedServer(**server)

    def _fields(self, task: Task, job: str, model: str) -> dict[str, str]:
        dataset = (task.metadata or {}).get("dataset")
        if not dataset:
            raise ComponentError(f"题目 {task.id} 没有 metadata.dataset（要配 RSIHubTaskEnv）")
        harbor_task = task.input.get("harbor_task", task.id) if isinstance(task.input, dict) else task.id
        return {"dataset": str(dataset), "task": str(harbor_task), "job": job, "jobs_dir": str(self.workdir),
                "repo": str(self.repo or ""), "model": model,
                "base_url": self.server.base_url if self.server else os.environ.get("OPENAI_BASE_URL", "")}

    def _env(self) -> dict[str, str]:
        env = {k: os.environ[k] for k in ("PATH", "HOME", "DOCKER_HOST", "TMPDIR") if k in os.environ}
        env.update({n: os.environ[n] for n in self.env_names if n in os.environ})
        if self.repo:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(self.repo), os.environ.get("PYTHONPATH", "")]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # 不在 harness 仓库里留 __pycache__
        return env

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        weights = next((i for i in context if i.kind == "weights"), None)
        model = weights.text if weights else self.model
        if weights is not None and self.server is not None:
            self.server.ensure(weights.text)
        job = f"{task.id}-{(snapshot_id or 'none')[:24]}-{uuid.uuid4().hex[:8]}".replace("/", "_")
        try:
            fields = self._fields(task, job, model)
        except ComponentError as e:
            return self._failed(task, snapshot_id, str(e))
        argv = [p.format(**fields) for p in [*self.command, *self.agent_args, *self.extra_args]]
        if argv and argv[0] in ("python", "python3"):
            argv[0] = sys.executable
        self.workdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout, env=self._env(),
                                  cwd=self.cwd)
        except subprocess.TimeoutExpired:
            return self._failed(task, snapshot_id, f"harbor 超时（{self.timeout}s）")
        except FileNotFoundError:
            return self._failed(task, snapshot_id, f"找不到命令 {argv[0]}（装了 harbor 吗？）")
        result_path = find_trial(self.workdir / job, fields["task"])
        if result_path is None:
            tail = (proc.stderr.strip().splitlines() or [f"退出码 {proc.returncode}"])[-1]
            return self._failed(task, snapshot_id, f"harbor 没有产出这道题的 result.json：{tail}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        steps, final = [], ""
        atif = result_path.parent / "agent" / "trajectory.json"
        if atif.is_file():
            try:
                steps, final = atif_steps(json.loads(atif.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, ValueError):
                steps = []
        exc = result.get("exception_info") if isinstance(result.get("exception_info"), dict) else None
        if exc:
            steps.append(Step(id="harbor-exception", type="observation", is_error=True,
                              content=f"{exc.get('exception_type')}: {str(exc.get('exception_message'))[:500]}"))
        agent_result = result.get("agent_result") if isinstance(result.get("agent_result"), dict) else {}
        harbor = {"result_path": str(result_path), "result_sha256": file_sha256(result_path),
                  "reward": read_reward(result), "trial_name": result.get("trial_name"), "job": job,
                  "exception_type": exc.get("exception_type") if exc else None,
                  "cost_usd": agent_result.get("cost_usd"), "n_input_tokens": agent_result.get("n_input_tokens"),
                  "n_output_tokens": agent_result.get("n_output_tokens"), "returncode": proc.returncode,
                  "task_digest": (task.metadata or {}).get("task_digest")}
        return Trajectory(id=f"traj-{job}", task_id=task.id, snapshot_id=snapshot_id, steps=steps, output=final,
                          used_items=[i.id for i in context if i.kind != "weights"],
                          metadata={"harbor": harbor, "model": model, "duration_s": round(time.monotonic() - start, 3)})

    @staticmethod
    def _failed(task: Task, snapshot_id: str | None, why: str) -> Trajectory:
        return Trajectory(id=f"traj-{task.id}-{snapshot_id}-error", task_id=task.id, snapshot_id=snapshot_id,
                          steps=[Step(id="s0", type="observation", content=why, is_error=True)], output=None,
                          metadata={"error": why})


# ---------------------------------------------------------------- judge


class HarborRewardJudge(Judge):
    """按 Harbor 验证脚本的奖励打分：score = 奖励（截到 0–1），奖励 ≥ pass_threshold 算通过。

    自己重新读 result.json：文件不在、哈希和 agent 记下的不一致、或者 task_name 对不上，都判失败。
    没有奖励（验证没跑完、agent 出错）也判失败，原因写进 rubric 的 note。
    """

    name = "anyrsi/harbor-reward-judge"
    version = "0.4.2"

    def __init__(self, pass_threshold: float = 1.0):
        self.pass_threshold = pass_threshold

    def _fail(self, task: Task, trajectory: Trajectory, why: str) -> Verdict:
        ev = Evidence.make("harbor_result", {"error": why, "task": task.id}, f"trajectory:{trajectory.id}", self.ref)
        return Verdict(passed=False, score=0.0, judge_ref=self.ref, evidence_ids=[ev.id], collected=[ev],
                       rubric=[RubricResult(id="verifier_reward", result="fail", required=True, evidence_ids=[ev.id],
                                            note=why)], notes=[why])

    def evaluate(self, task: Task, trajectory: Trajectory, evidence: list[Evidence] | None = None) -> Verdict:
        harbor = (trajectory.metadata or {}).get("harbor")
        if not isinstance(harbor, dict):
            return self._fail(task, trajectory, (trajectory.metadata or {}).get("error") or "没有 Harbor 结果")
        path = Path(str(harbor.get("result_path", "")))
        if not path.is_file():
            return self._fail(task, trajectory, f"result.json 不在了：{path}")
        if file_sha256(path) != harbor.get("result_sha256"):
            return self._fail(task, trajectory, "result.json 在 agent 记录之后被改过")
        result = json.loads(path.read_text(encoding="utf-8"))
        expected = task.input.get("harbor_task", task.id) if isinstance(task.input, dict) else task.id
        if result.get("task_name") != expected:
            return self._fail(task, trajectory, f"result.json 是 {result.get('task_name')} 的，不是 {expected}")
        reward = read_reward(result)
        ev = Evidence.make("harbor_result", {"task_name": result.get("task_name"), "trial_name": result.get("trial_name"),
                                             "reward": reward, "exception_info": result.get("exception_info")},
                           f"file:{PurePosixPath(*path.parts[-3:])}", self.ref, uri=path.as_uri())
        if reward is None:
            why = "验证脚本没有给出奖励"
            exc = result.get("exception_info")
            if isinstance(exc, dict) and exc.get("exception_type"):
                why += f"（{exc['exception_type']}）"
            return Verdict(passed=False, score=0.0, judge_ref=self.ref, evidence_ids=[ev.id], collected=[ev],
                           rubric=[RubricResult(id="verifier_reward", result="fail", required=True, evidence_ids=[ev.id],
                                                note=why)], notes=[why])
        passed = reward >= self.pass_threshold
        note = f"验证奖励 {reward:g}（通过线 {self.pass_threshold:g}）"
        return Verdict(passed=passed, score=max(0.0, min(1.0, reward)), judge_ref=self.ref, evidence_ids=[ev.id],
                       collected=[ev], rubric=[RubricResult(id="verifier_reward", result="pass" if passed else "fail",
                                                            required=True, evidence_ids=[ev.id], note=note)],
                       notes=[note])


__all__ = ["DEFAULT_COMMAND", "HarborAgent", "HarborRewardJudge", "PARITY_SPLIT", "RSIHubTaskEnv", "assign_splits",
           "atif_steps", "discover_task_names", "find_trial", "read_reward", "task_digest"]
