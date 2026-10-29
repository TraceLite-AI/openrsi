"""harness 可进化体：agent 本身的代码（工具、流程编排、实现）放在一个 git 仓库里进化。

- HarnessEvolvable：快照就是 git 提交；补丁是文件写入、删除或 unified diff，只能改声明的可改范围（mutable）；
  受保护的路径（protected）任何时候都不能改。回滚是切换提交，历史保存在 refs/openrsi/ 下不会丢。
- HarnessAgent：把仓库当前版本里的 agent 当子进程运行。题目和上下文从 stdin 以 JSON 传入，
  轨迹从 stdout 以 JSON 返回。可选在 Docker 容器里跑。

运行时在给某个版本打分前会把所有可进化体切到那个版本，所以 HarnessAgent 用的总是被评估的那份代码。
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

from openrsi import (Agent, ComponentError, Diff, Evolvable, EvolvableInfo, Item, Patch, Snapshot,
                     StaleSnapshotError, Step, Task, Trajectory, serve)
from openrsi.types import now_iso

REF_PREFIX = "refs/openrsi/snapshots/"


def _match(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) or (g.endswith("/**") and path.startswith(g[:-2])) for g in globs)


def _clean_path(raw: str) -> str:
    p = PurePosixPath(raw)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] == ".git":
        raise ComponentError(f"不允许的路径：{raw}")
    return p.as_posix()


class HarnessEvolvable(Evolvable):
    """with:
    repo        仓库目录；不存在就新建（从 template 复制初始文件）
    template    可选，初始文件所在目录
    mutable     可改范围（glob 列表），如 ["agent/**", "prompts/*.md"]；默认全部
    protected   任何时候都不能改的路径，如 ["eval/**", "tests/**"]
    export_max_bytes  导出给提案器时单个文件的大小上限
    """

    name = "openrsi/harness"
    version = "0.4.0"

    def __init__(self, repo: str | Path, target: str = "harness", template: str | Path | None = None,
                 mutable: list[str] | None = None, protected: list[str] | None = None,
                 export_max_bytes: int = 100_000):
        self.repo = Path(repo).resolve()
        self.target = target
        self.mutable = list(mutable or ["**"])
        self.protected = list(protected or [])
        self.export_max_bytes = export_max_bytes
        if not (self.repo / ".git").exists():
            self.repo.mkdir(parents=True, exist_ok=True)
            if template:
                shutil.copytree(Path(template).resolve(), self.repo, dirs_exist_ok=True)
            self._git("init", "-q")
            self._config()
            self._git("add", "-A")
            self._git("commit", "-q", "--allow-empty", "-m", "openrsi: initial harness")
        else:
            self._config()
        self._keep(self._head())

    # ------------------------------------------------------------ git

    def _git(self, *args: str, input: str | None = None) -> str:
        proc = subprocess.run(["git", "-C", str(self.repo), *args], input=input, text=True, capture_output=True)
        if proc.returncode != 0:
            raise ComponentError(f"git {' '.join(args[:2])} 失败：{proc.stderr.strip().splitlines()[-1:] or proc.returncode}")
        return proc.stdout

    def _config(self) -> None:
        self._git("config", "user.name", "openrsi")
        self._git("config", "user.email", "openrsi@localhost")

    def _head(self) -> str:
        return self._git("rev-parse", "HEAD").strip()

    def _resolve(self, snapshot_id: str) -> str:
        try:
            return self._git("rev-parse", "--verify", "-q", f"{snapshot_id}^{{commit}}").strip()
        except ComponentError as e:
            raise KeyError(f"没有快照 {snapshot_id}") from e

    def _keep(self, sha: str) -> None:
        self._git("update-ref", REF_PREFIX + sha, sha)

    def _dirty(self) -> bool:
        """只看已跟踪文件的修改；运行 agent 产生的缓存、日志等未跟踪文件不算。"""
        return bool(self._git("status", "--porcelain", "--untracked-files=no").strip())

    def _allowed(self, path: str) -> None:
        if _match(path, self.protected):
            raise ComponentError(f"{path} 受保护，不能修改")
        if not _match(path, self.mutable):
            raise ComponentError(f"{path} 不在可改范围 {self.mutable} 内")

    @staticmethod
    def _sid(sha: str) -> str:
        return sha[:12]

    # ------------------------------------------------------------ Evolvable

    def describe(self) -> EvolvableInfo:
        files = [f for f in self._git("ls-files").splitlines() if _match(f, self.mutable)]
        return EvolvableInfo(target=self.target, type="harness", head=self._sid(self._head()), surfaces=self.mutable,
                             size=len(files), example_ops=[{"op": "write", "path": self._probe_path(),
                                                            "content": "openrsi conformance probe\n"}])

    def _probe_path(self) -> str:
        for g in self.mutable:
            if g in ("**", "*"):
                return "__openrsi_probe__.txt"
            if g.endswith("/**"):
                return g[:-3] + "/__openrsi_probe__.txt"
        return "__openrsi_probe__.txt"

    def snapshot(self) -> Snapshot:
        return self._snapshot_of(self._head())

    def _snapshot_of(self, sha: str) -> Snapshot:
        fmt = self._git("show", "-s", "--format=%P%n%T%n%cI%n%s", sha).splitlines()
        parents, tree, created, subject = (fmt + ["", "", "", ""])[:4]
        return Snapshot(id=self._sid(sha), target=self.target, parent_ids=[self._sid(p) for p in parents.split()],
                        content_hash=tree, created_at=created or now_iso(), message=subject or None)

    def apply(self, patch: Patch) -> Snapshot:
        if patch.target != self.target:
            raise ValueError(f"补丁目标是 {patch.target}，本组件是 {self.target}")
        head = self._head()
        if patch.base_snapshot and self._resolve(patch.base_snapshot) != head:
            raise StaleSnapshotError(f"补丁基于 {patch.base_snapshot}，当前是 {self._sid(head)}")
        if self._dirty():
            raise ComponentError("harness 仓库有未提交的修改，先清理再进化")

        # 先检查全部 op，再动文件
        touched: list[str] = []
        for op in patch.ops:
            kind = op.get("op")
            if kind in ("write", "delete"):
                path = _clean_path(str(op.get("path", "")))
                self._allowed(path)
                touched.append(path)
                if kind == "delete" and not (self.repo / path).exists():
                    raise ComponentError(f"要删除的 {path} 不存在")
                if kind == "write" and not isinstance(op.get("content"), str):
                    raise ComponentError(f"写入 {path} 需要字符串 content")
            elif kind == "diff":
                diff = str(op.get("unified", ""))
                for line in self._git("apply", "--numstat", "-", input=diff).splitlines():
                    path = _clean_path(line.split("\t")[-1])
                    self._allowed(path)
                    touched.append(path)
                self._git("apply", "--check", "-", input=diff)
            else:
                raise ComponentError(f"不支持的 op：{kind}（可用 write / delete / diff）")

        created = [t for t in touched if not (self.repo / t).exists()]
        try:
            for op in patch.ops:
                if op["op"] == "write":
                    f = self.repo / _clean_path(op["path"])
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(op["content"], encoding="utf-8")
                elif op["op"] == "delete":
                    (self.repo / _clean_path(op["path"])).unlink()
                else:
                    self._git("apply", "-", input=op["unified"])
            self._git("add", "-A", "--", *sorted(set(touched)))  # 只提交补丁碰过的文件
            if not self._git("diff", "--cached", "--name-only").strip():
                raise ComponentError("补丁没有改变任何文件")
            self._git("commit", "-q", "-m", patch.rationale or "openrsi: harness patch")
        except Exception:
            self._git("reset", "-q", "--hard", head)  # 开始时已跟踪的文件是干净的，可以放心还原
            for t in created:  # 补丁新建的文件 reset 管不到，单独删掉
                (self.repo / t).unlink(missing_ok=True)
            raise
        sha = self._head()
        self._keep(sha)
        return self._snapshot_of(sha)

    def rollback(self, snapshot_id: str) -> Snapshot:
        sha = self._resolve(snapshot_id)
        if self._dirty():
            raise ComponentError("harness 仓库有未提交的修改，无法切换版本")
        self._git("checkout", "-q", "--detach", sha)
        return self._snapshot_of(sha)

    def diff(self, from_id: str, to_id: str) -> Diff:
        a, b = self._resolve(from_id), self._resolve(to_id)
        ops: list[dict[str, Any]] = []
        for line in self._git("diff", "--name-status", "--no-renames", a, b).splitlines():
            status, path = line.split("\t", 1)
            if status.startswith("D"):
                ops.append({"op": "delete", "path": path})
            else:
                ops.append({"op": "write", "path": path, "content": self._git("show", f"{b}:{path}")})
        return Diff(from_id=from_id, to_id=to_id, ops=ops)

    def export(self, snapshot_id: str | None = None) -> list[dict[str, Any]]:
        """可改范围内的文本文件，按 Item 格式导出（id 是路径），供提案器阅读。"""
        sha = self._resolve(snapshot_id) if snapshot_id else self._head()
        out = []
        for path in sorted(self._git("ls-tree", "-r", "--name-only", sha).splitlines()):
            if not _match(path, self.mutable) or _match(path, self.protected):
                continue
            size = int(self._git("cat-file", "-s", f"{sha}:{path}").strip())
            if size > self.export_max_bytes:
                continue
            try:
                text = self._git("show", f"{sha}:{path}")
            except (ComponentError, UnicodeDecodeError):
                continue
            out.append(Item(id=path, kind="file", text=text, data={"path": path}).model_dump(mode="json"))
        return out


class HarnessAgent(Agent):
    """with:
    repo          harness 仓库目录（和 HarnessEvolvable 用同一个）
    command       在仓库里启动 agent 的命令，默认 ["python", "agent.py"]
    timeout       每道题的超时（秒）
    env           传给 agent 的环境变量名列表（比如 ANTHROPIC_API_KEY），其余环境变量不传
    docker_image  可选；给了就在容器里运行（仓库只读挂载到 /work）
    """

    name = "openrsi/harness-agent"
    version = "0.4.0"

    def __init__(self, repo: str | Path, command: list[str] | None = None, timeout: float = 300.0,
                 env: list[str] | None = None, docker_image: str | None = None):
        self.repo = Path(repo).resolve()
        self.command = list(command or ["python", "agent.py"])
        self.timeout = timeout
        self.env_names = list(env or [])
        self.docker_image = docker_image

    def _argv(self) -> list[str]:
        if not self.docker_image:
            cmd = list(self.command)
            if cmd and cmd[0] in ("python", "python3"):
                import sys
                cmd[0] = sys.executable
            return cmd
        flags = [x for n in self.env_names for x in ("-e", n)]
        return ["docker", "run", "--rm", "-i", "--network", "none" if not self.env_names else "bridge",
                "-v", f"{self.repo}:/work:ro", "-w", "/work", *flags, self.docker_image, *self.command]

    def run(self, task: Task, context: list[Item], snapshot_id: str | None = None) -> Trajectory:
        payload = json.dumps({"task": task.model_dump(mode="json"),
                              "context": [i.model_dump(mode="json") for i in context],
                              "snapshot_id": snapshot_id}, ensure_ascii=False)
        env = {n: os.environ[n] for n in self.env_names if n in os.environ}
        env.setdefault("PATH", os.environ.get("PATH", ""))
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # 不在 harness 仓库里留 __pycache__
        start = time.monotonic()
        try:
            proc = subprocess.run(self._argv(), input=payload, capture_output=True, text=True, timeout=self.timeout,
                                  cwd=self.repo, env=env)
        except subprocess.TimeoutExpired:
            return self._failed(task, snapshot_id, f"agent 超时（{self.timeout}s）")
        if proc.returncode != 0:
            tail = (proc.stderr.strip().splitlines() or [f"退出码 {proc.returncode}"])[-1]
            return self._failed(task, snapshot_id, f"agent 进程失败：{tail}")
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return self._failed(task, snapshot_id, "agent 的 stdout 不是 JSON")
        data.setdefault("id", f"traj-{task.id}-{snapshot_id}")
        data.setdefault("task_id", task.id)
        data.setdefault("snapshot_id", snapshot_id)
        data.setdefault("metadata", {})["duration_s"] = round(time.monotonic() - start, 3)
        try:
            return Trajectory.model_validate(data)
        except ValueError as e:
            return self._failed(task, snapshot_id, f"agent 输出的轨迹格式不对：{e}")

    @staticmethod
    def _failed(task: Task, snapshot_id: str | None, why: str) -> Trajectory:
        return Trajectory(id=f"traj-{task.id}-{snapshot_id}-error", task_id=task.id, snapshot_id=snapshot_id,
                          steps=[Step(id="s0", type="observation", content=why, is_error=True)], output=None,
                          metadata={"error": why})


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="openrsi-harness")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mutable", action="append")
    ap.add_argument("--protected", action="append")
    ap.add_argument("--template")
    args = ap.parse_args()
    serve(HarnessEvolvable(args.repo, template=args.template, mutable=args.mutable, protected=args.protected))


__all__ = ["HarnessAgent", "HarnessEvolvable", "main"]
