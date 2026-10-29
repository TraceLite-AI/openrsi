"""基于证据的 rubric 判别器。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from openrsi import EvidenceCollector, Evidence, Judge, RubricResult, Task, Trajectory, Verdict

from .collector import TrajectoryCollector
from .llm import Checker, ClaudeChecker, LLMCheckError
from .rubric import CHECKS, Rubric, RubricItem, load_rubric


class RubricJudge(Judge):
    """按 rubric 逐项判断。

    - 调用方给的证据先复核哈希，篡改过的不采信
    - 没给证据时，用自带的 collector 补收集，结果放在 verdict.collected 里，调用方可以复核
    - 分数 = 通过项权重 / 有结论项权重；必选项全过且分数达到阈值才算通过
    """

    name = "openrsi/rubric-judge"
    version = "0.3.0"

    def __init__(self, rubric: Rubric | dict | str | Path, *, collector: EvidenceCollector | None = None,
                 checker: Checker | None = None):
        if isinstance(rubric, Rubric):
            self.rubric = rubric
        elif isinstance(rubric, dict):  # 从 openrsi.yaml 里直接写的 rubric
            self.rubric = Rubric.model_validate(rubric)
        else:
            self.rubric = load_rubric(rubric)
        self.collector = collector
        self._checker = checker

    @property
    def checker(self) -> Checker:
        if self._checker is None:
            self._checker = ClaudeChecker()
        return self._checker

    def evaluate(self, task: Task, trajectory: Trajectory, evidence: list[Evidence]) -> Verdict:
        notes: list[str] = []
        trusted = [e for e in evidence if e.verify()]
        rejected = [e.id for e in evidence if not e.verify()]
        if rejected:
            notes.append(f"以下证据哈希不符，未采信：{', '.join(rejected)}")

        collected: list[Evidence] = []
        if not trusted and self.collector is not None:
            collected = self.collector.collect(task, trajectory)
            trusted = collected
            notes.append("调用方未提供证据，由判别器自行收集")

        by_type: dict[str, list[Evidence]] = defaultdict(list)
        for e in trusted:
            by_type[e.type].append(e)

        results = [self._run(item, task, trajectory, by_type, trusted) for item in self.rubric.items]

        decided = [r for r in results if r.result in ("pass", "fail")]
        total = sum(r.weight for r in decided)
        score = sum(r.weight for r in decided if r.result == "pass") / total if total else 0.0
        required_ok = all(r.result == "pass" for r in results if r.required)
        has_llm = any(i.check == "llm" for i in self.rubric.items)

        return Verdict(
            score=round(score, 6),
            passed=required_ok and bool(decided) and score >= self.rubric.pass_threshold,
            rubric=results,
            confidence=None if has_llm else 1.0,
            judge_ref=self.ref,
            evidence_ids=[e.id for e in trusted],
            collected=collected,
            notes=notes,
        )

    def _run(self, item: RubricItem, task: Task, traj: Trajectory, by_type, trusted: list[Evidence]) -> RubricResult:
        base = {"id": item.id, "weight": item.weight, "required": item.required}
        if item.check == "llm":
            types = item.params.get("evidence_types")
            pool = [e for e in trusted if not types or e.type in types]
            try:
                r = self.checker.check(item.params["question"], task, pool)
            except LLMCheckError as e:
                return RubricResult(**base, result="error", note=str(e))
            return RubricResult(**base, result="pass" if r.passed else "fail",
                                evidence_ids=r.cited_evidence_ids, note=r.reason)
        try:
            result, ids, note = CHECKS[item.check](item, task, traj, by_type)
        except Exception as e:  # noqa: BLE001  单项出错不影响其他项
            return RubricResult(**base, result="error", note=f"{type(e).__name__}: {e}")
        return RubricResult(**base, result=result, evidence_ids=ids, note=note)


class EvidenceJudge(TrajectoryCollector, RubricJudge):
    """取证 + 判别合在一个组件里：既能单独提供证据，也能拿自己的证据判别。"""

    name = "openrsi/evidence-judge"
    version = "0.3.0"

    def __init__(self, rubric: Rubric | dict | str | Path, *, checker: Checker | None = None, **collector_config):
        TrajectoryCollector.__init__(self, **collector_config)
        RubricJudge.__init__(self, rubric, collector=self, checker=checker)
