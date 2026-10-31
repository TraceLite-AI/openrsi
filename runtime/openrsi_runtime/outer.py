"""外循环：外围组件的受控升级（two-loop-evolution 设计 §4.4、§7，P4）。默认关闭，要显式创建 OuterLoop 才会用。

    propose   提出升级：目标类型必须在白名单里；候选和验证计划在这一步冻结（各算一个哈希）
    validate  在冻结的资格验证集上，由固定的评估器打分；计划和资格集必须和提案时一模一样
    promote   必须有通过的验证 + 明确的批准人，才把活动指针切到候选；失败或崩溃时指针不动
    rollback  把活动指针切回上一版；所有记录保留

白名单首期只有 judge（判别器 / rubric）和 attribution（归因方法）。gate、guard、封存考试、人工标签、
预算、控制面都不能作为目标（不动点）。

judge 的验证：候选只做判决（evaluate），看不到人工标签，也不调用它自己的 calibrate；固定评估器拿判决对照
独立标签算一致率、误放率、误拒率。误放率是硬安全项：不能比当前版本高。
attribution 的验证：留出的干预样本，每个样本知道真正的原因（kind + ref）；看候选把最大责任放在原因上的比例。

验证计划（阈值、样本量、资格集哈希）在提案时锁定，看完结果再改会被拒绝。
所有提案、验证、升级、回滚都按内容寻址写进 RecordStore，只追加。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from openrsi import AnchorExample, Attribution, Episode, content_hash
from openrsi.types import now_iso

from .records import RecordStore

TargetKind = Literal["judge", "attribution"]
ALLOWED: tuple[str, ...] = ("judge", "attribution")


class OuterError(RuntimeError):
    """提案、验证或升级不合规。"""


class ValidationPlan(BaseModel):
    """事先登记的验证计划。提案时冻结，之后改任何一项都会让验证被拒。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qualification_hash: str = Field(description="资格验证集的哈希")
    min_examples: int = Field(default=20, ge=1)
    min_gain: float = Field(default=0.0, description="主指标至少要比当前版本高多少")
    max_false_pass_increase: float = Field(default=0.0, description="judge：误放率最多比当前版本高多少（硬安全项）")
    max_false_reject: float | None = Field(default=None, description="judge：误拒率上限")

    @property
    def plan_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class OuterProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    target_kind: TargetKind
    base_ref: str
    candidate_ref: str
    candidate_hash: str = Field(description="候选的冻结内容（配置、rubric）的哈希")
    hypothesis: str
    expected_effect: str = ""
    source_record_hashes: list[str] = Field(default_factory=list, description="提出这个升级所依据的记录")
    validation_plan_hash: str
    created_at: str = Field(default_factory=now_iso)


class OuterValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    validation_plan_hash: str
    evaluator_ref: str
    base_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    checks: dict[str, bool]
    accepted: bool
    detail: str = ""
    created_at: str = Field(default_factory=now_iso)


class PromotionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    validation_hash: str | None
    approver: str
    old_ref: str
    new_ref: str
    status: Literal["promoted", "rolled_back"]
    rollback_ref: str
    created_at: str = Field(default_factory=now_iso)


def qualification_hash(examples: list[Any]) -> str:
    return content_hash([e.model_dump(mode="json") if isinstance(e, BaseModel) else e for e in examples])


class AttributionFixture(BaseModel):
    """留出的干预样本：已知真正原因的一组 episode。"""

    episodes: list[Episode]
    culprit_kind: str
    culprit_ref: str


def _improves(base: float, cand: float, min_gain: float) -> bool:
    """必须真的变好：严格高于当前版本，且提升不小于计划登记的 min_gain。"""
    return cand > base and cand - base >= min_gain


# ---------------------------------------------------------------- 固定评估器

class FixedEvaluator:
    """不参与进化的评估器。只调用候选的 evaluate / explain，自己对照标签算指标。"""

    ref = "openrsi/fixed-evaluator:1"

    def judge_metrics(self, judge: Any, qualification: list[AnchorExample]) -> dict[str, float]:
        n = len(qualification)
        agree = false_pass = false_reject = 0
        neg = sum(1 for ex in qualification if not ex.label_passed) or 1
        pos = sum(1 for ex in qualification if ex.label_passed) or 1
        for ex in qualification:
            passed = bool(judge.evaluate(ex.task, ex.trajectory, list(ex.evidence)).passed)  # 不给标签
            if passed == ex.label_passed:
                agree += 1
            elif passed:
                false_pass += 1
            else:
                false_reject += 1
        return {"agreement": agree / n if n else 0.0, "false_pass_rate": false_pass / neg,
                "false_reject_rate": false_reject / pos, "n": float(n)}

    def attribution_metrics(self, attributor: Any, fixtures: list[AttributionFixture]) -> dict[str, float]:
        hits = 0
        for fx in fixtures:
            blame: dict[tuple[str, str], float] = {}
            for a in attributor.explain(fx.episodes):
                for t in Attribution.model_validate(a).targets:
                    if t.weight < 0:
                        blame[(t.kind, t.ref)] = blame.get((t.kind, t.ref), 0.0) - t.weight
            if blame and max(blame, key=lambda k: (blame[k], k)) == (fx.culprit_kind, fx.culprit_ref):
                hits += 1
        n = len(fixtures)
        return {"localization": hits / n if n else 0.0, "n": float(n)}


# ---------------------------------------------------------------- 外循环

class OuterLoop:
    def __init__(self, path: str | Path, evaluator: FixedEvaluator | None = None):
        self.root = Path(path)
        self.records = RecordStore(self.root / "records")
        self.evaluator = evaluator or FixedEvaluator()
        self._active = self.root / "active.json"
        self._plans: dict[str, ValidationPlan] = {}

    # 活动指针：每种目标当前用哪个版本，以及升级前是哪个（回滚用）
    def active(self) -> dict[str, dict[str, str]]:
        return json.loads(self._active.read_text(encoding="utf-8")) if self._active.exists() else {}

    def _set_active(self, data: dict[str, dict[str, str]]) -> None:
        tmp = self._active.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(self._active)

    def propose(self, target_kind: str, base_ref: str, candidate_ref: str, candidate_content: Any,
                hypothesis: str, plan: ValidationPlan, *, expected_effect: str = "",
                source_record_hashes: list[str] | None = None) -> OuterProposal:
        if target_kind not in ALLOWED:
            raise OuterError(f"{target_kind} 不能作为外循环的目标；白名单：{list(ALLOWED)}（gate、guard、封存考试、"
                             "人工标签、预算都是不动点）")
        for h in source_record_hashes or []:
            self.records.get(h)  # 依据的记录必须存在且没被改过
        body = {"target_kind": target_kind, "base_ref": base_ref, "candidate_ref": candidate_ref,
                "candidate_hash": content_hash(candidate_content), "hypothesis": hypothesis,
                "plan": plan.plan_hash}
        proposal = OuterProposal(proposal_id="op_" + content_hash(body)[:16], target_kind=target_kind,
                                 base_ref=base_ref, candidate_ref=candidate_ref,
                                 candidate_hash=content_hash(candidate_content), hypothesis=hypothesis,
                                 expected_effect=expected_effect, source_record_hashes=list(source_record_hashes or []),
                                 validation_plan_hash=plan.plan_hash)
        self.records.put(plan, "outer_plan")
        self.records.put(proposal, "outer_proposal")
        self._plans[proposal.proposal_id] = plan
        return proposal

    def validate(self, proposal: OuterProposal, plan: ValidationPlan, base: Any, candidate: Any,
                 candidate_content: Any, qualification: list[Any]) -> OuterValidation:
        if plan.plan_hash != proposal.validation_plan_hash:
            raise OuterError("验证计划和提案时登记的不一样：阈值不能看完结果再改")
        if content_hash(candidate_content) != proposal.candidate_hash:
            raise OuterError("候选和提案时冻结的内容不一样")
        if qualification_hash(qualification) != plan.qualification_hash:
            raise OuterError("资格验证集和计划登记的不一样：候选不能挑选、修改或删减验证集")
        if len(qualification) < plan.min_examples:
            raise OuterError(f"资格验证集只有 {len(qualification)} 条，计划要求至少 {plan.min_examples} 条")
        if proposal.target_kind == "judge":
            b = self.evaluator.judge_metrics(base, qualification)
            c = self.evaluator.judge_metrics(candidate, qualification)
            checks = {"improvement": _improves(b["agreement"], c["agreement"], plan.min_gain),
                      "no_more_false_pass": c["false_pass_rate"] <= b["false_pass_rate"] + plan.max_false_pass_increase}
            if plan.max_false_reject is not None:
                checks["false_reject_cap"] = c["false_reject_rate"] <= plan.max_false_reject
            detail = f"一致率 {b['agreement']:.3f} → {c['agreement']:.3f}，误放率 {b['false_pass_rate']:.3f} → " \
                     f"{c['false_pass_rate']:.3f}"
        else:
            b = self.evaluator.attribution_metrics(base, qualification)
            c = self.evaluator.attribution_metrics(candidate, qualification)
            checks = {"improvement": _improves(b["localization"], c["localization"], plan.min_gain)}
            detail = f"定位正确率 {b['localization']:.3f} → {c['localization']:.3f}"
        v = OuterValidation(proposal_id=proposal.proposal_id, validation_plan_hash=plan.plan_hash,
                            evaluator_ref=self.evaluator.ref, base_metrics=b, candidate_metrics=c, checks=checks,
                            accepted=all(checks.values()), detail=detail)
        self.records.put(v, "outer_validation")
        return v

    def promote(self, proposal: OuterProposal, validation: OuterValidation, approver: str) -> PromotionRecord:
        if not approver.strip():
            raise OuterError("升级必须有明确的批准人；自动化只到生成验证报告为止")
        if validation.proposal_id != proposal.proposal_id or not validation.accepted:
            raise OuterError("没有通过验证的提案不能升级")
        state = self.active()
        cur = state.get(proposal.target_kind, {}).get("ref", proposal.base_ref)
        if cur != proposal.base_ref:
            raise OuterError(f"当前活动版本是 {cur}，不是提案基于的 {proposal.base_ref}；请基于当前版本重新提案")
        rec = PromotionRecord(proposal_id=proposal.proposal_id, validation_hash=content_hash(validation.model_dump(
            mode="json")), approver=approver, old_ref=cur, new_ref=proposal.candidate_ref, status="promoted",
            rollback_ref=cur)
        self.records.put(rec, "outer_promotion")
        state[proposal.target_kind] = {"ref": proposal.candidate_ref, "previous": cur, "proposal": proposal.proposal_id}
        self._set_active(state)
        return rec

    def rollback(self, target_kind: str, approver: str) -> PromotionRecord:
        state = self.active()
        cur = state.get(target_kind)
        if not cur or not cur.get("previous"):
            raise OuterError(f"{target_kind} 没有可以回滚的上一版")
        rec = PromotionRecord(proposal_id=cur.get("proposal", ""), validation_hash=None, approver=approver,
                              old_ref=cur["ref"], new_ref=cur["previous"], status="rolled_back",
                              rollback_ref=cur["ref"])
        self.records.put(rec, "outer_promotion")
        state[target_kind] = {"ref": cur["previous"], "previous": "", "proposal": ""}
        self._set_active(state)
        return rec

    def history(self) -> list[dict[str, Any]]:
        return [{"kind": r["kind"], **self.records.get(r["hash"])} for r in self.records.entries()
                if r["kind"].startswith("outer_")]


__all__ = ["ALLOWED", "AttributionFixture", "FixedEvaluator", "OuterError", "OuterLoop", "OuterProposal",
           "OuterValidation", "PromotionRecord", "ValidationPlan", "qualification_hash"]
