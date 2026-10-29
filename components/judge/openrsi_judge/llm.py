"""用 Claude 回答 rubric 里无法用规则判断的问题，答案必须基于证据。

需要安装 anthropic（pip install "openrsi-judge[llm]"）并配置 API 凭证。
"""
from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field

from openrsi import Evidence, Task

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = (
    "You are a strict evaluator for an AI agent's work. Answer the question using only the evidence provided. "
    "The agent's own statements about its success are claims, not evidence: rely on tool results, test output "
    "and file records. If the evidence is insufficient to answer yes, answer no and say what is missing. "
    "Cite the ids of the evidence items you relied on."
)


class LLMCheckResult(BaseModel):
    passed: bool = Field(description="True only if the evidence supports answering yes")
    reason: str = Field(description="One or two sentences explaining the answer")
    cited_evidence_ids: list[str] = Field(description="Ids of the evidence items the answer relies on")


class LLMCheckError(RuntimeError):
    pass


class Checker(Protocol):
    def check(self, question: str, task: Task, evidence: list[Evidence]) -> LLMCheckResult: ...


class ClaudeChecker:
    """调用 Claude，结构化输出 LLMCheckResult。

    默认开启服务端拒答兜底（fallbacks="default"）：如果主模型因安全策略拒答，
    API 会在同一次调用里换一个模型重试。
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", client=None):
        self.model = model
        self.effort = effort
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - 取决于安装方式
                raise LLMCheckError('llm 检查需要 anthropic：pip install "openrsi-judge[llm]"') from e
            self._client = anthropic.Anthropic()
        return self._client

    def check(self, question: str, task: Task, evidence: list[Evidence]) -> LLMCheckResult:
        payload = {
            "task": task.model_dump(mode="json", exclude={"metadata"}),
            "evidence": [{"id": e.id, "type": e.type, "source": e.source, "content": e.content} for e in evidence],
        }
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort},
            system=SYSTEM,
            messages=[{"role": "user", "content": f"Question: {question}\n\n{json.dumps(payload, ensure_ascii=False)}"}],
            output_format=LLMCheckResult,
        )
        if response.stop_reason == "refusal":
            raise LLMCheckError("模型拒绝回答这个检查")
        result = response.parsed_output
        if result is None:
            raise LLMCheckError(f"没有得到结构化结果（stop_reason={response.stop_reason}）")
        known = {e.id for e in evidence}
        result.cited_evidence_ids = [i for i in result.cited_evidence_ids if i in known]
        return result
