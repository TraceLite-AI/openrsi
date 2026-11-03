"""One arm's HTTP-attempt ledger. No inputs, responses, credentials or headers are logged."""
from __future__ import annotations

import json
import threading
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar


class RequestBudget:
    def __init__(self, limit, path):
        if limit is not None and limit < 1:
            raise ValueError("request budget must be positive")
        self.limit, self.path = limit, path
        self.role = ContextVar("request_role", default="proposer")
        self.lock = threading.Lock()
        self.issued = Counter()
        self.tokens = Counter()
        self.role_tokens = {}
        self.missing_usage = Counter()
        self.denied = False

    @contextmanager
    def phase(self, role):
        mark = self.role.set(role)
        try:
            yield
        finally:
            self.role.reset(mark)

    def _event(self, data):
        with self.path.open("a") as f:
            f.write(json.dumps(data) + "\n")

    def before(self):
        role = self.role.get()
        with self.lock:
            spent = sum(n for key, n in self.issued.items() if key != "final")
            if role != "final" and self.limit is not None and spent >= self.limit:
                self.denied = True
                raise RuntimeError("search request budget exhausted")
            self.issued[role] += 1
            i = sum(self.issued.values())
            self._event({"id": i, "role": role, "issued": True})
            return i

    def after(self, i, usage, ok):
        with self.lock:
            self._event({"id": i, "role": self.role.get(), "ok": ok, "usage": usage})
            role = self.role.get()
            if not usage:
                self.missing_usage[role] += 1
            per_role = self.role_tokens.setdefault(role, Counter())
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                n = int(usage.get(k) or 0)
                self.tokens[k] += n
                per_role[k] += n

    def report(self):
        return {"unit": "HTTP attempts including retries", "search_limit": self.limit,
                "search_used": sum(n for k, n in self.issued.items() if k != "final"),
                "by_role": dict(self.issued), "reported_tokens": dict(self.tokens),
                "reported_tokens_by_role": {k: dict(v) for k, v in self.role_tokens.items()},
                "attempts_without_usage": dict(self.missing_usage), "budget_exhausted": self.denied}
