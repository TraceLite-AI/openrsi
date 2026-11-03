"""Disjoint content partitions (protocol v2). IDs alone do not distinguish expressions."""
from __future__ import annotations

import hashlib
import json
import random


class ExpressionPool:
    version = "secret-ops/disjoint-v2"

    def __init__(self, symbols, seed, hold=3, final=6):
        if not (0 < hold and 0 < final and hold + final < 64):
            raise ValueError("split sizes must leave a non-empty training pool per symbol")
        self.banks = {s: {} for s in symbols}
        for s in symbols:
            pairs = [(a, b) for a in range(2, 10) for b in range(2, 10)]
            random.Random(f"{self.version}:{seed}:{s}").shuffle(pairs)
            self.banks[s] = {"holdout": pairs[:hold], "final": pairs[hold:hold + final],
                             "train": pairs[hold + final:]}
        self.digest = hashlib.sha256(json.dumps(self.banks, sort_keys=True).encode()).hexdigest()

    def rows(self, split):
        return [(s, a, b) for s, bank in self.banks.items() for a, b in bank[split]]
