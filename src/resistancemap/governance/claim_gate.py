"""Falsification harness: a claim is only 'granted' if its pre-registered test passes.
Refusing unearned claims by construction is the discipline that distinguishes this
work from leaderboard-chasing. Mirrors the project's hard-won lesson that fabricated
or unverified metrics sink a submission."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import json


@dataclass
class ClaimGate:
    name: str
    test: Callable[[], bool]
    rationale: str

    def evaluate(self) -> dict:
        try:
            ok = bool(self.test())
            return {"claim": self.name, "status": "GRANTED" if ok else "BLOCKED",
                    "rationale": self.rationale}
        except Exception as e:
            return {"claim": self.name, "status": "ERROR",
                    "rationale": f"{self.rationale} | {type(e).__name__}: {e}"}


class Governance:
    def __init__(self):
        self.gates: list[ClaimGate] = []

    def add(self, gate: ClaimGate):
        self.gates.append(gate); return self

    def report(self) -> list[dict]:
        return [g.evaluate() for g in self.gates]

    def emit(self, path: str | None = None) -> list[dict]:
        rep = self.report()
        if path:
            json.dump(rep, open(path, "w"), indent=2)
        return rep
