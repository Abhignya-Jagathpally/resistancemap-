"""Minimal, legible stage runner — the replacement for the 40-task Airflow DAG.
Each stage is a (name, callable) pair run in order, with timing and a content hash
of any returned artifact path for a lightweight provenance trail."""
from __future__ import annotations
import time, hashlib, os
from typing import Callable


def _hash(path: str) -> str:
    if path and os.path.exists(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    return "-"


def run(stages: list[tuple[str, Callable]]):
    ledger = []
    for name, fn in stages:
        t0 = time.time()
        artifact = fn()
        dt = time.time() - t0
        row = {"stage": name, "seconds": round(dt, 3), "artifact_hash": _hash(artifact or "")}
        ledger.append(row)
        print(f"[{name:24s}] {dt:6.2f}s  {row['artifact_hash']}")
    return ledger
