"""SYNTHETIC CoMMpass-shaped survival data — FOR PIPELINE VALIDATION ONLY.

This is NOT a scientific dataset and NO biological conclusion may be drawn from it.
It draws right-censored time-to-event data from a known Cox-Weibull model so the
baseline + metric + governance code can be exercised end-to-end *before* the real
GDC-open CoMMpass run on the user's GPU machine. Because the data-generating
coefficients are known, it also serves as a recovery check: a correct Cox baseline
should rank the true linear predictor well (C-index clearly > 0.5).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["age_z", "iss_stage", "sex_male", "prog_score_1", "prog_score_2", "prog_score_3"]
TRUE_BETA = np.array([0.25, 0.45, 0.05, 0.60, -0.30, 0.15])


def make_synthetic(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age_z = rng.normal(0, 1, n)
    iss = rng.integers(1, 4, n).astype(float)
    sex = rng.integers(0, 2, n).astype(float)
    s = rng.normal(0, 1, (n, 3))
    X = np.column_stack([age_z, iss, sex, s])
    lp = X @ TRUE_BETA
    k, lam = 1.4, 80.0                      # Weibull baseline hazard
    u = rng.uniform(size=n)
    t_event = lam * (-np.log(u) * np.exp(-lp)) ** (1.0 / k)
    c = rng.uniform(10, 220, n)             # administrative + dropout censoring
    time = np.minimum(t_event, c)
    event = (t_event <= c).astype(int)
    df = pd.DataFrame(X, columns=FEATURES)
    df["duration"] = time
    df["event"] = event
    df["patient_id"] = [f"SYN{i:04d}" for i in range(n)]
    return df
