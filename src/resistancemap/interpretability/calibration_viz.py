"""Reliability diagrams + per-program partial dependence for the PFS model.

Reliability (landmark calibration)
----------------------------------
At a landmark horizon ``h`` (e.g. 365 / 730 days), the model predicts a
progression-by-h probability ``p = 1 - S(h | x)`` per patient. We bin patients
by predicted probability and, **respecting right-censoring**, compute the
observed event fraction within each bin using the bin-local Kaplan-Meier
estimate of ``1 - S(h)``. A perfectly calibrated model lies on the diagonal.
We also return the Expected Calibration Error (ECE) for the bin.

This is honest about censoring: rather than dropping censored-before-h patients
(which biases the observed rate downward) or counting them as non-events, each
bin's observed risk is the KM CIF at h within that bin.

Partial dependence
------------------
For a single program/feature, partial dependence sweeps that feature across its
empirical quantiles while holding all other features at their observed values
(Friedman 2001), then averages the model's risk. It shows the *direction and
shape* of the model's response to that program — monotone-up means "more of this
program => more predicted risk".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReliabilityPoint:
    """One reliability-diagram bin at a landmark horizon."""

    mean_pred: float       # mean predicted P(progress by h) in the bin
    obs_risk: float        # KM-estimated observed P(progress by h) in the bin
    n: int                 # patients in the bin


def _km_cif_at(durations: np.ndarray, events: np.ndarray, h: float) -> float:
    """Kaplan-Meier cumulative incidence 1 - S(h) for a set of patients.

    Censoring-honest: uses all patients, weighting by the KM product-limit
    estimator up to ``h``. Returns NaN if no patient is observed up to ``h``.
    """
    if len(durations) == 0:
        return float("nan")
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    s = 1.0
    for ut in np.unique(t[t <= h]):
        d_i = int(((t == ut) & (e == 1)).sum())
        n_i = int((t >= ut).sum())
        if n_i > 0:
            s *= (1.0 - d_i / n_i)
    return float(1.0 - s)


def landmark_reliability(
    surv_at_h: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    horizon: float,
    n_bins: int = 5,
) -> tuple[list[ReliabilityPoint], float]:
    """Reliability points + ECE at a landmark horizon.

    Parameters
    ----------
    surv_at_h:
        Predicted survival probability S(h | x) per patient (1 - this = risk).
    durations / events:
        Observed PFS time (days) and event indicator per patient.
    horizon:
        Landmark ``h`` in days.
    n_bins:
        Number of equal-width predicted-probability bins over [0, 1].

    Returns
    -------
    (points, ece)
        ``points`` is one :class:`ReliabilityPoint` per non-empty bin; ``ece``
        is the sample-weighted mean |obs - pred| over bins with a defined KM
        observed risk.
    """
    surv_at_h = np.asarray(surv_at_h, dtype=float)
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    pred = 1.0 - surv_at_h

    ok = np.isfinite(pred)
    pred, dur, ev = pred[ok], durations[ok], events[ok]

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1], right=False), 0, n_bins - 1)

    points: list[ReliabilityPoint] = []
    wsum = 0.0
    err = 0.0
    n_total = len(pred)
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        mean_pred = float(pred[mask].mean())
        obs = _km_cif_at(dur[mask], ev[mask], horizon)
        points.append(ReliabilityPoint(mean_pred=mean_pred, obs_risk=obs, n=nb))
        if np.isfinite(obs):
            err += (nb / n_total) * abs(obs - mean_pred)
            wsum += nb / n_total
    ece = float(err / wsum) if wsum > 0 else float("nan")
    return points, ece


def program_partial_dependence(
    model,
    df: pd.DataFrame,
    feature,
    features: list,
    n_grid: int = 20,
    q_lo: float = 0.02,
    q_hi: float = 0.98,
) -> pd.DataFrame:
    """Partial dependence of model risk on a single feature.

    Sweeps ``feature`` across ``n_grid`` evenly spaced values between its
    ``q_lo``/``q_hi`` empirical quantiles, holding every other feature at its
    observed value for each patient, and averages ``model.risk`` over patients.

    Returns a frame with columns ``grid_value`` and ``mean_risk`` (plus
    ``risk_std`` across patients), so the figure can show the response shape.
    """
    df = df.reset_index(drop=True).copy()
    vals = pd.to_numeric(df[feature], errors="coerce").to_numpy(float)
    finite = vals[np.isfinite(vals)]
    lo, hi = np.quantile(finite, [q_lo, q_hi])
    if lo == hi:
        hi = lo + 1e-6
    grid = np.linspace(lo, hi, n_grid)

    rows = []
    work = df.copy()
    for g in grid:
        work[feature] = g
        r = np.asarray(model.risk(work)).ravel()
        rows.append({
            "grid_value": float(g),
            "mean_risk": float(np.nanmean(r)),
            "risk_std": float(np.nanstd(r)),
        })
    return pd.DataFrame(rows)
