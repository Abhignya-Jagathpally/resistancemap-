"""Uncertainty + calibration utilities for risk scores and landmark tasks.

Two complementary tools:

* :func:`conformal_interval` -- split-conformal prediction intervals giving
  finite-sample, distribution-free marginal coverage ``>= 1 - alpha`` for a risk
  score, from a held-out calibration set.
* :func:`expected_calibration_error` -- the binned ECE for the *landmark* binary
  task (e.g. "progression by horizon H"), measuring the gap between predicted
  probabilities and observed frequencies.
"""
from __future__ import annotations

import numpy as np


def conformal_interval(
    scores_cal: np.ndarray,
    residuals_cal: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Split-conformal interval half-width for a risk score.

    Implements absolute-residual split conformal prediction (Lei et al. 2018,
    "Distribution-Free Predictive Inference for Regression"). Given calibration
    residuals ``|y - y_hat|`` from a model fit on a *disjoint* training split, the
    quantile

        q = the ceil((n+1)(1-alpha)) / n  empirical quantile of |residuals|

    yields a band ``[score - q, score + q]`` around any future point prediction
    with marginal coverage ``>= 1 - alpha`` (assuming exchangeability). The finite-
    sample correction ``(n+1)/n`` is what makes the guarantee exact rather than
    asymptotic.

    Parameters
    ----------
    scores_cal:
        Calibration-set risk scores (used only to report their range / centre;
        the interval *half-width* depends on the residuals, per split conformal).
    residuals_cal:
        Calibration-set absolute residuals ``|y - y_hat|`` (any non-negative
        conformity score works; absolute residual is the canonical choice).
    alpha:
        Target miscoverage; coverage is ``>= 1 - alpha``. Must be in ``(0, 1)``.

    Returns
    -------
    dict with keys:
        ``q`` -- the conformal quantile (interval half-width),
        ``alpha`` -- echoed target miscoverage,
        ``coverage_target`` -- ``1 - alpha``,
        ``n_cal`` -- calibration set size,
        ``lo_offset`` / ``hi_offset`` -- ``-q`` / ``+q`` to add to any prediction.

    Notes
    -----
    The returned ``q`` is a symmetric half-width. To band a concrete prediction
    ``y_hat`` use ``[y_hat - q, y_hat + q]``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    resid = np.abs(np.asarray(residuals_cal, dtype=float).ravel())
    n = resid.size
    if n == 0:
        raise ValueError("residuals_cal is empty")

    # Finite-sample conformal level: the ceil((n+1)(1-alpha))-th smallest residual.
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        # Not enough calibration points for the requested alpha -> widest band.
        q = float(np.max(resid))
    else:
        q = float(np.quantile(resid, level, method="higher"))

    scores = np.asarray(scores_cal, dtype=float).ravel()
    return {
        "q": q,
        "alpha": float(alpha),
        "coverage_target": float(1.0 - alpha),
        "n_cal": int(n),
        "lo_offset": -q,
        "hi_offset": q,
        "score_mean": float(scores.mean()) if scores.size else float("nan"),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) for a binary landmark task.

    Bins predicted probabilities into ``n_bins`` equal-width bins over ``[0, 1]``
    and returns the sample-size-weighted mean absolute gap between the average
    predicted probability and the empirical event rate in each bin (Naeini et al.
    2015; Guo et al. 2017):

        ECE = sum_b (n_b / N) * | acc(b) - conf(b) |,

    where ``conf(b)`` is the mean prediction in bin ``b`` and ``acc(b)`` the
    observed positive fraction. ``0`` is perfect calibration; larger is worse.

    Parameters
    ----------
    y_true:
        Binary outcomes in ``{0, 1}`` (e.g. progressed-by-horizon).
    p_pred:
        Predicted probabilities in ``[0, 1]`` for the positive class.
    n_bins:
        Number of equal-width probability bins.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(p_pred, dtype=float).ravel()
    if y.size != p.size:
        raise ValueError("y_true and p_pred must have the same length")
    if y.size == 0:
        raise ValueError("empty inputs")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins; clip so p == 1.0 lands in the last bin (idx n_bins-1).
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    n = y.size
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (nb / n) * abs(acc - conf)
    return float(ece)
