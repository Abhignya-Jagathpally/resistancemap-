"""Censoring-honest survival metrics with bootstrap confidence intervals."""
from __future__ import annotations
import numpy as np
from lifelines.utils import concordance_index


def cindex(durations, events, risk) -> float:
    """Harrell's C. `risk` is higher = worse prognosis (shorter survival)."""
    risk = np.asarray(risk, dtype=float)
    return concordance_index(np.asarray(durations, float), -risk, np.asarray(events, int))


def cindex_bootstrap(durations, events, risk, B: int = 400, seed: int = 1):
    rng = np.random.default_rng(seed)
    d = np.asarray(durations, float); e = np.asarray(events, int); r = np.asarray(risk, float)
    n = len(d); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if e[idx].sum() < 2:
            continue
        try:
            vals.append(cindex(d[idx], e[idx], r[idx]))
        except Exception:
            continue
    vals = np.asarray(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------------------
# Time-dependent, censoring-aware metrics (appended; backed by scikit-survival).
# These complement Harrell's C above with IPCW-corrected, time-resolved scores.
# ---------------------------------------------------------------------------
from collections.abc import Callable as _Callable


def _surv_array(durations, events):
    """Build a scikit-survival structured array from (durations, events)."""
    from sksurv.util import Surv

    return Surv.from_arrays(
        event=np.asarray(events, dtype=int).astype(bool),
        time=np.asarray(durations, dtype=float),
    )


def _clamp_times(
    train_durations,
    train_events,
    test_durations,
    test_events,
    times,
):
    """Restrict an evaluation time grid to the range sksurv considers valid.

    sksurv's IPCW estimators require evaluation times to lie strictly inside the
    follow-up support of the *test* set (between its smallest and largest event
    times, and below the largest observed test time). Times outside that window
    are dropped to keep the censoring weights well defined. The training
    censoring distribution must also cover them, so we additionally cap by the
    maximum *training* time.
    """
    td = np.asarray(test_durations, dtype=float)
    te = np.asarray(test_events, dtype=int).astype(bool)
    tr = np.asarray(train_durations, dtype=float)

    test_event_times = td[te]
    if test_event_times.size == 0:
        return np.asarray([], dtype=float)

    lo = float(test_event_times.min())
    # Strictly below the largest observed (any-cause) test time and the largest
    # training time, so the IPCW survival estimate of the censoring dist is > 0.
    hi = float(min(td.max(), tr.max()))
    times = np.atleast_1d(np.asarray(times, dtype=float))
    eps = 1e-8 * max(1.0, abs(hi))
    keep = (times > lo) & (times < hi - eps)
    return np.unique(times[keep])


def integrated_brier_score(
    train_durations,
    train_events,
    test_durations,
    test_events,
    survival,
    times,
) -> float:
    """Integrated Brier Score (IBS) over ``times`` via :mod:`sksurv.metrics`.

    The IBS is the time-average of the IPCW-weighted Brier score of the predicted
    *survival* probabilities; it is a proper scoring rule for right-censored data
    and rewards both discrimination and calibration. Lower is better; a model that
    always predicts the marginal survival sits near ~0.25 at the median.

    Parameters
    ----------
    train_durations, train_events:
        Follow-up times / event indicators of the data used to *fit* the model.
        These estimate the censoring distribution for the IPCW weights.
    test_durations, test_events:
        Follow-up times / event indicators of the evaluation set.
    survival:
        Either an array of shape ``(n_test, len(times))`` of survival
        probabilities ``S(times[j] | x_i)``, or a callable
        ``survival(times) -> (n_test, len(times))`` producing them. Columns must
        align with ``times`` (after internal clamping, see Notes).
    times:
        Requested evaluation time grid; silently clamped to the range sksurv
        accepts for the given test follow-up (see :func:`_clamp_times`).

    Returns
    -------
    float
        The integrated Brier score over the (clamped) grid, or ``nan`` if no
        valid evaluation time remains (e.g. too few events).

    Notes
    -----
    Because the valid grid is determined *after* clamping, prefer passing a
    callable for ``survival`` so the columns are computed at exactly the times
    used. If an array is passed it must already be aligned to ``times``; the same
    columns are then subset by the clamp mask.
    """
    from sksurv.metrics import integrated_brier_score as _sk_ibs

    times = np.atleast_1d(np.asarray(times, dtype=float))
    valid = _clamp_times(
        train_durations, train_events, test_durations, test_events, times
    )
    if valid.size == 0:
        return float("nan")
    if valid.size == 1:
        # IBS integrates over >=2 points; widen minimally if clamping left one.
        return float("nan")

    if isinstance(survival, _Callable):
        surv = np.asarray(survival(valid), dtype=float)
    else:
        surv = np.asarray(survival, dtype=float)
        # Subset the supplied columns to the kept times (align by membership).
        col_mask = np.isin(times, valid)
        surv = surv[:, col_mask]

    surv_train = _surv_array(train_durations, train_events)
    surv_test = _surv_array(test_durations, test_events)
    return float(_sk_ibs(surv_train, surv_test, surv, valid))


def time_dependent_auc(
    train_durations,
    train_events,
    test_durations,
    test_events,
    risk,
    times,
):
    """Cumulative/dynamic time-dependent AUC via :mod:`sksurv.metrics`.

    Wraps :func:`sksurv.metrics.cumulative_dynamic_auc`, the IPCW-corrected
    time-dependent ROC AUC of Uno et al. (2007). At each time ``t`` it scores how
    well ``risk`` separates subjects who have had the event by ``t`` (cases) from
    those still event-free (controls), correcting for censoring with weights from
    the training survival distribution. Higher is better; 0.5 is chance.

    Parameters
    ----------
    train_durations, train_events, test_durations, test_events:
        As in :func:`integrated_brier_score`.
    risk:
        A risk score where *higher == worse prognosis*. Either a 1-D array of
        length ``n_test`` (same ranking used at every time) or a 2-D array of
        shape ``(n_test, len(times))``, or a callable ``risk(times)`` returning
        such an array.
    times:
        Requested evaluation grid; clamped to the sksurv-valid range.

    Returns
    -------
    (auc_per_time, mean_auc):
        ``auc_per_time`` is an array aligned with the clamped grid;
        ``mean_auc`` is sksurv's time-integrated summary AUC. Both are ``nan``
        (and ``auc_per_time`` empty) if no valid time remains.
    """
    from sksurv.metrics import cumulative_dynamic_auc as _sk_auc

    times = np.atleast_1d(np.asarray(times, dtype=float))
    valid = _clamp_times(
        train_durations, train_events, test_durations, test_events, times
    )
    if valid.size == 0:
        return np.asarray([], dtype=float), float("nan")

    if isinstance(risk, _Callable):
        est = np.asarray(risk(valid), dtype=float)
    else:
        est = np.asarray(risk, dtype=float)
        if est.ndim == 2:
            col_mask = np.isin(times, valid)
            est = est[:, col_mask]

    surv_train = _surv_array(train_durations, train_events)
    surv_test = _surv_array(test_durations, test_events)
    auc, mean_auc = _sk_auc(surv_train, surv_test, est, valid)
    return np.asarray(auc, dtype=float), float(mean_auc)
