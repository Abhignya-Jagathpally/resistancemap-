"""Landmark cross-validation harness for Lane #2 (immortal-time-safe evaluation).

The whole methodological point of Lane #2 is that the treatment-trajectory covariate
``n_lines`` is partly a *consequence* of progression, so it cannot be used as a baseline
feature without immortal-time bias. The **landmark design** neutralizes this:

    For each landmark L (e.g. 180/365/730 days):
      1. Restrict to patients still AT-RISK and EVENT-FREE at L
         (duration > L). These patients "survived to the landmark".
      2. Re-origin time at L:  T' = duration - L,  delta unchanged.
      3. Features = static x (programs + clinical) + Z(L) built ONLY from history
         up to L  (lines accrued by L, first-line regimen one-hot,
         time-since-last-switch at L). No future information enters Z(L).
      4. Fit a survival model on (x, Z(L)) -> forward survival from L, and evaluate with
         censoring-aware time-dependent AUC + integrated Brier score under
         patient-disjoint k-fold CV.

This module supplies the harness; the *models* and the *feature attachment* are passed in
so it stays agnostic. Metrics are reused from :mod:`resistancemap.survival.metrics`, CV
splits from :mod:`resistancemap.survival.splits`. No fabrication: if a landmark has too few
at-risk patients or events for a metric, that cell is reported as ``nan`` with a reason.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import (
    cindex,
    integrated_brier_score,
    time_dependent_auc,
)
from .splits import patient_kfold

DEFAULT_LANDMARKS = (180.0, 365.0, 730.0)
# Forward evaluation horizons (days *after* the landmark) for td-AUC / IBS grids.
DEFAULT_HORIZONS = (180.0, 365.0, 730.0)


@dataclass
class LandmarkFoldResult:
    landmark: float
    fold: int
    model: str
    n_train: int
    n_test: int
    n_test_events: int
    td_auc_mean: float
    ibs: float
    cindex: float
    auc_grid: list = field(default_factory=list)
    auc_per_time: list = field(default_factory=list)


def at_risk_at_landmark(
    df: pd.DataFrame,
    landmark: float,
    duration_col: str = "duration",
) -> pd.DataFrame:
    """Return the sub-frame of patients still at risk and event-free at ``landmark``.

    A patient is at risk iff their observed follow-up extends strictly beyond the
    landmark (``duration > landmark``). The returned frame carries two extra columns:
    ``lm_duration`` (= duration - landmark, the re-origined forward time) and
    ``lm_landmark`` (the landmark, for provenance). The original ``event`` is kept as is
    (an event at duration > L is a forward event from L).
    """
    out = df[df[duration_col] > float(landmark)].copy()
    out["lm_landmark"] = float(landmark)
    out["lm_duration"] = out[duration_col].astype(float) - float(landmark)
    return out.reset_index(drop=True)


def run_landmark_cv(
    df: pd.DataFrame,
    models: dict[str, Callable[[], object]],
    feature_builder: Callable[[pd.DataFrame, float], tuple[pd.DataFrame, list[str]]],
    landmarks: Sequence[float] = DEFAULT_LANDMARKS,
    horizons: Sequence[float] = DEFAULT_HORIZONS,
    k: int = 5,
    seed: int = 0,
    duration_col: str = "duration",
    event_col: str = "event",
    min_test_events: int = 5,
) -> tuple[pd.DataFrame, list[LandmarkFoldResult]]:
    """Run immortal-time-safe landmark CV for a set of models.

    Parameters
    ----------
    df:
        Patient frame with at least ``patient_id``, ``duration`` (days from diagnosis),
        ``event``, plus any static covariates.
    models:
        Mapping ``name -> factory()`` where ``factory()`` returns a fresh, unfitted model
        exposing ``fit(df, features)``, ``risk(df) -> (n,)`` and
        ``survival_function(df, times) -> (n, len(times))``. A factory (not an instance)
        is required so each (landmark, fold) gets an independently-fitted model.
    feature_builder:
        ``feature_builder(landmark_frame, landmark) -> (augmented_frame, feature_cols)``.
        It must attach the static + Z(L) feature columns to the at-risk frame and return
        the list of feature column names to train on. Built per-landmark so Z(L) is
        landmark-specific. The harness re-origins time itself (``lm_duration``); the
        builder must NOT look past the landmark.
    landmarks, horizons:
        Landmark times (days from diagnosis) and forward horizons (days after the
        landmark) used to build the td-AUC / IBS evaluation grid.
    k, seed:
        Patient-disjoint CV configuration (delegates to ``patient_kfold``).
    min_test_events:
        Folds with fewer forward events in the test split than this are skipped for
        metric computation (reported, not fabricated).

    Returns
    -------
    (summary_table, fold_results):
        ``summary_table`` is one row per (landmark, model) with mean +/- std of td-AUC,
        IBS, C-index across folds and the number of contributing folds.
        ``fold_results`` is the raw per-fold list.
    """
    fold_results: list[LandmarkFoldResult] = []

    for L in landmarks:
        atrisk = at_risk_at_landmark(df, L, duration_col=duration_col)
        if atrisk.empty:
            continue
        # Build features once per landmark (Z(L) is landmark-specific, history<=L only).
        feat_df, feature_cols = feature_builder(atrisk, float(L))
        # Forward evaluation grid: horizons measured from the landmark, re-origined.
        grid = np.array(sorted({float(h) for h in horizons}), dtype=float)

        # Need enough distinct patients to make k folds; otherwise reduce k.
        n_patients = feat_df["patient_id"].nunique()
        eff_k = min(k, n_patients) if n_patients >= 2 else 0
        if eff_k < 2:
            continue

        for fold, (tr_idx, te_idx) in enumerate(
            patient_kfold(feat_df, k=eff_k, seed=seed)
        ):
            train = feat_df.iloc[tr_idx].reset_index(drop=True)
            test = feat_df.iloc[te_idx].reset_index(drop=True)

            # Survival models fit/eval against the re-origined FORWARD time. The frame
            # already carries the original diagnosis-origin `duration`; overwrite it with
            # `lm_duration` (= duration - L) so model.fit/eval use forward time, and align
            # the canonical `event` column. Drop lm_* helpers to avoid duplicate columns.
            train_fit = train.drop(columns=["lm_duration", "lm_landmark"]).copy()
            train_fit["duration"] = train["lm_duration"].to_numpy(float)
            train_fit["event"] = train[event_col].to_numpy(int)

            test_fit = test.drop(columns=["lm_duration", "lm_landmark"]).copy()
            test_fit["duration"] = test["lm_duration"].to_numpy(float)
            test_fit["event"] = test[event_col].to_numpy(int)

            tr_dur = train["lm_duration"].to_numpy(float)
            tr_evt = train[event_col].to_numpy(int)
            te_dur = test["lm_duration"].to_numpy(float)
            te_evt = test[event_col].to_numpy(int)
            n_te_events = int(te_evt.sum())

            for mname, factory in models.items():
                if n_te_events < min_test_events:
                    fold_results.append(LandmarkFoldResult(
                        landmark=float(L), fold=fold, model=mname,
                        n_train=len(train), n_test=len(test),
                        n_test_events=n_te_events,
                        td_auc_mean=float("nan"), ibs=float("nan"),
                        cindex=float("nan"),
                    ))
                    continue
                try:
                    model = factory()
                    model.fit(train_fit, features=feature_cols)
                    risk = np.asarray(model.risk(test_fit), dtype=float).ravel()
                except Exception:
                    fold_results.append(LandmarkFoldResult(
                        landmark=float(L), fold=fold, model=mname,
                        n_train=len(train), n_test=len(test),
                        n_test_events=n_te_events,
                        td_auc_mean=float("nan"), ibs=float("nan"),
                        cindex=float("nan"),
                    ))
                    continue

                # C-index (forward).
                try:
                    c = cindex(te_dur, te_evt, risk)
                except Exception:
                    c = float("nan")

                # Time-dependent AUC (IPCW), per-time + mean.
                try:
                    auc_pt, auc_mean = time_dependent_auc(
                        tr_dur, tr_evt, te_dur, te_evt, risk, grid
                    )
                    auc_grid = _valid_grid(tr_dur, tr_evt, te_dur, te_evt, grid)
                except Exception:
                    auc_pt, auc_mean, auc_grid = np.array([]), float("nan"), []

                # IBS via the model's survival function on the same grid.
                try:
                    def _surv(times, _m=model, _t=test_fit):
                        return np.asarray(_m.survival_function(_t, times), dtype=float)
                    ibs = integrated_brier_score(
                        tr_dur, tr_evt, te_dur, te_evt, _surv, grid
                    )
                except Exception:
                    ibs = float("nan")

                fold_results.append(LandmarkFoldResult(
                    landmark=float(L), fold=fold, model=mname,
                    n_train=len(train), n_test=len(test),
                    n_test_events=n_te_events,
                    td_auc_mean=float(auc_mean), ibs=float(ibs), cindex=float(c),
                    auc_grid=list(np.asarray(auc_grid, float)),
                    auc_per_time=list(np.asarray(auc_pt, float)),
                ))

    summary = _summarize(fold_results)
    return summary, fold_results


def _valid_grid(tr_dur, tr_evt, te_dur, te_evt, grid):
    """Mirror metrics._clamp_times so we can label which grid points were scored."""
    from .metrics import _clamp_times  # reuse the exact clamping logic
    return _clamp_times(tr_dur, tr_evt, te_dur, te_evt, grid)


def _summarize(fold_results: list[LandmarkFoldResult]) -> pd.DataFrame:
    """Aggregate per-fold results into a (landmark, model) summary table."""
    if not fold_results:
        return pd.DataFrame(columns=[
            "landmark", "model", "n_folds_scored",
            "td_auc_mean", "td_auc_std", "ibs_mean", "ibs_std",
            "cindex_mean", "cindex_std", "n_atrisk", "n_events_total",
        ])
    rows = []
    df = pd.DataFrame([r.__dict__ for r in fold_results])
    for (L, m), grp in df.groupby(["landmark", "model"], sort=True):
        scored = grp.dropna(subset=["td_auc_mean"])
        rows.append({
            "landmark": float(L),
            "model": m,
            "n_folds_scored": int(len(scored)),
            "td_auc_mean": float(scored["td_auc_mean"].mean()) if len(scored) else float("nan"),
            "td_auc_std": float(scored["td_auc_mean"].std(ddof=0)) if len(scored) else float("nan"),
            "ibs_mean": float(scored["ibs"].mean()) if len(scored) else float("nan"),
            "ibs_std": float(scored["ibs"].std(ddof=0)) if len(scored) else float("nan"),
            "cindex_mean": float(scored["cindex"].mean()) if len(scored) else float("nan"),
            "cindex_std": float(scored["cindex"].std(ddof=0)) if len(scored) else float("nan"),
            # at-risk total = max n_test summed across folds (one fold per patient)
            "n_atrisk": int(grp["n_test"].sum()),
            "n_events_total": int(grp["n_test_events"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["landmark", "model"]).reset_index(drop=True)


def fold_td_auc_vectors(
    fold_results: list[LandmarkFoldResult],
    landmark: float,
    model: str,
) -> list[float]:
    """Return the per-fold mean td-AUC values for a (landmark, model) cell.

    Used by the runner to bootstrap a paired CI of the model-vs-static td-AUC delta.
    """
    out = []
    for r in fold_results:
        if r.landmark == float(landmark) and r.model == model:
            if not np.isnan(r.td_auc_mean):
                out.append(float(r.td_auc_mean))
    return out
