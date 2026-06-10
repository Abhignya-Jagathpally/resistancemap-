"""Patient-disjoint CV harness for the survival module.

Runs 5-fold, patient-disjoint cross-validation on synthetic CoMMpass-shaped data
for BOTH the Cox proportional-hazards baseline and the torch-free discrete-time
hazard model, and reports, censoring-honestly and per fold:

* Harrell's C-index (lifelines),
* the Integrated Brier Score over an in-support time grid (scikit-survival, IPCW),
* the Expected Calibration Error of the landmark "progressed-by-median-time" task.

It also exercises ``patient_kfold`` (asserting no patient straddles a fold) and
the split-conformal interval helper. All printed numbers are the REAL output of
the run -- they are expected to be modest on synthetic data.

Run:
    cd <repo> && PYTHONPATH=src python3 tests/test_survival.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from resistancemap.data.synthetic import make_synthetic
from resistancemap.survival.baselines import CoxBaseline
from resistancemap.survival.calibration import (
    conformal_interval,
    expected_calibration_error,
)
from resistancemap.survival.competing_risk import DiscreteTimeHazard
from resistancemap.survival.metrics import (
    cindex,
    integrated_brier_score,
    time_dependent_auc,
)
from resistancemap.survival.splits import patient_kfold


def _landmark_labels(durations: np.ndarray, events: np.ndarray, horizon: float):
    """Binary 'progressed by `horizon`' labels with a censoring-honest mask.

    A subject counts as a positive if it had the event at/under the horizon. A
    subject CENSORED before the horizon has unknown status at the landmark and is
    excluded (mask=False) so the calibration metric is not polluted by guesses.
    """
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    progressed = (durations <= horizon) & (events == 1)
    censored_before = (durations < horizon) & (events == 0)
    mask = ~censored_before
    return progressed.astype(int), mask


def run_cv(
    df,
    model_factory,
    model_name: str,
    k: int = 5,
    seed: int = 0,
    n_time_points: int = 20,
):
    """Run patient-disjoint CV for one model; return a dict of mean metrics."""
    durations_all = df["duration"].to_numpy(float)
    landmark = float(np.median(durations_all))  # landmark horizon = global median time

    fold_c: list[float] = []
    fold_ibs: list[float] = []
    fold_auc: list[float] = []
    fold_ece: list[float] = []
    seen_test_patients: set[str] = set()

    for fold, (tr_idx, te_idx) in enumerate(patient_kfold(df, k=k, seed=seed)):
        train = df.iloc[tr_idx].reset_index(drop=True)
        test = df.iloc[te_idx].reset_index(drop=True)

        # --- patient-disjointness assertions (the whole point of the splitter) ---
        tr_pat = set(train["patient_id"])
        te_pat = set(test["patient_id"])
        assert tr_pat.isdisjoint(te_pat), f"patient leak in fold {fold}!"
        assert seen_test_patients.isdisjoint(te_pat), "test folds overlap!"
        seen_test_patients |= te_pat

        model = model_factory().fit(train)

        d_te = test["duration"].to_numpy(float)
        e_te = test["event"].to_numpy(int)
        d_tr = train["duration"].to_numpy(float)
        e_tr = train["event"].to_numpy(int)

        # ---- C-index (higher risk == worse) ----
        risk = model.risk(test)
        fold_c.append(cindex(d_te, e_te, risk))

        # ---- time grid inside the TEST follow-up support ----
        ev_te = d_te[e_te == 1]
        if ev_te.size >= 2:
            grid = np.linspace(ev_te.min(), ev_te.max(), n_time_points + 2)[1:-1]
        else:
            grid = np.asarray([landmark], dtype=float)

        # ---- IBS: needs survival probabilities S(t) on the grid ----
        if hasattr(model, "predict_survival_at"):
            surv_fn = lambda t, m=model: m.predict_survival_at(test, t)
        else:
            # Cox baseline -> survival from lifelines at requested times.
            def surv_fn(t, m=model):
                sf = m.cph.predict_survival_function(test[_cox_features(m)], times=t)
                return sf.values.T  # (n_test, len(t))

        ibs = integrated_brier_score(d_tr, e_tr, d_te, e_te, surv_fn, grid)
        fold_ibs.append(ibs)

        # ---- time-dependent AUC: risk score (1-D, constant ranking) ----
        _, mean_auc = time_dependent_auc(d_tr, e_tr, d_te, e_te, risk, grid)
        fold_auc.append(mean_auc)

        # ---- ECE on landmark progression-by-median task ----
        # Predicted P(progress by landmark) = 1 - S(landmark | x).
        s_land = _survival_at_landmark(model, test, landmark)
        p_prog = 1.0 - s_land
        y_land, lm_mask = _landmark_labels(d_te, e_te, landmark)
        if lm_mask.sum() > 0:
            fold_ece.append(
                expected_calibration_error(y_land[lm_mask], p_prog[lm_mask], n_bins=10)
            )

    def _nanmean(xs):
        xs = np.asarray(xs, dtype=float)
        return float(np.nanmean(xs)) if xs.size and not np.all(np.isnan(xs)) else float("nan")

    return {
        "model": model_name,
        "landmark": landmark,
        "cindex_folds": fold_c,
        "cindex_mean": _nanmean(fold_c),
        "ibs_mean": _nanmean(fold_ibs),
        "auc_mean": _nanmean(fold_auc),
        "ece_mean": _nanmean(fold_ece),
    }


def _cox_features(model):
    from resistancemap.survival.baselines import FEATURES

    return FEATURES


def _survival_at_landmark(model, test, landmark: float) -> np.ndarray:
    """S(landmark | x) for either model type, as a 1-D array of length n_test."""
    if hasattr(model, "predict_survival_at"):
        return model.predict_survival_at(test, np.asarray([landmark]))[:, 0]
    # Cox baseline via lifelines.
    sf = model.cph.predict_survival_function(test[_cox_features(model)], times=[landmark])
    return sf.values.ravel()


def main() -> int:
    df = make_synthetic(n=900, seed=7)
    n_patients = df["patient_id"].nunique()
    print("=== ResistanceMap survival CV (synthetic, FOR VALIDATION ONLY) ===")
    print(f"n_rows={len(df)}  n_patients={n_patients}  "
          f"events={int(df['event'].sum())}  "
          f"censoring={(1 - df['event'].mean()):.1%}")

    results = []
    results.append(run_cv(df, lambda: CoxBaseline(name="cox_ph"), "cox_ph", k=5, seed=0))
    results.append(
        run_cv(df, lambda: DiscreteTimeHazard(n_bins=10), "discrete_time_hazard", k=5, seed=0)
    )

    landmark = results[0]["landmark"]
    print(f"\nlandmark horizon (median observed time) = {landmark:.2f}")
    print("\n5-fold patient-disjoint CV (mean across folds):")
    header = f"{'model':24s} {'C-index':>9s} {'IBS':>9s} {'td-AUC':>9s} {'ECE':>9s}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:24s} {r['cindex_mean']:9.4f} {r['ibs_mean']:9.4f} "
            f"{r['auc_mean']:9.4f} {r['ece_mean']:9.4f}"
        )

    # Per-fold C-index detail (transparency: these are real, modest numbers).
    print("\nper-fold C-index:")
    for r in results:
        folds = ", ".join(f"{c:.4f}" for c in r["cindex_folds"])
        print(f"  {r['model']:24s} [{folds}]")

    # ---- split-conformal interval demo on the discrete-time risk score ----
    # Fit on first 600 patients, calibrate residuals on the rest (disjoint rows;
    # patient_id is unique per row here, so a row split is also patient-disjoint).
    dt = DiscreteTimeHazard(n_bins=10).fit(df.iloc[:600])
    cal = df.iloc[600:].reset_index(drop=True)
    cal_risk = dt.risk(cal)
    # Conformity target: did the subject progress by the landmark (honest mask)?
    y_cal, m_cal = _landmark_labels(
        cal["duration"].to_numpy(float), cal["event"].to_numpy(int), landmark
    )
    resid = np.abs(y_cal[m_cal] - cal_risk[m_cal])
    conf = conformal_interval(cal_risk[m_cal], resid, alpha=0.1)
    print(
        f"\nsplit-conformal (alpha=0.10) on discrete-time risk: "
        f"half-width q={conf['q']:.4f} over n_cal={conf['n_cal']} "
        f"(target coverage {conf['coverage_target']:.0%})"
    )

    # ---- assertions: the suite must PASS, not just print ----
    cox = next(r for r in results if r["model"] == "cox_ph")
    dth = next(r for r in results if r["model"] == "discrete_time_hazard")

    # Both models must recover signal clearly above chance on data with known beta.
    assert cox["cindex_mean"] > 0.55, f"cox C-index too low: {cox['cindex_mean']}"
    assert dth["cindex_mean"] > 0.55, f"DTH C-index too low: {dth['cindex_mean']}"
    # IBS must be a sensible probability-scale loss, comfortably below the 0.25 ref.
    assert 0.0 < cox["ibs_mean"] < 0.25, f"cox IBS out of range: {cox['ibs_mean']}"
    assert 0.0 < dth["ibs_mean"] < 0.25, f"DTH IBS out of range: {dth['ibs_mean']}"
    # ECE is a fraction in [0, 1].
    assert 0.0 <= cox["ece_mean"] <= 1.0
    assert 0.0 <= dth["ece_mean"] <= 1.0
    # Conformal half-width is a valid non-negative width.
    assert conf["q"] >= 0.0

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
