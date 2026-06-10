#!/usr/bin/env python
"""Lane #2 runner — immortal-time-safe landmark evaluation of treatment-conditioned,
PH-free progression forecasting on OPEN multiple-myeloma data.

At each landmark L in {180, 365, 730} days, among patients still at risk and event-free at
L, this compares (forward, re-origined survival from L):
    (a) static Cox       — programs + clinical only (no treatment trajectory)
    (b) time-varying Cox — static + Z(L)  [lines accrued by L, first-line regimen one-hot,
                            time-since-last-switch, current-regimen-class indicators]
    (c) treatment_basin  — PH-free basin mixture on the same (x, Z(L)) block (Ferle loss)

It reports time-dependent AUC + IBS (with bootstrap CIs over folds) and a Grambsch-Therneau
evidence table showing that number-of-lines violates proportional hazards. The pre-registered
gate is: *a treatment model beats static on forward td-AUC at a landmark with CI separation*.

HONESTY: no fabricated metrics. The landmark design (condition on history up to L, predict
forward) neutralizes immortal-time bias from ``n_lines``. If the treatment model does NOT beat
static, that is reported plainly and the gate stays BLOCKED.

Run from the pipeline3 root:
    venv/bin/python scripts/run_lane2_landmark.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

# Make THIS repo's `resistancemap` package importable when run from the pipeline3 root.
# A different (legacy v20) `resistancemap` is editable-installed in the venv via a custom
# meta-path finder that would otherwise shadow this package; drop that finder first so the
# local open-data edition under resistancemap/src wins.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
for _f in list(sys.meta_path):
    _nm = (type(_f).__module__ + "." + type(_f).__name__).lower()
    if "editable" in _nm and "resistancemap" in _nm:
        sys.meta_path.remove(_f)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from resistancemap.data.mmsygnal import load_ia12  # noqa: E402
from resistancemap.data.treatment_timeline import (  # noqa: E402
    DRUG_CLASS_ORDER,
    load_timeline,
)
from resistancemap.survival.baselines import CoxBaseline  # noqa: E402
from resistancemap.survival.landmark import (  # noqa: E402
    DEFAULT_HORIZONS,
    DEFAULT_LANDMARKS,
    fold_td_auc_vectors,
    run_landmark_cv,
)
from resistancemap.models.treatment_basin import TreatmentBasin  # noqa: E402

OUT_DIR = os.path.join("results", "lane2")
FIRST_LINE_CATEGORIES = ["PI", "IMiD", "CD38", "Alkyl", "PI+IMiD", "Other", "None"]
SEED = 0


# --------------------------------------------------------------------------- data
def load_lane2_frame():
    """Join mmSYGNAL programs+clinical (outcome) with the GDC treatment timeline.

    Returns (df, program_cols, clinical_cols, timeline). df has patient_id (MMRF_xxxx),
    duration (PFS days), event, programs, clinical.
    """
    data = load_ia12()
    timeline = load_timeline()

    df = data.df.copy()
    # mmSYGNAL program columns are *integer* names (0..140); rename to safe string
    # feature names so prefix checks and lifelines/sksurv column selection are robust.
    prog_rename = {c: f"prog_{int(c)}" for c in data.program_cols}
    df = df.rename(columns=prog_rename)
    program_cols = [prog_rename[c] for c in data.program_cols]

    df["patient_key"] = df["patient_id"].str.extract(r"(MMRF_\d+)")[0]
    df = df.dropna(subset=["patient_key"]).copy()
    # One row per patient_key (mmSYGNAL is baseline BM sample; de-dup defensively).
    df = df.drop_duplicates(subset=["patient_key"]).reset_index(drop=True)
    # Keep only patients present in the treatment timeline (need a treatment trajectory).
    tl_patients = set(timeline.patients())
    df = df[df["patient_key"].isin(tl_patients)].reset_index(drop=True)
    # Canonicalize the join/grouping key for the harness.
    df["patient_id"] = df["patient_key"]

    clinical_cols = [c for c in data.clinical_cols if c in df.columns]
    return df, program_cols, clinical_cols, timeline


# --------------------------------------------------- feature builder (static + Z(L))
def make_feature_builder(timeline, program_cols, clinical_cols):
    """Return a feature_builder(atrisk_df, landmark) -> (aug_df, feature_cols).

    Static block: programs + clinical. Treatment block Z(L) (prefixed ``z_``):
      z_lines_accrued, z_time_since_switch (scaled to years), z_first_<cat> one-hots,
      z_cur_<class> current-regimen-class indicators. Built from history <= L only.
    """
    def build(atrisk: pd.DataFrame, landmark: float):
        aug = atrisk.copy()
        # ---- static z-scoring is handled inside the models; just keep raw values ----
        static_cols = [c for c in (program_cols + clinical_cols) if c in aug.columns]

        # ---- time-varying covariates Z(L) ----
        lines, since_switch = [], []
        first_cat, cur_classes = [], []
        for pid in aug["patient_id"]:
            lines.append(timeline.lines_accrued_by(pid, landmark))
            since_switch.append(timeline.time_since_last_switch(pid, landmark))
            first_cat.append(timeline.first_line_regimen(pid))
            cur_classes.append(timeline.current_regimen_classes(pid, landmark))

        aug["z_lines_accrued"] = np.asarray(lines, dtype=float)
        aug["z_time_since_switch_yr"] = np.asarray(since_switch, dtype=float) / 365.25

        z_cols = ["z_lines_accrued", "z_time_since_switch_yr"]

        # First-line regimen one-hot (drop the most common to avoid collinearity).
        first_cat = pd.Series(first_cat, index=aug.index)
        for cat in FIRST_LINE_CATEGORIES:
            col = f"z_first_{cat.replace('+', '_')}"
            aug[col] = (first_cat == cat).astype(float)
        # Keep only one-hots that actually vary in this landmark cohort.
        for cat in FIRST_LINE_CATEGORIES:
            col = f"z_first_{cat.replace('+', '_')}"
            if aug[col].nunique() > 1:
                z_cols.append(col)
            else:
                aug.drop(columns=[col], inplace=True)

        # Current-regimen-class indicators (active line at landmark).
        for cls in DRUG_CLASS_ORDER:
            col = f"z_cur_{cls}"
            aug[col] = [1.0 if cls in cc else 0.0 for cc in cur_classes]
            if aug[col].nunique() > 1:
                z_cols.append(col)
            else:
                aug.drop(columns=[col], inplace=True)

        feature_cols = static_cols + z_cols
        # stash for per-model subsetting
        build.last_static = static_cols
        build.last_treatment = z_cols
        return aug, feature_cols

    build.last_static = []
    build.last_treatment = []
    return build


# ------------------------------------------------------- model factories (wrappers)
class _SubsetCox:
    """CoxBaseline that fits on a fixed *subset* (predicate) of the offered features.

    The harness passes the full augmented feature list to every model; this wrapper lets
    static-Cox keep only non-``z_`` columns while time-varying-Cox keeps all. Uses a small
    ridge penalty for numerical stability with the wide program block.
    """

    def __init__(self, name, keep_treatment: bool, penalizer: float = 1.0):
        self.name = name
        self.keep_treatment = keep_treatment
        self.penalizer = penalizer
        self._cox = None
        self.features = None
        self._impute = None  # train-fit per-column median (no leakage)

    def _select(self, features):
        if self.keep_treatment:
            return list(features)
        return [f for f in features if not str(f).startswith("z_")]

    def _prep(self, df, fit: bool):
        """Return a NaN-free feature frame; impute medians fit on TRAIN only."""
        X = df[self.features].apply(pd.to_numeric, errors="coerce")
        if fit:
            self._impute = X.median(numeric_only=True)
        return X.fillna(self._impute).fillna(0.0)

    def fit(self, df, features):
        self.features = self._select(features)
        # Drop zero-variance columns within this fold (lifelines is fragile to them).
        Xchk = df[self.features].apply(pd.to_numeric, errors="coerce")
        keep = [c for c in self.features if Xchk[c].nunique(dropna=True) > 1]
        self.features = keep
        X = self._prep(df, fit=True)
        fit_df = X.copy()
        fit_df["duration"] = df["duration"].values
        fit_df["event"] = df["event"].values
        self._cox = CoxBaseline(
            penalizer=self.penalizer, l1_ratio=0.0, name=self.name, features=keep,
        )
        self._cox.fit(fit_df, features=keep)
        return self

    def risk(self, df):
        X = self._prep(df, fit=False)
        return self._cox.cph.predict_partial_hazard(X).values.ravel()

    def survival_function(self, df, times):
        """Cox S(t|x) from the fitted baseline survival, evaluated at ``times``."""
        X = self._prep(df, fit=False)
        cph = self._cox.cph
        times = np.atleast_1d(np.asarray(times, dtype=float))
        # lifelines returns a (len(times) x n) frame; transpose to (n, len(times)).
        sf = cph.predict_survival_function(X, times=times)
        return np.asarray(sf.values.T, dtype=float)


def make_models(penalizer: float, basin_epochs: int):
    return {
        "static_cox": lambda: _SubsetCox("static_cox", keep_treatment=False, penalizer=penalizer),
        "timevarying_cox": lambda: _SubsetCox("timevarying_cox", keep_treatment=True, penalizer=penalizer),
        "treatment_basin": lambda: TreatmentBasin(
            n_basins=3, epochs=basin_epochs, penalty_weight=0.1, seed=SEED,
            require_treatment_features=True,
        ),
    }


# ------------------------------------------------- Grambsch-Therneau evidence table
def grambsch_therneau_evidence(df, program_cols, clinical_cols, timeline):
    """Cox PH test for n_lines (treatment trajectory) vs a baseline-only fit.

    n_lines here is total lines accrued over the whole follow-up — used ONLY for the PH
    diagnostic (Grambsch-Therneau), exactly the immortal-time-biased baseline use the
    landmark design avoids for prediction. We report the PH violation, then separately fit
    a baseline-clinical model to show baseline covariates do NOT violate PH.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    out = {"tests": [], "notes": (
        "n_lines is used here ONLY as a PH diagnostic. Its predictive use is "
        "immortal-time-biased and is handled time-varyingly in the landmark CV.")}

    work = df.copy()
    work["n_lines_total"] = [
        timeline.lines_accrued_by(pid, 1e9) for pid in work["patient_id"]
    ]

    # (1) n_lines alone.
    try:
        d1 = work[["duration", "event", "n_lines_total"]].dropna()
        cph1 = CoxPHFitter(penalizer=0.0)
        cph1.fit(d1, "duration", "event")
        zph1 = proportional_hazard_test(cph1, d1, time_transform="rank")
        row = zph1.summary.reset_index()
        for _, r in row.iterrows():
            out["tests"].append({
                "covariate": str(r.get("index", r.iloc[0])),
                "test_statistic": float(r["test_statistic"]),
                "p": float(r["p"]),
                "model": "n_lines_total_only",
                "violates_ph": bool(r["p"] < 0.05),
            })
    except Exception as e:  # pragma: no cover
        out["tests"].append({"covariate": "n_lines_total", "error": str(e)})

    # (2) clinical-baseline only (expect PH to HOLD => no violation).
    try:
        cc = [c for c in clinical_cols if c in work.columns and work[c].nunique() > 1]
        if cc:
            d2 = work[["duration", "event"] + cc].dropna()
            cph2 = CoxPHFitter(penalizer=0.1)
            cph2.fit(d2, "duration", "event")
            zph2 = proportional_hazard_test(cph2, d2, time_transform="rank")
            row = zph2.summary.reset_index()
            n_viol = int((row["p"] < 0.05).sum())
            out["clinical_baseline_n_covariates"] = len(cc)
            out["clinical_baseline_n_ph_violations"] = n_viol
    except Exception as e:  # pragma: no cover
        out["clinical_baseline_error"] = str(e)

    return out


# ----------------------------------------------------------------- bootstrap CI
def bootstrap_delta_ci(vec_treat, vec_static, B=2000, seed=0):
    """Paired bootstrap CI of mean(td-AUC_treat - td-AUC_static) across folds.

    Returns (delta_mean, lo, hi, frac_positive). Folds are paired (same split), so we
    resample fold indices. ``nan`` if too few paired folds.
    """
    a = np.asarray(vec_treat, float)
    b = np.asarray(vec_static, float)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a, b = a[:n], b[:n]
    d = a - b
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        means.append(float(d[idx].mean()))
    means = np.asarray(means)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), float((means > 0).mean()))


# ----------------------------------------------------------------------- figure
def make_figure(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    landmarks = sorted(summary["landmark"].unique())
    models = ["static_cox", "timevarying_cox", "treatment_basin"]
    colors = {"static_cox": "#888888", "timevarying_cox": "#1f77b4",
              "treatment_basin": "#d62728"}

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    width = 0.25
    x = np.arange(len(landmarks))
    for i, m in enumerate(models):
        means, errs = [], []
        for L in landmarks:
            cell = summary[(summary["landmark"] == L) & (summary["model"] == m)]
            if len(cell):
                means.append(float(cell["td_auc_mean"].iloc[0]))
                errs.append(float(cell["td_auc_std"].iloc[0]))
            else:
                means.append(np.nan); errs.append(0.0)
        ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=3,
               label=m, color=colors[m], alpha=0.9)
    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={int(L)}d" for L in landmarks])
    ax.set_ylabel("Forward time-dependent AUC (mean +/- std over folds)")
    ax.set_title("Lane #2 — landmark forward td-AUC (immortal-time-safe)")
    vals = summary["td_auc_mean"].to_numpy(float)
    top = 0.75 if not np.isfinite(vals).any() else max(0.75, float(np.nanmax(vals)) + 0.05)
    ax.set_ylim(0.4, top)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[lane2] loading mmSYGNAL IA12 + GDC treatment timeline ...")
    df, program_cols, clinical_cols, timeline = load_lane2_frame()
    print(f"[lane2] joined cohort: n={len(df)} patients, "
          f"events={int(df['event'].sum())}, "
          f"programs={len(program_cols)}, clinical={len(clinical_cols)}")
    print(f"[lane2] timeline summary: {timeline.summary}")

    builder = make_feature_builder(timeline, program_cols, clinical_cols)
    models = make_models(penalizer=1.0, basin_epochs=300)

    print("[lane2] running landmark CV (this trains a basin net per landmark/fold) ...")
    summary, fold_results = run_landmark_cv(
        df, models, builder,
        landmarks=DEFAULT_LANDMARKS, horizons=DEFAULT_HORIZONS,
        k=5, seed=SEED, min_test_events=5,
    )

    # --- pre-registered gate: treatment model beats static on forward td-AUC, CI-sep ---
    gate = {"name": "treatment_nonph_beats_static",
            "criterion": ("a treatment model (timevarying_cox or treatment_basin) beats "
                          "static_cox on forward td-AUC at >=1 landmark with paired-fold "
                          "bootstrap CI of the delta strictly > 0"),
            "per_landmark": [], "verdict": "BLOCKED"}

    any_pass = False
    for L in sorted(summary["landmark"].unique()):
        static_vec = fold_td_auc_vectors(fold_results, L, "static_cox")
        rec = {"landmark": float(L)}
        for m in ("timevarying_cox", "treatment_basin"):
            mvec = fold_td_auc_vectors(fold_results, L, m)
            dm, lo, hi, fp = bootstrap_delta_ci(mvec, static_vec, B=2000, seed=SEED)
            ci_separated = bool(lo > 0) if not np.isnan(lo) else False
            rec[m] = {"delta_mean": dm, "ci_lo": lo, "ci_hi": hi,
                      "frac_folds_positive": fp, "ci_separated_above_zero": ci_separated}
            any_pass = any_pass or ci_separated
        gate["per_landmark"].append(rec)
    gate["verdict"] = "PASS" if any_pass else "BLOCKED"

    gt = grambsch_therneau_evidence(df, program_cols, clinical_cols, timeline)

    # --- diagnostic: did the PH-free basin mixture collapse to a single basin? ---
    basin_diag = {"collapsed_at_landmarks": [], "note": (
        "A basin td-AUC pinned at ~0.500 with near-zero fold variance is the "
        "single-basin collapse documented in docs/LANE2_TREATMENT_NONPH.md: the "
        "partial-log-rank objective on this wide, low-signal substrate puts all soft "
        "mass on one basin, yielding a constant risk. Reported honestly, not hidden.")}
    for L in sorted(summary["landmark"].unique()):
        cell = summary[(summary["landmark"] == L) & (summary["model"] == "treatment_basin")]
        if len(cell):
            au = float(cell["td_auc_mean"].iloc[0]); sd = float(cell["td_auc_std"].iloc[0])
            if np.isfinite(au) and abs(au - 0.5) < 1e-3 and sd < 1e-3:
                basin_diag["collapsed_at_landmarks"].append(int(L))

    # --- persist ---
    results = {
        "cohort": {
            "n_patients": int(len(df)),
            "n_events": int(df["event"].sum()),
            "n_programs": len(program_cols),
            "n_clinical": len(clinical_cols),
            "timeline_summary": timeline.summary,
        },
        "landmarks": list(DEFAULT_LANDMARKS),
        "horizons": list(DEFAULT_HORIZONS),
        "summary_table": json.loads(summary.to_json(orient="records")),
        "grambsch_therneau": gt,
        "basin_collapse_diagnostic": basin_diag,
        "gate": gate,
    }
    json_path = os.path.join(OUT_DIR, "landmark_results.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # --- markdown table ---
    md = _markdown_report(summary, gate, gt, results["cohort"], basin_diag)
    md_path = os.path.join(OUT_DIR, "landmark_results.md")
    with open(md_path, "w") as fh:
        fh.write(md)

    # --- figure ---
    fig_path = os.path.join(OUT_DIR, "fig_lane2_landmark.png")
    try:
        make_figure(summary, fig_path)
    except Exception as e:
        print(f"[lane2] figure failed (non-fatal): {e}")
        fig_path = None

    print("\n" + md)
    print(f"\n[lane2] wrote {json_path}")
    print(f"[lane2] wrote {md_path}")
    if fig_path:
        print(f"[lane2] wrote {fig_path}")
    print(f"[lane2] GATE verdict: {gate['verdict']}")
    return 0


def _markdown_report(summary, gate, gt, cohort, basin_diag=None):
    lines = ["# Lane #2 — treatment-conditioned PH-free landmark results\n"]
    lines.append(f"- Cohort: n={cohort['n_patients']} patients, "
                 f"events={cohort['n_events']}, programs={cohort['n_programs']}, "
                 f"clinical={cohort['n_clinical']}.")
    lines.append("- Immortal-time-safe LANDMARK design: at each L, restrict to patients "
                 "event-free & at-risk at L; features = static x + Z(L) (history<=L only); "
                 "forward survival from L.\n")

    lines.append("## Forward time-dependent AUC by landmark (mean +/- std over folds)\n")
    lines.append("| Landmark (d) | static_cox | timevarying_cox | treatment_basin | "
                 "n_at_risk | n_events |")
    lines.append("|---|---|---|---|---|---|")
    for L in sorted(summary["landmark"].unique()):
        cells = {}
        nrisk = nev = 0
        for m in ("static_cox", "timevarying_cox", "treatment_basin"):
            c = summary[(summary["landmark"] == L) & (summary["model"] == m)]
            if len(c):
                cells[m] = f"{c['td_auc_mean'].iloc[0]:.3f}±{c['td_auc_std'].iloc[0]:.3f}"
                nrisk = int(c["n_atrisk"].iloc[0]); nev = int(c["n_events_total"].iloc[0])
            else:
                cells[m] = "n/a"
        lines.append(f"| {int(L)} | {cells['static_cox']} | {cells['timevarying_cox']} | "
                     f"{cells['treatment_basin']} | {nrisk} | {nev} |")

    lines.append("\n## Integrated Brier Score by landmark (lower better)\n")
    lines.append("| Landmark (d) | static_cox | timevarying_cox | treatment_basin |")
    lines.append("|---|---|---|---|")
    for L in sorted(summary["landmark"].unique()):
        cells = {}
        for m in ("static_cox", "timevarying_cox", "treatment_basin"):
            c = summary[(summary["landmark"] == L) & (summary["model"] == m)]
            cells[m] = f"{c['ibs_mean'].iloc[0]:.3f}" if len(c) else "n/a"
        lines.append(f"| {int(L)} | {cells['static_cox']} | {cells['timevarying_cox']} | "
                     f"{cells['treatment_basin']} |")

    lines.append("\n## Pre-registered gate: `treatment_nonph_beats_static`\n")
    lines.append(f"**Criterion:** {gate['criterion']}\n")
    lines.append("| Landmark | model | delta td-AUC vs static | 95% CI | CI>0? |")
    lines.append("|---|---|---|---|---|")
    for rec in gate["per_landmark"]:
        for m in ("timevarying_cox", "treatment_basin"):
            d = rec[m]
            lines.append(
                f"| {int(rec['landmark'])} | {m} | {d['delta_mean']:.3f} | "
                f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | "
                f"{'YES' if d['ci_separated_above_zero'] else 'no'} |")
    lines.append(f"\n**GATE VERDICT: {gate['verdict']}**\n")

    lines.append("## Grambsch-Therneau PH evidence\n")
    for t in gt.get("tests", []):
        if "error" in t:
            lines.append(f"- {t['covariate']}: error — {t['error']}")
        else:
            lines.append(f"- `{t['covariate']}` ({t['model']}): "
                         f"chi2={t['test_statistic']:.3f}, p={t['p']:.3g}, "
                         f"violates_PH={t['violates_ph']}")
    if "clinical_baseline_n_ph_violations" in gt:
        lines.append(f"- clinical baseline: "
                     f"{gt['clinical_baseline_n_ph_violations']}/"
                     f"{gt['clinical_baseline_n_covariates']} covariates violate PH "
                     f"(expect ~0 — baseline PH holds).")
    lines.append(f"\n_{gt.get('notes', '')}_\n")

    if basin_diag is not None:
        lines.append("## PH-free basin model — collapse diagnostic\n")
        coll = basin_diag.get("collapsed_at_landmarks", [])
        if coll:
            lines.append(f"- `treatment_basin` collapsed to a single basin at landmark(s) "
                         f"{coll} (td-AUC == 0.500, ~0 fold variance).")
        else:
            lines.append("- `treatment_basin` did not show the single-basin collapse "
                         "signature at any landmark.")
        lines.append(f"\n_{basin_diag.get('note', '')}_\n")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
