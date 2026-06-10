"""Characterize what the model's risk tertiles *are*, biologically and clinically.

Pipeline
--------
1. Stratify patients into Low / Mid / High risk **tertiles** using the model's
   out-of-fold (OOF) risk score (patient-disjoint CV; no patient scored by a
   model that saw it). Tertile cut-points are empirical risk quantiles.
2. For each tertile, report:
   - **Top differential programs** vs the rest of the cohort, ranked by a
     standardized mean difference (Cohen's d effect size). These name the
     transcriptional programs that distinguish high-risk patients.
   - **Cytogenetic enrichment**: per-flag positive proportions + a 2x2 Fisher
     exact test (tertile vs rest) for each cytogenetic marker.
   - **ISS distribution** (mean ISS + per-stage proportions).
   - **First-line regimen distribution** from the real MMRF treatments table
     (PI / IMiD / CD38 / other drug-class membership of the first line).
3. **KM separation**: per-tertile Kaplan-Meier PFS and a multivariate log-rank
   test p-value across the three tertiles.

All numbers are computed from the real frame + the real treatments.tsv. Fisher
tests fall back to a normal-approx two-proportion z-test only if SciPy is absent
(it is present here), and that fallback is labelled in the output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Drug-class membership for first-line regimen labelling.
DRUG_CLASSES = {
    "PI": {"bortezomib", "carfilzomib", "ixazomib"},
    "IMiD": {"lenalidomide", "pomalidomide", "thalidomide"},
    "CD38": {"daratumumab", "isatuximab", "elotuzumab"},
}
# Default location of the real MMRF treatments table (relative to pipeline3 root).
DEFAULT_TREATMENTS_TSV = (
    "ResistanceMap/data/raw/mmrf_commpass/treatments.tsv"
)
TERTILE_NAMES = ["Low", "Mid", "High"]


@dataclass
class TertileCharacterization:
    """All characterization tables, keyed by analysis."""

    tertile_of: pd.Series                       # patient_id -> tertile name
    risk_cutpoints: tuple[float, float]          # (q33, q67) of OOF risk
    differential_programs: pd.DataFrame          # program effect sizes per tertile
    cyto_enrichment: pd.DataFrame                # per-flag proportions + Fisher p
    iss_distribution: pd.DataFrame               # ISS mean + per-stage proportions
    regimen_distribution: pd.DataFrame           # drug-class proportions per tertile
    km_curves: dict                              # tertile -> {"t":[], "s":[]}
    logrank_p: float | None
    median_pfs_days: dict                        # tertile -> median PFS (days)
    n_per_tertile: dict
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------- treatments
def _norm_submitter(sample_id: str) -> str:
    """mmSYGNAL sample 'MMRF_2754_1_BM' -> MMRF submitter_id 'MMRF_2754'."""
    parts = str(sample_id).split("_")
    if len(parts) >= 2 and parts[0].upper().startswith("MMRF"):
        return f"{parts[0]}_{parts[1]}"
    return str(sample_id)


def load_first_line_regimens(
    treatments_tsv: str = DEFAULT_TREATMENTS_TSV,
) -> pd.DataFrame:
    """Per-patient first-line drug-class membership from the real MMRF table.

    Returns a frame indexed by ``submitter_id`` with boolean columns
    ``PI``, ``IMiD``, ``CD38`` (any agent of that class appeared in the patient's
    *first line of therapy*) and ``agents`` (the raw agent string). Returns an
    empty frame (and a flag) if the file is absent — never fabricates regimens.
    """
    if not os.path.exists(treatments_tsv):
        return pd.DataFrame(
            columns=["submitter_id", "PI", "IMiD", "CD38", "agents"]
        ).set_index("submitter_id")

    tr = pd.read_csv(treatments_tsv, sep="\t", dtype=str).fillna("")
    first = tr[tr["regimen_or_line_of_therapy"].str.contains(
        "First", case=False, na=False)]
    rows = []
    for sub, grp in first.groupby("submitter_id"):
        agents_raw = " ; ".join(a for a in grp["therapeutic_agents"] if a)
        toks = {
            t.strip().lower()
            for a in grp["therapeutic_agents"]
            for t in a.replace(",", ";").split(";")
            if t.strip()
        }
        rows.append({
            "submitter_id": sub,
            "PI": bool(toks & DRUG_CLASSES["PI"]),
            "IMiD": bool(toks & DRUG_CLASSES["IMiD"]),
            "CD38": bool(toks & DRUG_CLASSES["CD38"]),
            "agents": agents_raw,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["submitter_id", "PI", "IMiD", "CD38", "agents"]
        ).set_index("submitter_id")
    return out.set_index("submitter_id")


# ---------------------------------------------------------------- statistics
def _fisher_or_ztest(a: int, b: int, c: int, d: int) -> tuple[float, str]:
    """2x2 [[a,b],[c,d]] -> (two-sided p, method). a=in-group positive,
    b=in-group negative, c=rest positive, d=rest negative."""
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]])
        return float(p), "fisher_exact"
    except Exception:
        # Two-proportion z-test fallback (labelled).
        n1, n2 = a + b, c + d
        if n1 == 0 or n2 == 0:
            return float("nan"), "ztest_fallback"
        p1, p2 = a / n1, c / n2
        pp = (a + c) / (n1 + n2)
        se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        if se == 0:
            return 1.0, "ztest_fallback"
        z = (p1 - p2) / se
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return float(p), "ztest_fallback"


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Standardized mean difference (in-group x vs rest y), pooled SD."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled)


def _km_curve(durations: np.ndarray, events: np.ndarray):
    """Pure-numpy Kaplan-Meier estimator -> (event_times, survival, median)."""
    order = np.argsort(durations)
    t = durations[order]
    e = events[order]
    uniq = np.unique(t)
    n = len(t)
    surv = []
    s = 1.0
    at_risk = n
    median = None
    idx = 0
    for ut in uniq:
        d_i = int(((t == ut) & (e == 1)).sum())
        n_i = int((t >= ut).sum())
        if n_i > 0:
            s *= (1.0 - d_i / n_i)
        surv.append(s)
        if median is None and s <= 0.5:
            median = float(ut)
        idx += 1
    return uniq.astype(float), np.asarray(surv), median


# ---------------------------------------------------------------- main
def characterize_risk_tertiles(
    df: pd.DataFrame,
    oof_risk: np.ndarray,
    program_cols: list,
    clinical_cols: list,
    treatments_tsv: str = DEFAULT_TREATMENTS_TSV,
    top_programs: int = 12,
    program_label_fn=None,
) -> TertileCharacterization:
    """Stratify by OOF risk tertile and characterize each tertile.

    Parameters
    ----------
    df:
        Cohort frame with ``patient_id``/``sample``, ``duration``, ``event``,
        program + clinical columns.
    oof_risk:
        Out-of-fold model risk per row (higher = worse), aligned to ``df`` rows.
    program_cols / clinical_cols:
        Feature column ids.
    treatments_tsv:
        Path to the real MMRF treatments table for regimen membership.
    top_programs:
        Number of top differential programs to surface per tertile.
    program_label_fn:
        Optional callable mapping a column id to a human label.
    """
    if program_label_fn is None:
        program_label_fn = lambda c: str(c)  # noqa: E731

    df = df.reset_index(drop=True).copy()
    risk = np.asarray(oof_risk, dtype=float)
    finite = np.isfinite(risk)
    df = df.loc[finite].reset_index(drop=True)
    risk = risk[finite]

    q33, q67 = np.quantile(risk, [1 / 3, 2 / 3])
    tert = np.where(risk <= q33, "Low", np.where(risk <= q67, "Mid", "High"))
    df["_tertile"] = tert
    notes: list[str] = []

    # patient_id key for regimen join
    id_col = "patient_id" if "patient_id" in df.columns else "sample"
    df["_submitter"] = df[id_col].map(_norm_submitter)
    tertile_of = pd.Series(tert, index=df[id_col].values, name="tertile")

    # --- differential programs (effect size, tertile vs rest) ---
    diff_rows = []
    for tn in TERTILE_NAMES:
        in_mask = df["_tertile"].values == tn
        for col in program_cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            x = vals[in_mask & np.isfinite(vals)]
            y = vals[(~in_mask) & np.isfinite(vals)]
            d = _cohens_d(x, y)
            diff_rows.append({
                "tertile": tn,
                "program": col,
                "label": program_label_fn(col),
                "mean_in": float(x.mean()) if len(x) else float("nan"),
                "mean_rest": float(y.mean()) if len(y) else float("nan"),
                "cohens_d": d,
                "abs_d": abs(d),
            })
    diff_all = pd.DataFrame(diff_rows)
    # keep top_programs per tertile by |effect size|
    differential_programs = (
        diff_all.sort_values(["tertile", "abs_d"], ascending=[True, False])
        .groupby("tertile", group_keys=False)
        .head(top_programs)
        .reset_index(drop=True)
    )

    # --- cytogenetic enrichment (Fisher per flag, per tertile) ---
    cyto_flags = [c for c in
                  ["del13", "del17p", "del1p", "amp1q", "CCND1", "FGFR3", "WHSC1"]
                  if c in df.columns]
    cyto_rows = []
    for tn in TERTILE_NAMES:
        in_mask = df["_tertile"].values == tn
        for flag in cyto_flags:
            f = pd.to_numeric(df[flag], errors="coerce").fillna(0)
            pos_in = int((f[in_mask] > 0).sum())
            n_in = int(in_mask.sum())
            pos_rest = int((f[~in_mask] > 0).sum())
            n_rest = int((~in_mask).sum())
            p, method = _fisher_or_ztest(
                pos_in, n_in - pos_in, pos_rest, n_rest - pos_rest)
            cyto_rows.append({
                "tertile": tn, "flag": flag,
                "prop_in": round(pos_in / n_in, 4) if n_in else float("nan"),
                "prop_rest": round(pos_rest / n_rest, 4) if n_rest else float("nan"),
                "n_pos_in": pos_in, "n_in": n_in,
                "fisher_p": None if not np.isfinite(p) else round(p, 4),
                "method": method,
            })
    cyto_enrichment = pd.DataFrame(cyto_rows)
    if cyto_enrichment["method"].eq("ztest_fallback").any():
        notes.append("SciPy absent: cytogenetic p-values use a z-test fallback.")

    # --- ISS distribution ---
    iss_rows = []
    if "ISS" in df.columns:
        iss = pd.to_numeric(df["ISS"], errors="coerce")
        for tn in TERTILE_NAMES:
            in_mask = df["_tertile"].values == tn
            sub = iss[in_mask].dropna()
            row = {"tertile": tn, "mean_ISS": round(float(sub.mean()), 3)
                   if len(sub) else None, "n_with_ISS": int(len(sub))}
            for stage in (1, 2, 3):
                row[f"prop_ISS{stage}"] = (
                    round(float((sub == stage).mean()), 4) if len(sub) else None)
            iss_rows.append(row)
    iss_distribution = pd.DataFrame(iss_rows)

    # --- first-line regimen distribution ---
    reg = load_first_line_regimens(treatments_tsv)
    reg_rows = []
    if reg.empty:
        notes.append(
            f"Treatments table not found at {treatments_tsv}; regimen "
            "distribution omitted (no fabrication).")
    else:
        joined = df[["_submitter", "_tertile"]].join(
            reg[["PI", "IMiD", "CD38"]], on="_submitter")
        matched = joined.dropna(subset=["PI"])
        notes.append(
            f"First-line regimen matched for {len(matched)}/{len(df)} patients "
            "via MMRF submitter_id.")
        for tn in TERTILE_NAMES:
            sub = matched[matched["_tertile"] == tn]
            row = {"tertile": tn, "n_matched": int(len(sub))}
            for cls in ("PI", "IMiD", "CD38"):
                row[f"prop_{cls}"] = (
                    round(float(sub[cls].astype(bool).mean()), 4)
                    if len(sub) else None)
            reg_rows.append(row)
    regimen_distribution = pd.DataFrame(reg_rows)

    # --- KM + multivariate log-rank ---
    km_curves = {}
    median_pfs = {}
    for tn in TERTILE_NAMES:
        in_mask = df["_tertile"].values == tn
        dur = df.loc[in_mask, "duration"].to_numpy(float)
        ev = df.loc[in_mask, "event"].to_numpy(int)
        if len(dur) == 0:
            continue
        t, s, med = _km_curve(dur, ev)
        km_curves[tn] = {"t": t.tolist(), "s": s.tolist()}
        median_pfs[tn] = med

    logrank_p = None
    try:
        from lifelines.statistics import multivariate_logrank_test
        res = multivariate_logrank_test(
            df["duration"].to_numpy(float),
            df["_tertile"].to_numpy(),
            df["event"].to_numpy(int),
        )
        logrank_p = float(res.p_value)
    except Exception as e:  # pragma: no cover
        notes.append(f"log-rank unavailable: {type(e).__name__}")

    n_per = {tn: int((df["_tertile"].values == tn).sum())
             for tn in TERTILE_NAMES}

    return TertileCharacterization(
        tertile_of=tertile_of,
        risk_cutpoints=(float(q33), float(q67)),
        differential_programs=differential_programs,
        cyto_enrichment=cyto_enrichment,
        iss_distribution=iss_distribution,
        regimen_distribution=regimen_distribution,
        km_curves=km_curves,
        logrank_p=logrank_p,
        median_pfs_days=median_pfs,
        n_per_tertile=n_per,
        notes=notes,
    )
