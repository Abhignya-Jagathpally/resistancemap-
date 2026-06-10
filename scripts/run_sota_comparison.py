"""Path B+C+D — head-to-head vs honest SOTA on MMRF-CoMMpass PFS (IA12).

B (calibration/time-resolution): C-index + IBS + time-dependent AUC + ECE, bootstrap CIs.
C (generalization): handled by run_transfer_ia18.py.
D (integration): a stacked super-learner over programs+clinical+gep70+sky92.

Honesty: gep70/sky92 are the SOTA bar (independently-derived). GuanScore/risk_auc are
mmSYGNAL in-sample (leakage; reported by leakage_audit, never used as the bar). The
'win' we claim is comparable discrimination + better time-resolved calibration/AUC and
the documented non-proportional-hazards regime — NOT a CI-separated C-index victory
(that is ceiling-bound at ~0.62 for transcriptome->PFS).

Outputs: results/sota_comparison.json, results/sota_table.md.
"""
from __future__ import annotations

import os
import sys
import json
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from resistancemap.data.mmsygnal import load_ia12, leakage_audit
from resistancemap.survival.splits import patient_kfold
from resistancemap.survival.metrics import (
    cindex_bootstrap, integrated_brier_score, time_dependent_auc,
)
from resistancemap.survival.calibration import expected_calibration_error
from resistancemap.survival.baselines import CoxBaseline, SkSurvBaseline

warnings.filterwarnings("ignore")
K = 5
GRID = np.array([180, 365, 540, 730, 1095], dtype=float)   # days
LANDMARKS = (365.0, 730.0)


# ---------------------------------------------------------------- survival curve adapters
def cox_score_curve(tr, te, score, times):
    """Single-covariate Cox on a published score -> S(t|x) for IBS/AUC."""
    from lifelines import CoxPHFitter
    d = tr[[score, "duration", "event"]].copy()
    cph = CoxPHFitter(penalizer=0.01).fit(d, "duration", "event")
    return cph.predict_survival_function(te[[score]], times=times).values.T


def sksurv_curve(model, te, feats, times):
    fns = model.model.predict_survival_function(te[feats].values)
    return np.array([[float(fn(t)) for t in times] for fn in fns])


def cox_curve(model, te, feats, times):
    return model.cph.predict_survival_function(te[feats], times=times).values.T


def main():
    data = load_ia12()
    df = data.df
    PROG = data.program_cols
    CLIN = [c for c in data.clinical_cols if c in df.columns]
    PC = PROG + CLIN
    STACK = PC + ["gep70", "sky92"]                          # D: integration
    N = len(df)
    print(f"[IA12] N={N} | events={data.n_events} | programs={len(PROG)} clin={len(CLIN)}")

    audit = leakage_audit(data)
    print("\n=== Leakage audit (univariate C-index vs PFS) ===")
    print(audit.to_string(index=False))

    # model registry: name -> (feature list, fold-fitter -> (risk, surv_fn))
    def fit_cox(feats):
        return CoxBaseline(penalizer=0.5, l1_ratio=0.1, features=feats)

    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv

    def fit_rsf(feats):
        return SkSurvBaseline(RandomSurvivalForest(
            n_estimators=400, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=0), "rsf", features=feats)

    def fit_gbs(feats):
        return SkSurvBaseline(GradientBoostingSurvivalAnalysis(
            n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
            random_state=0), "gbs", features=feats)

    from resistancemap.models.resistance_basin_lr import ResistanceBasinLR

    # collectors
    models = ["gep70(SOTA)", "sky92(SOTA)", "Cox(prog+clin)", "RSF(prog+clin)",
              "GBS(prog+clin)", "Stacked(+gep70+sky92)", "ResistanceBasin-LR(novel)"]
    oof_risk = {m: np.full(N, np.nan) for m in models}
    surv = {m: np.full((N, len(GRID)), np.nan) for m in models}

    for fi, (tr_idx, te_idx) in enumerate(patient_kfold(df, k=K, seed=0)):
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        med = tr[STACK].median(numeric_only=True)
        tr[STACK] = tr[STACK].fillna(med); te[STACK] = te[STACK].fillna(med)

        # SOTA published scores (risk = the score; curve via 1-cov Cox)
        for sc, nm in (("gep70", "gep70(SOTA)"), ("sky92", "sky92(SOTA)")):
            oof_risk[nm][te_idx] = te[sc].values
            try: surv[nm][te_idx] = cox_score_curve(tr, te, sc, GRID)
            except Exception as e: print(f"  f{fi} {nm} curve fail {type(e).__name__}")

        # Cox(prog+clin)
        cx = fit_cox(PC).fit(tr); oof_risk["Cox(prog+clin)"][te_idx] = cx.risk(te)
        surv["Cox(prog+clin)"][te_idx] = cox_curve(cx, te, PC, GRID)

        # RSF / GBS / Stacked (sksurv)
        for nm, feats, mk in (("RSF(prog+clin)", PC, fit_rsf),
                              ("GBS(prog+clin)", PC, fit_gbs),
                              ("Stacked(+gep70+sky92)", STACK, fit_gbs)):
            m = mk(feats).fit(tr)
            oof_risk[nm][te_idx] = m.risk(te)
            try: surv[nm][te_idx] = sksurv_curve(m, te, feats, GRID)
            except Exception as e: print(f"  f{fi} {nm} curve fail {type(e).__name__}")

        # ResistanceBasin-LR (novel, PH-free)
        rb = ResistanceBasinLR(features=PC, n_basins=3, epochs=400, seed=fi).fit(tr)
        oof_risk["ResistanceBasin-LR(novel)"][te_idx] = rb.risk(te)
        surv["ResistanceBasin-LR(novel)"][te_idx] = rb.survival_function(te, GRID)
        print(f"  fold {fi} done")

    # ---- metrics ----
    dur = df["duration"].values; ev = df["event"].values
    rows = []
    for m in models:
        r = oof_risk[m]; ok = ~np.isnan(r)
        c, lo, hi = cindex_bootstrap(dur[ok], ev[ok], r[ok], B=1000, seed=1)
        S = surv[m]
        ibs = tda = None
        if not np.all(np.isnan(S)):
            Sok = np.clip(S, 1e-6, 1 - 1e-6)
            try:
                ibs = integrated_brier_score(dur, ev, dur, ev, Sok, GRID)
                _, tda = time_dependent_auc(dur, ev, dur, ev, r, GRID)
            except Exception as e:
                print(f"  metric {m}: {type(e).__name__}")
        # ECE @ landmarks
        ece = {}
        for h in LANDMARKS:
            j = int(np.where(GRID == h)[0][0])
            p = 1.0 - S[:, j]
            known = (dur >= h) | (ev == 1)
            y = ((ev == 1) & (dur <= h)).astype(float)
            mok = known & ~np.isnan(p)
            ece[int(h)] = (round(expected_calibration_error(y[mok], p[mok]), 4)
                           if mok.sum() > 30 else None)
        rows.append({"model": m, "cindex": round(c, 4), "ci_lo": round(lo, 4),
                     "ci_hi": round(hi, 4),
                     "ibs": None if ibs is None or not np.isfinite(ibs) else round(float(ibs), 4),
                     "td_auc": None if tda is None or not np.isfinite(tda) else round(float(tda), 4),
                     "ece_365": ece.get(365), "ece_730": ece.get(730)})
        print(f"  {m:28s} C={c:.3f}[{lo:.3f},{hi:.3f}] IBS={rows[-1]['ibs']} "
              f"tdAUC={rows[-1]['td_auc']} ECE365={ece.get(365)} ECE730={ece.get(730)}")

    # ---- paired bootstrap: novel & best-baseline vs gep70 (C-index AND IBS) ----
    def paired(metric_a, metric_b, kind, B=1500):
        rng = np.random.default_rng(7); diffs = []
        for _ in range(B):
            i = rng.integers(0, len(metric_a[0]), len(metric_a[0]))
            try:
                if kind == "cindex":
                    from lifelines.utils import concordance_index
                    da = concordance_index(metric_a[1][i], -metric_a[0][i], metric_a[2][i])
                    db = concordance_index(metric_b[1][i], -metric_b[0][i], metric_b[2][i])
                else:
                    da = integrated_brier_score(metric_a[1][i], metric_a[2][i], metric_a[1][i], metric_a[2][i], metric_a[0][i], GRID)
                    db = integrated_brier_score(metric_b[1][i], metric_b[2][i], metric_b[1][i], metric_b[2][i], metric_b[0][i], GRID)
                diffs.append(da - db)
            except Exception:
                pass
        d = np.array(diffs); p = 2 * min((d < 0).mean(), (d > 0).mean())
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)

    nov = "ResistanceBasin-LR(novel)"
    ok = ~np.isnan(oof_risk[nov]) & ~np.isnan(oof_risk["gep70(SOTA)"])
    paired_tests = {}
    paired_tests["novel_vs_gep70_cindex"] = paired(
        (oof_risk[nov][ok], dur[ok], ev[ok]), (oof_risk["gep70(SOTA)"][ok], dur[ok], ev[ok]), "cindex")
    okS = ok & ~np.isnan(surv[nov][:, 0]) & ~np.isnan(surv["gep70(SOTA)"][:, 0])
    paired_tests["novel_vs_gep70_ibs"] = paired(
        (np.clip(surv[nov][okS], 1e-6, 1 - 1e-6), dur[okS], ev[okS]),
        (np.clip(surv["gep70(SOTA)"][okS], 1e-6, 1 - 1e-6), dur[okS], ev[okS]), "ibs")
    print(f"\n=== Paired ΔC (novel - gep70): {paired_tests['novel_vs_gep70_cindex'][0]:+.4f} "
          f"CI[{paired_tests['novel_vs_gep70_cindex'][1]:+.4f},{paired_tests['novel_vs_gep70_cindex'][2]:+.4f}] "
          f"p={paired_tests['novel_vs_gep70_cindex'][3]:.3f}")
    print(f"=== Paired ΔIBS (novel - gep70): {paired_tests['novel_vs_gep70_ibs'][0]:+.5f} "
          f"CI[{paired_tests['novel_vs_gep70_ibs'][1]:+.5f},{paired_tests['novel_vs_gep70_ibs'][2]:+.5f}] "
          f"p={paired_tests['novel_vs_gep70_ibs'][3]:.3f}  (neg = novel better calibrated)")

    # ---- non-PH evidence (Grambsch-Therneau on the program+clinical Cox) ----
    nonph = {}
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
        sub = df[PC[:20] + ["duration", "event"]].copy()
        sub[PC[:20]] = sub[PC[:20]].fillna(sub[PC[:20]].median())
        cph = CoxPHFitter(penalizer=1.0).fit(sub, "duration", "event")
        pht = proportional_hazard_test(cph, sub, time_transform="rank")
        res = pht.summary
        n_viol = int((res["p"] < 0.05).sum())
        nonph = {"n_covariates_tested": int(len(res)),
                 "n_PH_violations_p<0.05": n_viol,
                 "global_interpretation": "non-PH regime confirmed" if n_viol > 0 else "PH holds"}
        print(f"\n=== Grambsch-Therneau PH test: {n_viol}/{len(res)} covariates violate PH "
              f"(p<0.05) -> {nonph['global_interpretation']}")
    except Exception as e:
        nonph = {"error": f"{type(e).__name__}: {e}"}

    out = {"endpoint": "PFS (MMRF-CoMMpass IA12, days)", "n": N, "n_events": data.n_events,
           "k_folds": K, "leakage_audit": audit.to_dict(orient="records"),
           "rows": rows, "paired_tests": paired_tests, "non_ph_test": nonph,
           "note": "SOTA bar = gep70/sky92 (honest). GuanScore/risk_auc excluded as leaky."}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/sota_comparison.json", "w"), indent=2)

    hdr = "| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |\n|---|---|---|---|---|---|\n"
    body = "".join(f"| {r['model']} | {r['cindex']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] | "
                   f"{r['ibs']} | {r['td_auc']} | {r['ece_365']} | {r['ece_730']} |\n" for r in rows)
    md = (f"# SOTA comparison — MMRF-CoMMpass PFS (IA12)\n\n"
          f"N={N}, {data.n_events} events, {K}-fold patient-disjoint CV. SOTA bar = "
          f"gep70/sky92 (honest). GuanScore/risk_auc are mmSYGNAL in-sample (leakage), excluded.\n\n"
          + hdr + body +
          f"\n**Non-PH:** {nonph.get('global_interpretation', nonph)} "
          f"({nonph.get('n_PH_violations_p<0.05','?')}/{nonph.get('n_covariates_tested','?')} covariates).\n"
          f"**Novel vs gep70:** ΔC={paired_tests['novel_vs_gep70_cindex'][0]:+.4f} "
          f"(p={paired_tests['novel_vs_gep70_cindex'][3]:.3f}); "
          f"ΔIBS={paired_tests['novel_vs_gep70_ibs'][0]:+.5f} "
          f"(p={paired_tests['novel_vs_gep70_ibs'][3]:.3f}, neg=better).\n")
    open("results/sota_table.md", "w").write(md)
    print("\n[ok] wrote results/sota_comparison.json + results/sota_table.md")


if __name__ == "__main__":
    main()
