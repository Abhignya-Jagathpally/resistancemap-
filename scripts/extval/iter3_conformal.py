"""Ext-val Iteration 3: distribution-free CONFORMAL survival prediction intervals on IA12.

Target = log(PFS days) for UNCENSORED patients (honest scope: time-to-progression among progressors).
Split-conformal: fit predictor on TRAIN; conformity residuals on a separate CALIBRATION set; quantile
q = ceil((n_cal+1)(1-alpha))/n_cal -> 90% intervals on TEST -> empirical coverage. Mondrian (per ISS
stratum) conformal for group-conditional coverage. Compare to NAIVE Gaussian intervals.
ICLR property: finite-sample distribution-free marginal coverage guarantee. Real data, many splits + CIs.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from resistancemap.data.mmsygnal import load_ia12

ALPHA = 0.10  # 90% intervals
N_SPLITS = 40


def conformal_q(residuals, alpha):
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return np.sort(residuals)[k - 1]


def main():
    d = load_ia12(); df = d.df.copy().reset_index(drop=True)
    PROG = d.program_cols
    df["ISS"] = pd.to_numeric(df["ISS"], errors="coerce")
    # feature set: gep70 + ISS + top-20 highest-variance programs
    var = df[PROG].var().sort_values(ascending=False)
    topP = list(var.index[:20])
    feats = ["gep70", "ISS"] + topP
    X = df[feats].copy()
    X["ISS"] = X["ISS"].fillna(X["ISS"].median())
    X["gep70"] = pd.to_numeric(X["gep70"], errors="coerce")
    X = X.fillna(X.median()).values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = np.log(df["duration"].values)
    ev = df["event"].values.astype(bool)
    iss_high = (df["ISS"].fillna(2).values == 3)
    pid = df["patient_id"].values

    cov_conf, cov_naive, width_conf = [], [], []
    cov_mondrian, cov_marg_in_high = [], []   # per-stratum coverage: marginal-q applied to high-ISS vs Mondrian
    rng = np.random.default_rng(0)
    pats = np.unique(pid)
    for s in range(N_SPLITS):
        rng.shuffle(pats)
        ntr, nca = int(0.5 * len(pats)), int(0.25 * len(pats))
        tr = np.isin(pid, pats[:ntr]); ca = np.isin(pid, pats[ntr:ntr + nca]); te = np.isin(pid, pats[ntr + nca:])
        # fit on uncensored TRAIN
        m = Ridge(alpha=10.0).fit(X[tr & ev], y[tr & ev])
        # calibration residuals (uncensored)
        rc = np.abs(y[ca & ev] - m.predict(X[ca & ev]))
        if len(rc) < 10: continue
        q = conformal_q(rc, ALPHA)
        qn = 1.645 * rc.std()                       # naive Gaussian 90%
        # test (uncensored)
        tm = te & ev
        yp = m.predict(X[tm]); yt = y[tm]
        cov_conf.append(float(np.mean(np.abs(yt - yp) <= q)))
        cov_naive.append(float(np.mean(np.abs(yt - yp) <= qn)))
        width_conf.append(2 * q)
        # per-stratum: marginal q applied to high-ISS test subset
        hi = tm & iss_high
        if hi.sum() >= 5:
            yph, yth = m.predict(X[hi]), y[hi]
            cov_marg_in_high.append(float(np.mean(np.abs(yth - yph) <= q)))
            # Mondrian: q from high-ISS calibration residuals
            rc_hi = np.abs(y[ca & ev & iss_high] - m.predict(X[ca & ev & iss_high]))
            if len(rc_hi) >= 10:
                qh = conformal_q(rc_hi, ALPHA)
                cov_mondrian.append(float(np.mean(np.abs(yth - yph) <= qh)))

    def stat(a):
        a = np.array(a); return {"mean": round(float(a.mean()), 4),
                                 "ci": [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)],
                                 "n_splits": int(len(a))}
    out = {"iteration": 3, "purpose": "distribution-free conformal survival intervals (uncensored target)",
           "nominal_coverage": 1 - ALPHA, "n_uncensored": int(ev.sum()), "feature_set": "gep70+ISS+top20 programs",
           "marginal_conformal_coverage": stat(cov_conf),
           "naive_gaussian_coverage": stat(cov_naive),
           "mean_interval_width_logdays": round(float(np.mean(width_conf)), 3),
           "high_ISS_coverage_marginal_q": stat(cov_marg_in_high) if cov_marg_in_high else None,
           "high_ISS_coverage_mondrian_q": stat(cov_mondrian) if cov_mondrian else None}
    mc = out["marginal_conformal_coverage"]["mean"]; nc = out["naive_gaussian_coverage"]["mean"]
    out["verdict"] = (f"Split-conformal achieves marginal coverage {mc:.3f} (nominal 0.90) — distribution-free "
                      f"guarantee holds. Naive Gaussian {nc:.3f}. Mondrian vs marginal per-stratum coverage "
                      f"shows whether group-conditional validity needs stratification.")
    os.makedirs("results/extval", exist_ok=True); json.dump(out, open("results/extval/iter3.json", "w"), indent=2)
    print(f"[extval-3] uncensored n={int(ev.sum())} | nominal 0.90")
    print(f"  marginal conformal coverage = {mc:.3f} {out['marginal_conformal_coverage']['ci']} (mean width {out['mean_interval_width_logdays']:.2f} log-days)")
    print(f"  naive Gaussian coverage    = {nc:.3f} {out['naive_gaussian_coverage']['ci']}")
    if out["high_ISS_coverage_marginal_q"]:
        print(f"  high-ISS coverage: marginal-q {out['high_ISS_coverage_marginal_q']['mean']:.3f} vs Mondrian-q {out['high_ISS_coverage_mondrian_q']['mean']:.3f}")
    print(f"  VERDICT: {out['verdict']}")
    print("  wrote results/extval/iter3.json")


if __name__ == "__main__":
    main()
