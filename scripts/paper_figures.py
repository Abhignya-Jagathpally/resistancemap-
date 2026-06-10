"""Paper figures from REAL CoMMpass results (no fabricated numbers).

Fig 1 (results/fig_paper_main.png), 3 panels:
  A  Kaplan-Meier overall survival stratified by Cox OOF risk tertile (+ log-rank p).
  B  Cross-validated C-index forest plot (5 models, bootstrap 95% CI) vs chance.
  C  2-year survival calibration (reliability) for Cox vs the novel program_basin.

All inputs are recomputed here with the same leakage-safe CV used in paper_metrics.py.
"""
from __future__ import annotations

import os
import sys
import json
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resistancemap.survival.baselines import get_baselines, infer_features, CoxBaseline
from resistancemap.survival.splits import patient_kfold
from resistancemap.models.program_basin import ProgramBasin

warnings.filterwarnings("ignore")
DATA = "data/processed/commpass.csv"
K = 5
H = 730.0  # 2-year landmark for calibration


def oof_risk_and_surv(df, feats, want_surv_at=None):
    """OOF Cox risk + (optional) OOF S(H|x) for cox_ph and program_basin."""
    oof = {"cox_ph": np.full(len(df), np.nan)}
    surv = {"cox_ph": np.full(len(df), np.nan), "program_basin": np.full(len(df), np.nan)}
    from lifelines import KaplanMeierFitter
    for tr_idx, te_idx in patient_kfold(df, k=K, seed=0):
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        med = tr[feats].median(numeric_only=True)
        tr[feats] = tr[feats].fillna(med); te[feats] = te[feats].fillna(med)

        cox = CoxBaseline(features=feats).fit(tr)
        oof["cox_ph"][te_idx] = cox.risk(te)
        if want_surv_at is not None:
            sf = cox.cph.predict_survival_function(te[feats], times=[want_surv_at])
            surv["cox_ph"][te_idx] = sf.values.ravel()
            pb = ProgramBasin(features=feats).fit(tr)
            r_tr = np.asarray(pb.risk(tr), float); mu, sd = r_tr.mean(), (r_tr.std() or 1.0)
            km = KaplanMeierFitter().fit(tr["duration"].values, tr["event"].values)
            s0 = float(np.clip(km.predict(want_surv_at), 1e-6, 1.0))
            lp = (np.asarray(pb.risk(te), float) - mu) / sd
            surv["program_basin"][te_idx] = s0 ** np.exp(lp)
    return oof, surv


def main():
    df = pd.read_csv(DATA).reset_index(drop=True)
    feats = infer_features(df)
    oof, surv = oof_risk_and_surv(df, feats, want_surv_at=H)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # ---- Panel A: KM by Cox risk tertile ----
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    r = oof["cox_ph"]; m = ~np.isnan(r)
    q = np.quantile(r[m], [1/3, 2/3])
    grp = np.where(r <= q[0], 0, np.where(r <= q[1], 1, 2))
    labels = {0: "Low risk (T1)", 1: "Mid risk (T2)", 2: "High risk (T3)"}
    colors = {0: "#2c7fb8", 1: "#7fbf7b", 2: "#d7301f"}
    for g in (0, 1, 2):
        gm = m & (grp == g)
        km = KaplanMeierFitter().fit(df["duration"].values[gm], df["event"].values[gm],
                                     label=f"{labels[g]} (n={int(gm.sum())})")
        km.plot_survival_function(ax=ax[0], ci_show=False, color=colors[g], lw=2)
    lr = multivariate_logrank_test(df["duration"].values[m], grp[m], df["event"].values[m])
    ax[0].set_title("A  Overall survival by Cox risk tertile")
    ax[0].set_xlabel("Days from diagnosis"); ax[0].set_ylabel("Survival probability")
    ax[0].set_ylim(0, 1.02)
    ax[0].text(0.04, 0.06, f"log-rank p = {lr.p_value:.1e}", transform=ax[0].transAxes,
               fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax[0].legend(loc="upper right", fontsize=8)

    # ---- Panel B: C-index forest ----
    pm = json.load(open("results/paper_metrics_real.json"))
    rows = pm["rows"]
    names = [x["model"] for x in rows]
    cidx = [x["cindex"] for x in rows]
    lo = [x["ci_lo"] for x in rows]; hi = [x["ci_hi"] for x in rows]
    y = np.arange(len(rows))[::-1]
    for yi, x, l, h, nm in zip(y, cidx, lo, hi, names):
        c = "#d7301f" if nm == "program_basin" else "#2c7fb8"
        ax[1].plot([l, h], [yi, yi], color=c, lw=2)
        ax[1].plot(x, yi, "o", color=c, ms=7)
    ax[1].axvline(0.5, ls="--", color="0.5", lw=1, label="chance")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([n + ("\n(novel)" if n == "program_basin" else "") for n in names], fontsize=8)
    ax[1].set_xlim(0.45, 0.82)
    ax[1].set_xlabel("CV C-index (95% bootstrap CI)")
    ax[1].set_title("B  Discrimination — patient-disjoint CV")
    ax[1].legend(loc="lower right", fontsize=8)

    # ---- Panel C: 2-year calibration ----
    for nm, col in (("cox_ph", "#2c7fb8"), ("program_basin", "#d7301f")):
        s = surv[nm]; mm = ~np.isnan(s)
        p_event = 1.0 - s[mm]
        dur, ev = df["duration"].values[mm], df["event"].values[mm]
        known = (dur >= H) | (ev == 1)
        yb = ((ev == 1) & (dur <= H)).astype(float)[known]
        pb = p_event[known]
        bins = np.quantile(pb, np.linspace(0, 1, 6))
        bins[-1] += 1e-9
        idx = np.clip(np.digitize(pb, bins[1:-1]), 0, 4)
        xs, ys = [], []
        for b in range(5):
            bm = idx == b
            if bm.sum() >= 5:
                xs.append(pb[bm].mean()); ys.append(yb[bm].mean())
        ax[2].plot(xs, ys, "o-", color=col, lw=2, ms=6,
                   label=f"{nm}" + (" (novel)" if nm == "program_basin" else ""))
    ax[2].plot([0, 1], [0, 1], ls="--", color="0.5", lw=1, label="ideal")
    ax[2].set_xlim(0, 0.8); ax[2].set_ylim(0, 0.8)
    ax[2].set_xlabel("Predicted 2-yr death probability")
    ax[2].set_ylabel("Observed 2-yr death fraction")
    ax[2].set_title("C  Calibration @ 2 years")
    ax[2].legend(loc="upper left", fontsize=8)

    fig.suptitle(f"ResistanceMap (open-data): MMRF-CoMMPASS overall survival "
                 f"(N={len(df)}, {int(df['event'].sum())} events)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("results/fig_paper_main.png", dpi=200, bbox_inches="tight")
    fig.savefig("results/fig_paper_main.pdf", bbox_inches="tight")
    print(f"[ok] wrote results/fig_paper_main.png/.pdf  (log-rank p={lr.p_value:.2e})")


if __name__ == "__main__":
    main()
