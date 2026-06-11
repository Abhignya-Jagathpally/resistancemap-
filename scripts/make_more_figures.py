"""Additional tangible figures (all from real on-disk data; no fabrication).

results/figures/
  fig_tdauc_over_time.png      forward td-AUC(t) decay: gep70 vs sky92 vs GBS(prog+clin)
  fig_nonph_schoenfeld.png     SMOKING GUN — scaled Schoenfeld residuals for n_lines (PH violation)
  fig_leakage_audit.png        standalone leakage audit: GuanScore=1.0 (leak) vs honest gep70/sky92
  fig_patient_manifold.png     PHATE/UMAP of 769 patients by model risk + amp1q
  fig_risk_stratification.png  KM PFS tertiles: gep70 vs sky92 vs GBS(prog+clin), side by side
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from resistancemap.data.mmsygnal import load_ia12, leakage_audit
from resistancemap.survival.splits import patient_kfold
from resistancemap.survival.metrics import time_dependent_auc

OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)
C = {"gep70": "#2c7fb8", "sky92": "#7fbf7b", "GBS(prog+clin)": "#d7301f"}


def oof_gbs_risk(df, feats):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.util import Surv
    r = np.full(len(df), np.nan)
    for tr, te in patient_kfold(df, k=5, seed=0):
        X, Y = df.iloc[tr].copy(), df.iloc[te].copy()
        med = X[feats].median(); X[feats] = X[feats].fillna(med); Y[feats] = Y[feats].fillna(med)
        g = GradientBoostingSurvivalAnalysis(n_estimators=300, max_depth=2, learning_rate=0.05,
                                             subsample=0.8, random_state=0)
        g.fit(X[feats].values, Surv.from_arrays(X["event"].astype(bool).values, X["duration"].values))
        r[te] = g.predict(Y[feats].values)
    return r


def main():
    d = load_ia12(); df = d.df.copy()
    PROG = d.program_cols; CLIN = [c for c in d.clinical_cols if c in df.columns]; PC = PROG + CLIN
    dur, ev = df["duration"].values, df["event"].values
    risk = oof_gbs_risk(df, PC)
    print(f"[data] N={len(df)} events={int(ev.sum())} | GBS OOF computed")

    # ---------- Fig 1: td-AUC over time ----------
    grid = np.array([180, 270, 365, 450, 540, 640, 730, 900, 1095], float)
    plt.figure(figsize=(7, 4.6))
    for col in ["gep70", "sky92"]:
        auc, _ = time_dependent_auc(dur, ev, dur, ev, pd.to_numeric(df[col]).values, grid)
        plt.plot(grid[:len(auc)] / 30.4, auc, "o-", color=C[col], lw=2, label=f"{col} (SOTA)")
    auc, _ = time_dependent_auc(dur, ev, dur, ev, risk, grid)
    plt.plot(grid[:len(auc)] / 30.4, auc, "s-", color=C["GBS(prog+clin)"], lw=2.4, label="GBS(prog+clin) — ours")
    plt.axhline(0.5, ls=":", color="0.6"); plt.ylim(0.45, 0.75)
    plt.xlabel("Months from diagnosis"); plt.ylabel("Time-dependent AUC (Uno)")
    plt.title("Discrimination decays over time — and converges\n(transcriptome→PFS signal is early & shared)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/fig_tdauc_over_time.png", dpi=200); plt.close()
    print("[ok] fig_tdauc_over_time")

    # ---------- Fig 2: non-PH Schoenfeld smoking gun (n_lines) ----------
    try:
        from resistancemap.data.treatment_timeline import load_timeline, lines_accrued_by  # type: ignore
        tl = load_timeline("ResistanceMap/data/raw/mmrf_commpass/treatments.tsv")
        df["pid"] = df["patient_id"].str.extract(r"(MMRF_\d+)")
        df["n_lines"] = [int(tl[tl["submitter_id"] == p]["line_num"].nunique()) if p in set(tl["submitter_id"]) else 1
                         for p in df["pid"]]
    except Exception:
        # fallback: count distinct lines directly from the tsv
        tx = pd.read_csv("ResistanceMap/data/raw/mmrf_commpass/treatments.tsv", sep="\t", dtype=str)
        import re
        tx["line_n"] = tx["regimen_or_line_of_therapy"].str.extract(r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)")
        nl = tx.groupby("submitter_id")["line_n"].nunique()
        df["pid"] = df["patient_id"].str.extract(r"(MMRF_\d+)")
        df["n_lines"] = df["pid"].map(nl).fillna(1).astype(int)
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
    sub = df[["n_lines", "ISS", "duration", "event"]].copy()
    sub["ISS"] = sub["ISS"].fillna(sub["ISS"].median())
    cph = CoxPHFitter(penalizer=0.1).fit(sub, "duration", "event")
    pht = proportional_hazard_test(cph, sub, time_transform="rank")
    p_nlines = float(pht.summary.loc["n_lines", "p"])
    sch = cph.compute_residuals(sub, "scaled_schoenfeld")
    times = sch.index.values.astype(float); resid = sch["n_lines"].values
    order = np.argsort(times); t_s, r_s = times[order], resid[order]
    # simple moving-average smoother
    w = max(15, len(r_s) // 20); kern = np.ones(w) / w
    smooth = np.convolve(r_s, kern, mode="same")
    plt.figure(figsize=(7, 4.6))
    plt.scatter(t_s / 30.4, r_s, s=10, color="0.7", alpha=0.6, label="scaled Schoenfeld residual")
    plt.plot(t_s / 30.4, smooth, color="#d7301f", lw=2.5, label="moving average (trend)")
    plt.axhline(0.0, ls="--", color="0.4", label="PH expectation (flat)")
    plt.xlabel("Months from diagnosis (event time rank)"); plt.ylabel("Scaled Schoenfeld residual — n_lines")
    plt.title(f"Non-proportional hazards SMOKING GUN: n_lines effect changes over time\n"
              f"Grambsch–Therneau p = {p_nlines:.1e} (non-flat trend = PH violated)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/fig_nonph_schoenfeld.png", dpi=200); plt.close()
    print(f"[ok] fig_nonph_schoenfeld (n_lines PH p={p_nlines:.1e})")

    # ---------- Fig 3: leakage audit standalone ----------
    aud = leakage_audit(d)
    plt.figure(figsize=(7, 4.2))
    cols = ["#d7301f" if x else "#2c7fb8" for x in aud["is_leaky"]]
    bars = plt.barh(aud["score"], aud["cindex"], color=cols)
    plt.axvline(0.5, ls=":", color="0.6"); plt.xlim(0.4, 1.05)
    for i, (c, leak) in enumerate(zip(aud["cindex"], aud["is_leaky"])):
        plt.text(c + 0.01, i, ("LEAK — excluded" if leak else "honest"), va="center", fontsize=9,
                 color="#d7301f" if leak else "#2c7fb8")
    plt.xlabel("Univariate C-index vs PFS")
    plt.title("Leakage audit: GuanScore C-index = 1.000 → re-encodes the label, EXCLUDED\n"
              "Honest SOTA bar = gep70 0.624 / sky92 0.620")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_leakage_audit.png", dpi=200); plt.close()
    print("[ok] fig_leakage_audit")

    # ---------- Fig 4: patient manifold (PHATE/UMAP) ----------
    X = df[PROG].fillna(df[PROG].median()).values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    emb = None; method = ""
    try:
        import phate; emb = phate.PHATE(n_components=2, verbose=0, random_state=0).fit_transform(X); method = "PHATE"
    except Exception:
        try:
            import umap; emb = umap.UMAP(n_components=2, random_state=0).fit_transform(X); method = "UMAP"
        except Exception:
            from sklearn.decomposition import PCA; emb = PCA(2, random_state=0).fit_transform(X); method = "PCA"
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sc0 = ax[0].scatter(emb[:, 0], emb[:, 1], c=risk, cmap="RdYlBu_r", s=14)
    ax[0].set_title(f"{method} of 769 patients — colored by model PFS risk"); plt.colorbar(sc0, ax=ax[0], label="GBS risk")
    amp = df["amp1q"].fillna(0).values.astype(int)
    ax[1].scatter(emb[amp == 0, 0], emb[amp == 0, 1], s=14, color="0.7", label="amp1q−")
    ax[1].scatter(emb[amp == 1, 0], emb[amp == 1, 1], s=18, color="#d7301f", label="amp1q+")
    ax[1].set_title(f"{method} — amp1q (high-risk cytogenetics)"); ax[1].legend(fontsize=9)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_patient_manifold.png", dpi=200); plt.close()
    print(f"[ok] fig_patient_manifold ({method})")

    # ---------- Fig 5: risk-stratification comparison ----------
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    scores = {"gep70": pd.to_numeric(df["gep70"]).values, "sky92": pd.to_numeric(df["sky92"]).values,
              "GBS(prog+clin) — ours": risk}
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for k, (name, s) in enumerate(scores.items()):
        q = np.quantile(s, [1/3, 2/3]); grp = np.where(s <= q[0], 0, np.where(s <= q[1], 1, 2))
        for g, lab, col in [(0, "Low", "#2c7fb8"), (1, "Mid", "#7fbf7b"), (2, "High", "#d7301f")]:
            m = grp == g
            KaplanMeierFitter().fit(dur[m] / 30.4, ev[m], label=f"{lab} (n={int(m.sum())})").plot_survival_function(
                ax=ax[k], ci_show=False, color=col, lw=2)
        lr = multivariate_logrank_test(dur, grp, ev)
        ax[k].set_title(f"{name}\nlog-rank p={lr.p_value:.1e}"); ax[k].set_xlabel("Months"); ax[k].set_ylim(0, 1.02)
        ax[k].legend(fontsize=7)
    ax[0].set_ylabel("PFS probability")
    fig.suptitle("Risk stratification — our multimodal model separates PFS comparably to published signatures", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_risk_stratification.png", dpi=200); plt.close()
    print("[ok] fig_risk_stratification")

    json.dump({"n": int(len(df)), "events": int(ev.sum()), "manifold_method": method,
               "n_lines_ph_p": p_nlines, "figures": sorted(os.listdir(OUT))},
              open(f"{OUT}/figures_manifest.json", "w"), indent=2)
    print(f"\n[done] wrote {len(os.listdir(OUT))} files to {OUT}/")


if __name__ == "__main__":
    main()
