"""Tangible interpretability pass for the GBS(prog+clin) PFS model (MMRF IA12).

Answers, with real numbers + figures: *what drives progression risk in this
model, biologically and clinically?*

Outputs (results/interpretability/):
  fig_risk_drivers.png            top-20 features by permutation C-index drop
  fig_risk_tertile_heatmap.png    top differential programs x risk tertile
  fig_tertile_km.png              KM PFS by risk tertile (+ log-rank, annotations)
  fig_calibration.png             reliability at 1y / 2y
  fig_program_partial_dependence.png  partial dependence of risk on top-3 programs
  interpretability.json           all ranked numbers

Model = sksurv GradientBoostingSurvivalAnalysis(n_estimators=300, max_depth=2,
learning_rate=0.05, subsample=0.8, random_state=0) on the 141 programs + 10
clinical features; PFS endpoint; 5-fold patient-disjoint CV (same features/CV as
run_sota_comparison.py). No fabricated numbers; the only RNG use is the seeded
column-shuffle inside permutation importance (the method itself).

Run from the pipeline3 ROOT:
    venv/bin/python resistancemap/scripts/interpret_pfs.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

# Allow running from pipeline3 root OR from resistancemap/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
sys.path.insert(0, _SRC)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from resistancemap.data.mmsygnal import load_ia12  # noqa: E402
from resistancemap.survival.splits import patient_kfold  # noqa: E402
from resistancemap.survival.baselines import SkSurvBaseline  # noqa: E402
from resistancemap.interpretability import (  # noqa: E402
    permutation_cindex_importance,
    program_label,
    characterize_risk_tertiles,
    landmark_reliability,
    program_partial_dependence,
)

warnings.filterwarnings("ignore")

K = 5
SEED = 0
LANDMARKS = (365.0, 730.0)        # 1y / 2y
GRID = np.array([180, 365, 540, 730, 1095], dtype=float)
# Anchor outputs to the resistancemap repo (scripts/ -> repo root), so the run
# is independent of the launch CWD. Data paths stay relative to the pipeline3
# root (where this is run from, per CLAUDE.md), so we resolve them too.
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT_DIR = os.path.join(_REPO_ROOT, "results", "interpretability")
# Treatments table lives under the pipeline3 root; try a few known locations.
_TREAT_CANDIDATES = [
    "ResistanceMap/data/raw/mmrf_commpass/treatments.tsv",
    os.path.join(_REPO_ROOT, "..", "ResistanceMap",
                 "data", "raw", "mmrf_commpass", "treatments.tsv"),
]
TREATMENTS_TSV = next((p for p in _TREAT_CANDIDATES if os.path.exists(p)),
                      _TREAT_CANDIDATES[0])

PALETTE = {"Low": "#2c7fb8", "Mid": "#fec44f", "High": "#d7301f"}


def make_gbs(features):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    return SkSurvBaseline(
        GradientBoostingSurvivalAnalysis(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=0),
        "gbs", features=features)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_ia12()
    df = data.df.reset_index(drop=True).copy()
    PROG = list(data.program_cols)
    CLIN = [c for c in data.clinical_cols if c in df.columns]
    PC = PROG + CLIN
    N = len(df)
    print(f"[IA12] N={N} events={data.n_events} programs={len(PROG)} clin={len(CLIN)}")

    dur = df["duration"].to_numpy(float)
    ev = df["event"].to_numpy(int)

    # ---- OOF risk + OOF S(h) curves (5-fold patient-disjoint) ----
    oof_risk = np.full(N, np.nan)
    oof_surv = np.full((N, len(GRID)), np.nan)   # S(t) at GRID per patient
    fold_cindex = []
    # we also stash one fold's (model, test_df) for permutation importance + PDP
    pi_model = None
    pi_test_df = None
    pi_train_df = None

    for fi, (tr_idx, te_idx) in enumerate(patient_kfold(df, k=K, seed=SEED)):
        tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
        med = tr[PC].median(numeric_only=True)
        tr[PC] = tr[PC].fillna(med)
        te[PC] = te[PC].fillna(med)
        m = make_gbs(PC).fit(tr)
        r = m.risk(te)
        oof_risk[te_idx] = r
        # survival curves at GRID
        fns = m.model.predict_survival_function(te[PC].values)
        oof_surv[te_idx] = np.array([[float(fn(t)) for t in GRID] for fn in fns])
        from sksurv.metrics import concordance_index_censored
        c = concordance_index_censored(
            te["event"].astype(bool).values, te["duration"].values, r)[0]
        fold_cindex.append(float(c))
        print(f"  fold {fi}: test C-index={c:.4f}")
        if fi == 0:
            pi_model, pi_test_df, pi_train_df = m, te, tr

    from resistancemap.survival.metrics import cindex_bootstrap
    oof_c, oof_lo, oof_hi = cindex_bootstrap(dur, ev, oof_risk, B=1000, seed=1)
    print(f"[OOF] C-index={oof_c:.4f} [{oof_lo:.4f},{oof_hi:.4f}] "
          f"mean-fold={np.mean(fold_cindex):.4f}")

    # ============================================================ (a) drivers
    print("\n[1/5] permutation C-index-drop importance (held-out fold 0) ...")
    pi = permutation_cindex_importance(
        pi_model, pi_test_df, PC, program_cols=set(PROG),
        n_repeats=10, seed=0)
    top20 = pi.top(20)
    print(f"  base held-out C-index={pi.base_cindex:.4f} (n_test={pi.n_test})")
    print("  top-10 risk-driving features:")
    for _, row in pi.top(10).iterrows():
        print(f"    {int(row['rank']):2d}. {row['label']:14s} "
              f"({row['block']:8s}) drop={row['importance']:+.4f} "
              f"+/-{row['importance_std']:.4f}")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    yv = np.arange(len(top20))[::-1]
    colors = ["#542788" if b == "clinical" else "#1b7837"
              for b in top20["block"]]
    ax.barh(yv, top20["importance"], xerr=top20["importance_std"],
            color=colors, ecolor="#555555", capsize=2)
    ax.set_yticks(yv)
    ax.set_yticklabels(top20["label"], fontsize=8)
    ax.set_xlabel("Permutation importance  (held-out C-index drop)")
    ax.set_title("Top-20 progression-risk drivers — GBS(prog+clin), PFS\n"
                 f"base held-out C-index = {pi.base_cindex:.3f}", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#1b7837", label="program"),
                       Patch(color="#542788", label="clinical")],
              loc="lower right", fontsize=8, frameon=False)
    ax.axvline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_risk_drivers.png", dpi=150)
    plt.close(fig)

    # ============================================================ (c) tertiles
    print("\n[2/5] characterize OOF risk tertiles ...")
    ch = characterize_risk_tertiles(
        df, oof_risk, PROG, CLIN, treatments_tsv=TREATMENTS_TSV,
        top_programs=12, program_label_fn=program_label)
    print(f"  tertile N: {ch.n_per_tertile}  risk cuts={tuple(round(x,3) for x in ch.risk_cutpoints)}")
    print(f"  log-rank p across tertiles = {ch.logrank_p}")
    print(f"  median PFS (days): {ch.median_pfs_days}")
    for nt in ch.notes:
        print("  note:", nt)

    # ----- heatmap: union of top differential programs x tertile -----
    dp = ch.differential_programs
    top_progs = (dp.sort_values("abs_d", ascending=False)
                 .drop_duplicates("program").head(15)["program"].tolist())
    order = ["Low", "Mid", "High"]
    mat = np.full((len(top_progs), 3), np.nan)
    labels = []
    for i, p in enumerate(top_progs):
        labels.append(program_label(p))
        for j, tn in enumerate(order):
            sel = dp[(dp["program"] == p) & (dp["tertile"] == tn)]
            if len(sel):
                mat[i, j] = sel["cohens_d"].values[0]
    fig, ax = plt.subplots(figsize=(5.4, 7.0))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(3)); ax.set_xticklabels(order)
    ax.set_yticks(range(len(top_progs))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Top differential programs by risk tertile\n"
                 "(Cohen's d, tertile vs rest)", fontsize=10)
    for i in range(len(top_progs)):
        for j in range(3):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(mat[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's d")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_risk_tertile_heatmap.png", dpi=150)
    plt.close(fig)

    # ----- KM by tertile + annotation -----
    fig, (axk, axa) = plt.subplots(
        1, 2, figsize=(11.5, 5.4), gridspec_kw={"width_ratios": [1.5, 1.0]})
    for tn in order:
        if tn not in ch.km_curves:
            continue
        t = np.array(ch.km_curves[tn]["t"]); s = np.array(ch.km_curves[tn]["s"])
        t = np.concatenate([[0.0], t]); s = np.concatenate([[1.0], s])
        med = ch.median_pfs_days.get(tn)
        lab = (f"{tn} risk (n={ch.n_per_tertile[tn]}, "
               f"med={int(med) if med else 'NR'}d)")
        axk.step(t / 30.44, s, where="post", color=PALETTE[tn], lw=2, label=lab)
    axk.set_xlabel("Months since baseline"); axk.set_ylabel("PFS (S(t))")
    axk.set_ylim(0, 1.02)
    pstr = (f"log-rank p = {ch.logrank_p:.2e}"
            if ch.logrank_p is not None else "log-rank p = n/a")
    axk.set_title(f"PFS by model risk tertile\n{pstr}", fontsize=10)
    axk.legend(fontsize=8, frameon=False, loc="upper right")
    axk.grid(alpha=0.25)

    # annotation panel: cyto + ISS + regimen proportions per tertile
    axa.axis("off")
    lines = ["High-risk-tertile enrichment (vs rest):", ""]
    high_cyto = ch.cyto_enrichment[ch.cyto_enrichment["tertile"] == "High"]
    high_cyto = high_cyto.sort_values("fisher_p")
    for _, r in high_cyto.head(4).iterrows():
        star = " *" if (r["fisher_p"] is not None and r["fisher_p"] < 0.05) else ""
        lines.append(f"  {r['flag']:7s}: {r['prop_in']*100:4.1f}% vs "
                     f"{r['prop_rest']*100:4.1f}%  (p={r['fisher_p']}){star}")
    if len(ch.iss_distribution):
        lines.append("")
        for tn in order:
            row = ch.iss_distribution[ch.iss_distribution["tertile"] == tn]
            if len(row):
                lines.append(f"  ISS mean [{tn:4s}]: {row['mean_ISS'].values[0]}")
    if len(ch.regimen_distribution):
        lines.append("")
        lines.append("  First-line drug class (% of tertile):")
        for tn in order:
            row = ch.regimen_distribution[
                ch.regimen_distribution["tertile"] == tn]
            if len(row):
                pi_ = row["prop_PI"].values[0]; im_ = row["prop_IMiD"].values[0]
                cd_ = row["prop_CD38"].values[0]
                fmt = lambda v: f"{v*100:.0f}%" if v is not None else "n/a"  # noqa: E731
                lines.append(f"    {tn:4s}: PI {fmt(pi_)}  IMiD {fmt(im_)}  "
                             f"CD38 {fmt(cd_)}")
    axa.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9,
             family="monospace")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_tertile_km.png", dpi=150)
    plt.close(fig)

    # ============================================================ (d) calibration
    print("\n[3/5] landmark reliability (1y / 2y) ...")
    reliability = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0))
    for ax, h in zip(axes, LANDMARKS):
        j = int(np.where(GRID == h)[0][0])
        Sh = oof_surv[:, j]
        ok = np.isfinite(Sh)
        pts, ece = landmark_reliability(Sh[ok], dur[ok], ev[ok], h, n_bins=5)
        reliability[int(h)] = {
            "ece": None if not np.isfinite(ece) else round(float(ece), 4),
            "bins": [{"mean_pred": round(p.mean_pred, 4),
                      "obs_risk": None if not np.isfinite(p.obs_risk)
                      else round(p.obs_risk, 4), "n": p.n} for p in pts],
        }
        xs = [p.mean_pred for p in pts]
        ys = [p.obs_risk for p in pts]
        ns = [p.n for p in pts]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(xs, ys, "o-", color="#d7301f", lw=2)
        for x, y, n in zip(xs, ys, ns):
            if np.isfinite(y):
                ax.annotate(f"n={n}", (x, y), fontsize=7,
                            xytext=(3, 4), textcoords="offset points")
        ax.set_xlabel("Predicted P(progress by h)")
        ax.set_ylabel("Observed (KM) P(progress by h)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Reliability @ {int(h/365)}y "
                     f"(ECE={reliability[int(h)]['ece']})", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(fontsize=8, frameon=False)
        print(f"  {int(h)}d: ECE={reliability[int(h)]['ece']}")
    fig.suptitle("Calibration of OOF progression-risk (censoring-honest KM bins)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_calibration.png", dpi=150)
    plt.close(fig)

    # ============================================================ (e) partial dep
    print("\n[4/5] partial dependence on top-3 programs ...")
    top3 = [r for r in pi.top(30)["feature"].tolist()
            if r in set(PROG)][:3]
    pdp_json = {}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    # build a fully-imputed frame for PDP (matches training imputation)
    pdp_df = pi_test_df  # already median-imputed within fold 0
    for ax, feat in zip(axes, top3):
        pdp = program_partial_dependence(pi_model, pdp_df, feat, PC, n_grid=20)
        ax.plot(pdp["grid_value"], pdp["mean_risk"], color="#1b7837", lw=2)
        ax.fill_between(pdp["grid_value"],
                        pdp["mean_risk"] - pdp["risk_std"],
                        pdp["mean_risk"] + pdp["risk_std"],
                        color="#1b7837", alpha=0.15)
        ax.set_xlabel(f"{program_label(feat)} activity")
        ax.set_ylabel("Mean predicted risk")
        ax.set_title(program_label(feat), fontsize=10)
        ax.grid(alpha=0.25)
        pdp_json[program_label(feat)] = pdp.to_dict(orient="records")
    fig.suptitle("Partial dependence of progression risk on top-3 programs "
                 "(fold-0 model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig_program_partial_dependence.png", dpi=150)
    plt.close(fig)

    # ============================================================ JSON
    print("\n[5/5] writing interpretability.json ...")
    out = {
        "model": "GBS(prog+clin) GradientBoostingSurvivalAnalysis "
                 "(n_estimators=300, max_depth=2, lr=0.05, subsample=0.8)",
        "endpoint": "PFS (MMRF-CoMMpass IA12, days)",
        "n_patients": int(N),
        "n_events": int(data.n_events),
        "n_programs": len(PROG),
        "n_clinical": len(CLIN),
        "k_folds": K,
        "oof_cindex": {"point": round(float(oof_c), 4),
                       "ci_lo": round(float(oof_lo), 4),
                       "ci_hi": round(float(oof_hi), 4),
                       "mean_fold": round(float(np.mean(fold_cindex)), 4),
                       "per_fold": [round(c, 4) for c in fold_cindex]},
        "permutation_importance": {
            "method": "held-out C-index drop (sksurv concordance), "
                      "10 repeats, seed=0, fold-0 test split",
            "base_cindex": round(pi.base_cindex, 4),
            "n_test": pi.n_test,
            "ranked": pi.table[["rank", "feature", "label", "block",
                                "importance", "importance_std"]]
                .round(5).to_dict(orient="records"),
            "top10": pi.top(10)[["rank", "label", "block",
                                 "importance"]].round(5)
                .to_dict(orient="records"),
        },
        "risk_tertiles": {
            "risk_cutpoints": [round(x, 5) for x in ch.risk_cutpoints],
            "n_per_tertile": ch.n_per_tertile,
            "median_pfs_days": {k: (None if v is None else round(v, 1))
                                for k, v in ch.median_pfs_days.items()},
            "logrank_p": ch.logrank_p,
            "differential_programs":
                ch.differential_programs.round(4).to_dict(orient="records"),
            "cyto_enrichment":
                ch.cyto_enrichment.to_dict(orient="records"),
            "iss_distribution":
                ch.iss_distribution.to_dict(orient="records"),
            "regimen_distribution":
                ch.regimen_distribution.to_dict(orient="records"),
            "notes": ch.notes,
        },
        "calibration": reliability,
        "partial_dependence": {"top3_programs": [program_label(f) for f in top3],
                               "curves": pdp_json},
        "figures": [
            "fig_risk_drivers.png", "fig_risk_tertile_heatmap.png",
            "fig_tertile_km.png", "fig_calibration.png",
            "fig_program_partial_dependence.png",
        ],
    }
    with open(f"{OUT_DIR}/interpretability.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[ok] wrote {OUT_DIR}/interpretability.json + 5 PNGs")
    return out


if __name__ == "__main__":
    main()
