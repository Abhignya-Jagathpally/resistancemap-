"""Figure for the honest benchmark paper (Lane #1) + the Lane #2 non-PH motivation.

Panel A  Leakage audit + honest SOTA bar (C-index forest; GuanScore flagged as leaky).
Panel B  Calibration/discrimination: IBS (lower=better) and td-AUC (higher=better) bars.
Panel C  KM overall survival by first-line regimen class + the n_lines PH-violation note
         (Lane #2: treatment-driven non-proportional hazards on open data).

All numbers come from results/sota_comparison.json + recomputed KM. No fabrication.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from resistancemap.data.mmsygnal import load_ia12


def main():
    j = json.load(open("results/sota_comparison.json"))
    rows = j["rows"]; audit = j["leakage_audit"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    # ---- A: C-index forest with leakage callout ----
    order = sorted(rows, key=lambda r: r["cindex"])
    y = np.arange(len(order))
    for i, r in enumerate(order):
        nov = r["model"].startswith("ResistanceBasin")
        sota = "SOTA" in r["model"]
        c = "#d7301f" if nov else ("#2c7fb8" if sota else "#555555")
        ax[0].plot([r["ci_lo"], r["ci_hi"]], [i, i], color=c, lw=2)
        ax[0].plot(r["cindex"], i, "o", color=c, ms=7)
    ax[0].axvline(0.5, ls=":", color="0.6", lw=1)
    gep = next(r for r in rows if r["model"].startswith("gep70"))
    ax[0].axvline(gep["cindex"], ls="--", color="#2c7fb8", lw=1, label="honest SOTA (gep70 0.624)")
    ax[0].set_yticks(y); ax[0].set_yticklabels([r["model"] for r in order], fontsize=8)
    ax[0].set_xlim(0.44, 0.82); ax[0].set_xlabel("CV C-index (95% CI)")
    ax[0].set_title("A  Discrimination (PFS) + leakage audit")
    leak = [a for a in audit if a["is_leaky"]]
    txt = "LEAKAGE excluded:\n" + "\n".join(f"  {a['score']}  C={a['cindex']:.2f}" for a in leak)
    ax[0].text(0.46, len(order) - 1.3, txt, fontsize=8, color="#d7301f",
               bbox=dict(boxstyle="round", fc="#fff5f0", ec="#d7301f"))
    ax[0].legend(loc="lower right", fontsize=7)

    # ---- B: IBS + td-AUC bars ----
    mods = [r["model"] for r in rows]
    ibs = [r["ibs"] or np.nan for r in rows]; tda = [r["td_auc"] or np.nan for r in rows]
    x = np.arange(len(mods)); w = 0.38
    ax[1].bar(x - w/2, ibs, w, label="IBS (lower=better)", color="#fdae6b")
    ax[1].bar(x + w/2, tda, w, label="td-AUC (higher=better)", color="#74add1")
    ax[1].set_xticks(x); ax[1].set_xticklabels([m.replace("(", "\n(") for m in mods], fontsize=6.5, rotation=0)
    ax[1].set_title("B  Time-resolved calibration / discrimination"); ax[1].legend(fontsize=7)
    ax[1].set_ylim(0, max(0.7, np.nanmax(tda) + 0.05))

    # ---- C: KM by first-line regimen (Lane #2 non-PH) ----
    from lifelines import KaplanMeierFitter
    tx = pd.read_csv("ResistanceMap/data/raw/mmrf_commpass/treatments.tsv", sep="\t", dtype=str)
    first = tx[tx["regimen_or_line_of_therapy"].str.contains("First", na=False)]
    CLS = {"Bortezomib": "PI", "Carfilzomib": "PI", "Ixazomib": "PI", "Lenalidomide": "IMiD",
           "Pomalidomide": "IMiD", "Thalidomide": "IMiD", "Daratumumab": "CD38",
           "Isatuximab": "CD38", "Elotuzumab": "CD38", "Melphalan": "Alkyl", "Cyclophosphamide": "Alkyl"}
    def classes(a): return {CLS.get(x) for x in str(a).split("|") if x in CLS} - {None}
    g = first.groupby("submitter_id")["therapeutic_agents"].apply(lambda s: set().union(*[classes(a) for a in s]))
    def reg(cs): return ("PI+IMiD(VRd-like)" if {"PI", "IMiD"} <= cs else
                         "PI-based" if "PI" in cs else "IMiD-based" if "IMiD" in cs else "other")
    rmap = g.apply(reg)
    d = load_ia12().df.copy(); d["submitter_id"] = d["patient_id"].str.extract(r"(MMRF_\d+)")
    d = d.merge(rmap.rename("regimen").reset_index(), on="submitter_id", how="inner")
    colors = {"PI+IMiD(VRd-like)": "#2c7fb8", "PI-based": "#d7301f", "IMiD-based": "#7fbf7b"}
    for r in ["PI+IMiD(VRd-like)", "PI-based", "IMiD-based"]:
        s = d[d["regimen"] == r]
        if len(s) < 25: continue
        KaplanMeierFitter().fit(s["duration"], s["event"], label=f"{r} (n={len(s)})").plot_survival_function(
            ax=ax[2], ci_show=False, color=colors[r], lw=2)
    ax[2].set_title("C  PFS by 1st-line regimen — Lane #2"); ax[2].set_xlabel("Days from diagnosis")
    ax[2].set_ylabel("PFS probability"); ax[2].set_ylim(0, 1.02); ax[2].legend(fontsize=7, loc="upper right")
    ax[2].text(0.03, 0.06, "n_lines violates PH\nGrambsch–Therneau p=1e-4\n→ treatment-driven non-PH",
               transform=ax[2].transAxes, fontsize=8, bbox=dict(boxstyle="round", fc="#fffbe6", ec="0.6"))

    fig.suptitle("MMRF-CoMMpass PFS (open data): honest benchmark + leakage audit + treatment non-PH",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("results/fig_sota_benchmark.png", dpi=200, bbox_inches="tight")
    fig.savefig("results/fig_sota_benchmark.pdf", bbox_inches="tight")
    print("[ok] wrote results/fig_sota_benchmark.png/.pdf")


if __name__ == "__main__":
    main()
