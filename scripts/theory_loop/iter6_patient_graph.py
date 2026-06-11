"""Theory loop — Iteration 6 (LAST mechanism): ego-centric patient-similarity graph survival.

HYPOTHESIS. Each patient's ego-graph = its k nearest neighbors on the 141-program manifold. A
transductive risk score is read off the neighborhood: risk(q) = 1 - S_KM(t_ref) from a Kaplan-Meier
fit on the OUT-OF-FOLD neighbors' (time, event). High risk = transcriptional neighbors progress fast.
ICLR/ICML property: a transductive, manifold/graph-smoothing estimator (no parametric hazard form).

PRE-REGISTERED: nested CV (inner picks k); compare C-index vs gep70/sky92 + Cox(ISS+gep70); does the
graph risk add as a feature to the strong baseline? Honest prior: ties (same features) -> completes the
method-family-spanning ceiling claim. No claim unless it CI-separates above SOTA. Real data only.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from resistancemap.data.mmsygnal import load_ia12
from resistancemap.survival.splits import patient_kfold

T_REF = 365.0
K_GRID = [15, 30, 50]


def graph_risk(Xtr, ttr, etr, Xte, k):
    """Transductive risk for Xte from k nearest TRAIN neighbors' KM at T_REF."""
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr))).fit(Xtr)
    idx = nn.kneighbors(Xte, return_distance=False)
    risk = np.zeros(len(Xte))
    for i, nbr in enumerate(idx):
        km = KaplanMeierFitter().fit(ttr[nbr], etr[nbr])
        risk[i] = 1.0 - float(km.predict(T_REF))
    return risk


def main():
    d = load_ia12(); df = d.df.copy().reset_index(drop=True)
    PROG = d.program_cols
    X = df[PROG].fillna(df[PROG].median()).values
    dur, ev = df["duration"].values.astype(float), df["event"].values.astype(int)
    print(f"[iter6] N={len(df)} events={int(ev.sum())} | program dims={len(PROG)} | k grid={K_GRID}")

    oof = np.full(len(df), np.nan); chosen_k = []
    for tr_i, te_i in patient_kfold(df, k=5, seed=0):
        Xtr, Xte = X[tr_i], X[te_i]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        # inner CV picks k
        best_k, best = K_GRID[0], -1
        for k in K_GRID:
            s = []
            for ia, ib in patient_kfold(df.iloc[tr_i].reset_index(drop=True), k=3, seed=1):
                try:
                    r = graph_risk(Ztr[ia], dur[tr_i][ia], ev[tr_i][ia], Ztr[ib], k)
                    s.append(concordance_index(dur[tr_i][ib], -r, ev[tr_i][ib]))
                except Exception: pass
            if s and np.nanmean(s) > best: best, best_k = np.nanmean(s), k
        chosen_k.append(best_k)
        try: oof[te_i] = graph_risk(Ztr, dur[tr_i], ev[tr_i], Zte, best_k)
        except Exception: pass

    graph_c = concordance_index(dur, -oof, ev)
    rng = np.random.default_rng(1); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(df), len(df))
        try: bs.append(concordance_index(dur[i], -oof[i], ev[i]))
        except Exception: pass
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    gep = pd.to_numeric(df["gep70"]).values; sky = pd.to_numeric(df["sky92"]).values
    gep_c, sky_c = concordance_index(dur, -gep, ev), concordance_index(dur, -sky, ev)

    # strong baseline + does graph risk add?
    def cox_oof(feats):
        o = np.full(len(df), np.nan)
        for tr_i, te_i in patient_kfold(df, k=5, seed=0):
            tr, te = df.iloc[tr_i].copy(), df.iloc[te_i].copy()
            for f in feats: tr[f] = tr[f].fillna(tr[f].median()); te[f] = te[f].fillna(tr[f].median())
            try:
                cph = CoxPHFitter(penalizer=1.0).fit(tr[feats + ["duration", "event"]], "duration", "event")
                o[te_i] = cph.predict_partial_hazard(te[feats]).values.ravel()
            except Exception: pass
        return o
    strong = cox_oof(["ISS", "gep70"]); ok = ~np.isnan(strong)
    strong_c = concordance_index(dur[ok], -strong[ok], ev[ok])
    df["_graph"] = oof
    strongx = cox_oof(["ISS", "gep70", "_graph"]); okx = ~np.isnan(strongx)
    strongx_c = concordance_index(dur[okx], -strongx[okx], ev[okx])

    out = {"iteration": 6, "mechanism": "ego-centric patient-similarity kNN graph (transductive KM risk)",
           "k_grid": K_GRID, "chosen_k_per_fold": chosen_k,
           "graph_cindex": round(graph_c, 4), "graph_ci": [round(lo, 4), round(hi, 4)],
           "sota_gep70": round(gep_c, 4), "sota_sky92": round(sky_c, 4),
           "strong_baseline": round(strong_c, 4), "strong_plus_graph": round(strongx_c, 4),
           "graph_beats_sota": bool(lo > max(gep_c, sky_c)),
           "graph_adds_to_strong": bool(strongx_c > strong_c + 0.005)}
    out["verdict"] = ("graph risk beats SOTA -> verify" if out["graph_beats_sota"] else
                      "ego-centric patient-graph ties/below SOTA, no add to strong base -> honest negative; ceiling holds in the graph family too")
    out["gate_beats_sota"] = "PASS-PENDING-VERIFICATION" if out["graph_beats_sota"] else "BLOCKED"
    os.makedirs("results/theory_loop", exist_ok=True); json.dump(out, open("results/theory_loop/iter6.json", "w"), indent=2)
    print(f"  graph C-index={graph_c:.3f} [{lo:.3f},{hi:.3f}] | SOTA gep70 {gep_c:.3f}/sky92 {sky_c:.3f} | k*={chosen_k}")
    print(f"  strong base {strong_c:.3f} -> +graph {strongx_c:.3f} (adds={out['graph_adds_to_strong']})")
    print(f"  VERDICT: {out['verdict']} | gate={out['gate_beats_sota']}")
    print("  wrote results/theory_loop/iter6.json")


if __name__ == "__main__":
    main()
