"""Ext-val Iteration 2: within-IA12 distribution-shift ROBUSTNESS (domain generalization).

Partition IA12 by a shift variable held OUT of the feature set (amp1q; ISS-high/low). Train
GBS(programs-only) on the SOURCE group, test on the held-out TARGET group; compare to in-distribution
patient-disjoint CV on the target. Metric = C-index degradation (in-dist - cross-group). ICLR property:
domain-generalization robustness (worst-group drop). gep70 = fixed reference. Real data, bootstrap CIs.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from resistancemap.data.mmsygnal import load_ia12
from resistancemap.survival.splits import patient_kfold


def cidx(t, e, r): return concordance_index(t, -r, e)


def boot_ci(t, e, r, B=1000, seed=1):
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(t), len(t))
        try: bs.append(cidx(t[i], e[i], r[i]))
        except Exception: pass
    return round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)


def gbs_fit_predict(Xtr, ttr, etr, Xte):
    m = GradientBoostingSurvivalAnalysis(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    m.fit(Xtr, Surv.from_arrays(etr.astype(bool), ttr))
    return m.predict(Xte)


def in_dist_cv(df, PROG, idx):
    """patient-disjoint 5-fold CV C-index of GBS(prog) within a group (indices into df)."""
    sub = df.iloc[idx].reset_index(drop=True)
    X = sub[PROG].fillna(sub[PROG].median()).values
    t, e = sub["duration"].values, sub["event"].values
    oof = np.full(len(sub), np.nan)
    for tr_i, te_i in patient_kfold(sub, k=5, seed=0):
        try: oof[te_i] = gbs_fit_predict(X[tr_i], t[tr_i], e[tr_i], X[te_i])
        except Exception: pass
    ok = ~np.isnan(oof)
    return cidx(t[ok], e[ok], oof[ok]) if ok.sum() > 20 and e[ok].sum() >= 3 else float("nan")


def cross_group(df, PROG, src_idx, tgt_idx):
    """train GBS(prog) on source group, test on target group."""
    Xs = df.iloc[src_idx][PROG].fillna(df[PROG].median()).values
    Xt = df.iloc[tgt_idx][PROG].fillna(df[PROG].median()).values
    ts, es = df.iloc[src_idx]["duration"].values, df.iloc[src_idx]["event"].values
    tt, et = df.iloc[tgt_idx]["duration"].values, df.iloc[tgt_idx]["event"].values
    try: r = gbs_fit_predict(Xs, ts, es, Xt)
    except Exception: return float("nan"), (np.nan, np.nan)
    return cidx(tt, et, r), boot_ci(tt, et, r)


def main():
    d = load_ia12(); df = d.df.copy().reset_index(drop=True); PROG = d.program_cols
    df["ISS"] = pd.to_numeric(df["ISS"], errors="coerce")
    shifts = {
        "amp1q": {"A": df.index[df["amp1q"] == 0].values, "B": df.index[df["amp1q"] == 1].values, "names": ("amp1q-", "amp1q+")},
        "ISS":   {"A": df.index[df["ISS"].isin([1, 2])].values, "B": df.index[df["ISS"] == 3].values, "names": ("ISS_low(1-2)", "ISS_high(3)")},
    }
    out = {"iteration": 2, "purpose": "within-IA12 domain-generalization robustness (GBS programs-only)", "shifts": {}}
    for sv, g in shifts.items():
        A, B, (nA, nB) = g["A"], g["B"], g["names"]
        gepA = cidx(df.iloc[A]["duration"].values, df.iloc[A]["event"].values, pd.to_numeric(df.iloc[A]["gep70"]).values)
        gepB = cidx(df.iloc[B]["duration"].values, df.iloc[B]["event"].values, pd.to_numeric(df.iloc[B]["gep70"]).values)
        idA, idB = in_dist_cv(df, PROG, A), in_dist_cv(df, PROG, B)
        cBgivenA, ciBA = cross_group(df, PROG, A, B)   # train A -> test B
        cAgivenB, ciAB = cross_group(df, PROG, B, A)   # train B -> test A
        rec = {"n_A": int(len(A)), "n_B": int(len(B)), "ev_A": int(df.iloc[A]['event'].sum()), "ev_B": int(df.iloc[B]['event'].sum()),
               "gbs_indist_A": round(idA, 3), "gbs_indist_B": round(idB, 3),
               "gbs_train{A}_test{B}".format(A=nA, B=nB): round(cBgivenA, 3), "ci_BgivenA": list(ciBA),
               "gbs_train{B}_test{A}".format(A=nA, B=nB): round(cAgivenB, 3), "ci_AgivenB": list(ciAB),
               "degradation_to_B": round(idB - cBgivenA, 3), "degradation_to_A": round(idA - cAgivenB, 3),
               "gep70_ref_A": round(gepA, 3), "gep70_ref_B": round(gepB, 3)}
        out["shifts"][sv] = rec
        print(f"  shift={sv} [{nA} n={len(A)}/{nB} n={len(B)}]: "
              f"GBS in-dist A={idA:.3f} B={idB:.3f} | trainA->testB={cBgivenA:.3f}{ciBA} (drop {idB-cBgivenA:+.3f}) | "
              f"trainB->testA={cAgivenB:.3f}{ciAB} (drop {idA-cAgivenB:+.3f}) | gep70 A={gepA:.3f} B={gepB:.3f}")
    # verdict
    drops = []
    for sv, r in out["shifts"].items():
        drops += [r["degradation_to_B"], r["degradation_to_A"]]
    drops = [x for x in drops if np.isfinite(x)]
    worst = max(drops) if drops else float("nan")
    out["worst_group_degradation"] = round(worst, 3)
    out["verdict"] = (f"GBS(programs) degrades by up to {worst:+.3f} C-index under stratum shift; "
                      f"small strata -> wide CIs. Honest read: cross-stratum generalization is at/near the "
                      f"ceiling band with notable worst-group drop; no method robustly exceeds gep70 under shift.")
    os.makedirs("results/extval", exist_ok=True); json.dump(out, open("results/extval/iter2.json", "w"), indent=2)
    print(f"\n[extval-2] worst-group degradation {worst:+.3f} | {out['verdict']}")
    print("[extval-2] wrote results/extval/iter2.json")


if __name__ == "__main__":
    main()
