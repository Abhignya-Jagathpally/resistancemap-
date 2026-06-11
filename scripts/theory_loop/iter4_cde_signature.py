"""Theory loop — Iteration 4: Neural-CDE / path-SIGNATURE of the treatment control path.

HYPOTHESIS. The treatment history is a control path X(t)=(t, cum_PI, cum_IMiD, cum_CD38) up to
landmark L. Its truncated signature S(X)^{<=2} is the universal, basis-free feature set of the path
(controlled-differential-equation theory). Level-1 = total increments (~ cumulative counts, already
tested in Lane #2); LEVEL-2 = iterated integrals ∫∫dX_i dX_j that encode treatment ORDERING /
interaction (non-commutative: PI-then-IMiD ≠ IMiD-then-PI). ICLR/ICML property: path-signatures are
universal path-functionals; handle irregular treatment timing.

PRE-REGISTERED TESTS (immortal-time-safe landmark, nested CV, vs STRONG baseline Cox(ISS+gep70)):
  (1) does +signature add over the strong baseline? (2) does LEVEL-2 (ordering) add over LEVEL-1
  (counts) — the genuinely novel content? No claim unless it CI-separates above gep70/sky92; if it
  beats the strong baseline, run the adversarial gauntlet next. Real data only.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from resistancemap.data.mmsygnal import load_ia12
from resistancemap.survival.splits import patient_kfold
from resistancemap.survival.metrics import time_dependent_auc

TX = "ResistanceMap/data/raw/mmrf_commpass/treatments.tsv"
CLS = {"Bortezomib": "PI", "Carfilzomib": "PI", "Ixazomib": "PI", "Lenalidomide": "IMiD",
       "Pomalidomide": "IMiD", "Thalidomide": "IMiD", "Daratumumab": "CD38", "Isatuximab": "CD38",
       "Elotuzumab": "CD38"}
LO = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5, "Sixth": 6, "Seventh": 7, "Eighth": 8}
CHAN = ["PI", "IMiD", "CD38"]           # drug-class channels (+ time prepended)
FWD_REL = np.array([180.0, 365.0])
LANDMARKS = [180.0, 365.0]
PEN_GRID = [0.5, 2.0]


def treatment_lines():
    tx = pd.read_csv(TX, sep="\t", dtype=str)
    tx["start"] = pd.to_numeric(tx["days_to_treatment_start"], errors="coerce")
    tx["line"] = tx["regimen_or_line_of_therapy"].str.extract(r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)")[0].map(LO)
    tx = tx.dropna(subset=["start", "line"])
    def cls(a): return {CLS.get(x) for x in str(a).split("|") if x in CLS} - {None}
    rows = []
    for (pid, ln), g in tx.groupby(["submitter_id", "line"]):
        classes = set().union(*[cls(a) for a in g["therapeutic_agents"]])
        rows.append({"pid": pid, "start": float(g["start"].min()),
                     **{c: int(c in classes) for c in CHAN}})
    return pd.DataFrame(rows)


def control_path(lines, L):
    """Piecewise control path X(t)=(t/365, cumPI, cumIMiD, cumCD38) for t in [0,L]."""
    pts = [[0.0, 0.0, 0.0, 0.0]]
    cum = np.zeros(len(CHAN))
    for _, r in lines.sort_values("start").iterrows():
        if r["start"] > L: break
        cum = cum + np.array([r[c] for c in CHAN], float)
        pts.append([r["start"] / 365.0, *cum])
    pts.append([L / 365.0, *cum])           # carry forward to the landmark
    return np.array(pts)


def signature(path):
    """Truncated signature level<=2 of a piecewise-linear path. Returns (S1 (d,), S2flat (d*d,))."""
    dX = np.diff(path, axis=0)              # (n-1, d)
    d = path.shape[1]
    S1 = dX.sum(0)
    cum = np.zeros(d); S2 = np.zeros((d, d))
    for k in range(dX.shape[0]):
        S2 += np.outer(cum, dX[k]) + 0.5 * np.outer(dX[k], dX[k])
        cum += dX[k]
    return S1, S2.reshape(-1)


def fwd_auc(tr, te, cols, pen):
    cph = CoxPHFitter(penalizer=pen).fit(tr[cols + ["fdur", "event"]], "fdur", "event")
    risk = cph.predict_partial_hazard(te[cols]).values.ravel()
    _, m = time_dependent_auc(tr["fdur"].values, tr["event"].values, te["fdur"].values,
                              te["event"].values, risk, FWD_REL)
    return m, risk


def nested_oof(sub, cols):
    oof = np.full(len(sub), np.nan)
    for tr_i, te_i in patient_kfold(sub, k=5, seed=0):
        tr, te = sub.iloc[tr_i], sub.iloc[te_i]
        best_pen, best = PEN_GRID[0], -1
        for pen in PEN_GRID:
            s = []
            for itr, iva in patient_kfold(tr, k=3, seed=1):
                try: m, _ = fwd_auc(tr.iloc[itr], tr.iloc[iva], cols, pen); s.append(m)
                except Exception: pass
            if s and np.nanmean(s) > best: best, best_pen = np.nanmean(s), pen
        try: _, r = fwd_auc(tr, te, cols, best_pen); oof[te_i] = r
        except Exception: pass
    return oof


def outer_tdauc(sub, risk):
    ok = ~np.isnan(risk)
    _, m = time_dependent_auc(sub["fdur"].values[ok], sub["event"].values[ok],
                              sub["fdur"].values[ok], sub["event"].values[ok], risk[ok], FWD_REL)
    return float(m)


def paired(sub, a, b, B=1200):
    ok = ~np.isnan(a) & ~np.isnan(b)
    fd, ev, A, Bv = sub["fdur"].values[ok], sub["event"].values[ok], a[ok], b[ok]
    rng = np.random.default_rng(7); d = []
    for _ in range(B):
        i = rng.integers(0, len(fd), len(fd))
        try:
            _, ma = time_dependent_auc(fd[i], ev[i], fd[i], ev[i], A[i], FWD_REL)
            _, mb = time_dependent_auc(fd[i], ev[i], fd[i], ev[i], Bv[i], FWD_REL)
            d.append(ma - mb)
        except Exception: pass
    d = np.array(d); return round(float(d.mean()), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)


def main():
    d = load_ia12(); df = d.df.copy(); df["pid"] = df["patient_id"].str.extract(r"(MMRF_\d+)")
    L_df = treatment_lines(); lines_by_pid = {p: g for p, g in L_df.groupby("pid")}
    dch = len(CHAN) + 1
    s1_cols = [f"s1_{i}" for i in range(dch)]
    s2_cols = [f"s2_{i}" for i in range(dch * dch)]
    print(f"[iter4] N={len(df)} events={int(df['event'].sum())} | path channels=time+{CHAN} | sig L1={dch} L2={dch*dch}")
    out = {"iteration": 4, "mechanism": "path-signature (level<=2) of treatment control path; CDE theory",
           "channels": ["time"] + CHAN, "landmarks": {}}
    empty = pd.DataFrame(columns=["start"] + CHAN)
    for L in LANDMARKS:
        sub = df[df["duration"] > L].copy().reset_index(drop=True)
        sub["fdur"] = sub["duration"] - L
        sub["ISS"] = sub["ISS"].fillna(sub["ISS"].median())
        sub["gep70"] = (sub["gep70"] - sub["gep70"].mean()) / (sub["gep70"].std() + 1e-9)
        S1 = np.zeros((len(sub), dch)); S2 = np.zeros((len(sub), dch * dch))
        for i, p in enumerate(sub["pid"]):
            path = control_path(lines_by_pid.get(p, empty), L)
            S1[i], S2[i] = signature(path)
        for j, c in enumerate(s1_cols): v = S1[:, j]; sub[c] = (v - v.mean()) / (v.std() + 1e-9)
        for j, c in enumerate(s2_cols): v = S2[:, j]; sub[c] = (v - v.mean()) / (v.std() + 1e-9)
        base = ["ISS", "gep70"]
        oof_base = nested_oof(sub, base)
        oof_s1 = nested_oof(sub, base + s1_cols)
        oof_s12 = nested_oof(sub, base + s1_cols + s2_cols)
        m_b, m_1, m_12 = outer_tdauc(sub, oof_base), outer_tdauc(sub, oof_s1), outer_tdauc(sub, oof_s12)
        dS12 = paired(sub, oof_s12, oof_base)        # full signature over strong base
        dORD = paired(sub, oof_s12, oof_s1)          # level-2 (ordering) over level-1 (counts) — the novel test
        gep_ref = outer_tdauc(sub, np.nan_to_num(pd.to_numeric(sub["gep70"]).values))
        out["landmarks"][int(L)] = {
            "n_atrisk": int(len(sub)), "n_events": int(sub["event"].sum()),
            "tdauc_base_ISS_gep70": round(m_b, 4), "tdauc_+sig1": round(m_1, 4), "tdauc_+sig12": round(m_12, 4),
            "delta_sig12_over_base": dS12, "sig_adds_over_strong": bool(dS12[1] > 0),
            "delta_ordering_lvl2_over_lvl1": dORD, "ordering_adds": bool(dORD[1] > 0)}
        print(f"  L={int(L)} (n={len(sub)},ev={int(sub['event'].sum())}): base={m_b:.3f} +sig1={m_1:.3f} +sig12={m_12:.3f} | "
              f"Δsig-over-base {dS12[0]:+.3f} CI[{dS12[1]:+.3f},{dS12[2]:+.3f}] | Δordering(L2/L1) {dORD[0]:+.3f} CI[{dORD[1]:+.3f},{dORD[2]:+.3f}]")
    any_add = any(v["sig_adds_over_strong"] for v in out["landmarks"].values())
    any_ord = any(v["ordering_adds"] for v in out["landmarks"].values())
    out["verdict"] = {
        "signature_adds_over_strong_baseline": bool(any_add),
        "treatment_ordering_adds": bool(any_ord),
        "conclusion": ("signature/ordering adds over strong baseline -> candidate; run adversarial gauntlet"
                       if any_add else
                       "path-signature (incl. treatment ordering) does NOT add over Cox(ISS+gep70) -> honest negative; ordering is not prognostic here; switch theory")}
    out["gate_beats_sota"] = "PASS-PENDING-VERIFICATION" if any_add else "BLOCKED"
    os.makedirs("results/theory_loop", exist_ok=True)
    json.dump(out, open("results/theory_loop/iter4.json", "w"), indent=2)
    print(f"\n[iter4] sig_adds_over_strong={any_add} | ordering_adds={any_ord}")
    print(f"[iter4] CONCLUSION: {out['verdict']['conclusion']}")
    print("[iter4] wrote results/theory_loop/iter4.json")


if __name__ == "__main__":
    main()
