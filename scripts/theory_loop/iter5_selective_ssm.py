"""Theory loop — Iteration 5: selective diagonal state-space (SSM) over treatment events.

HYPOTHESIS. A diagonal linear SSM with input-dependent timing reads the irregular treatment-event
sequence up to landmark L into a multi-timescale state:
    z_k = exp(-a ⊙ Δ_k) z_{k-1} + ι_k 1,   Δ_k = (t_k - t_{k-1})/365,  ι_k = line intensity,
for m time-constants a; state z(L) ∈ R^m + a velocity channel (z_fast - z_slow). This is the
identifiable diagonal SSM realization of the state-space framing (Mamba-style selectivity = the
input-dependent step Δ_k); distinct from Iter-1's single-exponential LTI (which was refuted).
ICLR/ICML property: stable/identifiable diagonal SSM.

PRE-REGISTERED (immortal-time-safe landmark, nested CV): does the SSM state add over the strong
baseline Cox(ISS+gep70)? No claim unless it CI-separates above gep70/sky92; if it beats the strong
baseline, run the adversarial gauntlet next. Real data only.
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
       "Pomalidomide": "IMiD", "Thalidomide": "IMiD", "Daratumumab": "CD38", "Isatuximab": "CD38", "Elotuzumab": "CD38"}
LO = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5, "Sixth": 6, "Seventh": 7, "Eighth": 8}
A = np.array([0.5, 1.0, 2.0, 4.0])      # SSM time-constants (1/yr)
FWD_REL = np.array([180.0, 365.0]); LANDMARKS = [180.0, 365.0]; PEN_GRID = [0.5, 2.0]


def treatment_lines():
    tx = pd.read_csv(TX, sep="\t", dtype=str)
    tx["start"] = pd.to_numeric(tx["days_to_treatment_start"], errors="coerce")
    tx["line"] = tx["regimen_or_line_of_therapy"].str.extract(r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)")[0].map(LO)
    tx = tx.dropna(subset=["start", "line"])
    def cls(a): return {CLS.get(x) for x in str(a).split("|") if x in CLS} - {None}
    rows = []
    for (pid, ln), g in tx.groupby(["submitter_id", "line"]):
        rows.append({"pid": pid, "start": float(g["start"].min()),
                     "intensity": max(1, len(set().union(*[cls(a) for a in g["therapeutic_agents"]])))})
    return pd.DataFrame(rows)


def ssm_state(lines, L):
    """Diagonal selective SSM state z(L) over treatment events; returns [z_a..., velocity]."""
    z = np.zeros(len(A)); prev = 0.0
    if lines is not None and len(lines):
        for _, r in lines.sort_values("start").iterrows():
            if r["start"] > L: break
            dt = (r["start"] - prev) / 365.0
            z = np.exp(-A * dt) * z + r["intensity"]
            prev = r["start"]
    z = np.exp(-A * ((L - prev) / 365.0)) * z      # decay to landmark
    velocity = z[-1] - z[0]                          # fast - slow
    return np.concatenate([z, [velocity]])


def fwd_auc(tr, te, cols, pen):
    cph = CoxPHFitter(penalizer=pen).fit(tr[cols + ["fdur", "event"]], "fdur", "event")
    risk = cph.predict_partial_hazard(te[cols]).values.ravel()
    _, m = time_dependent_auc(tr["fdur"].values, tr["event"].values, te["fdur"].values, te["event"].values, risk, FWD_REL)
    return m, risk


def nested_oof(sub, cols):
    oof = np.full(len(sub), np.nan)
    for tr_i, te_i in patient_kfold(sub, k=5, seed=0):
        tr, te = sub.iloc[tr_i], sub.iloc[te_i]
        best_pen, best = PEN_GRID[0], -1
        for pen in PEN_GRID:
            s = []
            for a, b in patient_kfold(tr, k=3, seed=1):
                try: s.append(fwd_auc(tr.iloc[a], tr.iloc[b], cols, pen)[0])
                except Exception: pass
            if s and np.nanmean(s) > best: best, best_pen = np.nanmean(s), pen
        try: _, r = fwd_auc(tr, te, cols, best_pen); oof[te_i] = r
        except Exception: pass
    return oof


def outer_tdauc(sub, risk):
    ok = ~np.isnan(risk)
    _, m = time_dependent_auc(sub["fdur"].values[ok], sub["event"].values[ok], sub["fdur"].values[ok], sub["event"].values[ok], risk[ok], FWD_REL)
    return float(m)


def paired(sub, a, b, B=1200):
    ok = ~np.isnan(a) & ~np.isnan(b); fd, ev, A_, B_ = sub["fdur"].values[ok], sub["event"].values[ok], a[ok], b[ok]
    rng = np.random.default_rng(7); d = []
    for _ in range(B):
        i = rng.integers(0, len(fd), len(fd))
        try:
            _, ma = time_dependent_auc(fd[i], ev[i], fd[i], ev[i], A_[i], FWD_REL)
            _, mb = time_dependent_auc(fd[i], ev[i], fd[i], ev[i], B_[i], FWD_REL)
            d.append(ma - mb)
        except Exception: pass
    d = np.array(d); return round(float(d.mean()), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)


def main():
    d = load_ia12(); df = d.df.copy(); df["pid"] = df["patient_id"].str.extract(r"(MMRF_\d+)")
    lines_by_pid = {p: g for p, g in treatment_lines().groupby("pid")}
    scols = [f"z_{i}" for i in range(len(A))] + ["z_vel"]
    print(f"[iter5] N={len(df)} ev={int(df['event'].sum())} | SSM state dim={len(scols)} timescales={A.tolist()}")
    out = {"iteration": 5, "mechanism": "selective diagonal SSM over treatment events", "timescales": A.tolist(), "landmarks": {}}
    for L in LANDMARKS:
        sub = df[df["duration"] > L].copy().reset_index(drop=True)
        sub["fdur"] = sub["duration"] - L; sub["ISS"] = sub["ISS"].fillna(sub["ISS"].median())
        sub["gep70"] = (sub["gep70"] - sub["gep70"].mean()) / (sub["gep70"].std() + 1e-9)
        St = np.array([ssm_state(lines_by_pid.get(p), L) for p in sub["pid"]])
        for j, c in enumerate(scols): v = St[:, j]; sub[c] = (v - v.mean()) / (v.std() + 1e-9)
        base = ["ISS", "gep70"]
        ob, os_ = nested_oof(sub, base), nested_oof(sub, base + scols)
        mb, ms = outer_tdauc(sub, ob), outer_tdauc(sub, os_)
        dS = paired(sub, os_, ob)
        out["landmarks"][int(L)] = {"n_atrisk": int(len(sub)), "n_events": int(sub["event"].sum()),
                                    "tdauc_base": round(mb, 4), "tdauc_+ssm": round(ms, 4),
                                    "delta_ssm_over_strong": dS, "ssm_adds": bool(dS[1] > 0)}
        print(f"  L={int(L)} (n={len(sub)},ev={int(sub['event'].sum())}): base={mb:.3f} +SSM={ms:.3f} | Δ {dS[0]:+.3f} CI[{dS[1]:+.3f},{dS[2]:+.3f}]")
    any_add = any(v["ssm_adds"] for v in out["landmarks"].values())
    out["verdict"] = ("SSM state adds over strong baseline -> candidate; run adversarial gauntlet" if any_add
                      else "selective SSM treatment-state does NOT add over Cox(ISS+gep70) -> honest negative; ceiling holds across the state-space family too")
    out["gate_beats_sota"] = "PASS-PENDING-VERIFICATION" if any_add else "BLOCKED"
    os.makedirs("results/theory_loop", exist_ok=True); json.dump(out, open("results/theory_loop/iter5.json", "w"), indent=2)
    print(f"\n[iter5] ssm_adds_over_strong={any_add} | {out['verdict']}")
    print("[iter5] wrote results/theory_loop/iter5.json")


if __name__ == "__main__":
    main()
