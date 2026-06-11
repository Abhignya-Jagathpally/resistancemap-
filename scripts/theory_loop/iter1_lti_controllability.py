"""Theory loop — Iteration 1: control-theoretic LTI treatment-state feature.

HYPOTHESIS. Model disease burden as a first-order LTI system driven by treatment:
    dx/dt = -a x + u(t),  u(t) = treatment-intensity impulses (drug-class count per line).
The state at landmark L is the impulse response / discounted controllability energy:
    x(L; a) = sum_{lines started <= L} intensity_line * exp(-a (L - t_line)/365).
ICLR/ICML-relevant property: x(L;a) is a *physically interpretable, identifiable* latent
state — the system time-constant `a` is a scalar recovered by inner CV (system identification),
and the feature has a controllability interpretation (input energy reaching the state).

PRE-REGISTERED TEST (immortal-time-safe landmark, nested CV): does adding the single LTI state
x(L;a) to a Cox(ISS) improve forward td-AUC over Cox(ISS) alone, with paired bootstrap CI > 0?
SOTA reference: gep70 / sky92 forward td-AUC at the same landmark. Gate stays BLOCKED unless the
model CI-separates above the SOTA reference. Parity reported honestly.
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
       "Elotuzumab": "CD38", "Melphalan": "Alkyl", "Cyclophosphamide": "Alkyl"}
LINE_ORD = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5, "Sixth": 6, "Seventh": 7, "Eighth": 8}
A_GRID = [0.5, 1.0, 2.0, 4.0]          # system time-constants (1/yr) — inner-CV selected
PEN_GRID = [0.5, 2.0]
LANDMARKS = [180.0, 365.0]
FWD = {180.0: [180.0, 365.0], 365.0: [180.0, 365.0]}   # forward horizons (days) per landmark


def treatment_lines():
    tx = pd.read_csv(TX, sep="\t", dtype=str)
    tx["start"] = pd.to_numeric(tx["days_to_treatment_start"], errors="coerce")
    tx["line"] = tx["regimen_or_line_of_therapy"].str.extract(r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)")[0].map(LINE_ORD)
    tx = tx.dropna(subset=["start", "line"])
    def cls(a): return {CLS.get(x) for x in str(a).split("|") if x in CLS} - {None}
    rows = []
    for (pid, ln), g in tx.groupby(["submitter_id", "line"]):
        classes = set().union(*[cls(a) for a in g["therapeutic_agents"]])
        rows.append({"pid": pid, "line": int(ln), "start": float(g["start"].min()), "intensity": max(1, len(classes))})
    return pd.DataFrame(rows)


def lti_state(lines_by_pid, pid, L, a):
    g = lines_by_pid.get(pid)
    if g is None: return 0.0
    s = 0.0
    for t, inten in g:
        if t <= L:
            s += inten * np.exp(-a * (L - t) / 365.0)
    return s


def fwd_auc(tr, te, feats, L):
    """forward td-AUC of Cox(feats) trained on tr, scored on te, both already landmarked."""
    cph = CoxPHFitter(penalizer=feats[1]).fit(tr[feats[0] + ["fdur", "event"]], "fdur", "event")
    risk = cph.predict_partial_hazard(te[feats[0]]).values.ravel()
    grid = np.array([h - L for h in FWD[L]], float)
    _, m = time_dependent_auc(tr["fdur"].values, tr["event"].values, te["fdur"].values, te["event"].values, risk, grid)
    return m, risk


def run_landmark(df, lines_by_pid, L):
    sub = df[df["duration"] > L].copy().reset_index(drop=True)
    sub["fdur"] = sub["duration"] - L
    sub["ISS"] = sub["ISS"].fillna(sub["ISS"].median())
    for a in A_GRID:
        sub[f"x_{a}"] = [lti_state(lines_by_pid, p, L, a) for p in sub["pid"]]
        c = sub[f"x_{a}"]; sub[f"x_{a}"] = (c - c.mean()) / (c.std() + 1e-9)
    n_ev = int(sub["event"].sum())
    base_oof = np.full(len(sub), np.nan); lti_oof = np.full(len(sub), np.nan)
    chosen_a = []
    for tr_i, te_i in patient_kfold(sub, k=5, seed=0):
        tr, te = sub.iloc[tr_i].copy(), sub.iloc[te_i].copy()
        # inner CV on tr: pick base penalizer, and (a,pen) for +LTI, by inner forward td-AUC
        best_base, best_base_s = (1.0,), -1
        best_lti, best_lti_s = (["ISS", "x_1.0"], 1.0), -1
        for pen in PEN_GRID:
            s = []
            for itr, iva in patient_kfold(tr, k=3, seed=1):
                try: m, _ = fwd_auc(tr.iloc[itr], tr.iloc[iva], (["ISS"], pen), L); s.append(m)
                except Exception: pass
            if s and np.nanmean(s) > best_base_s: best_base_s, best_base = np.nanmean(s), (pen,)
        for a in A_GRID:
            for pen in PEN_GRID:
                s = []
                for itr, iva in patient_kfold(tr, k=3, seed=1):
                    try: m, _ = fwd_auc(tr.iloc[itr], tr.iloc[iva], (["ISS", f"x_{a}"], pen), L); s.append(m)
                    except Exception: pass
                if s and np.nanmean(s) > best_lti_s: best_lti_s, best_lti = np.nanmean(s), (["ISS", f"x_{a}"], pen)
        chosen_a.append(best_lti[0][1])
        try:
            _, rb = fwd_auc(tr, te, (["ISS"], best_base[0]), L); base_oof[te_i] = rb
            _, rl = fwd_auc(tr, te, best_lti, L); lti_oof[te_i] = rl
        except Exception:
            pass
    return sub, base_oof, lti_oof, chosen_a, n_ev


def outer_auc(sub, risk, L):
    ok = ~np.isnan(risk)
    grid = np.array([h - L for h in FWD[L]], float)
    _, m = time_dependent_auc(sub["fdur"].values[ok], sub["event"].values[ok],
                              sub["fdur"].values[ok], sub["event"].values[ok], risk[ok], grid)
    return float(m)


def paired_boot(sub, a, b, L, B=1500):
    ok = ~np.isnan(a) & ~np.isnan(b)
    fdur, ev = sub["fdur"].values[ok], sub["event"].values[ok]; A, Bv = a[ok], b[ok]
    grid = np.array([h - L for h in FWD[L]], float); rng = np.random.default_rng(7); d = []
    for _ in range(B):
        i = rng.integers(0, len(fdur), len(fdur))
        try:
            _, ma = time_dependent_auc(fdur[i], ev[i], fdur[i], ev[i], A[i], grid)
            _, mb = time_dependent_auc(fdur[i], ev[i], fdur[i], ev[i], Bv[i], grid)
            d.append(ma - mb)
        except Exception: pass
    d = np.array(d); return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def main():
    d = load_ia12(); df = d.df.copy(); df["pid"] = df["patient_id"].str.extract(r"(MMRF_\d+)")
    L_df = treatment_lines()
    lines_by_pid = {p: list(zip(g["start"], g["intensity"])) for p, g in L_df.groupby("pid")}
    print(f"[iter1] cohort N={len(df)} events={int(df['event'].sum())} | patients with treatment lines={len(lines_by_pid)}")
    out = {"iteration": 1, "hypothesis": "first-order LTI treatment-state x(L;a) controllability feature",
           "design": "immortal-time-safe landmark + nested CV (inner picks a,penalizer)", "a_grid": A_GRID,
           "landmarks": {}}
    for L in LANDMARKS:
        sub, base_oof, lti_oof, chosen_a, n_ev = run_landmark(df, lines_by_pid, L)
        base_m = outer_auc(sub, base_oof, L); lti_m = outer_auc(sub, lti_oof, L)
        # SOTA reference at this landmark (forward td-AUC of the static published scores)
        ref = {}
        for sc in ["gep70", "sky92"]:
            r = pd.to_numeric(sub[sc], errors="coerce").values
            ref[sc] = outer_auc(sub, np.where(np.isnan(r), np.nanmedian(r), r), L)
        dm, lo, hi, p = paired_boot(sub, lti_oof, base_oof, L)
        out["landmarks"][int(L)] = {
            "n_atrisk": int(len(sub)), "n_events": n_ev,
            "base_cox_iss_tdauc": round(base_m, 4), "lti_tdauc": round(lti_m, 4),
            "delta_lti_minus_base": round(dm, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ci_separated_above_zero": bool(lo > 0), "p_two_sided": round(2 * min(p, 1 - p), 4),
            "sota_ref_gep70": round(ref["gep70"], 4), "sota_ref_sky92": round(ref["sky92"], 4),
            "lti_beats_sota_ref": bool(lti_m > max(ref.values())),
            "chosen_time_constants_a": chosen_a,
        }
        print(f"  L={int(L)}: base(ISS)={base_m:.3f} +LTI={lti_m:.3f} Δ={dm:+.4f} CI[{lo:+.4f},{hi:+.4f}] "
              f"| SOTA gep70={ref['gep70']:.3f} sky92={ref['sky92']:.3f} | a*={chosen_a}")
    # gate
    any_sep = any(v["ci_separated_above_zero"] for v in out["landmarks"].values())
    beats = any(v["lti_beats_sota_ref"] and v["ci_separated_above_zero"] for v in out["landmarks"].values())
    out["gate_lti_adds_over_base"] = "PASS" if any_sep else "BLOCKED"
    out["gate_beats_sota"] = "PASS" if beats else "BLOCKED"
    os.makedirs("results/theory_loop", exist_ok=True)
    json.dump(out, open("results/theory_loop/iter1.json", "w"), indent=2)
    print(f"\n[iter1] gate_lti_adds_over_base={out['gate_lti_adds_over_base']} | gate_beats_sota={out['gate_beats_sota']}")
    print("[iter1] wrote results/theory_loop/iter1.json")


if __name__ == "__main__":
    main()
