"""Theory loop — Iteration 2: ADVERSARIAL VERIFICATION of the Iter-1 LTI result.

Iter-1 produced a suspicious +0.072 forward td-AUC @L=180 from the control-theoretic LTI
treatment-state x(L;a). Treat it as guilty-until-proven-innocent. Four checks:
  (A) FIX forward-horizon grid (horizons in days-AFTER-L) and re-run L=180/365/730.
  (B) STRONG-BASELINE gate: does +x add over Cox(ISS+gep70), not just Cox(ISS)? (paired bootstrap)
  (C) NEGATIVE CONTROL: shuffle the treatment->patient mapping (break the patient-treatment link,
      preserve the treatment-history marginal), recompute x, re-evaluate. If the "win" persists under
      shuffling, it is an artifact of the feature distribution, not real patient-specific signal.
  (D) DIAGNOSTIC: what does x(L;a) correlate with among at-risk patients (forward time, event)?
All immortal-time-safe landmarks, nested CV (inner picks a). Real data only; write results JSON.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from lifelines import CoxPHFitter
from scipy.stats import spearmanr
from resistancemap.data.mmsygnal import load_ia12
from resistancemap.survival.splits import patient_kfold
from resistancemap.survival.metrics import time_dependent_auc

TX = "ResistanceMap/data/raw/mmrf_commpass/treatments.tsv"
CLS = {"Bortezomib": "PI", "Carfilzomib": "PI", "Ixazomib": "PI", "Lenalidomide": "IMiD",
       "Pomalidomide": "IMiD", "Thalidomide": "IMiD", "Daratumumab": "CD38", "Isatuximab": "CD38",
       "Elotuzumab": "CD38", "Melphalan": "Alkyl", "Cyclophosphamide": "Alkyl"}
LO = {"First": 1, "Second": 2, "Third": 3, "Fourth": 4, "Fifth": 5, "Sixth": 6, "Seventh": 7, "Eighth": 8}
A_GRID = [1.0, 2.0, 4.0]
FWD_REL = np.array([180.0, 365.0])      # days AFTER the landmark (the FIX: relative, always > 0)
LANDMARKS = [180.0, 365.0, 730.0]


def treatment_lines():
    tx = pd.read_csv(TX, sep="\t", dtype=str)
    tx["start"] = pd.to_numeric(tx["days_to_treatment_start"], errors="coerce")
    tx["line"] = tx["regimen_or_line_of_therapy"].str.extract(r"(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth)")[0].map(LO)
    tx = tx.dropna(subset=["start", "line"])
    def cls(a): return {CLS.get(x) for x in str(a).split("|") if x in CLS} - {None}
    rows = []
    for (pid, ln), g in tx.groupby(["submitter_id", "line"]):
        classes = set().union(*[cls(a) for a in g["therapeutic_agents"]])
        rows.append({"pid": pid, "start": float(g["start"].min()), "intensity": max(1, len(classes))})
    return pd.DataFrame(rows)


def lti(lines_by_pid, pid, L, a):
    g = lines_by_pid.get(pid)
    if not g: return 0.0
    return float(sum(inten * np.exp(-a * (L - t) / 365.0) for t, inten in g if t <= L))


def fwd_auc(tr, te, cols, pen=1.0):
    cph = CoxPHFitter(penalizer=pen).fit(tr[cols + ["fdur", "event"]], "fdur", "event")
    risk = cph.predict_partial_hazard(te[cols]).values.ravel()
    _, m = time_dependent_auc(tr["fdur"].values, tr["event"].values, te["fdur"].values,
                              te["event"].values, risk, FWD_REL)
    return m, risk


def build_sub(df, lines_by_pid, L, a_for_features=A_GRID):
    sub = df[df["duration"] > L].copy().reset_index(drop=True)
    sub["fdur"] = sub["duration"] - L
    sub["ISS"] = sub["ISS"].fillna(sub["ISS"].median())
    sub["gep70"] = (sub["gep70"] - sub["gep70"].mean()) / (sub["gep70"].std() + 1e-9)
    for a in a_for_features:
        c = np.array([lti(lines_by_pid, p, L, a) for p in sub["pid"]])
        sub[f"x_{a}"] = (c - c.mean()) / (c.std() + 1e-9)
    return sub


def nested_oof(sub, base_cols, add_x=True):
    """OOF risk for Cox(base_cols [+ inner-selected x_a]). Returns (oof_risk, chosen_a)."""
    oof = np.full(len(sub), np.nan); chosen = []
    for tr_i, te_i in patient_kfold(sub, k=5, seed=0):
        tr, te = sub.iloc[tr_i], sub.iloc[te_i]
        if add_x:
            best_a, best_s = A_GRID[0], -1
            for a in A_GRID:
                s = []
                for itr, iva in patient_kfold(tr, k=3, seed=1):
                    try: m, _ = fwd_auc(tr.iloc[itr], tr.iloc[iva], base_cols + [f"x_{a}"]); s.append(m)
                    except Exception: pass
                if s and np.nanmean(s) > best_s: best_s, best_a = np.nanmean(s), a
            cols = base_cols + [f"x_{best_a}"]; chosen.append(best_a)
        else:
            cols = base_cols
        try: _, r = fwd_auc(tr, te, cols); oof[te_i] = r
        except Exception: pass
    return oof, chosen


def outer_tdauc(sub, risk):
    ok = ~np.isnan(risk)
    _, m = time_dependent_auc(sub["fdur"].values[ok], sub["event"].values[ok],
                              sub["fdur"].values[ok], sub["event"].values[ok], risk[ok], FWD_REL)
    return float(m)


def paired_delta(sub, a_risk, b_risk, B=1200):
    ok = ~np.isnan(a_risk) & ~np.isnan(b_risk)
    fd, ev, A, Bv = sub["fdur"].values[ok], sub["event"].values[ok], a_risk[ok], b_risk[ok]
    rng = np.random.default_rng(7); d = []
    for _ in range(B):
        i = rng.integers(0, len(fd), len(fd))
        try:
            _, ma = time_dependent_auc(fd[i], ev[i], fd[i], ev[i], A[i], FWD_REL)
            _, mb = time_dependent_auc(fd[i], ev[i], fd[i], ev[i], Bv[i], FWD_REL)
            d.append(ma - mb)
        except Exception: pass
    d = np.array(d); return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    d = load_ia12(); df = d.df.copy(); df["pid"] = df["patient_id"].str.extract(r"(MMRF_\d+)")
    L_df = treatment_lines()
    lines_by_pid = {p: list(zip(g["start"], g["intensity"])) for p, g in L_df.groupby("pid")}
    print(f"[iter2] N={len(df)} events={int(df['event'].sum())} | treated patients={len(lines_by_pid)}")
    out = {"iteration": 2, "purpose": "adversarial verification of Iter-1 LTI (leakage check)",
           "forward_grid_fix": "horizons are days-after-landmark (relative); Iter-1 used absolute (bug)",
           "landmarks": {}}

    for L in LANDMARKS:
        sub = build_sub(df, lines_by_pid, L)
        n_ev = int(sub["event"].sum())
        # (A) re-run with fixed grid: base(ISS) vs +x ; (B) strong base(ISS+gep70) vs +x
        iss_oof, _ = nested_oof(sub, ["ISS"], add_x=False)
        iss_x_oof, chosen = nested_oof(sub, ["ISS"], add_x=True)
        strong_oof, _ = nested_oof(sub, ["ISS", "gep70"], add_x=False)
        strong_x_oof, _ = nested_oof(sub, ["ISS", "gep70"], add_x=True)
        m_iss, m_issx = outer_tdauc(sub, iss_oof), outer_tdauc(sub, iss_x_oof)
        m_str, m_strx = outer_tdauc(sub, strong_oof), outer_tdauc(sub, strong_x_oof)
        dW = paired_delta(sub, iss_x_oof, iss_oof)          # weak-base gate
        dS = paired_delta(sub, strong_x_oof, strong_oof)    # STRONG-base gate (the real test)
        # (D) diagnostics: x vs forward time and event
        xcol = f"x_{chosen[0] if chosen else 2.0}"
        rho_t, p_t = spearmanr(sub[xcol], sub["fdur"])
        rho_e, p_e = spearmanr(sub[xcol], sub["event"])

        entry = {"n_atrisk": int(len(sub)), "n_events": n_ev, "chosen_a": chosen,
                 "tdauc_ISS": round(m_iss, 4), "tdauc_ISS+x": round(m_issx, 4),
                 "delta_x_over_ISS": [round(v, 4) for v in dW], "ci_sep_weak": bool(dW[1] > 0),
                 "tdauc_ISS+gep70": round(m_str, 4), "tdauc_ISS+gep70+x": round(m_strx, 4),
                 "delta_x_over_STRONG": [round(v, 4) for v in dS], "ci_sep_strong": bool(dS[1] > 0),
                 "spearman_x_vs_fwdtime": [round(rho_t, 3), round(p_t, 4)],
                 "spearman_x_vs_event": [round(rho_e, 3), round(p_e, 4)]}
        out["landmarks"][int(L)] = entry
        print(f"  L={int(L)} (n={len(sub)},ev={n_ev}): ISS={m_iss:.3f} +x={m_issx:.3f} (Δweak {dW[0]:+.3f} CI[{dW[1]:+.3f},{dW[2]:+.3f}]) | "
              f"ISS+gep70={m_str:.3f} +x={m_strx:.3f} (Δstrong {dS[0]:+.3f} CI[{dS[1]:+.3f},{dS[2]:+.3f}]) | "
              f"x~fwdt rho={rho_t:+.2f}")

    # (C) NEGATIVE CONTROL at L=180: shuffle treatment->patient map, single-CV, fixed a=2, 20 reps
    print("  [negative control] shuffling treatment->patient mapping @L=180 ...")
    L = 180.0; sub0 = build_sub(df, lines_by_pid, L)
    real_oof, _ = nested_oof(sub0, ["ISS"], add_x=True); real_tdauc = outer_tdauc(sub0, real_oof)
    pids = list(lines_by_pid.keys()); hist = list(lines_by_pid.values())
    rng = np.random.default_rng(0); null = []
    for rep in range(20):
        perm = rng.permutation(len(hist))
        shuffled = {pids[i]: hist[perm[i]] for i in range(len(pids))}
        subS = build_sub(df, shuffled, L, a_for_features=[2.0])
        # single 5-fold, fixed a=2 (no inner) for speed
        oofS = np.full(len(subS), np.nan)
        for tr_i, te_i in patient_kfold(subS, k=5, seed=0):
            try: _, r = fwd_auc(subS.iloc[tr_i], subS.iloc[te_i], ["ISS", "x_2.0"]); oofS[te_i] = r
            except Exception: pass
        null.append(outer_tdauc(subS, oofS))
    null = np.array(null)
    p_perm = float((null >= real_tdauc).mean())
    out["negative_control_L180"] = {"real_ISS+x_tdauc": round(real_tdauc, 4),
                                    "shuffled_null_mean": round(float(null.mean()), 4),
                                    "shuffled_null_max": round(float(null.max()), 4),
                                    "p_perm_real_le_null": round(p_perm, 4),
                                    "interpretation": "LEAKAGE/ARTIFACT if null ≈ real (shuffling treatment doesn't hurt); REAL signal if real >> null"}
    print(f"  [neg control] real ISS+x={real_tdauc:.3f} | shuffled null mean={null.mean():.3f} max={null.max():.3f} | p={p_perm:.3f}")

    # VERDICT
    any_strong = any(v["ci_sep_strong"] for v in out["landmarks"].values())
    neg_ok = out["negative_control_L180"]["shuffled_null_mean"] < real_tdauc - 0.02
    out["verdict"] = {
        "strong_baseline_gate_pass": bool(any_strong),
        "survives_negative_control": bool(neg_ok),
        "conclusion": ("REAL treatment signal (survives strong-baseline AND negative-control)"
                       if (any_strong and neg_ok) else
                       "LEAKAGE/ARTIFACT or no add over strong baseline — Iter-1 win NOT credible; record as negative, switch theory")}
    os.makedirs("results/theory_loop", exist_ok=True)
    json.dump(out, open("results/theory_loop/iter2.json", "w"), indent=2)
    print(f"\n[iter2] strong_gate={any_strong} neg_control_survives={neg_ok}")
    print(f"[iter2] CONCLUSION: {out['verdict']['conclusion']}")
    print("[iter2] wrote results/theory_loop/iter2.json")


if __name__ == "__main__":
    main()
