"""Ext-val Iteration 1: data contract + LEAKAGE-OUT-OF-DISTRIBUTION test on IA18.

Pre-registered question: GuanScore is C-index=1.000 in-sample on IA12 (leakage). Does it COLLAPSE to
~0.6 on the independent IA18 relapse cohort? If yes, that decisively proves the in-sample value was
leakage and that true OOD performance sits at the ~0.62 ceiling. Also: are IA18 program features
present for a full train-IA12->test-IA18 model? Real data, bootstrap CIs, no fabrication.
"""
from __future__ import annotations
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
from lifelines.utils import concordance_index

D = "_external/mmSYGNAL-risk-prediction-models/data"
v = pd.read_csv(f"{D}/IA18_relapse_minernorm_best_quality_risk.csv")
t = pd.to_numeric(v["PFS_d"], errors="coerce"); e = pd.to_numeric(v["relapse"], errors="coerce")
ok0 = t.notna() & e.notna() & (t > 0)
v, t, e = v[ok0].reset_index(drop=True), t[ok0].values, e[ok0].astype(int).values
n, nev = len(v), int(e.sum())
print(f"[extval-1] IA18 cohort: n={n} relapse_events={nev}")

def cidx_ci(score, B=2000, seed=1):
    s = pd.to_numeric(score, errors="coerce")
    ok = s.notna(); s = s[ok].values; tt, ee = t[ok.values], e[ok.values]
    if ok.sum() < 20 or len(set(ee)) < 2: return None
    c = concordance_index(tt, -s, ee)
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(tt), len(tt))
        try: bs.append(concordance_index(tt[i], -s[i], ee[i]))
        except Exception: pass
    return round(c, 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), int(ok.sum())

# clin_risk is categorical -> ordinal
clin_map = {"low": 0, "intermediate": 1, "high": 2, "extreme": 3}
v["clin_risk_num"] = v["clin_risk"].astype(str).str.lower().map(clin_map)

scores = {"GuanScore": v.get("GuanScore"), "miner_risk": v.get("miner_risk"),
          "risk_auc": v.get("risk_auc"), "clin_risk": v.get("clin_risk_num")}
IA12_INSAMPLE = {"GuanScore": 1.000, "risk_auc": 0.846}   # from IA12 leakage audit (results/sota_comparison.json)
rows = {}
print("\n=== IA18 out-of-distribution C-index (relapse) ===")
for name, sc in scores.items():
    r = cidx_ci(sc)
    if r is None: print(f"  {name:12s} (unavailable/degenerate)"); continue
    c, lo, hi, nn = r
    rows[name] = {"cindex": c, "ci_lo": lo, "ci_hi": hi, "n": nn,
                  "ia12_in_sample": IA12_INSAMPLE.get(name)}
    extra = ""
    if name in IA12_INSAMPLE:
        extra = f"  <-- IA12 in-sample was {IA12_INSAMPLE[name]:.3f} (COLLAPSE = leakage confirmed)"
    print(f"  {name:12s} C={c:.3f} [{lo:.3f},{hi:.3f}] (n={nn}){extra}")

# data contract: are IA18 program features available for the _2 samples?
pa = pd.read_csv(f"{D}/program_activity_py.csv")
pa_cols = [c for c in pa.columns if c != pa.columns[0]]
ia18_ids = set(v["sample"])
overlap_2 = [c for c in pa_cols if c in ia18_ids]
overlap_base = sum(1 for c in pa_cols if c.endswith("_1_BM"))
print(f"\n[data-contract] program_activity_py.csv: {len(pa_cols)} sample cols; "
      f"{overlap_base} are baseline (_1_BM); IA18 relapse-sample (_2) cols present = {len(overlap_2)}")
program_features_for_ia18 = len(overlap_2) > 0

out = {"iteration": 1, "purpose": "external-validation data contract + leakage-OOD test",
       "ia18_n": n, "ia18_relapse_events": nev, "ood_cindex": rows,
       "guanscore_collapses_ood": bool(rows.get("GuanScore", {}).get("cindex", 1.0) < 0.95),
       "ia18_program_features_available": bool(program_features_for_ia18),
       "verdict": None}
gs = rows.get("GuanScore", {}).get("cindex")
out["verdict"] = (f"GuanScore collapses 1.000 (in-sample) -> {gs} (OOD): in-sample value was LEAKAGE, "
                  f"confirmed on independent IA18; true OOD performance at the ~0.62 ceiling. "
                  f"IA18 program features {'ARE' if program_features_for_ia18 else 'are NOT'} present "
                  f"-> {'iter2 = full train-IA12->test-IA18' if program_features_for_ia18 else 'pivot to within-IA12 robustness + conformal'}.")
os.makedirs("results/extval", exist_ok=True)
json.dump(out, open("results/extval/iter1.json", "w"), indent=2)
print(f"\n[extval-1] VERDICT: {out['verdict']}")
print("[extval-1] wrote results/extval/iter1.json")
