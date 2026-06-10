"""End-to-end pipeline = the replacement for the 40-task Airflow DAG.
Stages: data_validate -> evaluate (patient-disjoint k-fold CV; baselines + NOVEL model)
-> governance (pre-registered claim gate). Default = SYNTHETIC; --data for the real run."""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from resistancemap.data.synthetic import make_synthetic
from resistancemap.survival.baselines import get_baselines, infer_features
from resistancemap.survival.splits import patient_kfold
from resistancemap.survival.metrics import cindex_bootstrap
from resistancemap.models.program_basin import ProgramBasin
from resistancemap.governance.claim_gate import ClaimGate, Governance
from resistancemap.pipeline import run as run_pipeline


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None); ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    if a.data:
        df = pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)
        tag = f"REAL: {os.path.basename(a.data)}"
    else:
        df = make_synthetic(n=900, seed=7); tag = "SYNTHETIC (validation only)"
    df = df.reset_index(drop=True)
    feats = infer_features(df)
    state: dict = {}
    os.makedirs("results", exist_ok=True)

    def stage_validate():
        miss = {"duration", "event", "patient_id"} - set(df.columns)
        if miss: raise ValueError(f"missing columns: {miss}")
        if int(df["event"].sum()) < 2: raise ValueError("need >=2 observed events")
        if not (df["duration"] > 0).all(): raise ValueError("non-positive durations present")
        print(f"    {len(df)} patients | {int(df['event'].sum())} events | features={feats}")
        return None

    def stage_evaluate():
        models = dict(get_baselines(features=feats)); models["program_basin"] = ProgramBasin(features=feats)
        oof = {n: np.full(len(df), np.nan) for n in models}
        for tr_idx, te_idx in patient_kfold(df, k=a.k, seed=0):
            tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
            # Leakage-safe missing-covariate handling: impute medians fitted on TRAIN only.
            med = tr[feats].median(numeric_only=True)
            tr[feats] = tr[feats].fillna(med)
            te[feats] = te[feats].fillna(med)
            for n, m in models.items():
                try:
                    m.fit(tr); oof[n][te_idx] = m.risk(te)
                except Exception as e:
                    print(f"    [{n}] fold fail: {type(e).__name__}: {e}")
        rows = []
        for n, r in oof.items():
            mask = ~np.isnan(r)
            if mask.sum() < 10: continue
            mC, lo, hi = cindex_bootstrap(df["duration"].values[mask], df["event"].values[mask], r[mask], B=500, seed=1)
            rows.append({"model": n, "cindex": round(mC, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "n": int(mask.sum())})
            print(f"    {n:26s} CV C-index {mC:.3f} [{lo:.3f}, {hi:.3f}]")
        state["rows"] = rows
        json.dump({"substrate": tag, "k": a.k, "rows": rows}, open("results/cv_results.json", "w"), indent=2)
        return "results/cv_results.json"

    def stage_governance():
        rows = state.get("rows", [])
        best = max([x["cindex"] for x in rows if x["model"] != "program_basin"], default=0.0)
        nov = next((x for x in rows if x["model"] == "program_basin"), None)
        nov_lo = nov["ci_lo"] if nov else None
        gov = Governance()
        gov.add(ClaimGate("novel_model_beats_baselines",
                          lambda: nov_lo is not None and nov_lo > best,
                          f"program_basin CV C-index CI-lower must exceed best baseline C={best:.3f}"))
        for x in gov.emit("results/cv_governance.json"):
            print(f"    {x['status']:8s} {x['claim']}")
        return "results/cv_governance.json"

    print(f"[{tag}]")
    run_pipeline([("data_validate", stage_validate), ("evaluate", stage_evaluate), ("governance", stage_governance)])


if __name__ == "__main__":
    main()
