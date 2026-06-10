"""Single train/test split smoke run of baselines + the NOVEL program_basin.
Default = SYNTHETIC. Point --data at a real GDC-open CoMMpass CSV/parquet."""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from resistancemap.data.synthetic import make_synthetic
from resistancemap.survival.baselines import get_baselines, infer_features
from resistancemap.survival.metrics import cindex_bootstrap
from resistancemap.models.program_basin import ProgramBasin


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.data:
        df = pd.read_parquet(a.data) if a.data.endswith(".parquet") else pd.read_csv(a.data)
        tag = f"REAL DATA: {os.path.basename(a.data)}"
    else:
        df = make_synthetic(n=900, seed=7); tag = "SYNTHETIC SMOKE TEST - not a scientific result"
    df = df.reset_index(drop=True)
    feats = infer_features(df)
    rng = np.random.default_rng(a.seed)
    ids = df["patient_id"].unique().copy(); rng.shuffle(ids)
    train = set(ids[:int(0.7 * len(ids))])
    tr = df[df["patient_id"].isin(train)].reset_index(drop=True)
    te = df[~df["patient_id"].isin(train)].reset_index(drop=True)
    print(f"[{tag}]  train={len(tr)} test={len(te)}  features={feats}\n")
    models = dict(get_baselines(features=feats)); models["program_basin (NOVEL)"] = ProgramBasin(features=feats)
    rows = []
    for name, bl in models.items():
        try:
            bl.fit(tr); r = bl.risk(te)
            m, lo, hi = cindex_bootstrap(te["duration"], te["event"], r, B=400, seed=1)
            rows.append({"model": name, "cindex": m, "ci_lo": lo, "ci_hi": hi})
            print(f"{name:26s} C-index {m:.3f}  [{lo:.3f}, {hi:.3f}]")
        except Exception as e:
            print(f"{name:26s} FAILED: {type(e).__name__}: {e}")
    os.makedirs("results", exist_ok=True)
    json.dump({"substrate": tag, "rows": rows}, open("results/baseline_smoketest.json", "w"), indent=2)
    print("\nwrote results/baseline_smoketest.json")


if __name__ == "__main__":
    main()
