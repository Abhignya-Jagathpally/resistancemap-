"""Where does the first-hitting-time mechanism EARN ITS KEEP? Controlled comparison vs Cox.
PH regime (Cox-Weibull synthetic): expect parity. Non-PH regime (data simulated FROM the
FHT model so hazards cross): expect FHT to beat Cox on time-dependent metrics (IBS, td-AUC)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from scipy.stats import invgauss
from lifelines import CoxPHFitter
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc, concordance_index_censored
from resistancemap.data.synthetic import make_synthetic, FEATURES
from resistancemap.models.fht_threshold import FHTThreshold


def simulate_nonph(n=1200, seed=3):
    """Non-PH data: x1 drives DRIFT (early effect), x2 drives BOUNDARY (tail effect) -> crossing."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    mu = np.exp(0.0 + 0.8 * x1); y0 = np.exp(0.6 + 0.8 * x2)
    m, lam = y0 / mu, y0 ** 2
    T = invgauss.rvs(np.clip(m / lam, 1e-8, 1e8), scale=np.clip(lam, 1e-8, 1e12), random_state=rng)
    C = rng.uniform(0, np.quantile(T, 0.9), size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "duration": np.minimum(T, C), "event": (T <= C).astype(int)})
    df["patient_id"] = [f"P{i}" for i in range(n)]
    return df, ["x1", "x2"]


def split(df, seed=0, frac=0.7):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(df)); cut = int(frac * len(df))
    return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)


def evaluate(df, feats, label):
    tr, te = split(df)
    times = np.quantile(te.loc[te.event == 1, "duration"], [0.15, 0.3, 0.45, 0.6, 0.75])
    times = np.unique(times)
    y_tr = Surv.from_arrays(tr.event.astype(bool).values, tr.duration.values)
    y_te = Surv.from_arrays(te.event.astype(bool).values, te.duration.values)

    cph = CoxPHFitter(penalizer=0.01).fit(tr[feats + ["duration", "event"]], "duration", "event")
    S_cox = cph.predict_survival_function(te[feats], times=times).T.values
    r_cox = cph.predict_partial_hazard(te[feats]).values.ravel()

    fht = FHTThreshold(features=feats).fit(tr)
    S_fht = fht.survival(te, times); r_fht = fht.risk(te)

    def row(name, S, r):
        c = concordance_index_censored(te.event.astype(bool).values, te.duration.values, r)[0]
        ibs = integrated_brier_score(y_tr, y_te, S, times)
        try: _, auc = cumulative_dynamic_auc(y_tr, y_te, 1 - S, times)
        except Exception: auc = float("nan")
        print(f"  {label:8s} {name:6s}  C-index {c:.3f}   IBS {ibs:.4f}   mean td-AUC {auc:.3f}")
    print(f"[{label}]  n={len(df)}  events={int(df.event.mean()*100)}%  eval-times={np.round(times,1)}")
    row("Cox", S_cox, r_cox); row("FHT", S_fht, r_fht); print()


if __name__ == "__main__":
    ph = make_synthetic(1200, seed=7)
    evaluate(ph, FEATURES, "PH")
    nph, f2 = simulate_nonph()
    evaluate(nph, f2, "non-PH")
