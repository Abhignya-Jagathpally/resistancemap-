"""First-Hitting-Time (threshold-regression) survival model — the rigorous, *fittable*
form of "relapse = basin escape" (Lee & Whitmore 2006, Statistical Science).

A latent health process W(t) = y0 - mu*t + sigma*B(t) starts at reserve y0 > 0 and the
clinical event occurs at its FIRST passage to 0. The first-hitting time is Inverse
Gaussian: T ~ IG(mean = y0/mu, shape = y0^2/sigma^2); we fix sigma = 1 (identifiability).
Covariates modulate BOTH the drift and the boundary:

    mu_i = exp(b0 + x_i . beta)     # disease aggressiveness (drift toward relapse)
    y0_i = exp(g0 + x_i . gamma)    # resistance reserve (initial distance to threshold)

Fitted by PROPER censored maximum likelihood (events -> IG density; censored -> IG
survival), never a binary/logistic surrogate. When covariates act on BOTH mu and y0 the
model yields NON-proportional, time-varying hazards (crossing survival curves) that a Cox
PH model cannot represent. When only one parameter varies it reduces to a monotone
reparameterization (= parity with Cox). Same fit(df)/risk(df) API as the baselines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import invgauss

_NON_FEATURE = {"duration", "event", "patient_id", "case_submitter_id", "sample_id"}


class FHTThreshold:
    def __init__(self, features: list[str] | None = None, l2: float = 1e-3, name: str = "fht_threshold"):
        self.features = list(features) if features else None
        self.l2 = float(l2); self.name = name
        self.theta_ = None; self._mu = None; self._sd = None

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = np.asarray(df[self.features].values, float)
        if self._mu is None:
            self._mu = X.mean(0); self._sd = X.std(0); self._sd[self._sd == 0] = 1.0
        return (X - self._mu) / self._sd

    def _params(self, theta, Xs):
        p = Xs.shape[1]
        b0, beta, g0, gamma = theta[0], theta[1:1 + p], theta[1 + p], theta[2 + p:2 + 2 * p]
        mu = np.exp(np.clip(b0 + Xs @ beta, -8, 8))
        y0 = np.exp(np.clip(g0 + Xs @ gamma, -8, 8))
        return y0 / mu, y0 ** 2                      # mean m, shape lam (sigma=1)

    def _nll(self, theta, Xs, t, e):
        m, lam = self._params(theta, Xs)
        mu_s = np.clip(m / lam, 1e-10, 1e10); sc = np.clip(lam, 1e-10, 1e12)
        lp = invgauss.logpdf(t, mu_s, scale=sc); ls = invgauss.logsf(t, mu_s, scale=sc)
        lp = np.where(np.isfinite(lp), lp, -700.0); ls = np.where(np.isfinite(ls), ls, -700.0)
        return -(e * lp + (1 - e) * ls).sum() + self.l2 * np.sum(theta[1:] ** 2)

    def fit(self, df: pd.DataFrame, features: list[str] | None = None):
        self.features = list(features or self.features or
                             [c for c in df.columns if c not in _NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])])
        Xs = self._design(df); t = df["duration"].values.astype(float); e = df["event"].values.astype(int)
        p = Xs.shape[1]
        theta0 = np.zeros(2 + 2 * p); theta0[1 + p] = np.log(np.median(t) + 1e-6)   # y0~median(T), mu~1
        self.theta_ = minimize(self._nll, theta0, args=(Xs, t, e), method="L-BFGS-B",
                               options=dict(maxiter=500)).x
        return self

    def risk(self, df: pd.DataFrame) -> np.ndarray:        # higher = worse (shorter mean FHT)
        m, _ = self._params(self.theta_, self._design(df)); return -m

    def survival(self, df: pd.DataFrame, times) -> np.ndarray:
        m, lam = self._params(self.theta_, self._design(df))
        mu_s = np.clip(m / lam, 1e-10, 1e10); sc = np.clip(lam, 1e-10, 1e12)
        return np.column_stack([invgauss.sf(tt, mu_s, scale=sc) for tt in np.atleast_1d(times)])
