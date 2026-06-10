"""Program-Basin hazard model — the NOVEL, mechanistically-interpretable survival head.

Mechanistic chain (this is the conceptual novelty):

    program scores / covariates  --(fitted linear map)-->  scalar drug-pressure
                                 --(squash to risk in [0, ~0.4])-->  basin TILT
                                 --(Kramers escape over a double-well barrier)-->  hazard

Concretely, each patient's covariates (FEATURES; later, learned transcriptional
*program* activity scores) are projected onto a single scalar axis by a small
linear map. That scalar is interpreted as covariate-driven *drug pressure* / disease
aggressiveness. We squash it into a bounded ``risk`` in ``[0, ~0.4]`` and feed it as
the TILT of the double-well quasi-potential in :mod:`resistancemap.models.basin_sde`:
a higher tilt shallows the sensitive (z<0) well, lowers its escape barrier, and so
*raises* the Kramers first-passage hazard out of the sensitive basin. The returned
per-patient score is therefore a structured, mechanistic hazard — relapse modelled as
thermally-activated escape from a sensitive attractor — rather than a black-box
hazard regression.

The linear map is fitted with the SAME fit(df)/risk(df) API as the survival
baselines (:mod:`resistancemap.survival.baselines`) so it drops directly into a
``get_baselines()``-style evaluation loop. ``risk(df)`` returns a value that is
"higher = worse prognosis (shorter survival)", consistent with the baselines and
with :func:`resistancemap.survival.metrics.cindex` (which negates internally).

The map is fitted by ranking covariates against the observed event indicator. The
default fitter is an L2-regularised :class:`sklearn.linear_model.LogisticRegression`
on ``event``; if scikit-learn is unavailable we fall back to a closed-form
correlation (Cox-like univariate ranking) weighting. NO torch / deep learning is
used anywhere; everything here runs on numpy + scikit-learn only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .basin_sde import escape_rate_for_risk

# Same feature contract as the baselines so this drops into get_baselines()-style loops.
FEATURES = ["age_z", "iss_stage", "sex_male", "prog_score_1", "prog_score_2", "prog_score_3"]

# Diffusion constant for the latent resistance state (matches basin_sde default usage).
_D = 0.12
# Upper bound of the risk -> tilt band. basin_sde is bistable for small |tilt|; beyond
# ~0.385 the barrier vanishes (monostable) and the Kramers rate diverges, so we keep the
# squashed risk inside a safe, interpretable band in [0, RISK_MAX].
RISK_MAX = 0.4


class ProgramBasin:
    """Covariate/program scores -> landscape tilt -> basin-escape hazard.

    Same ``fit(df)`` / ``risk(df)`` API as the survival baselines. ``risk(df)`` is
    monotonically increasing in prognostic severity (higher = shorter survival),
    so it can be passed straight to the project's C-index helper.
    """

    def __init__(self, features=FEATURES, D: float = _D, risk_max: float = RISK_MAX,
                 C: float = 1.0, name: str = "program_basin"):
        self.features = list(features)
        self.D = float(D)
        self.risk_max = float(risk_max)
        self.C = float(C)            # LogisticRegression inverse-regularisation strength
        self.name = name
        # Fitted state (filled by .fit):
        self.coef_ = None            # linear map weights on standardized features
        self.mu_ = None              # feature means (standardisation)
        self.sd_ = None              # feature stds  (standardisation)
        self._lo = None              # raw-score lower band (for normalisation)
        self._hi = None              # raw-score upper band (for normalisation)
        self._fitter = None          # "logreg" or "corr"

    # ------------------------------------------------------------------ fitting
    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_) / self.sd_

    def fit(self, df: pd.DataFrame):
        X = np.asarray(df[self.features].values, dtype=float)
        y = np.asarray(df["event"].values, dtype=int)

        # Standardise features so the linear map is scale-free and the tilt band is stable.
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Xs = self._standardize(X)

        # Primary fitter: L2 LogisticRegression on the event indicator (Cox-like ranking
        # of who relapses). Coefficients define the single drug-pressure axis.
        coef = None
        try:
            from sklearn.linear_model import LogisticRegression
            if 0 < int(y.sum()) < len(y):     # need both classes present
                lr = LogisticRegression(C=self.C, penalty="l2", max_iter=2000)
                lr.fit(Xs, y)
                coef = lr.coef_.ravel().astype(float)
                self._fitter = "logreg"
        except Exception:
            coef = None

        # Fallback: closed-form univariate correlation of each feature with the event
        # (a robust, dependency-light Cox-like ranking direction).
        if coef is None:
            yc = y - y.mean()
            denom = np.sqrt((Xs ** 2).sum(axis=0) * (yc ** 2).sum() + 1e-12)
            coef = (Xs * yc[:, None]).sum(axis=0) / denom
            self._fitter = "corr"

        self.coef_ = coef

        # Calibrate the normalisation band from the training raw scores so risk lands
        # in [0, risk_max]. Use robust percentiles to resist outliers.
        raw = Xs @ self.coef_
        self._lo = float(np.percentile(raw, 2.0))
        self._hi = float(np.percentile(raw, 98.0))
        if self._hi - self._lo < 1e-9:
            self._hi = self._lo + 1.0
        return self

    # --------------------------------------------------------------- prediction
    def _raw_score(self, df: pd.DataFrame) -> np.ndarray:
        X = np.asarray(df[self.features].values, dtype=float)
        return self._standardize(X) @ self.coef_

    def tilt(self, df: pd.DataFrame) -> np.ndarray:
        """Per-patient basin TILT = risk in [0, risk_max] (the landscape control knob)."""
        raw = self._raw_score(df)
        # Min-max normalise into [0, 1] on the calibrated band, then scale to [0, risk_max].
        unit = (raw - self._lo) / (self._hi - self._lo)
        unit = np.clip(unit, 0.0, 1.0)
        return unit * self.risk_max

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        """Mechanistic hazard surrogate (higher = worse). Program score -> tilt -> Kramers
        basin-escape rate. Returned monotone-increasing in prognostic severity so it is
        directly comparable to the baseline ``.risk`` outputs and to the C-index helper.
        """
        if self.coef_ is None:
            raise RuntimeError("ProgramBasin.risk called before fit().")
        tilts = self.tilt(df)
        # Kramers escape rate rises steeply (exponentially) with tilt -> very large dynamic
        # range. Map to a log-hazard so the score is well-conditioned for ranking/C-index.
        rates = np.array([escape_rate_for_risk(float(t), D=self.D) for t in tilts], dtype=float)
        finite = np.isfinite(rates)
        if not finite.all():
            fill = np.nanmax(rates[finite]) if finite.any() else 1.0
            rates = np.where(finite, rates, fill)
        log_haz = np.log(rates + 1e-300)
        return log_haz.ravel()


def fit(df: pd.DataFrame, **kw) -> ProgramBasin:
    """Convenience: construct + fit a :class:`ProgramBasin` in one call."""
    return ProgramBasin(**kw).fit(df)


def risk(model: ProgramBasin, df: pd.DataFrame) -> np.ndarray:
    """Convenience functional wrapper mirroring the class method."""
    return model.risk(df)
