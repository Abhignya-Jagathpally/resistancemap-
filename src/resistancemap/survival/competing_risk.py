"""Discrete-time hazard survival model that runs WITHOUT torch.

The continuous follow-up axis is discretised into a fixed set of time bins
(deciles of the *observed* times by default). Each subject is expanded to
person-interval ("long") format: one row per bin the subject is still at risk in.
A single :class:`~sklearn.linear_model.LogisticRegression` then estimates the
per-interval discrete hazard

    h(t_j | x) = P(event in bin j | survived through bin j-1, x)
               = sigmoid( alpha_j + x . beta ),

where each bin gets its own intercept ``alpha_j`` (one-hot bin dummies) and the
covariate effect ``beta`` is shared across bins (a proportional-odds-in-discrete-
time assumption). The survival curve is reconstructed multiplicatively

    S(t_J | x) = prod_{j<=J} (1 - h(t_j | x)),

and a scalar risk score is ``risk = 1 - S(t_max | x)`` (higher == worse), matching
the ``risk(df)`` convention used by the Cox/sksurv baselines.

Right-censoring is handled honestly by construction. For a subject with observed
time ``t`` falling in bin ``b``:

* a subject who had the **event** (``event == 1``) contributes rows for bins
  ``0..b`` with label 0 on bins ``0..b-1`` and label **1** on its final bin ``b``;
* a **censored** subject (``event == 0``) contributes rows for bins ``0..b`` with
  label **0** on every bin, including ``b`` -- it was event-free for as long as it
  was observed and then simply drops out of the risk set. It never contributes a
  label-1 row, so censoring is not mistaken for an event.

This is the standard "person-period" / pooled-logistic discrete-time survival
model (Cox 1972 discrete logistic; Tutz & Schmid 2016; Gensheimer & Narasimhan
2019 use the same likelihood as the loss of a neural net -- here we just keep it
linear so no deep-learning framework is required).

Extending to TRUE competing risks
----------------------------------
For ``m`` mutually exclusive competing causes, replace the binary per-interval
logistic with a *multinomial* per-interval model over ``{censored-survives,
cause_1, ..., cause_m}`` (a softmax / multinomial ``LogisticRegression`` on the
same long format, with the long-format label set to the cause index in a
subject's event bin and to the "survives" category otherwise). The cause-specific
discrete hazards ``h_c(t_j)`` then give the cumulative incidence function
``CIF_c(t_J) = sum_{j<=J} h_c(t_j) * S(t_{j-1})`` with overall survival
``S(t_j) = prod_{l<=j}(1 - sum_c h_c(t_l))``. The expansion / censoring logic in
:func:`_to_long` is unchanged; only the label (multiclass instead of binary) and
the link (softmax instead of sigmoid) differ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .baselines import FEATURES


class DiscreteTimeHazard:
    """Pooled-logistic discrete-time hazard model (single-event, torch-free).

    Parameters
    ----------
    n_bins:
        Number of discrete time intervals. Bin edges are the empirical quantiles
        of the training observed times, so each bin holds a comparable number of
        observations.
    features:
        Covariate column names. Defaults to the project-wide ``FEATURES``.
    C:
        Inverse-regularisation strength passed to ``LogisticRegression``.
    name:
        Human-readable identifier (mirrors the baseline models).
    """

    def __init__(
        self,
        n_bins: int = 10,
        features: list[str] | None = None,
        C: float = 1.0,
        name: str = "discrete_time_hazard",
    ) -> None:
        self.n_bins = int(n_bins)
        self.features = list(features) if features is not None else list(FEATURES)
        self.C = float(C)
        self.name = name
        # learned at fit time
        self.bin_edges_: np.ndarray | None = None       # interior edges, len n_bins-1
        self.bin_mids_: np.ndarray | None = None         # representative time per bin
        self.t_max_: float | None = None
        self.scaler_: StandardScaler | None = None
        self.clf_: LogisticRegression | None = None

    # ------------------------------------------------------------------ binning
    def _make_bins(self, durations: np.ndarray) -> None:
        """Set bin edges from quantiles of observed training times."""
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(durations, qs)
        # Deduplicate (ties at the boundaries collapse bins); keep interior edges.
        edges = np.unique(edges)
        if edges.size < 2:  # degenerate: all times identical
            edges = np.array([durations.min(), durations.max() + 1e-9])
        self.bin_edges_ = edges[1:-1]            # interior cut points
        self.t_max_ = float(durations.max())
        # Representative time within each realised bin (used as the curve's x-axis).
        full = np.concatenate(([durations.min()], self.bin_edges_, [self.t_max_]))
        self.bin_mids_ = 0.5 * (full[:-1] + full[1:])

    def _bin_index(self, durations: np.ndarray) -> np.ndarray:
        """Map each duration to a 0-based bin index in ``[0, n_realised_bins-1]``."""
        assert self.bin_edges_ is not None
        # np.searchsorted with the interior edges => index of the containing bin.
        idx = np.searchsorted(self.bin_edges_, durations, side="right")
        n_real = self.n_realised_bins
        return np.clip(idx, 0, n_real - 1).astype(int)

    @property
    def n_realised_bins(self) -> int:
        assert self.bin_edges_ is not None
        return int(self.bin_edges_.size + 1)

    # -------------------------------------------------------------- long format
    def _to_long(
        self, X: np.ndarray, last_bin: np.ndarray, event: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand to person-interval rows.

        Returns ``(X_long, bin_long, y_long)`` where a subject with final bin ``b``
        and event indicator ``e`` contributes rows for bins ``0..b``; the label is
        1 only on bin ``b`` when ``e == 1`` and 0 everywhere else.
        """
        x_rows: list[np.ndarray] = []
        bin_rows: list[int] = []
        y_rows: list[int] = []
        for xi, b, e in zip(X, last_bin, event):
            for j in range(b + 1):
                x_rows.append(xi)
                bin_rows.append(j)
                y_rows.append(1 if (j == b and e == 1) else 0)
        return (
            np.asarray(x_rows, dtype=float),
            np.asarray(bin_rows, dtype=int),
            np.asarray(y_rows, dtype=int),
        )

    def _design(self, X_scaled: np.ndarray, bin_long: np.ndarray) -> np.ndarray:
        """Concatenate one-hot bin dummies (per-bin intercepts) with covariates."""
        n_real = self.n_realised_bins
        dummies = np.zeros((bin_long.size, n_real), dtype=float)
        dummies[np.arange(bin_long.size), bin_long] = 1.0
        return np.hstack([dummies, X_scaled])

    # --------------------------------------------------------------------- API
    def fit(self, df: pd.DataFrame) -> "DiscreteTimeHazard":
        """Fit the discrete-time hazard on a frame with ``duration``/``event``."""
        durations = df["duration"].to_numpy(dtype=float)
        event = df["event"].to_numpy(dtype=int)
        X = df[self.features].to_numpy(dtype=float)

        self._make_bins(durations)
        last_bin = self._bin_index(durations)

        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)

        X_long, bin_long, y_long = self._to_long(X_scaled, last_bin, event)
        design = self._design(X_long, bin_long)

        # Fit intercepts via the bin dummies => disable the model's own intercept.
        self.clf_ = LogisticRegression(
            C=self.C, fit_intercept=False, max_iter=2000, solver="lbfgs"
        )
        # If the long format happens to be single-class (no events at all), guard.
        if np.unique(y_long).size < 2:
            self._degenerate = True
        else:
            self._degenerate = False
            self.clf_.fit(design, y_long)
        return self

    def _hazard_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Per-subject, per-bin discrete hazard ``h(t_j | x)`` (shape ``n x B``)."""
        if self.clf_ is None or self.scaler_ is None:
            raise RuntimeError("model is not fitted; call fit() first")
        X = df[self.features].to_numpy(dtype=float)
        X_scaled = self.scaler_.transform(X)
        n, n_real = X_scaled.shape[0], self.n_realised_bins

        if getattr(self, "_degenerate", False):
            # No events observed in training => zero hazard everywhere.
            return np.zeros((n, n_real), dtype=float)

        # Predict hazard for every (subject, bin) combination.
        bins = np.tile(np.arange(n_real), n)              # 0..B-1 repeated per subj
        X_rep = np.repeat(X_scaled, n_real, axis=0)
        design = self._design(X_rep, bins)
        haz = self.clf_.predict_proba(design)[:, 1]
        return haz.reshape(n, n_real)

    def predict_survival(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times, S)`` where ``S`` is ``n x B`` survival probabilities.

        ``times`` is the per-bin representative time grid (length ``B``); column
        ``j`` of ``S`` is ``S(times[j] | x) = prod_{l<=j} (1 - h(t_l | x))``.
        """
        haz = self._hazard_matrix(df)
        surv = np.cumprod(1.0 - haz, axis=1)
        assert self.bin_mids_ is not None
        return self.bin_mids_.copy(), surv

    def predict_survival_at(self, df: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Survival probabilities evaluated at arbitrary ``times`` (shape ``n x len(times)``).

        Uses a right-continuous step interpolation of the bin survival curve: for
        each requested time we take ``S`` at the last bin whose representative time
        is ``<=`` it (and ``S = 1`` before the first bin midpoint).
        """
        grid, surv = self.predict_survival(df)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        # index of last grid point <= t  (=> -1 means "before grid" -> S=1)
        cols = np.searchsorted(grid, times, side="right") - 1
        out = np.ones((surv.shape[0], times.size), dtype=float)
        valid = cols >= 0
        if valid.any():
            out[:, valid] = surv[:, cols[valid]]
        return out

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Scalar risk score ``1 - S(t_max | x)`` (higher == worse prognosis)."""
        _, surv = self.predict_survival(df)
        return 1.0 - surv[:, -1]

    # Alias to match the baseline ``risk(df)`` calling convention.
    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_risk(df)
