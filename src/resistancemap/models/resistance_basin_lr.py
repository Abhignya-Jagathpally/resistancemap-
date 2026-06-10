"""ResistanceBasin-LR — mechanism-anchored, proportional-hazards-FREE survival model.

Pipeline (the contribution):
    program/clinical features  --φ_θ (MLP)-->  soft assignment s ∈ Δ^K over K resistance
    basins  --(trained with the partial multivariate log-rank loss, Ferle 2026)-->
    per-basin baseline survival S0_k(t)  -->  S(t|x) = Σ_k s_k · S0_k(t).

Why it is novel and the right fit (vs mmSYGNAL / gep70 / Cox):
  * The published scores are a single static rank → they ASSUME proportional hazards.
  * Here each basin carries its OWN baseline survival S0_k(t); the mixture S(t|x) is a
    genuinely NON-proportional, time-resolved hazard — so two patients' hazard curves
    can CROSS (exactly the MM reality as CAR-T/bispecifics move earlier). That is the
    axis where this model can beat a static score even when C-index is ceiling-bound.
  * The objective is the partial multivariate log-rank statistic — the score test of the
    Cox model under the null — maximised over soft clusters: PH-free and censoring-honest.

Same fit(df)/risk(df) API as resistancemap.survival.baselines, plus survival_function().
risk(df) is "higher = worse prognosis" so it drops into the project C-index helper.
Requires torch; the partial log-rank loss is vendored from resistancemap.loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:                       # pragma: no cover
    _HAS_TORCH = False


class _BasinNet(nn.Module):
    def __init__(self, in_dim: int, n_basins: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_basins),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class ResistanceBasinLR:
    """PH-free basin-mixture survival model trained with the partial log-rank loss."""

    def __init__(self, features=None, n_basins: int = 3, hidden: int = 64,
                 dropout: float = 0.3, epochs: int = 400, lr: float = 1e-3,
                 weight_decay: float = 1e-4, penalty_weight: float = 0.1,
                 seed: int = 0, name: str = "resistance_basin_lr"):
        if not _HAS_TORCH:
            raise ImportError("ResistanceBasinLR needs torch. pip install torch")
        self.features = list(features) if features else None
        self.n_basins = n_basins
        self.hidden = hidden
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.penalty_weight = penalty_weight
        self.seed = seed
        self.name = name
        # fitted state
        self.net_ = None
        self.mu_ = None
        self.sd_ = None
        self.times_ = None            # sorted unique training event times (grid for S0_k)
        self.S0_ = None               # (K, T) per-basin baseline survival
        self.basin_risk_ = None       # (K,) scalar risk per basin (higher=worse)

    # ---------------------------------------------------------------- helpers
    def _X(self, df: pd.DataFrame) -> "torch.Tensor":
        X = np.asarray(df[self.features].values, dtype=float)
        X = (X - self.mu_) / self.sd_
        return torch.tensor(X, dtype=torch.float32)

    @staticmethod
    def _weighted_km(times_grid, durations, events, weights):
        """Per-basin baseline survival via a weight (soft-assignment) Kaplan-Meier.

        S0(t) = Π_{t_j <= t} (1 - d_j / n_j), with weighted counts. Pure numpy.
        """
        order = np.argsort(durations)
        d = durations[order]; e = events[order]; w = weights[order]
        S = np.ones(len(times_grid)); s = 1.0
        # cumulative at-risk weight from the right
        # iterate event times
        uniq = np.unique(d[e == 1])
        gi = 0
        surv_at_uniq = {}
        for ut in uniq:
            at_risk = w[d >= ut].sum()
            d_j = w[(d == ut) & (e == 1)].sum()
            if at_risk > 0:
                s *= max(0.0, 1.0 - d_j / at_risk)
            surv_at_uniq[ut] = s
        # step-function evaluate on grid
        out = np.ones(len(times_grid))
        last = 1.0
        ui = 0
        usorted = uniq
        for k, t in enumerate(times_grid):
            while ui < len(usorted) and usorted[ui] <= t:
                last = surv_at_uniq[usorted[ui]]; ui += 1
            out[k] = last
        return np.clip(out, 1e-6, 1.0)

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame, features=None):
        from resistancemap.loss import PartialMultivariateLogRankLoss
        self.features = list(features or self.features)
        torch.manual_seed(self.seed); np.random.seed(self.seed)

        Xraw = np.asarray(df[self.features].values, dtype=float)
        self.mu_ = np.nanmean(Xraw, axis=0)
        self.sd_ = np.nanstd(Xraw, axis=0); self.sd_[self.sd_ == 0] = 1.0
        X = torch.tensor((np.nan_to_num(Xraw, nan=0.0) - self.mu_) / self.sd_, dtype=torch.float32)
        dur = torch.tensor(df["duration"].values, dtype=torch.float32)
        ev = torch.tensor(df["event"].values, dtype=torch.float32)

        self.net_ = _BasinNet(X.shape[1], self.n_basins, self.hidden, self.dropout)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = PartialMultivariateLogRankLoss(penalty_weight=self.penalty_weight)

        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            s = self.net_(X)
            loss = loss_fn(s, dur, ev)
            if not torch.isfinite(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
            opt.step()

        # freeze: per-basin baseline survival on training data (soft-weighted KM)
        self.net_.eval()
        with torch.no_grad():
            s_tr = self.net_(X).numpy()                       # (N, K)
        d_np = df["duration"].values.astype(float)
        e_np = df["event"].values.astype(int)
        self.times_ = np.unique(d_np)
        S0 = np.stack([self._weighted_km(self.times_, d_np, e_np, s_tr[:, k])
                       for k in range(self.n_basins)])          # (K, T)
        self.S0_ = S0
        # scalar basin risk = mean cumulative incidence over the grid (higher = worse)
        self.basin_risk_ = (1.0 - S0).mean(axis=1)
        return self

    # ---------------------------------------------------------------- predict
    def _soft(self, df: pd.DataFrame) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            return self.net_(self._X(df.fillna(0.0) if df[self.features].isna().any().any() else df)).numpy()

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        """Higher = worse prognosis: mixture-weighted basin risk."""
        s = self._soft(df)
        return s @ self.basin_risk_

    def survival_function(self, df: pd.DataFrame, times) -> np.ndarray:
        """S(t|x) = Σ_k s_k S0_k(t), evaluated at `times`. Shape (n, len(times))."""
        s = self._soft(df)                                    # (n, K)
        times = np.atleast_1d(np.asarray(times, dtype=float))
        # interpolate each basin's step S0 onto requested times
        S0_t = np.stack([np.interp(times, self.times_, self.S0_[k], left=1.0, right=self.S0_[k][-1])
                         for k in range(self.n_basins)])       # (K, T)
        return s @ S0_t                                        # (n, T)

    def basin_assignments(self, df: pd.DataFrame) -> np.ndarray:
        return self._soft(df)
