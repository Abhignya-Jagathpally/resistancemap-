"""C-index-drop permutation importance for a fitted survival model.

Why this method
---------------
SHAP is not importable in this environment, and tree-SHAP for sksurv gradient
boosting is not standardized. The model-agnostic, always-available, and honest
alternative is **permutation importance against the survival metric we care
about**: fit once on a patient-disjoint train split, then for each feature
shuffle that feature's column on the held-out test split and measure how far the
test C-index drops. A larger drop => the model relies more on that feature to
rank patients by progression risk.

This is the survival analogue of sklearn's ``permutation_importance`` (Breiman
2001; Fisher, Rudin & Dominici 2019), using Harrell's C-index (via sksurv's
``concordance_index_censored``) as the score instead of accuracy/R^2.

Honesty notes
-------------
- The permutation RNG is *the method* (we must shuffle a column). It is seeded.
  No reported quantity is invented by ``np.random``.
- Importance is measured on a held-out split, so it reflects generalizable
  reliance, not in-sample overfitting.
- We average over ``n_repeats`` independent shuffles per feature and report the
  mean drop + its std, so the reader can see stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


def program_label(col: str | int) -> str:
    """Human-facing label for a feature column.

    mmSYGNAL program columns are bare integers (0..140). We render those as
    ``Program_007`` so figures/tables are legible; clinical columns pass through.
    """
    s = str(col)
    if s.isdigit():
        return f"Program_{int(s):03d}"
    return s


@dataclass
class PermutationImportanceResult:
    """Ranked permutation importance over all model features.

    Attributes
    ----------
    table:
        DataFrame with columns ``feature`` (raw column id), ``label``
        (human label), ``block`` ("program"/"clinical"), ``importance``
        (mean C-index drop when the feature is permuted), ``importance_std``
        (std over repeats), and ``rank`` (1 = most important).
    base_cindex:
        Held-out test C-index of the un-permuted model.
    n_test:
        Number of test patients used to score importance.
    n_repeats:
        Number of independent shuffles averaged per feature.
    """

    table: pd.DataFrame
    base_cindex: float
    n_test: int
    n_repeats: int

    def top(self, k: int = 20) -> pd.DataFrame:
        return self.table.head(k).reset_index(drop=True)


def _cindex(model, df_eval: pd.DataFrame, features: list) -> float:
    risk = np.asarray(model.risk(df_eval)).ravel()
    return float(
        concordance_index_censored(
            df_eval["event"].astype(bool).to_numpy(),
            df_eval["duration"].to_numpy(float),
            risk,
        )[0]
    )


def permutation_cindex_importance(
    model,
    df_test: pd.DataFrame,
    features: list,
    program_cols: set | None = None,
    n_repeats: int = 10,
    seed: int = 0,
) -> PermutationImportanceResult:
    """Compute C-index-drop permutation importance on a held-out split.

    Parameters
    ----------
    model:
        Any fitted object exposing ``model.risk(df) -> higher-is-worse array``
        (e.g. ``SkSurvBaseline`` wrapping a sksurv estimator). Must already be
        ``fit`` on a *different* (patient-disjoint) split than ``df_test``.
    df_test:
        Held-out evaluation frame with ``duration``/``event`` and ``features``.
    features:
        Feature column ids the model was fit on (order need not match).
    program_cols:
        Set of column ids that are mmSYGNAL programs (for the "block" label).
        Anything not in this set is labelled "clinical".
    n_repeats:
        Independent shuffles per feature; the mean drop is the importance.
    seed:
        Reproducible RNG seed for the shuffles.

    Returns
    -------
    PermutationImportanceResult
        Sorted descending by mean importance (C-index drop).
    """
    program_cols = set(program_cols or set())
    rng = np.random.default_rng(seed)
    df_test = df_test.reset_index(drop=True)
    base = _cindex(model, df_test, features)

    rows = []
    for col in features:
        original = df_test[col].to_numpy(copy=True)
        drops = np.empty(n_repeats, dtype=float)
        permuted = df_test.copy()
        for r in range(n_repeats):
            permuted[col] = rng.permutation(original)
            drops[r] = base - _cindex(model, permuted, features)
        permuted[col] = original  # restore (defensive; permuted is local anyway)
        rows.append(
            {
                "feature": col,
                "label": program_label(col),
                "block": "program" if col in program_cols else "clinical",
                "importance": float(drops.mean()),
                "importance_std": float(drops.std(ddof=0)),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        "importance", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return PermutationImportanceResult(
        table=table,
        base_cindex=base,
        n_test=int(len(df_test)),
        n_repeats=int(n_repeats),
    )
