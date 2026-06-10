"""Patient-disjoint cross-validation splits.

Time-to-event models leak optimistically if two rows from the *same* patient
(e.g. repeated visits / longitudinal samples) land on opposite sides of a
train/test split. Every splitter here groups by ``patient_id`` so that the set
of patients in train and the set in test are disjoint.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def patient_kfold(
    df: pd.DataFrame,
    k: int = 5,
    seed: int = 0,
    patient_col: str = "patient_id",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` positional index arrays for ``k``-fold CV
    grouped by ``patient_col`` so no patient appears on both sides of a fold.

    Indices are *positional* (0..len(df)-1), suitable for ``df.iloc[idx]``.

    A patient (group) is assigned wholly to exactly one test fold. We shuffle the
    *group order* with ``seed`` before delegating to scikit-learn's
    :class:`~sklearn.model_selection.GroupKFold`, which guarantees group-disjoint
    folds and balances fold sizes greedily. The shuffle makes the partition depend
    on ``seed`` (plain ``GroupKFold`` is otherwise deterministic and ignores it).

    Parameters
    ----------
    df:
        Frame containing ``patient_col``.
    k:
        Number of folds. Must not exceed the number of distinct patients.
    seed:
        Controls the (reproducible) shuffling of group order.
    patient_col:
        Column holding the grouping key.
    """
    if patient_col not in df.columns:
        raise KeyError(f"{patient_col!r} not in dataframe columns")

    n = len(df)
    groups_raw = df[patient_col].to_numpy()
    uniq = pd.unique(groups_raw)
    n_groups = len(uniq)
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    if n_groups < k:
        raise ValueError(
            f"cannot make {k} patient-disjoint folds from only {n_groups} patients"
        )

    # Reproducibly permute the *group* labels so the fold assignment depends on seed.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    relabel = {orig: int(new) for orig, new in zip(uniq, perm)}
    shuffled_groups = np.array([relabel[g] for g in groups_raw])

    pos = np.arange(n)
    gkf = GroupKFold(n_splits=k)
    # GroupKFold needs X/y only for length; pass positional indices as X.
    for train_idx, test_idx in gkf.split(pos, groups=shuffled_groups):
        yield train_idx, test_idx
