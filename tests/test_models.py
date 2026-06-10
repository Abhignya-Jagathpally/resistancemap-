"""Tests for the novel models added under models/ and viz/.

Fits the NOVEL `program_basin` model on synthetic CoMMpass-shaped data, reports its
Harrell C-index (via lifelines), and exercises the numpy/sklearn `pca_latent` fallback.
Everything here runs on numpy/scikit-learn/lifelines only (NO torch).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from lifelines.utils import concordance_index

from resistancemap.data.synthetic import make_synthetic, FEATURES
from resistancemap.models.program_basin import ProgramBasin
from resistancemap.models.scvi_latent import pca_latent


def _cindex_higher_is_worse(durations, events, risk) -> float:
    """Harrell's C with the project convention: higher risk = shorter survival,
    so negate risk before handing to lifelines.concordance_index."""
    return concordance_index(np.asarray(durations, float),
                             -np.asarray(risk, float),
                             np.asarray(events, int))


def test_program_basin_cindex():
    df = make_synthetic(n=900, seed=7)
    model = ProgramBasin().fit(df)
    r = model.risk(df)
    assert r.shape == (len(df),), f"risk shape mismatch: {r.shape}"
    assert np.all(np.isfinite(r)), "risk produced non-finite values"

    ci = _cindex_higher_is_worse(df["duration"], df["event"], r)
    print(f"[program_basin] fitter={model._fitter}  C-index = {ci:.4f}")
    # Must rank meaningfully better than chance on a known-signal generator.
    assert ci > 0.55, f"program_basin C-index too low: {ci:.4f}"
    return ci


def test_pca_latent():
    df = make_synthetic(n=900, seed=7)
    X = df[FEATURES].values
    Z = pca_latent(X, k=4)
    assert Z.shape == (len(df), 4), f"pca_latent shape mismatch: {Z.shape}"
    assert np.all(np.isfinite(Z)), "pca_latent produced non-finite values"
    print(f"[pca_latent] latent shape = {Z.shape}  (first-row norm {np.linalg.norm(Z[0]):.3f})")
    return Z.shape


if __name__ == "__main__":
    print("=== program_basin (NOVEL model) C-index on make_synthetic(n=900) ===")
    ci = test_program_basin_cindex()

    print("\n=== pca_latent (numpy/sklearn latent fallback) ===")
    shape = test_pca_latent()

    print("\nALL MODEL TESTS PASSED")
    print(f"program_basin C-index = {ci:.4f}")
