"""scVI latent-embedding wrapper with a numpy/scikit-learn PCA fallback.

Two entry points:

* :func:`scvi_latent` — a *lazy-import* wrapper around scVI (single-cell variational
  inference; Lopez et al., Nat. Methods 2018). ``scvi``/``torch`` are imported INSIDE
  the function so this module never hard-fails on import when the deep-learning stack
  is absent. If scVI/torch are missing it raises a clear ImportError telling the user
  what to install (and noting that this environment intentionally does NOT ship torch).

* :func:`pca_latent` — a dependency-light PCA latent space that runs HERE (numpy /
  scikit-learn only). It is the drop-in fallback used for development/CI when scVI is
  not available, returning a (n_samples x k) latent matrix suitable for downstream
  PHATE embedding or trajectory comparators.
"""
from __future__ import annotations

import numpy as np

_SCVI_INSTALL_MSG = (
    "scVI is not available in this environment (it requires torch + scvi-tools, "
    "which are intentionally NOT installed here). Install with "
    "`pip install scvi-tools` on a machine with a working PyTorch, or use "
    "`pca_latent(X, k=...)` as the numpy/scikit-learn fallback that runs here."
)


def scvi_latent(adata, n_latent: int = 10, n_epochs: int = 200,
                batch_key=None, **kw) -> np.ndarray:
    """Train an scVI model on an AnnData and return its latent representation.

    scVI/torch are imported lazily inside this function. Raises ImportError (with
    install instructions) if the deep-learning stack is absent. ``adata`` is an
    :class:`anndata.AnnData` of raw counts; returns an (n_cells x n_latent) array.
    """
    try:
        import scvi  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(_SCVI_INSTALL_MSG) from exc

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent, **kw)
    model.train(max_epochs=n_epochs)
    return np.asarray(model.get_latent_representation())


def pca_latent(X, k: int = 10, standardize: bool = True, random_state: int = 0) -> np.ndarray:
    """numpy/scikit-learn PCA latent space (the fallback that RUNS in this environment).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature / expression matrix.
    k : int
        Number of latent components (clipped to ``min(n_samples, n_features)``).
    standardize : bool
        Z-score features before PCA (recommended for heterogeneous scales).

    Returns
    -------
    np.ndarray, shape (n_samples, k)
        Latent embedding, suitable as input to PHATE or a trajectory comparator.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"pca_latent expects a 2-D array, got shape {X.shape}")
    n, d = X.shape
    k_eff = int(max(1, min(k, n, d)))

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    try:
        from sklearn.decomposition import PCA
        return np.asarray(PCA(n_components=k_eff, random_state=random_state).fit_transform(X))
    except Exception:
        # Pure-numpy SVD fallback (no sklearn dependency at all).
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        return np.asarray(U[:, :k_eff] * S[:k_eff])
