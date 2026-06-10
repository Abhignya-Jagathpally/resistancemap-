"""PHATE / T-PHATE embedding wrapper.

PHATE (Moon et al., Nat. Biotechnol. 2019; Krishnaswamy Lab) is a manifold-learning
/ dimensionality-reduction method that preserves both local and global structure of
high-dimensional data via a potential of an information-theoretic diffusion operator.
T-PHATE is its temporally-aware variant for trajectory data. PHATE is lightweight
(pure numpy / scipy / scikit-learn under the hood) so it can run in this environment.

This module uses a *lazy import* of ``phate`` so that importing
:mod:`resistancemap.viz` never hard-fails if the package is absent. If ``phate`` is
not installed, :func:`phate_embed` / :func:`plot_phate` raise an ImportError with a
clear ``pip install phate`` message instead of crashing at import time.
"""
from __future__ import annotations

import numpy as np

_INSTALL_MSG = (
    "PHATE is not installed. Install it with `pip install phate` "
    "(it is lightweight: numpy/scipy/scikit-learn only, no torch)."
)


def _import_phate():
    try:
        import phate  # noqa: F401  (lazy: only imported when actually needed)
        return phate
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(_INSTALL_MSG) from exc


def phate_embed(X, n_components: int = 2, knn: int = 5, decay: int = 40,
                random_state: int = 0, **kw) -> np.ndarray:
    """Embed ``X`` (n_samples x n_features) into ``n_components`` PHATE coordinates.

    Extra keyword args are forwarded to :class:`phate.PHATE`. Returns an
    (n_samples x n_components) numpy array. Raises ImportError (with install
    instructions) if PHATE is unavailable.
    """
    phate = _import_phate()
    X = np.asarray(X, dtype=float)
    op = phate.PHATE(n_components=n_components, knn=knn, decay=decay,
                     random_state=random_state, verbose=False, **kw)
    return np.asarray(op.fit_transform(X))


def plot_phate(X, color=None, out_png: str = "results/phate_demo.png",
               title: str = "PHATE embedding", cmap: str = "coolwarm", **kw):
    """Compute a 2-D PHATE embedding of ``X`` and save a scatter to ``out_png``.

    ``color`` is an optional per-sample array used to colour points (e.g. the event
    indicator). Returns the embedding array. Raises ImportError if PHATE is absent.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    emb = phate_embed(X, n_components=2, **kw)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(emb[:, 0], emb[:, 1],
                    c=(None if color is None else np.asarray(color)),
                    cmap=cmap, s=14, alpha=0.85, edgecolors="none")
    if color is not None:
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("color")
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return emb


def _demo():
    """Run a PHATE demo on make_synthetic features coloured by event; save PNG.

    Skips (prints a clear message) if PHATE is not installed.
    """
    import os
    import sys
    # Make the package importable when run as a standalone script.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from resistancemap.data.synthetic import make_synthetic, FEATURES

    df = make_synthetic(n=900, seed=7)
    X = df[FEATURES].values
    out = os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "phate_demo.png")
    out = os.path.abspath(out)
    try:
        emb = plot_phate(X, color=df["event"].values, out_png=out,
                         title="PHATE of synthetic CoMMpass-shaped features (colour = event)")
        print(f"PHATE demo OK -> wrote {out}  (embedding shape {emb.shape})")
    except ImportError as exc:
        print(f"PHATE demo SKIPPED: {exc}")


if __name__ == "__main__":
    _demo()
