"""Single-cell RNA-seq quality control: a scanpy/scverse path and a numpy fallback.

The repo's heavy single-cell stack (scanpy / anndata) is NOT installed in the offline
sandbox, so scanpy is imported LAZILY inside ``mad_qc_anndata`` and raises a clear,
actionable message if it is absent. A pure-numpy ``basic_qc_counts`` fallback computes
per-cell summaries and simple percentile filters with no optional dependency, so this
module is always importable and partially usable.
"""
from __future__ import annotations

import numpy as np


def _is_outlier_mad(values: np.ndarray, n_mads: float = 5.0) -> np.ndarray:
    """Boolean mask of MAD-based outliers (scverse best-practice rule).

    A value is an outlier if it lies more than ``n_mads`` median-absolute-deviations from
    the median. Works on a 1-D metric vector (e.g. log1p total counts).
    """
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        # Degenerate spread: nothing is an outlier under the MAD rule.
        return np.zeros_like(values, dtype=bool)
    lower = med - n_mads * mad
    upper = med + n_mads * mad
    return (values < lower) | (values > upper)


def mad_qc_anndata(
    adata,
    n_mads: float = 5.0,
    pct_counts_mt_max: float = 8.0,
    mt_prefix: str = "MT-",
):
    """MAD-based per-cell QC filtering on an AnnData, following scverse best practice.

    Flags cells whose ``log1p_total_counts``, ``log1p_n_genes_by_counts`` or
    ``pct_counts_in_top_20_genes`` are >``n_mads`` MADs from the median, plus a hard cap
    on mitochondrial fraction, then returns the filtered AnnData.

    scanpy is imported HERE (lazily). If it is missing, a clear ``ImportError`` explains
    how to install it or to fall back to ``basic_qc_counts``.
    """
    try:
        import scanpy as sc  # noqa: F401  (used below)
    except ModuleNotFoundError as exc:
        raise ImportError(
            "mad_qc_anndata requires scanpy (and anndata), which are NOT installed in "
            "this environment. Install the optional omics extras on a suitable machine:\n"
            "    pip install 'resistancemap[omics]'   # or: pip install scanpy anndata\n"
            "For a dependency-free per-cell summary, use basic_qc_counts(matrix) instead."
        ) from exc

    # Annotate mitochondrial genes and compute standard QC metrics in place.
    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], inplace=True, percent_top=[20], log1p=True
    )

    outlier = (
        _is_outlier_mad(adata.obs["log1p_total_counts"].values, n_mads)
        | _is_outlier_mad(adata.obs["log1p_n_genes_by_counts"].values, n_mads)
        | _is_outlier_mad(adata.obs["pct_counts_in_top_20_genes"].values, n_mads)
    )
    mt_outlier = adata.obs["pct_counts_mt"].values > pct_counts_mt_max

    adata.obs["qc_outlier"] = outlier | mt_outlier
    keep = ~adata.obs["qc_outlier"].values
    return adata[keep].copy()


def basic_qc_counts(matrix, percentile_low: float = 1.0, percentile_high: float = 99.0) -> dict:
    """Pure-numpy per-cell QC summary + simple percentile filter (no scanpy needed).

    Parameters
    ----------
    matrix:
        cells x genes count matrix (array-like; densified with ``np.asarray``).
    percentile_low, percentile_high:
        keep cells whose ``total_counts`` AND ``n_genes`` fall within these percentile
        bounds (inclusive). Defaults drop the extreme 1%% tails on each metric.

    Returns
    -------
    dict with:
      - ``n_genes`` (int array): non-zero genes per cell
      - ``total_counts`` (float array): summed counts per cell
      - ``keep`` (bool array): cells passing both percentile filters
      - ``n_cells`` / ``n_cells_kept`` (int): before/after counts
      - ``thresholds`` (dict): the resolved numeric cutoffs
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D cells x genes matrix, got shape {m.shape}")

    total_counts = m.sum(axis=1)
    n_genes = (m > 0).sum(axis=1)

    tc_lo, tc_hi = np.percentile(total_counts, [percentile_low, percentile_high])
    ng_lo, ng_hi = np.percentile(n_genes, [percentile_low, percentile_high])

    keep = (
        (total_counts >= tc_lo)
        & (total_counts <= tc_hi)
        & (n_genes >= ng_lo)
        & (n_genes <= ng_hi)
    )

    return {
        "n_genes": n_genes,
        "total_counts": total_counts,
        "keep": keep,
        "n_cells": int(m.shape[0]),
        "n_cells_kept": int(keep.sum()),
        "thresholds": {
            "total_counts_low": float(tc_lo),
            "total_counts_high": float(tc_hi),
            "n_genes_low": float(ng_lo),
            "n_genes_high": float(ng_hi),
        },
    }
