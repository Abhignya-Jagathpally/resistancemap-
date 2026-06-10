"""Patient-level immune-microenvironment features from the MMRF Immune Atlas (Path A).

This module turns the MMRF Immune Atlas single-cell bone-marrow annotation into a small,
interpretable, **patient-level** feature table that can be *late-fused* onto the existing
bulk-tumor mmSYGNAL program features for PFS prediction. See ``docs/PATH_A_IMMUNE_FUSION.md``
for the full scoping rationale, accessions, and the non-redundancy argument.

Honesty contract (binding — mirrors ``data/gdc_clinical.py``)
------------------------------------------------------------
- We do **not** fabricate immune values. ``load_immune_features`` raises
  ``FileNotFoundError`` with the *exact* request/download path if the data is absent.
- The real single-cell matrices are **not openly downloadable**:
    * Zenodo ``10.5281/zenodo.14624955`` hosts an annotated Seurat object, but the files are
      **RESTRICTED** (CC-BY license, yet the download is request-gated behind Zenodo login).
    * The clinical metadata needed to attach PFS labels is **CONTROLLED** via the MMRF VLAB
      (``https://mmrfvirtuallab.org``).
  So feature *derivation* is a user-machine step; this module gives you the offline-testable
  derivation logic plus an honest loader stub.

Two entry points
----------------
- ``load_immune_features(path)``: read a previously-derived per-patient feature CSV/Parquet
  (index = ``MMRF_xxxx``). Raises ``FileNotFoundError`` with actionable instructions if absent.
- ``derive_immune_features_from_anndata(adata, patient_col, celltype_col, ...)``: pure,
  offline-testable transform from a scanpy ``AnnData`` (cell annotations) into the per-patient
  feature frame. No optional heavy dependency at import time; ``anndata`` is only *duck-typed*.

The derived features mirror the MMRF Immune Atlas's own Figure-7 survival design:
cell-type **proportions** (non-tumor compartments) + a few T-cell **state scores**
(naive / cytotoxic / exhaustion / dysfunction), aggregated per ``public_id`` (``MMRF_xxxx``).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Canonical feature vocabulary (see docs/PATH_A_IMMUNE_FUSION.md §3)
# --------------------------------------------------------------------------------------

#: Coarse, interpretable lineages we compute proportions for. Tumor plasma cells are
#: intentionally EXCLUDED from the immune feature block (they re-encode the redundant
#: tumor-burden axis); keep them only as an optional purity covariate, not a feature.
DEFAULT_LINEAGES: tuple[str, ...] = (
    "CD8_T",
    "CD4_T",
    "Treg",
    "NK",
    "NKT",
    "CD14_Mono",
    "CD16_Mono",
    "cDC",
    "pDC",
    "B",
    "Erythroid",
    "HSC_Prog",
)

#: T-cell state signatures scored per cell, then averaged per patient. Gene-set provenance:
#: naive/cytotoxic/exhaustion = Chu et al. pan-cancer T-cell atlas (Tables S4+S6); the
#: dysfunctional set is the atlas's custom marker list, reproduced verbatim so it is
#: reconstructable without the external xlsx. "+"/"-" denote up/down markers; for a simple
#: mean-z module score we use the "+" (up) markers as the positive set.
TCELL_STATE_SIGNATURES: dict[str, list[str]] = {
    # NOTE: naive/cytotoxic/exhaustion full gene lists live in the Chu et al. supplement
    # (pan-cancer_tcell_atlas_chu_et_al.xlsx). Representative anchor markers are listed here
    # so the function is runnable/testable; swap in the full lists when the xlsx is available.
    "tcell_naive": ["CCR7", "TCF7", "LEF1", "SELL", "IL7R"],
    "tcell_cytotoxic": ["GZMB", "GZMK", "PRF1", "NKG7", "GNLY", "IFNG"],
    "tcell_exhaustion": ["PDCD1", "TIGIT", "LAG3", "HAVCR2", "CTLA4", "TOX"],
    "tcell_dysfunctional": [
        "B3GAT1", "ZEB2", "KLRG1", "KLRK1", "TIGIT",
        "PDCD1", "LAG3", "CTLA4", "HAVCR2", "TOX", "CD160",
    ],
}

#: Substrings (case-insensitive) used to decide which cells are T cells when computing the
#: T-state scores. Mirrors the atlas restricting state scoring to CD8/CD4 lineages.
_TCELL_CELLTYPE_HINTS: tuple[str, ...] = ("cd8", "cd4", "t cell", "t_cell", "tcell", "nkt")


# --------------------------------------------------------------------------------------
# Loader (honest stub)
# --------------------------------------------------------------------------------------

_DOWNLOAD_INSTRUCTIONS = """\
Immune-atlas feature table not found at: {path}

The MMRF Immune Atlas single-cell data is NOT openly downloadable. To produce this file:

  1. Annotated single-cell object (cell annotations, NO clinical metadata):
       Zenodo 10.5281/zenodo.14624955  ->  https://zenodo.org/records/14624955
       ACCESS = RESTRICTED. The record is CC-BY-4.0 but the FILES are request-gated:
       log in to Zenodo and request access, then download the .rds Seurat object
       (MMRF_ImmuneAtlas_Full_With_Corrected_Censored_Metadata.rds).

  2. Clinical metadata / PFS labels (to attach MMRF_xxxx -> (ttcpfs, censpfs)):
       MMRF VLAB (CONTROLLED ACCESS)  ->  https://mmrfvirtuallab.org  (submit a request form).
       CoMMpass open clinical can alternatively come from GDC project MMRF-COMMPASS
       (phs000748, OPEN) via src/resistancemap/data/gdc_clinical.py.

  3. Convert RDS -> h5ad with the ExportH5AD() helper in the atlas repo
       (_external/MMRF_ImmuneAtlas/manuscript_figures/figure_7/.../MMRF_immune_dataset.R),
       then derive per-patient features:

         import anndata, pandas as pd
         from resistancemap.data.immune_atlas import derive_immune_features_from_anndata
         adata = anndata.read_h5ad("imm_atlas.h5ad")
         feats = derive_immune_features_from_anndata(
             adata,
             patient_col="public_id",          # or derive from sample_id (MMRF_xxxx)
             celltype_col="celltype_subclusters_label_transferring_Yizhe_v1",
         )
         feats.to_csv("{path}")

  See docs/PATH_A_IMMUNE_FUSION.md for the full feature list and fusion plan.
  This loader will NOT synthesize values.
"""


def load_immune_features(path: str) -> pd.DataFrame:
    """Load a previously-derived per-patient immune feature table.

    Parameters
    ----------
    path : str
        Path to a CSV or Parquet file whose first column is the patient id (``MMRF_xxxx``)
        and whose remaining columns are immune features.

    Returns
    -------
    pandas.DataFrame
        Indexed by patient id (``MMRF_xxxx``), one column per immune feature.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist — with the exact, actionable download/request
        instructions (no values are fabricated).
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(_DOWNLOAD_INSTRUCTIONS.format(path=path))

    if path.endswith((".parquet", ".pq")):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0)

    df.index = df.index.astype(str)
    df.index.name = "patient_id"
    return df


# --------------------------------------------------------------------------------------
# Offline-testable derivation
# --------------------------------------------------------------------------------------


def _patient_id_from_sample(sample_id: str) -> str:
    """Recover the CoMMpass patient id (``MMRF_xxxx``) from a sample/aliquot id.

    The atlas's own convention (``surv_models_and_abbundance.R``) truncates ``sample_id``
    (e.g. ``MMRF_1817_1_BM``) to recover the patient. We strip to the ``MMRF_<digits>`` stem,
    which is robust to either ``MMRF_1817`` or ``MMRF_1817_1_BM`` inputs.
    """
    s = str(sample_id)
    parts = s.split("_")
    if len(parts) >= 2 and parts[0].upper().startswith("MMRF"):
        return f"{parts[0]}_{parts[1]}"
    return s


def _map_celltype_to_lineage(label: str, lineages: Sequence[str]) -> str | None:
    """Map a fine cell-type annotation to one of the coarse ``lineages`` by substring.

    Returns ``None`` if the label does not map to any requested lineage (e.g. tumor plasma
    cells, which we intentionally exclude from the immune feature block).
    """
    lab = str(label).lower()
    # Ordered hint table: first hit wins. Kept explicit for interpretability/audit.
    hints: list[tuple[str, tuple[str, ...]]] = [
        ("Treg", ("treg", "regulatory")),
        ("CD8_T", ("cd8",)),
        ("CD4_T", ("cd4",)),
        ("NKT", ("nkt",)),
        ("NK", ("nk",)),
        ("CD14_Mono", ("cd14", "classical mono")),
        ("CD16_Mono", ("cd16", "non-classical", "nonclassical")),
        ("pDC", ("pdc", "plasmacytoid")),
        ("cDC", ("cdc", "dendritic", "dc1", "dc2")),
        ("HSC_Prog", ("hsc", "progenitor", "gmp", "cmp", "mpp")),
        ("Erythroid", ("eryth", "erythro", "ery_")),
        ("B", ("b cell", "b_cell", "bcell", "memory b", "naive b", "plasmablast")),
    ]
    lineset = set(lineages)
    for lineage, keys in hints:
        if lineage in lineset and any(k in lab for k in keys):
            return lineage
    return None


def _module_score(
    expr: np.ndarray,
    var_names: Sequence[str],
    gene_set: Sequence[str],
) -> np.ndarray:
    """Simple per-cell mean-expression module score for ``gene_set``.

    This is a lightweight stand-in for Seurat ``AddModuleScore`` / ``UCell`` so the function
    is dependency-light and testable. The atlas uses UCell; swap that in on the real data for
    parity. Genes absent from ``var_names`` are ignored; an all-absent set yields zeros.
    """
    name_to_idx = {str(g): i for i, g in enumerate(var_names)}
    idx = [name_to_idx[g] for g in gene_set if g in name_to_idx]
    if not idx:
        return np.zeros(expr.shape[0], dtype=float)
    return np.asarray(expr[:, idx], dtype=float).mean(axis=1)


def _to_dense_matrix(X) -> np.ndarray:
    """Coerce an AnnData ``.X`` (dense or scipy-sparse) to a 2-D float ndarray."""
    if hasattr(X, "toarray"):  # scipy sparse
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


def derive_immune_features_from_anndata(
    adata,
    patient_col: str = "public_id",
    celltype_col: str = "celltype_subclusters_label_transferring_Yizhe_v1",
    *,
    sample_col: str | None = None,
    lineages: Sequence[str] = DEFAULT_LINEAGES,
    state_signatures: Mapping[str, Sequence[str]] | None = None,
    baseline_mask_col: str | None = None,
    baseline_value: str = "Baseline",
) -> pd.DataFrame:
    """Derive a per-patient immune feature table from an annotated scanpy ``AnnData``.

    Pure / offline-testable: only duck-types the AnnData interface (``.obs``, ``.X``,
    ``.var_names``). No values are fabricated — every number is an aggregate of real cells.

    Parameters
    ----------
    adata
        scanpy ``AnnData`` of annotated bone-marrow cells (cells x genes). ``.X`` should be
        log-normalized expression for the module scores; counts also work for proportions.
    patient_col
        ``.obs`` column holding the patient id (``MMRF_xxxx``). If absent and ``sample_col``
        is given, the patient id is derived from the sample id via ``_patient_id_from_sample``.
    celltype_col
        ``.obs`` column with the fine cell-type annotation used for both proportions and the
        T-cell restriction of the state scores.
    sample_col
        Optional ``.obs`` column with sample/aliquot ids (e.g. ``MMRF_1817_1_BM``); used to
        derive ``patient_col`` when it is not already present.
    lineages
        Coarse lineages to compute proportions for (default :data:`DEFAULT_LINEAGES`).
    state_signatures
        Mapping ``score_name -> gene list``. Defaults to :data:`TCELL_STATE_SIGNATURES`.
    baseline_mask_col, baseline_value
        If given, restrict to baseline cells (avoids PFS time-leakage), e.g.
        ``baseline_mask_col="VJ_INTERVAL", baseline_value="Baseline"``.

    Returns
    -------
    pandas.DataFrame
        Index = patient id (``MMRF_xxxx``). Columns: ``prop_<lineage>`` (sum to ~1 over the
        requested lineages per patient), the state scores, and ``n_cells`` (support).
    """
    if state_signatures is None:
        state_signatures = TCELL_STATE_SIGNATURES

    obs = adata.obs.copy()

    # --- resolve patient id ---
    if patient_col in obs.columns:
        pid = obs[patient_col].astype(str)
    elif sample_col is not None and sample_col in obs.columns:
        pid = obs[sample_col].astype(str).map(_patient_id_from_sample)
    else:
        raise KeyError(
            f"patient_col '{patient_col}' not in adata.obs and no usable sample_col given; "
            f"available obs columns: {list(obs.columns)}"
        )
    pid = pid.map(_patient_id_from_sample)  # normalize MMRF_1817_1_BM -> MMRF_1817

    if celltype_col not in obs.columns:
        raise KeyError(
            f"celltype_col '{celltype_col}' not in adata.obs; "
            f"available obs columns: {list(obs.columns)}"
        )
    celltype = obs[celltype_col].astype(str)

    # --- optional baseline restriction ---
    keep = np.ones(len(obs), dtype=bool)
    if baseline_mask_col is not None and baseline_mask_col in obs.columns:
        keep = obs[baseline_mask_col].astype(str).str.lower().values == str(baseline_value).lower()
    pid = pid[keep].reset_index(drop=True)
    celltype = celltype[keep].reset_index(drop=True)
    X = _to_dense_matrix(adata.X)[keep, :]
    var_names = list(map(str, adata.var_names))

    # --- cell-type proportions per patient ---
    lineage = celltype.map(lambda c: _map_celltype_to_lineage(c, lineages))
    comp = pd.DataFrame({"patient_id": pid.values, "lineage": lineage.values})
    comp_known = comp.dropna(subset=["lineage"])
    counts = (
        comp_known.groupby(["patient_id", "lineage"]).size().unstack(fill_value=0)
    )
    # Ensure all requested lineages are present as columns (zero where unobserved).
    for lin in lineages:
        if lin not in counts.columns:
            counts[lin] = 0
    counts = counts[list(lineages)]
    totals = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(totals, axis=0).fillna(0.0)
    props.columns = [f"prop_{c}" for c in props.columns]

    # --- T-cell state scores (mean over a patient's T cells) ---
    is_t = celltype.str.lower().apply(lambda s: any(h in s for h in _TCELL_CELLTYPE_HINTS))
    state_cols: dict[str, pd.Series] = {}
    for score_name, genes in state_signatures.items():
        per_cell = _module_score(X, var_names, list(genes))
        sdf = pd.DataFrame(
            {"patient_id": pid.values, "score": per_cell, "is_t": is_t.values}
        )
        t_only = sdf[sdf["is_t"]]
        # Mean per patient over T cells; patients with no T cells -> NaN (honest missing).
        state_cols[score_name] = t_only.groupby("patient_id")["score"].mean()

    states = pd.DataFrame(state_cols)

    # --- assemble ---
    n_cells = comp.groupby("patient_id").size().rename("n_cells")
    out = props.join(states, how="outer").join(n_cells, how="outer")
    out.index.name = "patient_id"
    # Proportions are honest 0 where a lineage is unobserved; state scores stay NaN when a
    # patient has no T cells (do not fabricate). n_cells fills 0 for patients with no cells.
    out["n_cells"] = out["n_cells"].fillna(0).astype(int)
    return out.sort_index()
