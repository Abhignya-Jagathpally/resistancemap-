"""Named multiple-myeloma program gene-sets + a transparent program-scoring function.

These are small, human-readable marker panels used to turn a genes x patients expression
matrix into a compact patients x programs covariate block (a transparent, ssGSEA-style
stand-in: z-score each gene across patients, then average within a program). They are
curated marker lists for pipeline interpretability, NOT a validated signature; treat the
``mmsygnal_like`` set in particular as a labelled PLACEHOLDER, not the proprietary
mSignal/MMprofiler panel.

Use ``program_scores(expr_df, MM_PROGRAMS)`` for the default panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Cell-cycle / proliferation program (proliferation index drivers).
proliferation: list[str] = [
    "MKI67", "TOP2A", "CCNB1", "CCNB2", "CDK1", "CCNA2", "BUB1", "AURKA", "AURKB",
    "PLK1", "CENPA", "CENPE", "TYMS", "RRM2", "PCNA", "MCM2", "MCM6", "BIRC5",
]

# Proteasome / unfolded-protein-response program (proteasome-inhibitor biology).
proteasome_UPR: list[str] = [
    "PSMB5", "PSMB1", "PSMB2", "PSMA1", "PSMA5", "PSMC1", "PSMD1",
    "HSPA5", "DDIT3", "ATF4", "XBP1", "ERN1", "EIF2AK3", "ATF6", "HERPUD1", "SEC61A1",
]

# Stemness / progenitor program. Per task spec this MUST include the following six genes.
stemness: list[str] = [
    "PTPRG", "E2F8", "E2F7", "FOXM1", "E2F1", "TIMELESS",
    "SOX2", "POU5F1", "NANOG", "PROM1", "ALDH1A1", "KLF4", "MYC",
]

# Immune microenvironment program (T/NK/myeloid infiltration, checkpoint).
immune_microenv: list[str] = [
    "CD3D", "CD3E", "CD8A", "CD4", "NKG7", "GZMB", "PRF1", "IFNG",
    "CD68", "CD163", "ITGAM", "PDCD1", "CD274", "CTLA4", "LAG3", "FOXP3",
]

# PLACEHOLDER mimicking the *shape* of a proprietary MM prognostic signature
# (e.g. mSignal / MMprofiler-style high-risk panels). NOT the real panel -- a stand-in
# of well-known MM high-risk-associated genes so downstream code has a named slot to fill.
mmsygnal_like: list[str] = [
    "FGFR3", "WHSC1", "CCND1", "CCND2", "MAF", "MAFB", "MMSET",
    "TP53", "RB1", "CKS1B", "DKK1", "FRZB", "TNFRSF17", "SDC1",
]

# Default panel exposed to the pipeline.
MM_PROGRAMS: dict[str, list[str]] = {
    "proliferation": proliferation,
    "proteasome_UPR": proteasome_UPR,
    "stemness": stemness,
    "immune_microenv": immune_microenv,
    "mmsygnal_like": mmsygnal_like,
}


def program_scores(expr_df: pd.DataFrame, gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    """Compute z-scored mean-expression program scores.

    Parameters
    ----------
    expr_df:
        genes x patients expression matrix (rows = gene symbols, columns = patient ids),
        assumed to be log1p TPM (or any roughly-symmetric per-gene scale).
    gene_sets:
        mapping of program name -> list of gene symbols.

    Returns
    -------
    A patients x programs DataFrame. Each gene is z-scored across patients (genes with
    zero variance map to 0, never NaN); each program score is the mean z over the genes
    of that program that are actually present in ``expr_df``. Missing genes are dropped
    silently; a program with no overlapping genes yields an all-zero column (with a
    runtime warning) rather than crashing.
    """
    # Per-gene z-score across patients (axis=1). Guard zero-variance genes -> 0.
    mean = expr_df.mean(axis=1)
    std = expr_df.std(axis=1).replace(0.0, np.nan)
    z = expr_df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)

    patients = expr_df.columns
    scores: dict[str, pd.Series] = {}
    for name, genes in gene_sets.items():
        present = [g for g in genes if g in z.index]
        if not present:
            import warnings

            warnings.warn(
                f"program '{name}': none of its {len(genes)} genes were found in expr_df; "
                "emitting an all-zero score column.",
                RuntimeWarning,
                stacklevel=2,
            )
            scores[name] = pd.Series(0.0, index=patients)
        else:
            scores[name] = z.loc[present].mean(axis=0)

    out = pd.DataFrame(scores, index=patients)
    out.index.name = expr_df.columns.name or "patient_id"
    return out
