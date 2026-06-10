"""Loader for the OPEN tier of MMRF CoMMpass (no dbGaP / Researcher Gateway needed).

Download on your machine (the sandbox has no S3/GDC network):

    # RNA-seq STAR counts (open S3, no AWS account):
    aws s3 cp --no-sign-request --recursive \
        s3://gdc-mmrf-commpass-phs000748-2-open/  data/raw/gdc_open/

    # Harmonized clinical (vital status, days_to_death/follow_up, ISS, age, sex)
    # via the GDC API for project MMRF-COMMPASS (open clinical is downloadable).

Then build a survival frame: one row per patient with `duration`, `event`, and
covariates (clinical + gene-set program scores from the STAR counts).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_survival_frame(clinical: pd.DataFrame) -> pd.DataFrame:
    """Map GDC open clinical fields to (duration, event). Expects columns like
    vital_status, days_to_death, days_to_last_follow_up, ajcc/iss stage, age, gender."""
    c = clinical.copy()
    dead = c["vital_status"].str.lower().eq("dead")
    c["event"] = dead.astype(int)
    c["duration"] = np.where(dead, c.get("days_to_death"), c.get("days_to_last_follow_up"))
    c = c[c["duration"].notna() & (c["duration"] > 0)].copy()
    return c


def program_scores(expr: pd.DataFrame, gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    """Interpretable z-scored mean-expression program scores (a transparent ssGSEA-style
    stand-in). expr: genes x patients (log1p TPM). Returns patients x programs."""
    z = expr.sub(expr.mean(1), axis=0).div(expr.std(1).replace(0, np.nan), axis=0).fillna(0.0)
    out = {name: z.reindex(genes).dropna().mean(0) for name, genes in gene_sets.items()}
    return pd.DataFrame(out)
