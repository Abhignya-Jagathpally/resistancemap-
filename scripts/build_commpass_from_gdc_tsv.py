"""Assemble the analysis-ready CoMMpass survival table from the GDC-OPEN MMRF dump
already on disk (TSV layout produced by the GDC API + STAR gene-counts merge), rather
than the live GDC /cases CSV that scripts/build_commpass_dataset.py expects.

Reuses, does not reinvent:
  - resistancemap.data.gdc_clinical.build_survival_frame  (OS endpoint from open clinical)
  - resistancemap.data.gene_sets.{MM_PROGRAMS, program_scores}

Inputs (real, open-access; --raw points at .../mmrf_commpass):
  clinical.tsv          one row/case: submitter_id, vital_status, days_to_death,
                        days_to_last_follow_up, age_at_diagnosis_days, gender, iss_stage
  gene_expression.tsv   gene_id (Ensembl, version-stripped) x aliquot TPM matrix
  rna/<uuid>.*star_gene_counts.tsv  any one carries gene_id<->gene_name (offline ENSG->symbol)

Endpoint note: GDC OPEN tier exposes overall survival (vital_status / days_to_death /
days_to_last_follow_up), NOT IMWG-PFS. This builder therefore writes an OS table and
labels it as such; PFS requires the gated Researcher Gateway and stays off by default.

Output: data/processed/commpass.csv with columns
  patient_id, duration, event, age_z, sex_male, iss_stage, + 5 program-score columns.
"""
from __future__ import annotations

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd

from resistancemap.data.gdc_clinical import build_survival_frame
from resistancemap.data.gene_sets import MM_PROGRAMS, program_scores


def load_clinical(raw: str) -> pd.DataFrame:
    """Adapt the on-disk clinical.tsv to the schema build_survival_frame consumes."""
    c = pd.read_csv(os.path.join(raw, "clinical.tsv"), sep="\t", dtype=str)
    adapted = pd.DataFrame()
    adapted["case_submitter_id"] = c["submitter_id"]
    adapted["vital_status"] = c["vital_status"]
    adapted["days_to_death"] = pd.to_numeric(c["days_to_death"], errors="coerce")
    adapted["days_to_last_follow_up"] = pd.to_numeric(c["days_to_last_follow_up"], errors="coerce")
    # _encode_clinical treats `age` as DAYS and /365.25 -> years, so pass days through.
    adapted["age"] = pd.to_numeric(c["age_at_diagnosis_days"], errors="coerce")
    adapted["gender"] = c["gender"]
    adapted["iss_stage"] = c["iss_stage"]
    return adapted


def encode_clinical(sf: pd.DataFrame) -> pd.DataFrame:
    """Same encoding contract as scripts/build_commpass_dataset.py::_encode_clinical."""
    out = pd.DataFrame()
    out["patient_id"] = sf["case_submitter_id"].astype(str)
    out["duration"] = pd.to_numeric(sf["duration"], errors="coerce")
    out["event"] = pd.to_numeric(sf["event"], errors="coerce").astype("Int64")
    age_years = pd.to_numeric(sf.get("age"), errors="coerce") / 365.25
    sd = age_years.std() or 1.0
    out["age_z"] = (age_years - age_years.mean()) / sd
    out["sex_male"] = (sf.get("gender").astype(str).str.lower() == "male").astype(float)
    iss = (sf.get("iss_stage").astype(str).str.upper()
           .str.replace("STAGE", "", regex=False).str.strip())
    out["iss_stage"] = iss.map({"I": 1, "II": 2, "III": 3, "1": 1, "2": 2, "3": 3}).astype(float)
    return out


def ensembl_to_symbol(raw: str) -> dict[str, str]:
    """Build a version-stripped ENSG -> gene_name map from any one STAR gene-counts TSV."""
    star = sorted(glob.glob(os.path.join(raw, "rna", "*star_gene_counts.tsv")))
    if not star:
        raise FileNotFoundError(f"no STAR gene-counts TSV under {raw}/rna for ENSG->symbol map")
    d = pd.read_csv(star[0], sep="\t", comment="#")
    d = d[d["gene_id"].astype(str).str.startswith("ENSG")]
    ids = d["gene_id"].astype(str).str.split(".").str[0]
    return dict(zip(ids, d["gene_name"].astype(str)))


def patient_of(col: str) -> str:
    """MMRF_1817_1_BM_CD138pos_T2_... -> MMRF_1817 (case submitter id)."""
    parts = col.split("_")
    return "_".join(parts[:2])


def visit_of(col: str) -> int:
    """Third underscore token is the visit index; lower = earlier (baseline)."""
    parts = col.split("_")
    try:
        return int(parts[2])
    except (IndexError, ValueError):
        return 9999


def build_baseline_expression(raw: str) -> pd.DataFrame:
    """genes(symbol) x patients log1p-TPM, restricted to MM program genes, baseline visit."""
    sym = ensembl_to_symbol(raw)
    program_genes = {g for genes in MM_PROGRAMS.values() for g in genes}
    # Reverse map: only Ensembl IDs whose symbol is a program gene (keeps the read tiny).
    wanted_ensg = {e for e, s in sym.items() if s in program_genes}

    expr = pd.read_csv(os.path.join(raw, "gene_expression.tsv"), sep="\t", index_col=0)
    expr.index = expr.index.astype(str).str.split(".").str[0]
    expr = expr.loc[expr.index.isin(wanted_ensg)]
    expr = expr.rename(index=sym)
    # Collapse duplicate symbols (paralog probes) by max.
    expr = expr.groupby(level=0).max()

    # Aliquot columns -> baseline (earliest visit) per patient.
    cols = list(expr.columns)
    by_patient: dict[str, str] = {}
    for c in cols:
        p = patient_of(c)
        if p not in by_patient or visit_of(c) < visit_of(by_patient[p]):
            by_patient[p] = c
    baseline_cols = list(by_patient.values())
    sub = expr[baseline_cols].copy()
    sub.columns = [patient_of(c) for c in baseline_cols]
    # Average any residual duplicate patient columns.
    sub = sub.T.groupby(level=0).mean().T
    sub.columns.name = "patient_id"
    return np.log1p(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../ResistanceMap/data/raw/mmrf_commpass")
    ap.add_argument("--out", default="data/processed/commpass.csv")
    a = ap.parse_args()

    base = encode_clinical(build_survival_frame(load_clinical(a.raw)))
    print(f"[clinical] {len(base)} patients with usable OS; events={int(base['event'].sum())}")

    expr = build_baseline_expression(a.raw)
    found = sum(1 for genes in MM_PROGRAMS.values() for g in genes if g in expr.index)
    total = sum(len(g) for g in MM_PROGRAMS.values())
    print(f"[expr] {expr.shape[0]} program genes x {expr.shape[1]} patients "
          f"(matched {found}/{total} program genes to expression)")

    ps = program_scores(expr, MM_PROGRAMS)
    ps.index.name = "patient_id"
    ps = ps.reset_index()
    ps["patient_id"] = ps["patient_id"].astype(str)

    df = base.merge(ps, on="patient_id", how="inner")
    df = df.dropna(subset=["duration", "event"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    feats = [c for c in df.columns if c not in ("patient_id", "duration", "event")]
    print(f"[ok] wrote {a.out}: {df.shape[0]} patients x {df.shape[1]} cols; "
          f"events={int(df['event'].sum())}")
    print(f"[ok] endpoint=overall_survival (GDC open tier); features={feats}")


if __name__ == "__main__":
    main()
