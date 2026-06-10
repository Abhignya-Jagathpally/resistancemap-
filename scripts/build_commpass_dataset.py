"""Assemble the analysis-ready CoMMpass survival table from GDC-open downloads.
Run AFTER scripts/download_gdc_open.sh. Reuses resistancemap.data.gdc_clinical and
resistancemap.data.gene_sets. Output: data/processed/commpass.csv with columns
patient_id, duration, event, age_z, sex_male, iss_stage, + program-score columns.

Inputs (on your machine):
  data/raw/mmrf_open_clinical.csv   (from fetch_open_clinical)
  data/raw/expression_matrix.tsv    OPTIONAL genes x patients log1p TPM
  data/raw/gdc_sample_sheet.tsv     OPTIONAL (to assemble expression from raw STAR files)
If no expression is available, writes a clinical-only table so the pipeline still runs."""
from __future__ import annotations
import os, sys, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from resistancemap.data.gdc_clinical import build_survival_frame
from resistancemap.data.gene_sets import MM_PROGRAMS, program_scores


def _encode_clinical(sf: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["patient_id"] = sf["case_submitter_id"].astype(str)
    out["duration"] = pd.to_numeric(sf["duration"], errors="coerce")
    out["event"] = pd.to_numeric(sf["event"], errors="coerce").astype("Int64")
    age_days = pd.to_numeric(sf.get("age"), errors="coerce")          # GDC age_at_diagnosis is in DAYS
    age_years = age_days / 365.25
    sd = age_years.std() or 1.0
    out["age_z"] = (age_years - age_years.mean()) / sd
    g = sf.get("gender").astype(str).str.lower()
    out["sex_male"] = (g == "male").astype(float)
    iss = (sf.get("iss_stage").astype(str).str.upper()
           .str.replace("STAGE", "", regex=False).str.strip())
    out["iss_stage"] = iss.map({"I": 1, "II": 2, "III": 3, "1": 1, "2": 2, "3": 3}).astype(float)
    return out


def assemble_expression(raw_dir: str, sample_sheet: str) -> pd.DataFrame | None:
    """Build genes x patients log1p TPM from GDC STAR per-sample TSVs + the GDC sample sheet."""
    if not os.path.exists(sample_sheet):
        return None
    ss = pd.read_csv(sample_sheet, sep="\t")
    file2case = dict(zip(ss.get("File Name", []), ss.get("Case ID", [])))
    cols: dict[str, pd.Series] = {}
    for f in glob.glob(os.path.join(raw_dir, "**", "*star_gene_counts.tsv"), recursive=True):
        case = file2case.get(os.path.basename(f))
        if case is None:
            continue
        d = pd.read_csv(f, sep="\t", comment="#")
        if "gene_name" not in d or "tpm_unstranded" not in d:
            continue
        d = d[d["gene_name"].notna()]
        s = pd.to_numeric(d["tpm_unstranded"], errors="coerce")
        s.index = d["gene_name"].values
        cols[str(case).split(",")[0].strip()] = np.log1p(s.groupby(level=0).max())
    if not cols:
        return None
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", default="data/raw/mmrf_open_clinical.csv")
    ap.add_argument("--expr", default="data/raw/expression_matrix.tsv")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--sample_sheet", default="data/raw/gdc_sample_sheet.tsv")
    ap.add_argument("--out", default="data/processed/commpass.csv")
    a = ap.parse_args()
    base = _encode_clinical(build_survival_frame(pd.read_csv(a.clinical)))
    expr = pd.read_csv(a.expr, sep="\t", index_col=0) if os.path.exists(a.expr) else assemble_expression(a.raw_dir, a.sample_sheet)
    if expr is not None:
        ps = program_scores(expr, MM_PROGRAMS); ps.index.name = "patient_id"; ps = ps.reset_index()
        ps["patient_id"] = ps["patient_id"].astype(str)
        df = base.merge(ps, on="patient_id", how="inner")
        print(f"[ok] merged expression program scores: {ps.shape[1]-1} programs")
    else:
        print("[warn] no expression found -> clinical-only covariates")
        df = base
    df = df.dropna(subset=["duration", "event"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.shape[0]} patients x {df.shape[1]} cols; events={int(df['event'].sum())}")
    print("feature cols:", [c for c in df.columns if c not in ('patient_id', 'duration', 'event')])


if __name__ == "__main__":
    main()
