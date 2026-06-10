"""Offline tests for the data layer. NO internet required.

We synthesize a tiny genes x patients expression matrix and a small clinical table
(both clearly labelled SYNTHETIC -- no biological meaning) and exercise:
  - resistancemap.data.gene_sets.program_scores
  - resistancemap.data.gdc_clinical.build_survival_frame  (+ parse_cases_response)
  - resistancemap.data.scrna_qc.basic_qc_counts            (numpy fallback path)
  - resistancemap.data.scrna_qc.mad_qc_anndata             (asserts it errors w/o scanpy)
asserting shapes, columns and dtypes. Run:
    cd resistancemap && PYTHONPATH=src python3 tests/test_data.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from resistancemap.data.gene_sets import MM_PROGRAMS, program_scores
from resistancemap.data.gdc_clinical import build_survival_frame, parse_cases_response
from resistancemap.data.scrna_qc import basic_qc_counts, mad_qc_anndata


def synth_expression(n_genes: int = 60, n_patients: int = 25, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC genes x patients log1p-TPM-shaped matrix. No biological meaning."""
    rng = np.random.default_rng(seed)
    # Include some real program gene symbols so overlap is non-trivial, plus filler genes.
    named = ["PTPRG", "E2F8", "E2F7", "FOXM1", "E2F1", "TIMELESS", "MKI67", "PSMB5",
             "CD3D", "TNFRSF17"]  # >=1 gene from every default program -> no QC warning
    filler = [f"GENE{i:04d}" for i in range(n_genes - len(named))]
    genes = named + filler
    patients = [f"SYN{i:03d}" for i in range(n_patients)]
    data = rng.gamma(shape=2.0, scale=1.0, size=(n_genes, n_patients))
    expr = pd.DataFrame(np.log1p(data), index=genes, columns=patients)
    expr.columns.name = "patient_id"
    return expr


def synth_clinical(seed: int = 1) -> pd.DataFrame:
    """SYNTHETIC GDC-open-shaped clinical table. No biological meaning."""
    return pd.DataFrame(
        {
            "case_submitter_id": ["MMRF_0001", "MMRF_0002", "MMRF_0003", "MMRF_0004", "MMRF_0005"],
            "vital_status": ["Dead", "Alive", "alive", "Dead", "Alive"],
            "days_to_death": [420.0, np.nan, np.nan, 130.0, np.nan],
            "days_to_last_follow_up": [np.nan, 800.0, 0.0, np.nan, -5.0],
            "age": [61.0, 70.0, 55.0, 48.0, 66.0],
            "gender": ["male", "female", "male", "female", "male"],
            "iss_stage": ["I", "III", "II", "III", "I"],
        }
    )


def test_program_scores():
    expr = synth_expression()
    scores = program_scores(expr, MM_PROGRAMS)

    # patients x programs
    assert scores.shape == (expr.shape[1], len(MM_PROGRAMS)), scores.shape
    assert list(scores.columns) == list(MM_PROGRAMS.keys())
    assert list(scores.index) == list(expr.columns)
    # numeric, finite, never NaN
    assert all(np.issubdtype(dt, np.floating) for dt in scores.dtypes), scores.dtypes.tolist()
    assert np.isfinite(scores.to_numpy()).all()

    # Required stemness genes are all present in the panel and in our synthetic matrix,
    # so the stemness column must be a genuine (non-degenerate) score, not the zero
    # fallback. With per-gene z-scoring the column mean is ~0 but it must vary across
    # patients.
    for g in ("PTPRG", "E2F8", "E2F7", "FOXM1", "E2F1", "TIMELESS"):
        assert g in MM_PROGRAMS["stemness"], g
        assert g in expr.index, g
    assert scores["stemness"].std() > 0
    print(f"[ok] program_scores -> {scores.shape[0]} patients x {scores.shape[1]} programs")


def test_program_scores_missing_genes():
    # A program whose genes are entirely absent must yield an all-zero column, not crash.
    expr = synth_expression()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scores = program_scores(expr, {"ghost": ["NOPE1", "NOPE2"], "stemness": MM_PROGRAMS["stemness"]})
    assert scores.shape == (expr.shape[1], 2)
    assert (scores["ghost"] == 0.0).all()
    print("[ok] program_scores handles missing genes (all-zero column)")


def test_build_survival_frame():
    clin = synth_clinical()
    surv = build_survival_frame(clin)

    # Rows with NaN/non-positive duration dropped: patient 3 (follow_up=0) and
    # patient 5 (follow_up=-5) go; patients 1, 2, 4 remain.
    assert len(surv) == 3, len(surv)
    for col in ("duration", "event", "case_submitter_id", "age", "gender", "iss_stage"):
        assert col in surv.columns, col

    assert surv["event"].dtype == np.dtype("int64") or np.issubdtype(surv["event"].dtype, np.integer)
    assert np.issubdtype(surv["duration"].dtype, np.floating)
    assert set(surv["event"].unique()).issubset({0, 1})
    assert (surv["duration"] > 0).all()

    # Dead patients take days_to_death; alive take days_to_last_follow_up.
    row1 = surv[surv["case_submitter_id"] == "MMRF_0001"].iloc[0]
    assert row1["event"] == 1 and row1["duration"] == 420.0
    row2 = surv[surv["case_submitter_id"] == "MMRF_0002"].iloc[0]
    assert row2["event"] == 0 and row2["duration"] == 800.0
    print(f"[ok] build_survival_frame -> {len(surv)} usable patients, cols={list(surv.columns)}")


def test_parse_cases_response_then_survival():
    # Hand-built GDC-shaped payload (the structure fetch_open_clinical would return),
    # proving the parser -> survival-frame path end to end without any network.
    payload = {
        "data": {
            "hits": [
                {
                    "submitter_id": "MMRF_9001",
                    "demographic": {"vital_status": "Dead", "days_to_death": 300, "gender": "male"},
                    "diagnoses": [{"age_at_diagnosis": 365.25 * 60, "days_to_last_follow_up": 250, "iss_stage": "II"}],
                },
                {
                    "submitter_id": "MMRF_9002",
                    "demographic": {"vital_status": "Alive", "days_to_death": None, "gender": "female"},
                    "diagnoses": [{"age_at_diagnosis": 365.25 * 72, "days_to_last_follow_up": 900, "iss_stage": "I"}],
                },
            ]
        }
    }
    clin = parse_cases_response(payload)
    assert list(clin.columns) == [
        "case_submitter_id", "vital_status", "days_to_death", "gender",
        "age", "days_to_last_follow_up", "iss_stage",
    ], list(clin.columns)
    assert clin.loc[0, "age"] == 60.0  # 365.25*60 days -> 60 years

    surv = build_survival_frame(clin)
    assert len(surv) == 2
    assert surv.loc[surv["case_submitter_id"] == "MMRF_9001", "duration"].iloc[0] == 300.0
    print("[ok] parse_cases_response -> build_survival_frame (synthetic GDC payload)")


def test_basic_qc_counts():
    rng = np.random.default_rng(2)
    # SYNTHETIC cells x genes count matrix.
    mat = rng.poisson(lam=1.0, size=(200, 40)).astype(float)
    qc = basic_qc_counts(mat)
    assert qc["n_genes"].shape == (200,)
    assert qc["total_counts"].shape == (200,)
    assert qc["keep"].dtype == bool
    assert qc["n_cells"] == 200
    assert 0 < qc["n_cells_kept"] <= 200
    assert set(qc["thresholds"]) == {
        "total_counts_low", "total_counts_high", "n_genes_low", "n_genes_high"
    }
    print(f"[ok] basic_qc_counts -> kept {qc['n_cells_kept']}/{qc['n_cells']} cells")


def test_mad_qc_requires_scanpy():
    # scanpy is not installed here; the function must raise a clear, actionable ImportError.
    try:
        mad_qc_anndata(object())
    except ImportError as exc:
        assert "scanpy" in str(exc).lower()
        print("[ok] mad_qc_anndata raises actionable ImportError when scanpy is absent")
    else:  # pragma: no cover - only if scanpy somehow present
        print("[ok] scanpy present; mad_qc_anndata import path available")


def main():
    print("=== resistancemap data-layer tests (SYNTHETIC data, no internet) ===")
    test_program_scores()
    test_program_scores_missing_genes()
    test_build_survival_frame()
    test_parse_cases_response_then_survival()
    test_basic_qc_counts()
    test_mad_qc_requires_scanpy()
    print("ALL DATA TESTS PASSED")


if __name__ == "__main__":
    main()
