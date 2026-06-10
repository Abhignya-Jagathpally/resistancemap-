"""Loader for the mmSYGNAL open risk-prediction substrate (Wall et al., Baliga Lab).

Source repo (public, git-LFS): github.com/baliga-lab/mmSYGNAL-risk-prediction-models
Data are MMRF-CoMMpass interim analyses IA12 (training) and IA18 (validation),
derived via the miner causal/regulon framework. We consume the *open derived
features + labels* — we do NOT redistribute them; point ``root`` at your local clone.

What this gives the pipeline
----------------------------
- X_programs : 141 mmSYGNAL transcriptional **program-activity** scores per patient
  (mechanistic, interpretable; the published feature space).
- X_clinical : ISS + 7 cytogenetic flags (del13/del17p/del1p/amp1q/CCND1/FGFR3/WHSC1)
  + total_muts + cyto_count.
- target     : (duration=D_PFS in **days**, event=D_PFS_FLAG)  — IMWG-PFS, the proper
  MM endpoint (fixes the OS-only limitation of the GDC-open loader).
- comparators: gep70, sky92  (HONEST published signatures), and GuanScore / risk_auc
  (mmSYGNAL's own scores — **outcome-derived / in-sample; treated as leakage**, see
  ``leakage_audit``: do not quote them as a beatable SOTA bar).

No fabrication: if files are absent, raise FileNotFoundError with the clone command.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULT_ROOT = "_external/mmSYGNAL-risk-prediction-models/data"

CLINICAL_COLS = ["ISS", "del13", "del17p", "del1p", "amp1q",
                 "CCND1", "FGFR3", "WHSC1", "total_muts", "cyto_count"]
HONEST_SOTA = ["gep70", "sky92"]          # independently-derived published signatures
LEAKY_SCORES = ["GuanScore", "risk_auc"]  # mmSYGNAL in-sample / outcome-derived


@dataclass
class MMSygnalData:
    df: pd.DataFrame
    program_cols: list[str]
    clinical_cols: list[str] = field(default_factory=lambda: list(CLINICAL_COLS))

    @property
    def n(self) -> int:
        return len(self.df)

    @property
    def n_events(self) -> int:
        return int(self.df["event"].sum())

    def features(self, blocks=("programs", "clinical")) -> list[str]:
        cols: list[str] = []
        if "programs" in blocks:
            cols += self.program_cols
        if "clinical" in blocks:
            cols += [c for c in self.clinical_cols if c in self.df.columns]
        return cols


def _require(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"mmSYGNAL file not found: {path}\n"
            "Clone the open repo and pull its LFS CSVs:\n"
            "  git clone https://github.com/baliga-lab/mmSYGNAL-risk-prediction-models "
            "_external/mmSYGNAL-risk-prediction-models\n"
            "  cd _external/mmSYGNAL-risk-prediction-models && git lfs pull\n"
            "(If git-lfs is unavailable, resolve pointers via the GitHub LFS batch API.)"
        )
    return path


def load_ia12(root: str = DEFAULT_ROOT) -> MMSygnalData:
    """Load IA12 training cohort: programs x patients + PFS + clinical + comparators."""
    pa = pd.read_csv(_require(os.path.join(root, "program_activity_IA12_py.csv")))
    pa = pa.set_index("program").T.reset_index().rename(columns={"index": "sample"})
    program_cols = [c for c in pa.columns if c != "sample"]

    ph = pd.read_csv(_require(os.path.join(root, "ia12_pheno_01012023.csv")))
    keep = ["sample", "D_PFS", "D_PFS_FLAG"] + CLINICAL_COLS + HONEST_SOTA + LEAKY_SCORES
    ph = ph[[c for c in keep if c in ph.columns]].copy()

    m = ph.merge(pa, on="sample", how="inner")
    m = m[m["D_PFS"] > 0].dropna(subset=["D_PFS", "D_PFS_FLAG"]).reset_index(drop=True)
    m["patient_id"] = m["sample"].astype(str)
    m["duration"] = m["D_PFS"].astype(float)          # DAYS
    m["event"] = m["D_PFS_FLAG"].astype(int)
    return MMSygnalData(df=m, program_cols=program_cols)


def load_ia18_validation(root: str = DEFAULT_ROOT) -> pd.DataFrame:
    """Independent IA18 relapse cohort for external generalization (Path C).

    Returns a frame with patient_id, duration (PFS_d, days), event (relapse), and the
    miner_risk score. Program activity for IA18 lives in program_activity_py.csv.
    """
    v = pd.read_csv(_require(os.path.join(root, "IA18_relapse_minernorm_best_quality_risk.csv")))
    out = pd.DataFrame()
    out["patient_id"] = v["sample"].astype(str)
    out["duration"] = pd.to_numeric(v["PFS_d"], errors="coerce")
    out["event"] = pd.to_numeric(v["relapse"], errors="coerce").astype("Int64")
    out["miner_risk"] = pd.to_numeric(v["miner_risk"], errors="coerce")
    out = out[out["duration"] > 0].dropna(subset=["duration", "event"]).reset_index(drop=True)
    return out


def leakage_audit(data: MMSygnalData) -> pd.DataFrame:
    """Univariate C-index of each precomputed score vs PFS.

    A C-index at/near 1.0 flags an outcome-derived (leaky) score that MUST NOT be used
    as a beatable SOTA bar. Returns a tidy table with an `is_leaky` verdict (C >= 0.95).
    """
    from lifelines.utils import concordance_index
    rows = []
    for col, kind in ([(c, "honest_published") for c in HONEST_SOTA]
                      + [(c, "mmSYGNAL_internal") for c in LEAKY_SCORES]):
        if col not in data.df.columns:
            continue
        s = pd.to_numeric(data.df[col], errors="coerce")
        ok = s.notna()
        if ok.sum() < 20:
            continue
        c = concordance_index(data.df["duration"][ok], -s[ok], data.df["event"][ok])
        rows.append({"score": col, "kind": kind, "cindex": round(c, 4),
                     "is_leaky": bool(c >= 0.95)})
    return pd.DataFrame(rows)
