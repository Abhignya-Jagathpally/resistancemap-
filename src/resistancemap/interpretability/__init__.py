"""Interpretability subtree for MM progression-risk forecasting (PFS, IA12).

Goal: make the GBS(prog+clin) survival model *tangible* — answer, in plain terms,
"what actually drives progression risk in this model, biologically and clinically?"

Every quantity here is computed from real on-disk data with real methods:
- C-index-drop permutation importance (``risk_drivers``),
- OOF risk-tertile basin characterization with Fisher enrichment + log-rank
  (``basin_characterization``),
- landmark reliability diagrams + per-program partial dependence
  (``calibration_viz``).

No fabricated numbers; no ``np.random`` for any *reported* quantity (the only RNG
use is feature-shuffling inside permutation importance, which is the method itself,
and is seeded for reproducibility).
"""
from __future__ import annotations

from .risk_drivers import (
    PermutationImportanceResult,
    permutation_cindex_importance,
    program_label,
)
from .basin_characterization import (
    TertileCharacterization,
    characterize_risk_tertiles,
    load_first_line_regimens,
)
from .calibration_viz import (
    ReliabilityPoint,
    landmark_reliability,
    program_partial_dependence,
)

__all__ = [
    "PermutationImportanceResult",
    "permutation_cindex_importance",
    "program_label",
    "TertileCharacterization",
    "characterize_risk_tertiles",
    "load_first_line_regimens",
    "ReliabilityPoint",
    "landmark_reliability",
    "program_partial_dependence",
]
