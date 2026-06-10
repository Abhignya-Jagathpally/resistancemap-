"""TreatmentBasin — PH-free basin-mixture survival conditioned on (x, Z(L)).

Lane #2 model. It is the :class:`~resistancemap.models.resistance_basin_lr.ResistanceBasinLR`
basin mixture (soft assignment to K resistance basins, each with its OWN baseline survival
S0_k(t), trained with the vendored Ferle partial multivariate log-rank loss) — but fed the
**augmented** feature block ``[static x | Z(L)]`` produced by the landmark harness.

Why reuse rather than rewrite: ``ResistanceBasinLR`` already accepts an arbitrary feature
list through ``fit(df, features=...)`` and keeps the ``fit / risk / survival_function`` API
the landmark harness expects. The basin mixture is exactly the PH-free machinery whose edge
appears when hazards CROSS — which is the regime the treatment covariate ``Z(L)`` creates
(number-of-lines violates proportional hazards at Grambsch-Therneau p=1e-4). So all this
class adds over its parent is (a) a Lane-#2 name and (b) a thin guard that the caller passed
treatment columns, so a misuse that silently drops Z(L) is caught loudly rather than
fabricating a "treatment" model that saw no treatment.

API (identical to the survival baselines so it drops into the harness):
    fit(df, features)            -> self
    risk(df) -> (n,)             higher = worse prognosis
    survival_function(df, times) -> (n, len(times))
"""
from __future__ import annotations

import pandas as pd

from .resistance_basin_lr import ResistanceBasinLR

# Prefix used by the Lane #2 feature builder for time-varying covariates Z(L).
TREATMENT_FEATURE_PREFIX = "z_"


class TreatmentBasin(ResistanceBasinLR):
    """Basin-mixture survival model that consumes static + treatment Z(L) features.

    Parameters mirror :class:`ResistanceBasinLR`. ``require_treatment_features`` (default
    ``True``) makes ``fit`` raise if no column with the treatment prefix is present, so the
    model cannot be silently reduced to a static basin model.
    """

    def __init__(
        self,
        features=None,
        n_basins: int = 3,
        hidden: int = 64,
        dropout: float = 0.3,
        epochs: int = 400,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        penalty_weight: float = 0.1,
        seed: int = 0,
        name: str = "treatment_basin",
        require_treatment_features: bool = True,
        treatment_prefix: str = TREATMENT_FEATURE_PREFIX,
    ):
        super().__init__(
            features=features, n_basins=n_basins, hidden=hidden, dropout=dropout,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            penalty_weight=penalty_weight, seed=seed, name=name,
        )
        self.require_treatment_features = require_treatment_features
        self.treatment_prefix = treatment_prefix

    def fit(self, df: pd.DataFrame, features=None):
        feats = list(features or self.features or [])
        if self.require_treatment_features:
            has_tx = any(f.startswith(self.treatment_prefix) for f in feats)
            if not has_tx:
                raise ValueError(
                    "TreatmentBasin expects at least one treatment covariate "
                    f"(column prefixed {self.treatment_prefix!r}) in `features`; got "
                    f"{feats[:8]}{'...' if len(feats) > 8 else ''}. This guard prevents "
                    "silently training a static model and calling it a treatment model."
                )
        return super().fit(df, features=feats)
