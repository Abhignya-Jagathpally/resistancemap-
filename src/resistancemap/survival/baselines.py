"""Numbers-to-beat: censoring-honest survival baselines with a uniform fit/risk API.
A model claiming novelty must beat these on identical patient-disjoint splits.

Feature handling: each baseline uses an explicit feature list. If none is given it
falls back to FEATURES (the synthetic contract), so existing code keeps working; on
real data pass `features=infer_features(df)` so the SAME harness runs on GDC-open
columns (age_z, sex_male, iss_stage + program scores) without edits."""
from __future__ import annotations
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# Default (synthetic) feature contract.
FEATURES = ["age_z", "iss_stage", "sex_male", "prog_score_1", "prog_score_2", "prog_score_3"]
_NON_FEATURE = {"duration", "event", "patient_id", "case_submitter_id", "sample_id"}


def infer_features(df: pd.DataFrame, drop: set[str] | None = None) -> list[str]:
    """Numeric columns of `df` that are model features (everything except id/target cols)."""
    drop = _NON_FEATURE if drop is None else set(drop)
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


class CoxBaseline:
    def __init__(self, penalizer: float = 0.0, l1_ratio: float = 0.0,
                 name: str = "cox_ph", features: list[str] | None = None):
        self.name = name
        self.features = list(features) if features else None
        self.cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)

    def fit(self, df: pd.DataFrame, features: list[str] | None = None):
        self.features = list(features or self.features or FEATURES)
        self.cph.fit(df[self.features + ["duration", "event"]], "duration", "event")
        return self

    def risk(self, df: pd.DataFrame) -> np.ndarray:           # higher = worse
        return self.cph.predict_partial_hazard(df[self.features]).values.ravel()


class SkSurvBaseline:
    def __init__(self, model, name: str, features: list[str] | None = None):
        self.model = model; self.name = name
        self.features = list(features) if features else None

    def fit(self, df: pd.DataFrame, features: list[str] | None = None):
        from sksurv.util import Surv
        self.features = list(features or self.features or FEATURES)
        y = Surv.from_arrays(event=df["event"].astype(bool).values, time=df["duration"].values)
        self.model.fit(df[self.features].values, y)
        return self

    def risk(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(df[self.features].values)).ravel()


def get_baselines(features: list[str] | None = None) -> dict:
    bl = {
        "cox_ph": CoxBaseline(name="cox_ph", features=features),
        "cox_elasticnet": CoxBaseline(penalizer=0.1, l1_ratio=0.5, name="cox_elasticnet", features=features),
    }
    try:
        from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
        bl["random_survival_forest"] = SkSurvBaseline(
            RandomSurvivalForest(n_estimators=200, min_samples_leaf=15, n_jobs=-1, random_state=0),
            "random_survival_forest", features=features)
        bl["gradient_boosted_survival"] = SkSurvBaseline(
            GradientBoostingSurvivalAnalysis(n_estimators=200, max_depth=3, random_state=0),
            "gradient_boosted_survival", features=features)
    except Exception:
        pass
    return bl
