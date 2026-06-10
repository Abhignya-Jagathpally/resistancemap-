"""Client + parser for the OPEN tier of MMRF-COMMPASS clinical data via the GDC REST API.

The Genomic Data Commons (GDC) exposes *open-access* harmonized clinical records for
project MMRF-COMMPASS (phs000748) WITHOUT any dbGaP / Researcher Gateway credentials.
This module issues the real GDC query on a machine WITH internet; it is intentionally
NOT runnable inside the offline sandbox (the network call is guarded and surfaces a
clear, actionable error instead of hanging).

Two entry points
----------------
- ``fetch_open_clinical(out_csv)``: hit ``https://api.gdc.cancer.gov/`` and write the
  raw open clinical table (one row per case) to ``out_csv``. Run this on your machine.
- ``build_survival_frame(clinical_df)``: pure-pandas transform of that table into the
  ``(duration, event)`` survival frame the rest of ``resistancemap`` consumes. This is
  fully offline-testable and is exercised in ``tests/test_data.py``.

Bulk RNA-seq / supplementary open files (STAR counts etc.) are NOT fetched here; pull
them from the open S3 mirror with the AWS CLI (no AWS account / credentials needed):

    aws s3 cp --no-sign-request --recursive s3://gdc-mmrf-commpass-phs000748-2-open/ data/raw/

See <https://docs.gdc.cancer.gov/API/Users_Guide/Getting_Started/> for the API contract.
"""
from __future__ import annotations

import json
import pandas as pd

GDC_API = "https://api.gdc.cancer.gov/"

# Open-tier filter: harmonized Clinical category for MMRF-COMMPASS, access == "open".
OPEN_CLINICAL_FILTERS: dict = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["MMRF-COMMPASS"]}},
        {"op": "in", "content": {"field": "files.access", "value": ["open"]}},
        {"op": "in", "content": {"field": "files.data_category", "value": ["Clinical"]}},
    ],
}

# Case-level clinical fields we expand from the /cases endpoint. Kept small and explicit
# so the parser below has a stable contract regardless of GDC schema churn elsewhere.
CASE_FIELDS: list[str] = [
    "submitter_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.gender",
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.iss_stage",
]


def build_cases_params(size: int = 5000) -> dict:
    """Return the JSON-encoded query string params for the GDC ``/cases`` endpoint.

    Separated out so it can be unit-tested without any network access.
    """
    return {
        "filters": json.dumps(OPEN_CLINICAL_FILTERS),
        "fields": ",".join(CASE_FIELDS),
        "format": "JSON",
        "size": str(size),
    }


def _flatten_case(case: dict) -> dict:
    """Flatten one nested GDC case hit into a flat row.

    GDC nests ``demographic`` as an object and ``diagnoses`` as a list; we take the
    first diagnosis (CoMMpass cases carry a single primary MM diagnosis).
    """
    demo = case.get("demographic", {}) or {}
    diags = case.get("diagnoses", []) or [{}]
    diag = diags[0] if diags else {}
    age_days = diag.get("age_at_diagnosis")
    return {
        "case_submitter_id": case.get("submitter_id"),
        "vital_status": demo.get("vital_status"),
        "days_to_death": demo.get("days_to_death"),
        "gender": demo.get("gender"),
        # GDC stores age at diagnosis in DAYS; convert to years for a human covariate.
        "age": (age_days / 365.25) if age_days is not None else None,
        "days_to_last_follow_up": diag.get("days_to_last_follow_up"),
        "iss_stage": diag.get("iss_stage"),
    }


def parse_cases_response(payload: dict) -> pd.DataFrame:
    """Parse a GDC ``/cases`` JSON response body into a flat clinical DataFrame.

    Pure function (no network); the offline test feeds it a hand-built payload.
    """
    hits = payload.get("data", {}).get("hits", [])
    rows = [_flatten_case(h) for h in hits]
    return pd.DataFrame(rows)


def fetch_open_clinical(out_csv: str, size: int = 5000, timeout: int = 60) -> pd.DataFrame:
    """Query the live GDC API for MMRF-COMMPASS open clinical records and write a CSV.

    Runs on a machine WITH internet. Inside the offline sandbox the import of/call to
    ``requests`` is allowed but the HTTP request will fail with no route to the GDC host;
    we catch that and raise a clear, actionable ``RuntimeError`` rather than hanging.

    Returns the parsed DataFrame (also written to ``out_csv``).
    """
    try:
        import requests  # local import: keeps module importable where requests is absent
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "fetch_open_clinical needs the 'requests' package. Install it (pip install "
            "requests) and run this on a machine with internet access to api.gdc.cancer.gov."
        ) from exc

    params = build_cases_params(size=size)
    try:
        resp = requests.get(GDC_API + "cases", params=params, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # broad: network errors, DNS failure, HTTP errors, bad JSON
        raise RuntimeError(
            "Could not reach the GDC API at "
            f"{GDC_API} (this is EXPECTED in the offline sandbox). Run fetch_open_clinical "
            "on a machine with internet access. Underlying error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    df = parse_cases_response(payload)
    df.to_csv(out_csv, index=False)
    return df


def build_survival_frame(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Map GDC open clinical fields to a survival frame.

    Input columns (any extras are ignored): ``vital_status``, ``days_to_death``,
    ``days_to_last_follow_up`` and covariates ``age``, ``gender``, ``iss_stage``.

    Output: one row per usable patient with
      - ``duration`` (float): days_to_death if dead else days_to_last_follow_up
      - ``event`` (int): 1 if vital_status == "dead" (case-insensitive) else 0
      - retained covariates ``age``, ``gender``, ``iss_stage`` (when present)
      - ``case_submitter_id`` (when present) as a stable patient key

    Rows with NaN or non-positive duration are dropped (uninformative for time-to-event).
    """
    c = clinical_df.copy()

    vital = c["vital_status"].astype("string").str.lower()
    dead = vital.eq("dead")
    c["event"] = dead.astype(int)

    death = pd.to_numeric(c.get("days_to_death"), errors="coerce")
    follow = pd.to_numeric(c.get("days_to_last_follow_up"), errors="coerce")
    # Dead -> use days_to_death; alive/unknown -> use days_to_last_follow_up.
    c["duration"] = death.where(dead, follow).astype(float)

    keep = ["duration", "event"]
    for col in ("case_submitter_id", "age", "gender", "iss_stage"):
        if col in c.columns:
            keep.append(col)

    out = c[keep]
    out = out[out["duration"].notna() & (out["duration"] > 0)].reset_index(drop=True)
    return out
