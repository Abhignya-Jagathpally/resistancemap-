"""Offline unit tests for Lane #2 treatment-timeline parsing and the immortal-time-safe
landmark filtering. No real data, no fabricated metrics: everything runs on a tiny
synthetic treatment timeline so the *logic* (not any specific outcome) is verified.

Run:
    PYTHONPATH=src python -m pytest tests/test_lane2_treatment.py -q
or:
    PYTHONPATH=src python tests/test_lane2_treatment.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from resistancemap.data.treatment_timeline import (  # noqa: E402
    build_timeline,
    patient_join_key,
)
from resistancemap.survival.landmark import at_risk_at_landmark  # noqa: E402


def _toy_raw():
    """A tiny synthetic treatments.tsv-shaped frame.

    P0001: 1st line PI+IMiD (Bortezomib|Lenalidomide) start day 1; 2nd line CD38
           (Daratumumab) start day 400.
    P0002: 1st line Alkyl (Melphalan) start day 5; a row with NO start day (dropped);
           a non-numbered "Liposomal Doxorubicin Regimen" (no line number, dropped).
    """
    return pd.DataFrame([
        # P0001 first line (two agents -> one (patient,line) record)
        {"submitter_id": "MMRF_0001", "regimen_or_line_of_therapy": "First line of therapy",
         "therapeutic_agents": "Bortezomib", "days_to_treatment_start": "1",
         "days_to_treatment_end": "60"},
        {"submitter_id": "MMRF_0001", "regimen_or_line_of_therapy": "First line of therapy",
         "therapeutic_agents": "Lenalidomide", "days_to_treatment_start": "1",
         "days_to_treatment_end": "120"},
        # P0001 second line
        {"submitter_id": "MMRF_0001", "regimen_or_line_of_therapy": "Second line of therapy",
         "therapeutic_agents": "Daratumumab", "days_to_treatment_start": "400",
         "days_to_treatment_end": ""},
        # P0002 first line
        {"submitter_id": "MMRF_0002", "regimen_or_line_of_therapy": "First line of therapy",
         "therapeutic_agents": "Melphalan", "days_to_treatment_start": "5",
         "days_to_treatment_end": "70"},
        # P0002 a row with no start day -> must be dropped (no immortal-time imputation)
        {"submitter_id": "MMRF_0002", "regimen_or_line_of_therapy": "Second line of therapy",
         "therapeutic_agents": "Pomalidomide", "days_to_treatment_start": "",
         "days_to_treatment_end": ""},
        # P0002 non-numbered regimen -> dropped from line counting
        {"submitter_id": "MMRF_0002", "regimen_or_line_of_therapy": "Liposomal Doxorubicin Regimen",
         "therapeutic_agents": "Doxorubicin", "days_to_treatment_start": "200",
         "days_to_treatment_end": "210"},
    ])


def test_patient_join_key():
    assert patient_join_key("MMRF_2754_1_BM") == "MMRF_2754"
    assert patient_join_key("MMRF_0001") == "MMRF_0001"
    assert patient_join_key("garbage") is None
    assert patient_join_key(None) is None


def test_timeline_parses_classes_and_collapses_lines():
    tl = build_timeline(_toy_raw())
    p1 = tl.patient("MMRF_0001")
    # First line collapses two agents into one record with both classes.
    line1 = p1[p1["line"] == 1].iloc[0]
    assert line1["start_day"] == 1.0
    assert set(line1["classes"]) == {"PI", "IMiD"}
    # Second line is CD38.
    line2 = p1[p1["line"] == 2].iloc[0]
    assert line2["start_day"] == 400.0
    assert set(line2["classes"]) == {"CD38"}


def test_no_start_and_nonnumbered_rows_dropped():
    tl = build_timeline(_toy_raw())
    p2 = tl.patient("MMRF_0002")
    # Only the first line (Melphalan, start=5) survives; the no-start 2nd line and the
    # non-numbered Doxorubicin regimen are excluded.
    assert list(p2["line"]) == [1]
    assert set(p2.iloc[0]["classes"]) == {"Alkyl"}
    # Summary records the exclusions honestly.
    assert tl.summary["n_rows_no_start_day"] >= 1
    assert tl.summary["n_rows_no_line_number"] >= 1


def test_first_line_regimen_category():
    tl = build_timeline(_toy_raw())
    assert tl.first_line_regimen("MMRF_0001") == "PI+IMiD"
    assert tl.first_line_regimen("MMRF_0002") == "Alkyl"
    assert tl.first_line_regimen("MMRF_9999") == "None"  # unknown patient


def test_lines_accrued_is_monotone_and_time_varying():
    tl = build_timeline(_toy_raw())
    pid = "MMRF_0001"
    # Before any line starts.
    assert tl.lines_accrued_by(pid, 0) == 0
    # After first line (start day 1) but before second (start day 400).
    assert tl.lines_accrued_by(pid, 1) == 1
    assert tl.lines_accrued_by(pid, 399) == 1
    # After second line starts.
    assert tl.lines_accrued_by(pid, 400) == 2
    assert tl.lines_accrued_by(pid, 5000) == 2
    # Monotone non-decreasing across a sweep of times.
    prev = -1
    for t in range(0, 600, 25):
        cur = tl.lines_accrued_by(pid, t)
        assert cur >= prev, f"non-monotone at t={t}: {cur} < {prev}"
        prev = cur


def test_time_since_last_switch_backward_only():
    tl = build_timeline(_toy_raw())
    pid = "MMRF_0001"
    # No line started yet -> time since diagnosis day 0.
    assert tl.time_since_last_switch(pid, 0) == 0.0
    # 10 days after first line start (day 1).
    assert abs(tl.time_since_last_switch(pid, 11) - 10.0) < 1e-9
    # Just before second switch (day 400): measured from day 1.
    assert abs(tl.time_since_last_switch(pid, 399) - 398.0) < 1e-9
    # Right at second switch: resets to 0.
    assert abs(tl.time_since_last_switch(pid, 400) - 0.0) < 1e-9
    # Always >= 0.
    for t in range(0, 600, 17):
        assert tl.time_since_last_switch(pid, t) >= 0.0


def test_current_regimen_classes_at_landmark():
    tl = build_timeline(_toy_raw())
    pid = "MMRF_0001"
    assert tl.current_regimen_classes(pid, 0) == frozenset()
    assert tl.current_regimen_classes(pid, 100) == frozenset({"PI", "IMiD"})
    assert tl.current_regimen_classes(pid, 450) == frozenset({"CD38"})


def test_landmark_at_risk_filtering():
    # Three patients with different follow-ups; landmark filter keeps only duration > L.
    df = pd.DataFrame({
        "patient_id": ["MMRF_0001", "MMRF_0002", "MMRF_0003"],
        "duration": [100.0, 400.0, 800.0],
        "event": [1, 1, 0],
    })
    L = 365.0
    atrisk = at_risk_at_landmark(df, L)
    # Only patients with duration > 365 survive to the landmark.
    assert set(atrisk["patient_id"]) == {"MMRF_0002", "MMRF_0003"}
    # Re-origined forward time = duration - L.
    row2 = atrisk[atrisk["patient_id"] == "MMRF_0002"].iloc[0]
    assert abs(row2["lm_duration"] - (400.0 - L)) < 1e-9
    assert (atrisk["lm_landmark"] == L).all()
    # A patient who progressed before the landmark is excluded (no immortal-time credit).
    assert "MMRF_0001" not in set(atrisk["patient_id"])


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} Lane #2 tests passed.")


if __name__ == "__main__":
    _run_all()
