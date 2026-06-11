"""Treatment-timeline parser for the GDC CoMMpass ``treatments.tsv`` (Lane #2).

The non-proportional-hazards structure that justifies a PH-free survival model in
multiple myeloma is *treatment-driven* (Grambsch-Therneau chi2=14.7, p=1e-4 on number
of lines of therapy; see ``docs/LANE2_TREATMENT_NONPH.md``). This module turns the open
GDC ``treatments.tsv`` into a tidy per-(patient, line) timeline and exposes the
**time-varying** covariate builders Z(t) that Lane #2 needs.

CRITICAL honesty constraint (binding, see LANE2 doc and CLAUDE.md):
    ``n_lines`` is partly a CONSEQUENCE of progression. Using lines accrued over the
    whole follow-up as a *baseline* feature is immortal-time bias / reverse causation.
    Therefore every builder here is a function of an *evaluation time* ``t_days``:
    ``lines_accrued_by(patient, t)`` counts only therapy lines that had STARTED by ``t``,
    and ``time_since_last_switch(patient, t)`` looks only backwards from ``t``. These are
    meant to be evaluated at a fixed landmark, never with future information.

No fabrication: the parser raises ``FileNotFoundError`` if the file is absent; missing
``days_to_treatment_start`` rows are dropped (a line with no known start cannot enter a
time-varying covariate honestly) and recorded in the summary, never imputed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Default location of the open GDC CoMMpass treatment dump (relative to repo root /
# pipeline3 working dir). Override via the ``path`` argument.
DEFAULT_TREATMENTS_PATH = (
    "ResistanceMap/data/raw/mmrf_commpass/treatments.tsv"
)

# Drug-class lexicon (Lane #2 spec). Names are matched case-insensitively against the
# pipe-separated ``therapeutic_agents`` field.
DRUG_CLASSES: dict[str, set[str]] = {
    "PI": {"bortezomib", "carfilzomib", "ixazomib"},
    "IMiD": {"lenalidomide", "pomalidomide", "thalidomide"},
    "CD38": {"daratumumab", "isatuximab", "elotuzumab"},
    "Alkyl": {"melphalan", "cyclophosphamide"},
}
DRUG_CLASS_ORDER = ["PI", "IMiD", "CD38", "Alkyl"]

# Map the verbose ``regimen_or_line_of_therapy`` strings to an integer line number.
_LINE_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10,
}

PATIENT_KEY_RE = re.compile(r"(MMRF_\d+)")


def patient_join_key(sample_or_patient_id: str) -> str | None:
    """Extract the canonical ``MMRF_xxxx`` join key from any CoMMpass id.

    mmSYGNAL samples look like ``MMRF_2754_1_BM``; the treatment timeline keys on
    ``MMRF_2754``. Returns ``None`` if no key can be parsed (never fabricates one).
    """
    if sample_or_patient_id is None:
        return None
    m = PATIENT_KEY_RE.search(str(sample_or_patient_id))
    return m.group(1) if m else None


def _line_to_int(line_str: object) -> int | None:
    """Map a ``regimen_or_line_of_therapy`` value to an integer line number.

    Returns ``None`` for values that are not a numbered line of therapy (e.g.
    "Liposomal Doxorubicin Regimen"), which are excluded from line counting.
    """
    if not isinstance(line_str, str):
        return None
    low = line_str.strip().lower()
    for word, num in _LINE_WORDS.items():
        if low.startswith(word):
            return num
    return None


def _classes_for_agents(agents: object) -> frozenset[str]:
    """Return the set of drug classes present in a pipe-separated agent string."""
    if not isinstance(agents, str) or not agents.strip():
        return frozenset()
    tokens = [a.strip().lower() for a in agents.split("|") if a.strip()]
    out: set[str] = set()
    for cls, members in DRUG_CLASSES.items():
        if any(tok in members for tok in tokens):
            out.add(cls)
    return frozenset(out)


@dataclass
class TreatmentTimeline:
    """Tidy per-(patient, line) treatment timeline plus a parse summary.

    ``table`` columns:
        patient_id   : canonical MMRF_xxxx key
        line         : integer line of therapy (1..N)
        start_day    : earliest days_to_treatment_start across the line's agents
        end_day      : latest days_to_treatment_end (may be NaN if unknown)
        classes      : frozenset of drug classes used in that line
        agents       : sorted tuple of raw agent names in that line
    """

    table: pd.DataFrame
    summary: dict = field(default_factory=dict)

    # ---- per-patient slices -------------------------------------------------
    def patient(self, patient_id: str) -> pd.DataFrame:
        key = patient_join_key(patient_id)
        sub = self.table[self.table["patient_id"] == key]
        return sub.sort_values("line").reset_index(drop=True)

    def patients(self) -> list[str]:
        return sorted(self.table["patient_id"].unique().tolist())

    # ---- regimen / line helpers --------------------------------------------
    def first_line_regimen(self, patient_id: str) -> str:
        """Coarse first-line regimen *category* for one patient.

        Returns one of the explicit single-class names (``"PI"``/``"IMiD"``/
        ``"CD38"``/``"Alkyl"``), ``"PI+IMiD"`` etc. for common doublets, ``"Other"``
        for any first line whose class set is non-empty but uncategorised, or
        ``"None"`` if the patient has no first-line record in the timeline. This is a
        fixed *baseline* descriptor (first line is established at/near diagnosis, so it
        is immortal-time safe), unlike ``lines_accrued_by`` which is time-varying.
        """
        sub = self.patient(patient_id)
        first = sub[sub["line"] == 1]
        if first.empty:
            return "None"
        classes: set[str] = set()
        for c in first["classes"]:
            classes |= set(c)
        if not classes:
            return "Other"
        # Prefer the canonical backbone categories; collapse to a stable label.
        ordered = [c for c in DRUG_CLASS_ORDER if c in classes]
        if len(ordered) == 1:
            return ordered[0]
        # Common backbone doublet (PI+IMiD = VRd-like); keep it as its own category.
        if set(ordered) == {"PI", "IMiD"}:
            return "PI+IMiD"
        return "+".join(ordered) if ordered else "Other"

    def lines_accrued_by(self, patient_id: str, t_days: float) -> int:
        """Number of distinct therapy lines that had STARTED on/before ``t_days``.

        This is the core time-varying covariate. It is monotone non-decreasing in
        ``t_days`` by construction (a line, once started, stays counted). Lines whose
        ``start_day`` is unknown were dropped at parse time and never counted.
        """
        sub = self.patient(patient_id)
        if sub.empty:
            return 0
        started = sub[sub["start_day"] <= float(t_days)]
        return int(started["line"].nunique())

    def time_since_last_switch(self, patient_id: str, t_days: float) -> float:
        """Days since the most recent therapy-line *start* on/before ``t_days``.

        A "switch" is the start of a new line of therapy. Returns ``t_days - (last
        start <= t_days)``. If no line has started by ``t_days`` returns ``t_days``
        (time since diagnosis day 0). Always >= 0; looks only backwards, so it carries
        no future information.
        """
        sub = self.patient(patient_id)
        if sub.empty:
            return float(t_days)
        started = sub[sub["start_day"] <= float(t_days)]
        if started.empty:
            return float(t_days)
        last_start = float(started["start_day"].max())
        return max(0.0, float(t_days) - last_start)

    def current_regimen_classes(self, patient_id: str, t_days: float) -> frozenset[str]:
        """Drug classes of the most recently STARTED line on/before ``t_days``.

        Returns the class set of the active line at the landmark; empty set if no line
        has started yet. Backward-looking only.
        """
        sub = self.patient(patient_id)
        if sub.empty:
            return frozenset()
        started = sub[sub["start_day"] <= float(t_days)]
        if started.empty:
            return frozenset()
        last = started.loc[started["start_day"].idxmax()]
        return frozenset(last["classes"])

    # ---- ENRICHED time-varying builders (Lane #2 v2) -----------------------
    # All of the following are strictly backward-looking functions of an evaluation
    # time ``t_days`` (the landmark), exactly like ``lines_accrued_by``: they count or
    # describe only therapy-line history that had STARTED on/before ``t_days``. No future
    # information enters, so they are immortal-time safe when evaluated at a landmark.

    def n_switches_by(self, patient_id: str, t_days: float) -> int:
        """Number of *therapy switches* (line transitions) that occurred on/before ``t``.

        A switch is the start of a new line of therapy beyond the first. With L distinct
        lines accrued by ``t``, there have been ``max(0, L - 1)`` switches. Monotone
        non-decreasing in ``t``. Looks only backwards (uses ``lines_accrued_by``).
        """
        return max(0, self.lines_accrued_by(patient_id, t_days) - 1)

    def escalated_by(self, patient_id: str, t_days: float) -> int:
        """1 if the cumulative drug-CLASS set *grew* across accrued lines by ``t``, else 0.

        "Escalation" here means a line that started on/before ``t`` introduced at least one
        drug class not seen in the union of all *earlier* started lines. This captures
        treatment intensification (e.g. adding CD38 mAb on relapse) without using future
        information: it only inspects lines whose ``start_day <= t``. Returns 0 if zero or
        one line has started (nothing to escalate from).
        """
        sub = self.patient(patient_id)
        if sub.empty:
            return 0
        started = sub[sub["start_day"] <= float(t_days)].sort_values(["start_day", "line"])
        if len(started) < 2:
            return 0
        seen: set[str] = set()
        escalated = False
        for _, row in started.iterrows():
            cur = set(row["classes"])
            if seen and (cur - seen):
                escalated = True
                break
            seen |= cur
        return int(escalated)

    def n_distinct_classes_by(self, patient_id: str, t_days: float) -> int:
        """Count of distinct drug classes used across all lines started on/before ``t``.

        Monotone non-decreasing in ``t``; a coarse measure of cumulative treatment
        breadth. Backward-looking only.
        """
        sub = self.patient(patient_id)
        if sub.empty:
            return 0
        started = sub[sub["start_day"] <= float(t_days)]
        if started.empty:
            return 0
        classes: set[str] = set()
        for c in started["classes"]:
            classes |= set(c)
        return int(len(classes))

    def enriched_covariates(self, patient_id: str, t_days: float) -> dict:
        """Bundle the enriched, immortal-time-safe Z(L) scalars for one patient at ``t``.

        Returns a dict with:
            n_lines_accrued        : distinct lines started by t (monotone)
            n_switches             : line transitions by t (= max(0, n_lines-1))
            escalated              : 1 if cumulative class set grew by t, else 0
            n_distinct_classes     : distinct drug classes used by t
            time_since_switch_days : days since the most recent line start <= t
        and one current-regimen-class membership flag ``cur_<CLASS>`` per canonical class.

        These are the raw building blocks the v2 feature builder turns into model columns
        (scaling, one-hots, and interaction terms are applied there). Every value is a
        function of history <= t only.
        """
        cur = self.current_regimen_classes(patient_id, t_days)
        out = {
            "n_lines_accrued": float(self.lines_accrued_by(patient_id, t_days)),
            "n_switches": float(self.n_switches_by(patient_id, t_days)),
            "escalated": float(self.escalated_by(patient_id, t_days)),
            "n_distinct_classes": float(self.n_distinct_classes_by(patient_id, t_days)),
            "time_since_switch_days": float(self.time_since_last_switch(patient_id, t_days)),
        }
        for cls in DRUG_CLASS_ORDER:
            out[f"cur_{cls}"] = 1.0 if cls in cur else 0.0
        return out


def build_timeline(table_or_df) -> TreatmentTimeline:
    """Construct a :class:`TreatmentTimeline` from an already-loaded raw frame.

    Pure function (no I/O) so it is unit-testable on a tiny synthetic frame. ``df`` must
    have columns ``submitter_id``, ``regimen_or_line_of_therapy``, ``therapeutic_agents``,
    ``days_to_treatment_start``, ``days_to_treatment_end``.
    """
    df = table_or_df.copy()
    n_raw = len(df)

    df["patient_id"] = df["submitter_id"].map(patient_join_key)
    df["line"] = df["regimen_or_line_of_therapy"].map(_line_to_int)
    df["start_day"] = pd.to_numeric(df["days_to_treatment_start"], errors="coerce")
    df["end_day"] = pd.to_numeric(df.get("days_to_treatment_end"), errors="coerce")
    df["classes"] = df["therapeutic_agents"].map(_classes_for_agents)

    n_no_line = int(df["line"].isna().sum())
    n_no_start = int(df["start_day"].isna().sum())
    n_no_key = int(df["patient_id"].isna().sum())

    # Honest exclusions: a row cannot enter a time-varying covariate without a patient
    # key, a numbered line, and a known start day. We do NOT impute starts.
    df = df.dropna(subset=["patient_id", "line", "start_day"]).copy()
    df["line"] = df["line"].astype(int)

    # Collapse to one row per (patient, line): earliest start, latest end, union classes.
    rows = []
    for (pid, line), grp in df.groupby(["patient_id", "line"], sort=True):
        classes: set[str] = set()
        for c in grp["classes"]:
            classes |= set(c)
        agents: set[str] = set()
        for a in grp["therapeutic_agents"].dropna():
            agents |= {tok.strip() for tok in str(a).split("|") if tok.strip()}
        end_vals = grp["end_day"].dropna()
        rows.append({
            "patient_id": pid,
            "line": int(line),
            "start_day": float(grp["start_day"].min()),
            "end_day": float(end_vals.max()) if len(end_vals) else np.nan,
            "classes": frozenset(classes),
            "agents": tuple(sorted(agents)),
        })

    table = pd.DataFrame(rows, columns=[
        "patient_id", "line", "start_day", "end_day", "classes", "agents",
    ])
    if not table.empty:
        table = table.sort_values(["patient_id", "line"]).reset_index(drop=True)

    summary = {
        "n_rows_raw": n_raw,
        "n_rows_no_patient_key": n_no_key,
        "n_rows_no_line_number": n_no_line,
        "n_rows_no_start_day": n_no_start,
        "n_patient_line_records": int(len(table)),
        "n_unique_patients": int(table["patient_id"].nunique()) if not table.empty else 0,
        "max_lines_observed": int(table["line"].max()) if not table.empty else 0,
    }
    return TreatmentTimeline(table=table, summary=summary)


def load_timeline(path: str = DEFAULT_TREATMENTS_PATH) -> TreatmentTimeline:
    """Load and parse the GDC CoMMpass ``treatments.tsv`` into a timeline.

    Raises ``FileNotFoundError`` (never fabricates) if the file is absent.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"treatments.tsv not found: {path}\n"
            "Lane #2 expects the open GDC CoMMpass treatment dump at "
            f"{DEFAULT_TREATMENTS_PATH} (run from the pipeline3 root)."
        )
    raw = pd.read_csv(path, sep="\t", dtype=str)
    return build_timeline(raw)
