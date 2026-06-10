# Systematic Literature Review Protocol (SPAR-4-SLR)

Following Paul, Lim, O'Cass, Hao & Bresciani (Int. J. Consumer Studies, 2021) — the
SPAR-4-SLR protocol — so the review process is reproducible and auditable.

## Stage 1 — Assembling
**Identification.** Domain: AI/ML for multiple-myeloma progression & drug-resistance
forecasting from molecular + routine-lab + clinical data.
Research questions: see `docs/CONTRIBUTION.md` (RQ1 forecasting, RQ2 measurement inversion,
RQ3 mechanistic hazard).
Source databases: PubMed, Semantic Scholar/Scopus (via Consensus), bioRxiv/medRxiv, GDC docs.

**Acquisition.** Search strings: `literature/search_strings.md`. Window: 2018–2026.

## Stage 2 — Arranging
**Organization.** Code each paper by: method family (landmark Cox / RNN / curated-signature /
transformer / cell-line DL), data (CoMMpass / GEO / cell-line), endpoint (PFS/OS/IC50),
interpretability, calibration reported (Y/N), measurement model (Y/N).
**Purification.** Inclusion: MM or pan-heme progression/resistance forecasting with a
quantitative model + outcome. Exclusion: pure diagnosis-from-images, review-only, no outcome,
no quantitative evaluation.

## Stage 3 — Assessing
**Evaluation.** Per-paper extraction: cohort n, censoring handling, CV scheme, metric (C-index/
AUROC/IBS), external validation (Y/N), reported value. Contradiction & lineage mapping.
**Reporting.** Landscape map + contradiction table + gap statement (see project landscape docs).

## PRISMA-style flow (fill from runs in `literature/references.csv`)
identified: __  ·  screened: __  ·  eligible: __  ·  included: __

## Reference registry
Each included paper is linked to the model component it informs in `literature/references.csv`
(columns: key, year, claim, method_family, informs_component).
