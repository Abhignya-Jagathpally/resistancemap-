# PRISMA-style flow

**Date of search:** 2026-06-10
**Tool:** WebSearch (US-only web index)
**Reviewer:** automated seeding pass following the SPAR-4-SLR spirit (see `literature/PROTOCOL.md`)
**Scope:** AI/ML forecasting of multiple-myeloma progression / relapse / drug resistance (survival models);
single-cell trajectory & optimal-transport methods; survival-analysis foundations; reference cohorts and clinical criteria.

## Flow counts

| Stage | Count | Notes |
|---|---|---|
| Records identified | 165 | sum of result links returned across all 20 queries (see per-query table below) |
| Records after de-duplication (screened) | 96 | many hits repeat across queries (e.g. the SCOPE / "Joint AI-driven" paper appears in 3 queries; the WashU/MMRF immune atlas, MMRF CoMMpass, and the npj Ferle paper each appear in several); duplicates and obvious non-papers (vendor pages, patent PDFs, news/blog posts, retailer listings, generic tool docs) removed |
| Records eligible (full-text/abstract assessed) | 27 | unique scholarly works (peer-reviewed articles, preprints, a monograph, and the two reference data resources) assessed against scope |
| Records included | 18 | key papers/resources mapped to a model component in `references.csv` |

Excluded at screening/eligibility (examples): vendor/marketing pages (TIBCO, M-inSight), news & press (ConsultQD, GEN, OncLive, ISB/Springer community posts), USPTO patent PDFs, retailer listings (Amazon/B&N), duplicate mirrors of the same paper (PMC vs. publisher vs. bioRxiv vs. ResearchGate), and adjacent-but-out-of-scope ML/drug-response papers not among the named comparators (e.g. DeepCDR-style cell-line IC50 nets, DAGFormer, scATD, IAC-50, six-gene ASCO abstract).

## Per-query records identified

| # | Query | Records |
|---|---|---|
| 1 | machine learning multiple myeloma progression relapse drug resistance survival prediction | 9 |
| 2 | PANGEA landmark Cox model smoldering multiple myeloma progression prediction laboratory trajectories | 9 |
| 3 | Ferle LSTM CRBM multiple myeloma progression forecasting routine labs | 8 |
| 4 | mmSYGNAL transcriptional signature multiple myeloma risk stratification | 9 |
| 5 | SCOPE transformer multiple myeloma PFS OS prediction longitudinal event npj digital medicine | 6 |
| 6 | DrugFormer single-cell transformer drug resistance prediction cancer | 8 |
| 7 | CancerFoundation foundation model single-cell drug response cancer | 6 |
| 8 | MIOFlow manifold interpolating optimal transport single-cell trajectory Krishnaswamy | 10 |
| 9 | PHATE visualization dimensionality reduction high-dimensional biological data Moon Krishnaswamy Nature Biotechnology | 9 |
| 10 | Waddington-OT optimal transport reprogramming developmental trajectories Schiebinger Cell 2019 | 8 |
| 11 | Cox 1972 regression models life tables proportional hazards journal royal statistical society | 9 |
| 12 | Random Survival Forests Ishwaran Kogalur 2008 Annals of Applied Statistics | 9 |
| 13 | Vovk Gammerman Shafer algorithmic learning in a random world conformal prediction book | 8 |
| 14 | T-PHATE temporal PHATE manifold learning fMRI brain dynamics trajectory | 7 |
| 15 | MMRF CoMMpass study multiple myeloma genomics longitudinal molecular profiling | 10 |
| 16 | IMWG International Myeloma Working Group uniform response criteria Kumar 2016 Lancet Oncology | 9 |
| 17 | Washington University WashU bone marrow immune atlas multiple myeloma single-cell immune cells drug resistance Nature | 7 |
| 18 | single-cell atlas bone marrow myeloma progression immune microenvironment newly diagnosed precursor 2024 2025 | 8 |
| 19 | "International Myeloma Working Group consensus criteria for response and minimal residual disease" Lancet Oncology 2016 e328 | 9 |
| 20 | Cleveland Clinic machine learning model risk prediction multiple myeloma progressing first line therapy outperforms ISS | 9 |
| | **Total** | **165** |

## Included papers (registry)

The 18 included records are listed with key, year, one-sentence claim, method family, the model component
they inform, and a real URL in `literature/references.csv`.

Coverage check against the seeding targets:
- ML/AI MM forecasting (survival): PANGEA, PANGEA 2.0, Ferle, mmSYGNAL, SCOPE.
- Single-cell drug-resistance models: DrugFormer, CancerFoundation.
- Single-cell trajectory / optimal transport: MIOFlow, MIOFlow 2.0, PHATE, T-PHATE, Waddington-OT.
- Survival-analysis foundations: Cox 1972, Random Survival Forests, conformal prediction (Vovk et al.).
- Reference cohorts / clinical criteria: MMRF CoMMpass, WashU/MMRF bone-marrow immune atlas, IMWG (Kumar 2016) response criteria.
