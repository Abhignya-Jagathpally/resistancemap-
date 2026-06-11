# Real-vs-pending ledger (single source of truth for claims)

## SOTA benchmark (PFS, mmSYGNAL substrate) — 2026-06-10
**PRIMARY endpoint.** Source: `results/sota_comparison.json` (mirror `results/sota_table.md`);
narrative in `docs/RESULTS_LANE1.md`. MMRF-CoMMpass IA12, mmSYGNAL program-activity substrate,
IMWG **PFS**, **N=769 / 323 events**, 5-fold patient-disjoint CV.

**Leakage audit** (`leakage_audit`): honest published bar = gep70 0.6242 / sky92 0.620;
**GuanScore in-sample C-index = 1.000 → flagged `is_leaky=true`, EXCLUDED** (re-encodes the label,
not a predictor); risk_auc 0.8462 also excluded as mmSYGNAL-internal.

| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| gep70 (SOTA, honest bar) | 0.624 [0.589, 0.659] | 0.1769 | 0.6452 | 0.0149 | 0.0656 |
| sky92 (SOTA, honest bar) | 0.620 [0.586, 0.655] | 0.1806 | 0.6472 | 0.0180 | 0.0537 |
| Cox (prog+clin) | 0.632 [0.599, 0.667] | 0.1818 | 0.6593 | 0.0476 | 0.0489 |
| RSF (prog+clin) | 0.631 [0.597, 0.665] | 0.1792 | 0.6611 | 0.0211 | 0.0586 |
| GBS (prog+clin) | 0.643 [0.609, 0.675] | 0.1772 | 0.6772 | 0.0238 | 0.0591 |
| **Stacked (+gep70+sky92)** | **0.644 [0.610, 0.678]** | **0.1765** | **0.6806** | 0.0473 | 0.0568 |
| ResistanceBasin-LR (novel) | 0.482 [0.448, 0.516] | 0.1931 | 0.4976 | 0.0113 | 0.0612 |

- **Honest verdict — PARITY, not a win.** Best honest model Stacked 0.644 [0.610, 0.678] vs gep70
  0.624 [0.589, 0.659] / sky92 0.620 [0.586, 0.655]: **CIs overlap → comparable discrimination, NOT
  CI-separated.** No SOTA C-index claimed.
- **PH holds on baseline** (`non_ph_test`): Grambsch–Therneau **0/20** covariate violations →
  the ~0.62–0.64 ceiling is a **modality** property (every method shares the band), not a method
  limit. The novel PH-free model has no edge and **collapses to chance (C=0.482)** — honest negative.
- **Novel vs gep70 paired bootstrap** (B=1500, `paired_tests`): ΔC-index −0.142, 95% CI
  [−0.193, −0.093], p<0.001 (worse); ΔIBS +0.016, 95% CI [+0.008, +0.023], p<0.001 (worse). No novel win.
- **Where the non-PH signal lives:** treatment trajectory — n_lines violates PH at **χ²=14.7,
  p=1e-4** (`docs/LANE2_TREATMENT_NONPH.md`); motivates treatment-conditioned PH-free modeling
  (counting-process / landmark, immortal-time-safe). Immune fusion is the access-gated next axis
  (`docs/PATH_A_IMMUNE_FUSION.md`; Zenodo restricted + VLAB controlled — scoped, not claimed).
- **Governance: `novel_model_beats_baselines` BLOCKED** — paired tests show the novel model is worse;
  parity is the honest, submittable outcome. (OS secondary run below is also BLOCKED, best_baseline
  0.7369.)

## Lane #2 — treatment-conditioned non-PH (landmark, immortal-time-safe) — 2026-06-10
Source: `results/lane2/landmark_results.json` (mirror `landmark_results.md`, fig `fig_lane2_landmark.png`);
plan in `docs/LANE2_TREATMENT_NONPH.md`. Cohort = mmSYGNAL IA12 ∩ open `treatments.tsv`, **n=712 / 311 events**.
Forward td-AUC at each landmark (mean±std over patient-disjoint folds); time re-origined at L (immortal-time-safe):

| Landmark | static_cox | timevarying_cox (+treatment Z(L)) | treatment_basin |
|---|---|---|---|
| 180 d | 0.644±0.063 | **0.656±0.065** | 0.500 (collapsed) |
| 365 d | 0.572±0.068 | **0.582±0.068** | 0.500 (collapsed) |
| 730 d | 0.518±0.033 | 0.495±0.016 | 0.500 (collapsed) |

- **Gate `treatment_nonph_beats_static`: PASS (narrow, honest).** v1 (single-CV) found a CI-separated
  win at L=365 (Δ=+0.0097); the rigorous **nested-CV refinement (v2) supersedes this** — see below.
- **Mechanism confirmed:** Grambsch–Therneau `n_lines` violates PH (χ²=19.1, p=1.2e-5); clinical baseline
  0/10 (PH holds). The non-PH structure is treatment-driven, on open data.
- **PH-free `treatment_basin` collapses** to a single basin (td-AUC=0.500) at every landmark — same failure
  mode as ResistanceBasin-LR; the partial-log-rank objective finds no stable multi-basin structure on this
  wide low-signal substrate. Reported plainly, not hidden. 8/8 Lane #2 unit tests pass.

### v2 — nested-CV refinement (RIGOROUS; supersedes v1) — `results/lane2/landmark_results_v2.json`
Outer 5-fold patient-disjoint CV; inner CV (on training folds only) selects every penalizer/config; outer
fold touched once. Enriched time-varying covariates (n_switches, escalation, n_distinct_classes, regimen
class at L) vs static Cox:

| Landmark | static | enriched_tv | Δ (enriched−static) | 95% CI | CI>0 |
|---|---|---|---|---|---|
| 180 d | 0.635 | **0.649** | **+0.0145** | **[+0.0018, +0.0259]** | **YES (98.2% boot)** |
| 365 d | 0.567 | 0.574 | +0.0076 | [−0.0084, +0.0212] | no |
| 730 d | 0.494 | 0.472 | −0.0216 | [−0.0470, +0.0003] | no |

- **Honest verdict (nested CV):** the time-varying treatment model gives a **small CI-separated forward
  td-AUC gain at L=180 (+0.0145)**, but the effect is **landmark-sensitive** — NOT robust at L=365 (the v1
  L=365 win does not survive nesting) and reverses by L=730. A real but **fragile, modest** treatment-driven
  non-PH signal; do not overclaim a stable win. Immortal-time-safe; no fabricated values.
- **Basin fix (Q2): partially mitigated, NOT fixed.** With program compression (top-k / PCA-16) + KMeans
  warm-start, the PH-free basin no longer pins at 0.500 (occupies ≥2 basins on 3–4/5 folds, max td-AUC ~0.68)
  but is fold-unstable and **still loses to static Cox** at the informative landmarks (0.563 vs 0.635 @180;
  0.530 vs 0.567 @365). The PH-free basin remains unsuccessful on this substrate — honest negative.

## Interpretability — what drives the model's PFS risk — 2026-06-10
Source: `results/interpretability/interpretability.json` + 5 figures; narrative `docs/RESULTS_INTERPRETABILITY.md`.
GBS(prog+clin), N=769/323 ev, OOF C-index 0.643. Real permutation importance / KM / Fisher; no fabrication.
- **Top risk driver = ISS** (C-index drop 0.0334, ~2× any program); discrimination otherwise spreads across a
  long tail of weakly-predictive programs — **no single "resistance program."**
- **Risk tertiles** (median PFS **1436 / 914 / 715 d**, log-rank **p=4.8e-12**): high tertile enriched (Fisher)
  for **WHSC1 19.9% vs 10.5% (p=5e-4)**, **FGFR3 15.6% vs 7.2% (p=5e-4)** [both t(4;14)], **CCND1 (p=3e-3)**
  [t(11;14)], **amp1q (p=3e-3)**; ISS mean 1.51→2.11→2.33; **first-line drug class flat across tertiles** →
  separation is disease biology, not a regimen confound. 2-yr ECE = 0.034 (well-calibrated).

## REAL GDC-OPEN RUN (2026-06-10) — MMRF-CoMMpass, overall survival
**SECONDARY endpoint** (separate, lower-event-count OS run; NOT the PFS primary above).
Source: `ResistanceMap/data/raw/mmrf_commpass/{clinical,gene_expression}.tsv` (GDC open tier,
project phs000748). Builder `scripts/build_commpass_from_gdc_tsv.py` → `data/processed/commpass.csv`
(781 patients with RNA+clinical, **161 deaths**, 8 features = age_z/sex/ISS + 5 RNA program scores;
75/77 program genes matched). **Endpoint = overall survival** (open tier exposes OS, NOT IMWG-PFS;
PFS needs the gated Researcher Gateway — kept off). 5-fold patient-disjoint CV, train-fold-only
median imputation. Reproduce: `python scripts/paper_metrics.py` ; figure: `python scripts/paper_figures.py`.

| Model | CV C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| cox_ph | 0.737 [0.700, 0.771] | 0.129 | 0.749 | 0.014 | 0.052 |
| cox_elasticnet | 0.727 [0.691, 0.762] | 0.130 | 0.740 | 0.013 | 0.041 |
| random_survival_forest | 0.721 [0.682, 0.757] | 0.129 | 0.746 | 0.005 | 0.035 |
| gradient_boosted_survival | 0.696 [0.657, 0.733] | 0.152 | 0.705 | 0.047 | 0.068 |
| **program_basin (NOVEL)** | **0.698 [0.659, 0.740]** | 0.144 | 0.703 | 0.026 | **0.030** |

- Risk stratification (Cox OOF tertiles): log-rank **p = 1.1e-22** (`results/fig_paper_main.png`).
- Governance on REAL data: **`novel_model_beats_baselines` BLOCKED** — program_basin reaches parity
  inside the baseline band via an interpretable Kramers basin-escape mechanism and has the BEST
  2-yr calibration (ECE 0.030), but does NOT beat Cox. The gate honestly refuses the unearned claim.
- Mechanistic gates (synthetic, `tests/test_modules.py`): PK-inverse R²=1.000 GRANTED;
  Kramers↔MC consistent + risk→hazard monotone GRANTED.
- Artifacts: `results/{paper_metrics_real.json,paper_table1.md,cv_results.json,cv_governance.json,fig_paper_main.png/.pdf}`.


## RAN in this build (reproducible: `python tests/test_modules.py`, `python scripts/run_baselines.py`)
- Baseline harness on SYNTHETIC smoke data (labelled; no biological claim):
  cox_ph 0.676 [0.636,0.713] · cox_elasticnet 0.672 · RSF 0.663 · GBS 0.644
  -> note: plain Cox >= ensembles, the expected "simple-wins-on-modest-data" pattern.
- PK-inverse recovery (IgG, 21 d): R^2 = 1.000.
- Kramers vs Monte-Carlo first-passage: same direction, order-of-magnitude agreement.
- Risk -> basin-escape hazard: monotonically increasing (0.028, 0.071, 0.097).
- Governance: pk_inverse_recovers_signal GRANTED · hazard_mechanistic GRANTED ·
  novel_model_beats_baselines BLOCKED (honest: no novel model trained yet).

## PENDING (run on your GPU box — sandbox is 2-CPU/2.8 GB, no S3/HF network)
- Download GDC-open CoMMpass (see src/.../data/gdc_open.py) and run
  `python scripts/run_baselines.py --data <commpass.parquet>` for the REAL baseline table.
- Train the program-score + basin model; only then may the "beats baselines" gate flip.
- Conformal calibration + ECE.
- PK/lab trajectory layer: requires MMRF Researcher Gateway access (no redistribution; 30-day notice).

## DO NOT CLAIM until the gate passes
- Any SOTA / "beats baselines" statement.
- Any longitudinal-lab or PK-on-real-data result.
- Any biological conclusion from synthetic data.

---
## VERIFIED BUILD — parallel-agent subtrees + integration gate (2026-06-10)
Compile-check: all .py pass ast.parse. All test suites green. No fabricated numbers.

5-fold patient-disjoint CV on SYNTHETIC (labelled; no biological meaning):
- cox_ph              C-index 0.678 | IBS 0.115 | td-AUC 0.761 | ECE 0.084
- discrete_time_hazard C-index 0.677 | IBS 0.119 | td-AUC 0.761 | ECE 0.072
NOVEL program_basin (program scores -> basin-escape Kramers hazard): C-index 0.645
  -> lands inside the baseline band (0.64-0.68), reached through an interpretable
     landscape-tilt mechanism. The "beats baselines" claim remains BLOCKED. Honest.

Added & verified: data/{gdc_clinical,gene_sets,scrna_qc}; survival/{splits,competing_risk,
calibration} + metrics(IBS, td-AUC); models/{program_basin,scvi_latent} + viz/phate (PHATE 2.0
installed, results/phate_demo.png); comparators/MIOFLOW.md; literature/references.csv (18 papers,
real URLs) + PRISMA.md (165 identified -> 18 included).

PENDING (your GPU box): real GDC-open run, scVI/MIOFlow (need torch), Gateway lab/PK layer.

## Theory loop — CONVERGED (2026-06-10)
Six theory families (LTI/SSM/diffusion/CDE/log-rank/graph) tie at the ~0.62-0.66 PFS ceiling; none beats Cox(ISS+gep70). Residual signal = power-limited ~+0.01 td-AUC treatment effect (Lane #2 CI-separated only @180d). Synthesis: docs/THEORY_LOOP_SYNTHESIS.md; per-iter: docs/THEORY_LOOP_LOG.md + results/theory_loop/. Gate beats_SOTA BLOCKED throughout (honest).
