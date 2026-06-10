# Real-vs-pending ledger (single source of truth for claims)

## REAL GDC-OPEN RUN (2026-06-10) — MMRF-CoMMpass, overall survival
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
