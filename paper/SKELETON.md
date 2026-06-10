# Paper skeleton (IMRaD) — fill the [RUN] slots from real GDC-open results

**Title (working):** Inverting the Measurement: Interpretable, Censoring-Honest Forecasting of
Multiple-Myeloma Progression on Open Data.

**Abstract.** Problem; gap (biomarkers treated as features, not inverted); contribution
(PK-inverse observation + basin-escape hazard + claim-gated open-data benchmark); honest
result framing (parity with strong baselines + interpretability/calibration gains).
**[RUN 2026-06-10]** On MMRF-CoMMpass (GDC open tier; N=781, 161 deaths; overall survival;
5-fold patient-disjoint CV) the interpretable basin-escape hazard reaches CV C-index
0.698 [0.659, 0.740] — parity inside the strong-baseline band (Cox 0.737, EN-Cox 0.727,
RSF 0.721, GBS 0.696) — with the best 2-yr calibration (ECE 0.030). Cox-risk tertiles
separate survival at log-rank p=1.1e-22. The pre-registered governance gate
`novel_model_beats_baselines` returns BLOCKED, by design.

**1. Introduction.** Relapse is the clinical problem; current models forecast but don't invert
measurement or model relapse as a first-passage event. RQ1–RQ3 (docs/CONTRIBUTION.md).

**2. Related work.** PANGEA / Ferle / mmSYGNAL / SCOPE / cell-line DL — and the shared blind spot.
(From the SPAR-4-SLR review, `literature/`.)

**3. Methods.**
3.1 Latent state-space + PK-inverse observation operator (one-compartment PK; literature half-lives).
3.2 Basin-escape hazard (double-well quasi-potential; Kramers rate; covariate -> barrier).
3.3 Censoring-honest evaluation (patient-disjoint CV, C-index/IBS, bootstrap CIs, conformal calibration).
3.4 Claim-gate governance (pre-registered falsification).

**4. Data.** GDC-open CoMMpass (MMRF-COMMPASS, phs000748, open tier). **[RUN]** 781 patients
carrying both harmonized STAR RNA-seq and clinical OS; 161 observed deaths (20.6%). Features:
age (z), sex, ISS stage + 5 RNA program scores (proliferation, proteasome/UPR, stemness, immune
microenvironment, MM-high-risk-like; 75/77 marker genes matched, ssGSEA-style z-mean). Endpoint =
overall survival (the open tier does not expose IMWG-PFS; the gated Researcher Gateway lab/PK layer
is OFF). Builder: `scripts/build_commpass_from_gdc_tsv.py`. Synthetic used for portability only.

**5. Results.** **[RUN — `results/paper_table1.md`, `results/paper_metrics_real.json`]**
Table 1 (5-fold patient-disjoint CV, train-fold median imputation):

| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| cox_ph | 0.737 [0.700, 0.771] | 0.129 | 0.749 | 0.014 | 0.052 |
| cox_elasticnet | 0.727 [0.691, 0.762] | 0.130 | 0.740 | 0.013 | 0.041 |
| random_survival_forest | 0.721 [0.682, 0.757] | 0.129 | 0.746 | 0.005 | 0.035 |
| gradient_boosted_survival | 0.696 [0.657, 0.733] | 0.152 | 0.705 | 0.047 | 0.068 |
| **program_basin (novel)** | **0.698 [0.659, 0.740]** | 0.144 | 0.703 | 0.026 | **0.030** |

Fig 1 (`results/fig_paper_main.png`): (A) KM by Cox risk tertile, log-rank p=1.1e-22;
(B) C-index forest; (C) 2-yr calibration (program_basin best-calibrated). Mechanistic checks:
PK-inverse recovery R²=1.000; Kramers↔Monte-Carlo first-passage consistent; risk→hazard monotone.
Governance: `pk_inverse_recovers_signal` GRANTED, `hazard_is_mechanistic_and_monotone` GRANTED,
`novel_model_beats_baselines` **BLOCKED** (honest parity).

**6. Discussion.** Parity not SOTA; interpretability + honesty as the contribution; generality
beyond MM (any indirect-biomarker, censored forecasting problem).

**7. Limitations (pre-registered).** No open longitudinal labs today; low signal ceiling;
synthetic carries no biological weight; cell-line != patient.

**8. Reproducibility.** Repo, configs, seeds, claim-gate report.
