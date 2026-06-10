# Paper skeleton (IMRaD) — [RUN] slots filled from real results

**Title (working):** Inverting the Measurement: Interpretable, Censoring-Honest Forecasting of
Multiple-Myeloma Progression on Open Data.

> **Honesty contract for this skeleton.** Every numeric `[RUN]` slot is copied from an on-disk
> results file and cited inline. The PRIMARY endpoint is **progression-free survival (PFS)** on the
> mmSYGNAL substrate (`results/sota_comparison.json`). Overall survival (OS) on the GDC-open tier
> (`results/paper_metrics_real.json`) is a clearly-labeled SECONDARY endpoint. No SOTA C-index win
> is claimed anywhere — the headline is **comparable discrimination** (parity, overlapping CIs).

**Abstract.** Problem; gap (transcriptomic biomarkers treated as static features, never inverted to
a censoring-honest hazard); contribution (PK-inverse observation + basin-escape hazard + a
leakage-audited, claim-gated open-data benchmark); honest result framing.
**[RUN 2026-06-10 — `results/sota_comparison.json`]** On MMRF-CoMMpass IA12 (mmSYGNAL
program-activity substrate; PFS; N=769, 323 events; 5-fold patient-disjoint CV), the strongest
*honest* model in the panel reaches CV C-index **0.644 [0.610, 0.678]** (Stacked) — **comparable
discrimination** to the published transcriptomic SOTA bar **gep70 0.624 [0.589, 0.659]** and
**sky92 0.620 [0.586, 0.655]**, with **overlapping confidence intervals (NOT CI-separated)**. A
leakage audit removes the mmSYGNAL-internal **GuanScore (in-sample C-index = 1.000)** as a circular
score. On baseline features **proportional hazards holds** (Grambsch–Therneau: 0/20 covariates
violate at p<0.05), so the ~0.62–0.64 static-transcriptome ceiling is a property of the
**modality**, not of the **method**. The non-PH structure that would justify a PH-free model is
instead **treatment-driven** (Lane #2: number of lines of therapy violates PH at p=1e-4),
motivating treatment-conditioned modeling; **immune-microenvironment fusion** is the access-gated
extension. The pre-registered governance gate `novel_model_beats_baselines` returns **BLOCKED**, by
design.

**1. Introduction.** Relapse is the clinical problem; existing transcriptomic risk scores forecast
ordinally but do not invert the measurement or model relapse as a first-passage (escape) event. We
ask: (RQ1) can an interpretable, censoring-honest hazard reach parity with published transcriptomic
SOTA on open data; (RQ2) is the discrimination ceiling a method limit or a modality limit; (RQ3)
where does non-proportional-hazards structure actually live? (docs/CONTRIBUTION.md).

**2. Related work.** PANGEA / Ferle / mmSYGNAL (gep70, sky92, GuanScore) / SCOPE / cell-line DL —
and the shared blind spot: static, PH, leaderboard-chasing on small N. (From the SPAR-4-SLR review,
`literature/`.)

**3. Methods.**
3.1 Latent state-space + PK-inverse observation operator (one-compartment PK; literature half-lives).
3.2 Basin-escape hazard (double-well quasi-potential; Kramers rate; covariate → barrier tilt).
3.3 Censoring-honest evaluation (patient-disjoint CV; C-index / IBS / time-dependent AUC; bootstrap
    CIs; conformal/ECE calibration; Grambsch–Therneau PH test).
3.4 **Leakage audit** — every candidate score is checked for in-sample circularity before it enters
    the SOTA bar; a score that recovers the label perfectly (C-index = 1.000) is excluded.
3.5 Claim-gate governance (pre-registered falsification; refuse unearned claims by construction).

**4. Data.**
**4.1 Primary — PFS substrate.** MMRF-CoMMpass IA12, mmSYGNAL program-activity features.
**[RUN — `results/sota_comparison.json`]** N=769 patients, **323 PFS events**, 5-fold
patient-disjoint CV. Endpoint = IMWG progression-free survival (days). The honest SOTA bar is the
published transcriptomic scores **gep70** and **sky92**; the mmSYGNAL-internal **GuanScore** and
**risk_auc** are excluded by the leakage audit (§5.1).
**4.2 Secondary — OS substrate.** GDC-open CoMMpass (MMRF-COMMPASS, phs000748, open tier).
**[RUN — `results/paper_metrics_real.json`]** N=781 patients with harmonized STAR RNA-seq + clinical
OS, **161 observed deaths**; 8 features (age z, sex, ISS + 5 RNA program scores). Endpoint = overall
survival (the open tier does not expose IMWG-PFS; the gated Researcher Gateway lab/PK layer is OFF).
OS is reported only as a secondary, lower-event-count view; it is NOT the primary benchmark.

**5. Results.**

**5.1 Leakage audit (a contribution, not a footnote).**
**[RUN — `results/sota_comparison.json` → `leakage_audit`]** Of four candidate transcriptomic
scores, two are honest published scores (gep70 in-sample C-index 0.6242; sky92 0.620) and two are
mmSYGNAL-internal. **GuanScore reports in-sample C-index = 1.000 → it perfectly recovers the label,
is flagged `is_leaky=true`, and is EXCLUDED** from the SOTA bar; risk_auc (0.8462) is also internal
and excluded. The honest SOTA bar is therefore **gep70 0.624 / sky92 0.620**. Reporting a 1.000
C-index as a "result" is exactly the failure this audit prevents.

**5.2 Primary benchmark — Table 1 (PFS).**
**[RUN — `results/sota_comparison.json` → `rows`; mirror of `results/sota_table.md`]**
N=769, 323 events, 5-fold patient-disjoint CV. Lower IBS / higher td-AUC / lower ECE = better.

| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| gep70 (SOTA, honest bar) | 0.624 [0.589, 0.659] | 0.1769 | 0.6452 | 0.0149 | 0.0656 |
| sky92 (SOTA, honest bar) | 0.620 [0.586, 0.655] | 0.1806 | 0.6472 | 0.0180 | 0.0537 |
| Cox (prog+clin) | 0.632 [0.599, 0.667] | 0.1818 | 0.6593 | 0.0476 | 0.0489 |
| RSF (prog+clin) | 0.631 [0.597, 0.665] | 0.1792 | 0.6611 | 0.0211 | 0.0586 |
| GBS (prog+clin) | 0.643 [0.609, 0.675] | 0.1772 | 0.6772 | 0.0238 | 0.0591 |
| **Stacked (+gep70+sky92)** | **0.644 [0.610, 0.678]** | **0.1765** | **0.6806** | 0.0473 | 0.0568 |
| ResistanceBasin-LR (novel) | 0.482 [0.448, 0.516] | 0.1931 | 0.4976 | 0.0113 | 0.0612 |

**Honest reading.** The best honest model (Stacked, 0.644) is **comparable** to gep70 (0.624) and
sky92 (0.620): the confidence intervals **overlap — this is parity, NOT a CI-separated win**. The
paired bootstrap tests (B=1500) confirm the *novel* PH-free model does NOT win:
**[RUN — `results/sota_comparison.json` → `paired_tests`]** ResistanceBasin-LR vs gep70
ΔC-index = **−0.142, 95% CI [−0.193, −0.093], p<0.001** (novel worse on discrimination);
ΔIBS = **+0.016, 95% CI [+0.008, +0.023], p<0.001** (novel worse survival-function calibration).
The novel PH-free learner **collapses to a single basin (C ≈ 0.482, ~chance)** — an honest negative
result, and exactly what §5.3 predicts.

**5.3 Proportional hazards holds on baseline → the ceiling is a modality property.**
**[RUN — `results/sota_comparison.json` → `non_ph_test`]** Grambsch–Therneau test on the 20 baseline
covariates: **0/20 violate PH at p<0.05; global interpretation = "PH holds."** Because PH holds on
the baseline modality, a PH-free / time-varying model has **no structural edge** there — and indeed
ResistanceBasin-LR collapses (§5.2). The ~0.62–0.64 ceiling shared by gep70, sky92, Cox, RSF, GBS,
and Stacked is therefore a property of the **static tumor-transcriptome modality**, not of any one
method. No method in this panel escapes it, including the strong ensembles.

**5.4 Where the non-PH structure actually lives → the motivated extension (Lane #2).**
**[RUN — `docs/LANE2_TREATMENT_NONPH.md`]** Adding the **treatment trajectory** from open
`treatments.tsv`, the **number of lines of therapy violates PH decisively — Grambsch–Therneau
χ²=14.7, p=1e-4** (regimen class alone does not; log-rank across 1st-line regimens p=1.3e-3). So the
non-PH structure the static benchmark cannot exhibit is **treatment-driven**, recoverable from
**open data alone**. This is the empirical justification a treatment-conditioned, PH-free hazard
model needs to earn its keep. NOTE: n_lines is partly a *consequence* of progression and must be
modeled as a **time-varying covariate** in a counting-process / landmark design (never a baseline
feature) to avoid immortal-time bias — this is the methodological core of Lane #2, not a footnote.

**5.5 Access-gated extension — immune-microenvironment fusion (Path A).**
**[RUN — `docs/PATH_A_IMMUNE_FUSION.md`]** The only principled way to lift the static ceiling is to
add a modality on a **different biological axis**: the bone-marrow immune microenvironment (MMRF
Immune Atlas, Nature Cancer 2025). The single-cell matrices are **request-gated (Zenodo)** and the
clinical PFS join is **controlled (MMRF VLAB)**; the loader fails closed (`FileNotFoundError`) until
access is granted. Pre-registered as a CI-separated incremental-value claim on the immune subset
(~300–340 patients) — scoped, not claimed.

**5.6 Secondary endpoint — OS (GDC-open).**
**[RUN — `results/paper_metrics_real.json`; mirror of `results/paper_table1.md`]** On the GDC-open OS
substrate (N=781, **161 deaths**, 5-fold patient-disjoint CV), Cox 0.737 [0.700, 0.771], EN-Cox
0.727, RSF 0.721, GBS 0.696, and the novel **program_basin 0.698 [0.659, 0.740]** (best 2-yr
calibration, ECE 0.030). program_basin reaches parity inside the baseline band but does NOT beat Cox;
the governance gate `novel_model_beats_baselines` is **BLOCKED**
(`results/paper_metrics_real.json → governance`: rule "program_basin C-index CI-lower must exceed
best baseline 0.737"; best_baseline_cindex 0.7369). This OS view has fewer events (161) than the PFS
primary (323) and is supporting evidence only.

**6. Discussion.** Parity, not SOTA. The contribution is a **method + an honest, leakage-audited
open-data benchmark**: measurement inversion, a mechanistic interpretable hazard, censoring-honest
calibration, and pre-registered falsification. The static ceiling is a modality limit (PH holds);
the actionable non-PH signal is treatment-driven; immune fusion is the access-gated next axis.
Generality: any indirect-biomarker, censored forecasting problem (CKD, heart failure, transplant).

**7. Limitations (pre-registered).** No open longitudinal labs today (PK/lab layer gated); low
static-transcriptome signal ceiling (~0.62–0.64); the novel PH-free model collapses where PH holds
(honest negative); OS secondary endpoint has only 161 events; immune fusion is access-blocked;
synthetic data carries no biological weight; cell-line ≠ patient.

**8. Reproducibility.** Repo, configs, seeds, leakage audit, claim-gate report; primary numbers from
`results/sota_comparison.json` (+ `results/sota_table.md`), secondary from
`results/paper_metrics_real.json` (+ `results/paper_table1.md`).
