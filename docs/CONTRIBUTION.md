# Research design & contribution

## 1. Problem
Multiple myeloma is treatable but relapses. Predicting *when* a patient's disease will escape
therapy, and *through which mechanism*, **before it manifests clinically**, would let clinicians
pre-empt. The data that exist openly are imperfect for this; the honest task is to extract the
most defensible, interpretable signal from them — not to overclaim.

## 2. Research questions
- **RQ1 (forecasting).** From an open-access molecular + clinical snapshot, can we forecast
  time-to-progression with calibrated uncertainty, *at parity* with strong baselines, but through
  an **interpretable mechanism**?
- **RQ2 (measurement inversion).** If routine serum biomarkers are treated as half-life-blurred
  *shadows* of a latent production signal and **deconvolved** by a PK observation model, do they
  carry more usable signal than the raw values?
- **RQ3 (mechanistic hazard).** Does framing relapse as **basin escape** (Kramers) in a learned
  quasi-potential yield a hazard that is both competitive and decomposable into named resistance
  programs, versus a black-box head?

## 3. Current methods and where they fall short
| Method | What it does | Blind spot |
|---|---|---|
| PANGEA / landmark Cox | strong landmark Cox on lab deltas | labs are raw **features**; snapshot, not trajectory; no mechanism |
| Ferle (LSTM+CRBM) | trajectory-aware density model | black box; no PK; no mechanistic hazard |
| mmSYGNAL | curated transcriptional-program risk score | static score; no measurement physics |
| SCOPE / transformers | joint PFS/OS | opaque; labs are features |
| DeepCDR / cell-line DL | IC50 prediction | wrong endpoint; cell-line != patient |

**Shared blind spot:** every method treats the biomarker as a *feature*. None inverts the
measurement; none models relapse as a first-passage event with a calibrated, mechanistic hazard;
none ships a falsification harness.

## 4. The gap (one sentence)
> No open-data MM model (a) inverts serum-biomarker measurement physics to recover the latent
> disease signal, (b) models relapse as basin escape with a Kramers hazard, and (c) gates every
> claim by pre-registered falsification — at honest parity with strong baselines.

## 5. Contribution (parity-not-leaderboard; all four are runnable code)
- **C1 — PK-inverse observation operator** (`models/pk_observation.py`). One-compartment PK with
  literature half-lives; inversion `p_hat = dC/dt + ke*C`. Sandbox recovery R^2 = 1.000.
- **C2 — Basin-escape hazard** (`models/basin_sde.py`). Relapse = first-passage in a double-well
  quasi-potential; Kramers rate is the hazard; covariates set the barrier. Kramers <-> Monte-Carlo
  consistent in direction & order of magnitude; risk monotonically raises the hazard.
- **C3 — Censoring-honest open-data benchmark** (`survival/`). Cox, EN-Cox, RSF, GBS;
  patient-disjoint CV; bootstrap CIs. The numbers-to-beat.
- **C4 — Claim-gate governance** (`governance/claim_gate.py`). Pre-registered falsification;
  refuses unearned claims by construction (it currently, correctly, **BLOCKS** "beats baselines").

## 6. Mathematical grounding
- **State-space + observation.** Latent state z(t); serum biomarker C is the PK-filtered shadow of
  production p(t): `dC/dt = p - ke*C`, `ke = ln2 / t_half`. Inversion recovers p from serial C.
- **Dynamics + hazard.** `dz = -U'(z) dt + sqrt(2D) dW`, double-well U. Relapse = first passage;
  hazard ~ Kramers rate `r = sqrt(U''_min |U''_b|)/(2π) * exp(-ΔU/D)`. (Asymptotically exact for
  ΔU/D >> 1; we report direction + order-of-magnitude agreement with Monte-Carlo at finite D.)
- **Calibration.** Conformal prediction for distribution-free coverage; ECE for risk calibration.
- **Falsification.** Each claim <-> test <-> threshold; granted iff the test passes.

## 7. Data plan (open-access)
**Primary — GDC-open MMRF CoMMpass** (`gdc-mmrf-commpass-phs000748-2-open`, no dbGaP):
| Source | Columns used | Why |
|---|---|---|
| Open clinical | vital_status, days_to_death, days_to_last_follow_up -> (duration, event); ISS stage, age, sex | defines the survival target + covariates without gated access |
| RNA-seq STAR counts | tumour gene expression -> interpretable **program scores** (proliferation, mmSYGNAL-style gene sets) | the molecular signal that is actually downloadable |

**Transforms:** log1p TPM; z-score; transparent ssGSEA-style program scoring (`gdc_open.program_scores`);
patient-disjoint splits; leakage audit. **Integration:** late fusion of clinical + program scores
into the survival models and as the read-in to the basin model.

**Pending-access layer (architected, off by default):** longitudinal labs via Researcher Gateway
-> PK-inverse observation -> trajectory + mechanism decomposition. License forbids redistribution
and requires 30-day pre-publication notice (documented); switches on when access lands.

**Demonstration-only:** published CC-BY synthetic (electricsheepafrica) for portability checks —
the dataset self-labels as unsuitable for empirical analysis, so **no biological claim** is drawn.

## 8. Baselines for comparison
Cox PH (full panel), elastic-net Cox, RandomSurvivalForest, GradientBoostedSurvival, landmark Cox.
Literature anchors quoted as *context only* (not our numbers): PANGEA, Ferle, mmSYGNAL, SCOPE.

## 9. Significance & generality beyond MM
The **PK-inverse + basin-escape + claim-gate** template is disease-agnostic: any chronic disease
monitored through indirect, half-life-blurred biomarkers under censoring (CKD, heart failure,
diabetes, transplant rejection) inherits the same machinery. The contribution is *a method for
honest, interpretable forecasting from indirect biomarkers*, demonstrated on MM.

## 10. Limitations & pre-registered claim-gates
- No open longitudinal labs today -> PK layer validated on the forward model / synthetic only;
  real-lab validation is **BLOCKED** until Gateway access.
- Expect **parity, not SOTA** (low signal ceiling). "Beats baselines" stays **BLOCKED** unless the
  novel-model C-index CI-lower exceeds the best baseline.
- Cell-line != patient; IC50 != clinical resistance — not used as an endpoint.
- Synthetic results carry no biological weight by construction.

## 11. Enhancement roadmap
1. GDC-open run on the user's GPU box -> real baseline table + program-score model.
2. Conformal calibration layer + ECE reporting.
3. Wire PK layer to Gateway labs when access lands -> trajectory model + mechanism decomposition.
4. External generalization check (independent cohort / published synthetic portability).
