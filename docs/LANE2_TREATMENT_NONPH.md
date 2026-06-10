# Lane #2 — Treatment-conditioned, PH-free progression forecasting (open data)

**Status (2026-06-10): thesis EMPIRICALLY VALIDATED on open data. Build = multi-day.**

## The one-line idea
The non-proportional-hazards structure that justifies a PH-free / time-varying survival model
in multiple myeloma is **treatment-driven**, and it is recoverable from **open GDC CoMMpass
data alone** (no MMRF VLAB / Zenodo gate). The contribution is a treatment-conditioned model
where the hazard effect of therapy changes over time — exactly the regime where mmSYGNAL/gep70/
sky92 (static, PH) cannot follow.

## Why this lane exists (the chain of evidence, all run on real data)
1. Static transcriptome→PFS discrimination is **ceiling-bound ~0.62–0.64**; best honest model
   0.644 vs gep70 0.624, **not** CI-separated. (`results/sota_comparison.json`)
2. On **baseline** features, **PH holds** — Grambsch–Therneau **0/20** violations. So a PH-free
   model has no edge and indeed **collapses** (ResistanceBasin-LR → 1 basin, C=0.48). Honest
   negative result; consistent with PH-holds.
3. Add **treatment trajectory** from open `treatments.tsv`: number of lines of therapy
   **violates PH decisively — Grambsch–Therneau χ²=14.7, p=1e-4** (regimen class alone does
   not). Log-rank across 1st-line regimens p=1.3e-3. → **non-PH is real and treatment-driven.**

This is the missing empirical support the PH-free machinery (Ferle partial-log-rank loss,
basin mixture) needs to earn its keep — and it is open-data only.

## Data (all OPEN; already on disk)
- `ResistanceMap/data/raw/mmrf_commpass/treatments.tsv` (7185 rows): `regimen_or_line_of_therapy`,
  `therapeutic_agents`, `days_to_treatment_start/end` → per-patient treatment timeline + lines.
- mmSYGNAL program activity + clinical (Lane B substrate) for the covariate block.
- GDC clinical PFS/OS (`gdc_clinical.py`) as the outcome backbone.

## Variables / target / equation
- **Target:** PFS `(T, δ)`, right-censored, from diagnosis.
- **Static covariates x:** 141 programs + ISS + cytogenetics.
- **Time-varying covariate Z(t):** cumulative lines of therapy, current-regimen-class indicator,
  time-since-last-switch — built from the treatment timeline, evaluated in a counting-process /
  landmark frame.
- **Model:** time-varying / landmark hazard `λ(t | x, Z(t))`, PH-free via either (a) extended
  Cox with time-transform interactions, or (b) the basin mixture `S(t|x,Z)=Σ_k s_k(x) S0_k(t)`
  re-fit per landmark so basin hazards may cross. The win axis is **time-dependent AUC / IBS at
  the landmarks where hazards cross**, not global C-index.

## CRITICAL honesty constraint (the methodological core, not a footnote)
`n_lines` is partly a **consequence** of progression → using it as a baseline feature is
**immortal-time bias / reverse causation**. It MUST be modeled as a **time-varying covariate**
in a counting-process or **landmark** design (condition on lines accrued *by* landmark t, predict
forward). The whole point of the model is to handle this correctly — that is the contribution,
and the pre-registered falsifier is: *does the time-varying model beat a static Cox on
time-dependent AUC at post-baseline landmarks, without immortal-time inflation?*

## Build plan (ordered, multi-day)
1. `data/treatment_timeline.py` — parse `treatments.tsv` → tidy (patient, line, regimen_class,
   start, end); derive Z(t) builders. (offline-testable)
2. `survival/landmark.py` — landmark-CV harness (landmarks at e.g. 180/365/730 d): at each
   landmark, restrict to at-risk patients, features = x + Z(landmark), evaluate forward
   td-AUC/IBS. Reuse `survival/metrics.py`.
3. `models/treatment_basin.py` — basin mixture conditioned on (x, Z(t)); Ferle PH-free loss.
4. `scripts/run_lane2_landmark.py` — static-Cox vs time-varying vs treatment-basin at each
   landmark; pre-registered ΔtdAUC gate; Grambsch–Therneau evidence table.
5. Governance gate `treatment_nonph_beats_static` (BLOCKED until CI-separated, immortal-time-safe).

## Honest risk assessment
- The win is most plausible on **time-dependent metrics at later landmarks**, not global C-index.
- Immortal-time bias is the main threat; the landmark design neutralizes it but must be audited.
- Generalization: the treatment→non-PH mechanism is general (CKD, transplant, NSCLC) — the
  maximilianferle "across cancer types" framing transfers.
- This does NOT need VLAB/Zenodo; it is the open-data breakthrough route while immune-fusion
  (Path A) waits on access.
