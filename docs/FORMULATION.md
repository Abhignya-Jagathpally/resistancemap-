# Problem formalization (the precise framing your paper needs)

## 1. Task type — it is SURVIVAL, not plain classification
- **Primary:** right-censored **time-to-event (survival)** modelling. Output is a patient-specific
  hazard `lambda(t|x)` / survival curve `S(t|x)` over time — not one label.
- **Secondary (for AUROC/calibration reporting only):** **landmark binary classification** —
  "progression by horizon h in {6,12,18} months".
- **Why not plain classification:** progression times are **censored** (many patients are
  event-free at last follow-up). Forcing a fixed "relapsed/not" label throws away censoring and
  biases everything — this is *exactly* the v20 TT2L / event-gate bug. Survival is the honest frame.

## 2. Target / dependent variable
Observed per patient `i`: the pair **`(T_i, delta_i)`** where
`T_i = min(time-to-progression-or-death, censoring_time)` and `delta_i = 1` if the event was
observed else `0` (**endpoint = IMWG-PFS**, not time-to-second-line).
Modelled latents (our framing): resistance state `z(t)`, deconvolved biomarker production `p(t)`,
and the hazard `lambda(t|x)`.

## 3. Independent variables (features)
- **Clinical:** age, sex, ISS stage, cytogenetics (del17p, t(4;14), t(14;16), 1q gain, t(11;14)).
- **Molecular** (GDC-open RNA-seq -> interpretable **program scores**): proliferation,
  proteasome/UPR, **stemness** (e.g. PTPRG / E2F8 / FOXM1), immune-microenvironment, mmSYGNAL-style.
- **Treatment exposure (regimen class)** — CD38-triplet / PI / IMiD / CAR-T / bispecific.
  **Required**, because RRMM standard-of-care is shifting CAR-T & bispecifics *earlier* (CARTITUDE-4,
  MajesTEC-3/9, DREAMM-7/8), so the hazard is treatment-dependent and the label distribution drifts.
- **(pending Gateway) longitudinal serum:** M-protein, FLC kappa/lambda + ratio, B2M, albumin, Ig —
  entered through the PK-inverse operator, not as raw features.

## 4. Core equations
1. **Cox baseline:** `lambda(t|x) = lambda0(t) * exp(beta^T x)`; estimate beta by partial likelihood
   (Cox 1972); ranking quality = Harrell C-index.
2. **Mechanistic (ours):** relapse = first-passage of `z(t)` escaping the sensitive basin under
   `dz = -grad U_x(z) dt + sqrt(2D) dW`. Hazard approx the **Kramers rate**
   `lambda(x) ~ (sqrt(U''_min |U''_b|)/2pi) * exp(-dU(x)/D)`, so
   `log lambda(x) ~ const - dU(x)/D`. The term `-dU(x)/D` is a *physically interpretable*
   stand-in for the Cox linear predictor `beta^T x`.
3. **Observation (PK inverse):** `dC/dt = p - ke*C`, `ke = ln2/t_half`  =>  `p_hat = dC/dt + ke*C`.
4. **Likelihood:** Cox partial likelihood (baselines) or discrete-time hazard Bernoulli with a
   censoring mask (neural head). **Calibration:** split-conformal coverage + ECE.

## 5. Assumptions (chosen because they buy identifiability or interpretability)
- **A1 Proportional hazards** — Cox baseline only; test via Schoenfeld residuals; the basin model relaxes PH.
- **A2 Non-informative censoring** — censoring independent of event time given covariates.
- **A3 One-compartment first-order PK** with literature half-lives — makes the observation operator invertible in closed form.
- **A4 Manifold hypothesis** — `z` lies on a low-dimensional manifold (justifies scVI / PHATE latent).
- **A5 Metastability** — sensitive vs resistant are basins; transitions are rare events (justifies Kramers / Freidlin-Wentzell).
- **A6 Snapshot -> dynamics via OT** — population dynamics are recoverable from cross-sectional snapshots (Waddington-OT / MIOFlow regularity). *This is the formal license for "no longitudinal data".*
- **A7 Treatment-as-tilt** — regimen modulates the barrier `dU(x)`; enables treatment-conditioned / counterfactual hazard.

## 6. Theorems / lemmas that do real work
- **Kramers escape-rate (1940):** mean first-passage over a barrier in the small-noise limit -> the hazard form. Asymptotic in `dU/D >> 1` (we report direction + order-of-magnitude vs Monte-Carlo, not exact).
- **Freidlin-Wentzell large deviations:** rigorous quasi-potential / rare-transition foundation beyond Kramers.
- **Optimal-transport trajectory inference** (Benamou-Brenier; Schiebinger Waddington-OT; Huguet/Krishnaswamy MIOFlow): recover continuous dynamics from snapshots.
- **Cox partial likelihood (1972):** `beta` estimable without specifying `lambda0`.
- **Split-conformal coverage (Vovk):** distribution-free finite-sample interval coverage; Mondrian/stratified for per-subgroup.
- **Diffusion maps / PHATE / T-PHATE:** manifold geometry of the latent (and the temporal manifold).
- **Stone minimax rate (identifiability caveat):** bounds learnable complexity at small N -> we do **not** claim per-patient long-horizon forecasting.

## 7. Keywords
survival analysis; right-censoring; IMWG-PFS; competing risks; Harrell C / IBS / time-dependent AUC;
landmark analysis; Waddington quasi-potential; first-passage / Kramers / Freidlin-Wentzell; optimal
transport; manifold learning (PHATE / diffusion maps); latent Neural-SDE/ODE; conformal prediction;
ECE calibration; treatment-conditioned hazard; counterfactual inference.

## 8. The persistent data problems -> how this design answers each
| Challenge (you named these) | Design response |
|---|---|
| Biology is context-dependent | per-stratum reporting; external-cohort check; **no cross-disease transfer claim** (MM proteasome biology did not transfer to AML) |
| Longitudinal signal from snapshots | OT / MIOFlow dynamics + PK inversion; calendar-time mapping flagged as needing clinical anchors |
| Multimodal, missing modalities | late fusion + product-of-experts missing-modality handling; start with open modalities |
| Consent / data provenance | open-tier GDC only for submission; license documented; MMRF acknowledgment + 30-day notice for Gateway labs |
| Endpoint validity | IMWG-PFS (not time-to-2nd-line); pre-registered |
| Regulatory credibility | TRIPOD+AI reporting checklist; claim-gate refuses unearned claims |
| Clinical-pathway drift (CAR-T/bispecifics earlier) | **treatment-conditioned hazard**; explicitly acknowledge label-distribution drift over calendar time |
