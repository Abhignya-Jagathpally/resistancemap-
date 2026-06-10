# Model diagnosis: why `program_basin` underperforms, and the fix

## Symptom
`program_basin` CV C-index (~0.64) sits at or below the baselines (Cox ~0.68); it never beats them.

## Root cause — a theorem, not a tuning issue
**Harrell's concordance index is invariant under any strictly monotone transform of the risk
score** — it scores the *ranking*, not the values (Elgui et al. 2023; Gandy 2025). `program_basin`
computes `risk = log KramersRate(tilt(w·x_tilde))`, where `w` comes from a LogisticRegression on the
binary `event`. That is a monotone map of a single linear score, so it **cannot out-rank that
score**. Proven on synthetic n=900:

| score | C-index |
|---|---|
| CoxPH (proper survival fit) | 0.682 |
| internal linear score s = w·x_tilde | 0.676 |
| program_basin.risk = wrapper(s) | 0.645 |

`Spearman(program_basin, s) = 0.85` (NOT 1.0). Three findings:
1. **The linear score is already Cox-level (0.676).** The Kramers wrapper has no headroom to add
   ranking signal (monotone-invariance).
2. **The wrapper is worse than a no-op:** near the double-well bifurcation the Kramers prefactor
   `sqrt(U''_min * |U''_b|)` collapses (`U'' -> 0`), so the rate is **non-monotone** in tilt and
   *reorders* the highest-risk patients — costing ~3 C-index points. Clipping tilt to [0, 0.4] also
   ties ~4% of patients.
3. **It is fit on the binary event label** (logistic), discarding event times + censoring; on real,
   more-censored data this is strictly weaker than Cox's partial likelihood.

**Bottom line:** as a cross-sectional *ranking* model on proportional-hazards data, a basin/Kramers
score-wrapper can only match (at best) or hurt. **C-index is the wrong objective for this mechanism.**

## The fix — make the first-passage idea a proper survival model (literature-grounded)
"Relapse = escape from a sensitive basin" *is* **first-hitting-time / threshold regression** (Lee &
Whitmore, *Statistical Science*, 2006): a latent Wiener health process `W(t) = y0 - mu*t + sigma*B(t)`
hits 0; the first-hitting time is **Inverse-Gaussian**, `T ~ IG(mean = y0/mu, shape = y0^2/sigma^2)`.
Covariates modulate drift `mu` and boundary `y0`. This is a rigorous, published family (R packages
`threg`, `invGauss`) — the tractable, fittable cousin of Kramers escape.

Implemented in `models/fht_threshold.py`:
- `mu_i = exp(b0 + x.beta)` (disease aggressiveness), `y0_i = exp(g0 + x.gamma)` (resistance reserve), `sigma = 1`.
- **Proper censored MLE** (events -> IG density; censored -> IG survival) — never a binary surrogate.

## Where it earns its keep — non-proportional hazards
Threshold regression is explicitly used for **non-PH** survival (Bayesian random-effects TR,
*Biostatistics* 2010). MM relapse IS non-PH: the FIRST trial's PFS curves cross (~18 mo); early vs
late relapse have different biology. When covariates act on BOTH `mu` and `y0`, FHT produces
**crossing, time-varying hazards** that Cox cannot represent.

Demonstrated (`scripts/experiment_fht.py`, n=1200, held-out test):

| regime | model | C-index | IBS (lower better) | mean td-AUC (higher better) |
|---|---|---|---|---|
| PH (Cox-Weibull) | Cox | 0.688 | **0.177** | **0.745** |
| PH | FHT | 0.692 | 0.193 | 0.734 |
| **non-PH** (crossing) | Cox | **0.842** | 0.112 | 0.916 |
| **non-PH** | **FHT** | 0.833 | **0.102** | **0.930** |

Read honestly: under PH, Cox wins calibration (it is the true model) and FHT ties on C. Under the
**non-PH** regime MM actually exhibits, **FHT beats Cox on IBS (calibration) and time-dependent AUC**
— even though C-index (PH-oriented, monotone-invariant) slightly favors Cox. The gains are in
calibration and time-resolved discrimination, NOT C-index.

## The publishable claim (reframed)
**DO claim:** "A first-hitting-time threshold-regression model for MM relapse whose covariates
modulate a latent resistance process's drift (aggressiveness) and boundary (reserve). (1) It matches
Cox under proportional hazards; (2) under the non-proportional hazards documented in MM it improves
calibration (IBS) and time-dependent AUC; (3) its parameters are mechanistically interpretable;
(4) it extends to longitudinal biomarkers via joint FHT modeling (the PK layer)." Cite Lee-Whitmore
2006, the non-PH threshold-regression work, and the 2025 joint longitudinal-FHT paper.
**DO NOT claim:** beating Cox on C-index (impossible for a monotone score; the wrong yardstick).

## Concrete code changes
1. **Adopt `models/fht_threshold.py` as the proposed model** (the headline). Keep `basin_sde.py` as
   the conceptual/mechanistic link and for the latent-trajectory layer; **demote `program_basin.py`**
   to a "mechanistic illustration," not the evaluated ranking model.
2. **Evaluation protocol:** report **IBS, time-dependent AUC, calibration (ECE), and the Schoenfeld
   PH test** as PRIMARY; C-index SECONDARY. On real GDC data run the PH test FIRST: if PH holds, Cox
   is optimal and the contribution is the framework + interpretability; if PH is violated, FHT is the
   empirical claim.
3. Add FHT to the evaluation harness alongside the baselines for IBS/td-AUC (not only C-index).
4. **Extensions:** treatment-conditioned drift `mu(x, regimen)`; joint longitudinal FHT using the
   deconvolved-biomarker slope (the PK layer) when Gateway labs land; per-cytogenetic-stratum reporting.

## Sources
- Lee & Whitmore 2006, Threshold Regression — https://projecteuclid.org/euclid.ss/1177334526 (arXiv 0708.0346)
- Bayesian random-effects threshold regression (non-PH) — https://academic.oup.com/biostatistics/article/11/1/111/226250
- Joint longitudinal-biomarker + FHT (2025) — https://arxiv.org/pdf/2503.24146
- C-index monotone invariance — https://arxiv.org/pdf/2302.12059 ; crossing-hazard C-index — https://onlinelibrary.wiley.com/doi/10.1111/sjos.70000
- R packages: threg https://rdrr.io/cran/threg/man/threg.html ; invGauss https://rdrr.io/cran/invGauss/man/invGauss.html
