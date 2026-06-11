# Ext-val synthesis — trustworthy & generalizable MM-progression forecasting

**Companion to** `docs/THEORY_LOOP_SYNTHESIS.md` (the discrimination-ceiling result). Where that
document shows *discrimination is method-invariantly capped (~0.62)* on open multi-omic+treatment data,
this one shows what the open data **can** still contribute honestly: a rigorous **trustworthiness +
generalization** characterization. Substrate: MMRF-CoMMpass IA12 (N=769, PFS) + the independent IA18
relapse cohort (n=86, 64 events). All numbers cite `results/extval/*.json`. No fabricated metrics.

## The thesis
When point-prediction discrimination is **information-limited** (not model-limited), the scientific
contribution moves from "predict better" to "predict **honestly and trustworthily**": audit leakage,
characterize generalization under shift, and deliver **distribution-free uncertainty with a coverage
guarantee**. We do all three on open data.

## Finding 1 — the leakage critique generalizes (decisive)
`results/extval/iter1.json`. Pre-registered hypothesis: the leaky in-sample `GuanScore` (IA12 C-index
= 1.000) would *collapse* to ~0.6 out-of-distribution on IA18. **Refuted — informatively:**
- `GuanScore` is **still C-index = 1.000 on IA18** → the IA18 published risk table *also* carries an
  outcome-fit score. **Leakage lives in BOTH published cohort tables**, not just one — a stronger
  finding than hypothesized. Naively using these columns as "predictions" yields a fake perfect score.
- The **genuine** mmSYGNAL prediction, `miner_risk`, generalizes to **C-index 0.612 [0.531, 0.692]** —
  exactly the ceiling. `risk_auc` collapses 0.846 → 0.655 (partly leaky).
- *Caveat:* IA18 is small and relapse-enriched (74%); directional, not a precise generalization estimate.
- *Lesson:* a pre-registered leakage audit is essential before any cross-cohort "validation"; both
  published tables fail it.

## Finding 2 — the program representation is shift-robust
`results/extval/iter2.json`. Train GBS(programs-only) on one stratum, test on a held-out stratum
(variable kept out of the feature set):
- amp1q: train(amp1q−)→test(amp1q+) **0.679 [0.613, 0.745]** (no drop); train(amp1q+)→test(amp1q−) 0.581.
- ISS: degradation ±0.003. **Worst-group degradation +0.009 → no detectable cross-stratum drop.**
- *Caveat:* wide CIs (small strata), near-ceiling band; **`gep70` is equally robust and stronger in
  amp1q+ (0.704)** → no method robustly *exceeds* gep70 under shift. The honest claim is *robustness*,
  not superiority.

## Finding 3 — distribution-free conformal coverage holds where naive fails (positive)
`results/extval/iter3.json`. Split-conformal 90% intervals for log time-to-progression (uncensored),
40 patient-disjoint splits:
- **Marginal coverage 0.906 [0.819, 0.977]** — the finite-sample distribution-free guarantee holds.
- **Naive Gaussian intervals UNDER-cover at 0.765** — overconfident by ~13 points; conformal restores
  validity (necessity demonstrated).
- Per-stratum (high-ISS): marginal-q 0.902 vs Mondrian-q 0.909 — group-conditional coverage maintained.
- *Caveat:* intervals are **wide** (~3 log-day width ≈ 20× multiplicative range) — a *calibrated,
  honest* expression of irreducible uncertainty, not a defect; target is uncensored-only.

## The combined ICLR/ICML contribution (two synthesis docs together)
1. **A method-invariant discrimination ceiling** for MM-PFS on open multi-omic+treatment data, proven
   across six theory families with a pre-registered, adversarial-verification harness that refuted its
   own best positive (`THEORY_LOOP_SYNTHESIS.md`).
2. **A trustworthiness layer on the same substrate:** the field's published scores fail a leakage audit
   in *both* cohort tables; the program representation is shift-robust; and split/Mondrian conformal
   delivers valid distribution-free prediction intervals where naive uncertainty is overconfident.
3. **A reusable, theory-grounded evaluation harness** (immortal-time-safe landmarks, nested CV,
   negative-control/permutation/strong-baseline gates, leakage audit, conformal coverage) — the
   transferable artifact, applicable to any censored, treatment-conditioned, biology-context-dependent
   forecasting problem (CKD, transplant rejection, NSCLC immunotherapy).
4. **The one credible path to higher discrimination** is new orthogonal information — the bone-marrow
   immune microenvironment — which is access-gated (`docs/PATH_A_IMMUNE_FUSION.md`), scoped not claimed.

## Honest non-claims
- No discrimination win anywhere (gate BLOCKED; ceiling proven).
- The conformal positive is about *coverage validity*, not sharper prediction (intervals are wide).
- IA18 generalization is directional (n=86, relapse-enriched), not a precise external-validation number.

**Status:** both research directions (theory-mechanism search; external-validation/robustness/calibration)
have produced their honest contributions. Further mechanisms on this open substrate would be thrashing.
The next real move is GitHub-auth refresh (to push the local commits) and/or the access-gated immune
modality, or a new user-directed direction.
