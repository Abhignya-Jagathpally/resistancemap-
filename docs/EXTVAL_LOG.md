# External-validation / robustness / calibration loop (post-convergence direction)

Beyond the converged theory-mechanism search (docs/THEORY_LOOP_SYNTHESIS.md). Axis = generalization
under distribution shift + trustworthy uncertainty, not discrimination. No fabricated metrics.

## Iteration 1 — external-validation data contract + leakage-OOD test (IA18)
**Pre-registered hypothesis:** GuanScore (C-index=1.000 in-sample on IA12 = leakage) would COLLAPSE to
~0.6 out-of-distribution on the independent IA18 relapse cohort (n=86, 64 events).
**Outcome: hypothesis REFUTED — informatively.** `results/extval/iter1.json`.
- **GuanScore is STILL C=1.000 on IA18** → the IA18 published table's GuanScore is *also* an in-sample /
  outcome-fit score. Leakage lives in **both** published cohort tables, not just IA12 — a *stronger*
  leakage finding than pre-registered.
- **miner_risk (genuine prediction): C=0.612 [0.531, 0.692] OOD** → mmSYGNAL's real generalizable signal
  sits at the ~0.62 ceiling.
- **risk_auc collapses 0.846 → 0.655 [0.598, 0.715]** (partly leaky, lands at ceiling).
- clin_risk 0.774 [0.719, 0.823] — SUSPICIOUS on a small, relapse-enriched (74%) cohort (selection /
  construction effects); **not claimed**.
- **Data contract:** IA18 program features for the `_2` relapse samples are NOT present (0/881 cols) →
  a full train-IA12→test-IA18 transcriptomic model is not possible on this open dump.
**Honest takeaway:** the leakage critique generalizes (both cohort tables carry outcome-fit scores), and
the genuine OOD signal is at the ceiling. **Caveat:** n=86, relapse-enriched, wide CIs — directional, not
precise. **Next (Iter 2):** pivot — within-IA12 distribution-shift ROBUSTNESS (train on one ISS/cyto
stratum, test on held-out strata; which method degrades least), then Iter 3 = distribution-free CONFORMAL
survival with finite-sample coverage (the trustworthy-uncertainty contribution).
