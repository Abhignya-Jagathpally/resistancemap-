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

## Iteration 2 — within-IA12 domain-generalization robustness  [no detectable degradation; honest]
**Design:** partition IA12 by a shift variable held OUT of the feature set (amp1q±; ISS high/low); train
GBS(programs-only) on source group, test on held-out target; degradation = in-dist CV − cross-group.
`results/extval/iter2.json`.
**Result:**
- amp1q: train(amp1q−)→test(amp1q+) **0.679 [0.613, 0.745]** (vs in-dist 0.630 — no drop, larger source
  set helps); train(amp1q+)→test(amp1q−) 0.581 (drop +0.009). gep70 ref: amp1q− 0.598, **amp1q+ 0.704**.
- ISS: drops ±0.003 (none). gep70 ref: low 0.578, high 0.631.
- **Worst-group degradation +0.009 → no detectable cross-stratum degradation.** The program
  representation generalizes across cytogenetic/ISS strata without breaking.
**Honest caveats:** wide CIs (small strata); everything in the ~0.58–0.68 band (little to degrade);
**gep70 is equally robust and stronger in amp1q+ (0.704)** → no method robustly EXCEEDS gep70 under
shift. Legitimate result: stratum-shift robustness characterized rigorously, not a discrimination win.
**Next (Iter 3):** distribution-free CONFORMAL survival on IA12 — split-conformal + Mondrian
(per-stratum) coverage; the trustworthy-uncertainty contribution with a finite-sample guarantee.
