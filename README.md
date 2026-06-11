# ResistanceMap (open-data edition)

**Interpretable, censoring-honest forecasting of multiple-myeloma (MM) progression on open-access data
— where the contribution is rigorous *honesty + calibrated uncertainty + generalization*, not a
leaderboard win.** Every number in this repo is produced by code here and traces to a `results/*.json`;
nothing is fabricated, and no SOTA discrimination claim is made (it is provably information-limited).

> **One-figure summary:** [`paper/figures/fig_capstone.png`](paper/figures/fig_capstone.png) —
> (A) discrimination is method-invariantly ceiling-bound; (B) the treatment-conditioned signal is a
> small, power-limited ~+0.01 td-AUC; (C) conformal prediction gives valid coverage where naive
> intervals under-cover.

---

## The result, in three honest claims

**1. Discrimination on open multi-omic + treatment data is method-invariantly ceiling-bound (~0.62–0.66).**
On MMRF-CoMMpass IA12 (N=769, IMWG-PFS, patient-disjoint CV), the strongest *honest* model reaches
C-index **0.644 [0.610, 0.678]** — *comparable*, with overlapping CIs, to the published transcriptomic
signatures **gep70 0.624** / **sky92 0.620**, and below the strong baseline **Cox(ISS+gep70) ≈ 0.653**.
A **leakage audit** excludes mmSYGNAL's internal `GuanScore` (in-sample C-index **= 1.000**). Six
distinct theory families — control-theoretic LTI/SSM, score-based **diffusion-SDE**, **Neural-CDE /
path-signature**, log-rank clustering, transductive **graph** — *all* tie at this band. The limit is the
**information, not the model**. → `docs/THEORY_LOOP_SYNTHESIS.md`.

**2. The only residual signal is treatment-driven and power-limited.** Proportional hazards *holds* on
baseline features (Grambsch–Therneau 0/20); it is *violated* by treatment trajectory (`n_lines`,
p=1e-4). An immortal-time-safe, nested-CV time-varying model gains **+0.0145 td-AUC [+0.0018, +0.0259]
at 6 months** — real, CI-separated, but landmark-sensitive; every treatment-conditioned mechanism gives
the same ~+0.01–0.015, none reaching significance beyond it (a power limit at N≈650). → `docs/LANE2_TREATMENT_NONPH.md`.

**3. Where the open data still contributes: trustworthiness & generalization.**
- **Leakage generalizes** — `GuanScore` is *also* 1.000 on the independent IA18 cohort (both published
  tables are outcome-fit); the genuine prediction `miner_risk` generalizes to **0.612** (the ceiling).
- **Shift-robust** — the program representation transfers across amp1q/ISS strata with no detectable
  degradation (worst-group +0.009).
- **Valid distribution-free uncertainty** — split/Mondrian **conformal** survival intervals achieve
  **0.906 coverage** (nominal 0.90) where **naive Gaussian under-covers at 0.765**. → `docs/EXTVAL_SYNTHESIS.md`.

**Interpretability:** model risk is ISS-led (no single "resistance program"); risk tertiles separate
PFS at **log-rank p = 4.8e-12** (median 47/30/24 mo), with the high-risk tertile enriched for canonical
high-risk lesions — t(4;14) (WHSC1/FGFR3), t(11;14) (CCND1), amp1q — and *flat* treatment distribution
(biology, not a regimen confound). 2-yr ECE = 0.034. → `docs/RESULTS_INTERPRETABILITY.md`.

---

## Why this is a methods contribution (ICLR/ICML-style)
When point-prediction is information-capped, the transferable artifact is **the rigorous evaluation
harness + the honest characterization**, not a single model: a pre-registered, leakage-audited,
immortal-time-safe, nested-CV protocol with **adversarial verification that refuted its own most
promising positive** (an LTI feature that looked like +0.072 but was immortal-time leakage), plus
distribution-free conformal coverage. It generalizes to any **censored, treatment-conditioned,
biology-context-dependent** forecasting problem (CKD, transplant rejection, NSCLC immunotherapy). The
one credible path to higher discrimination is new orthogonal information — the bone-marrow immune
microenvironment — which is **access-gated** and scaffolded, not claimed (`docs/PATH_A_IMMUNE_FUSION.md`).

## Data
| Source | Access | Use |
|---|---|---|
| **mmSYGNAL** program activity + clinical (CoMMpass IA12/IA18) | open (git-LFS) | **primary** PFS benchmark + external validation; comparator scores gep70/sky92 |
| GDC-open CoMMpass (`phs000748`, expression + clinical) | open | secondary OS run + treatment timeline (`treatments.tsv`) |
| MMRF Immune Atlas scRNA (immune microenvironment) | gated (Zenodo/VLAB) | the access-gated path to higher discrimination — scaffolded, off by default |

## Reproduce
```bash
pip install -e .                                            # core deps (lifelines, sksurv, ...)
# Primary PFS benchmark + leakage audit + figure
python resistancemap/scripts/run_sota_comparison.py
python resistancemap/scripts/run_sota_figures.py
# Treatment-conditioned non-PH (immortal-time-safe, nested CV)
python resistancemap/scripts/run_lane2_v2.py
# Interpretability + the supplementary + capstone figures
python resistancemap/scripts/interpret_pfs.py
python resistancemap/scripts/make_more_figures.py
python resistancemap/scripts/make_capstone_figure.py
# Theory-mechanism loop (6 families) and external-validation loop
python resistancemap/scripts/theory_loop/iter*.py
python resistancemap/scripts/extval/iter*.py
```
*(mmSYGNAL substrate: clone `baliga-lab/mmSYGNAL-risk-prediction-models` into `_external/` and `git lfs pull`.)*

## Layout
```
src/resistancemap/{data,survival,models,loss,stats,interpretability,governance}/   pipeline.py
scripts/{run_sota_comparison, run_lane2_v2, interpret_pfs, make_*_figure, theory_loop/, extval/}
docs/   THEORY_LOOP_SYNTHESIS · EXTVAL_SYNTHESIS · LANE2_TREATMENT_NONPH · RESULTS_* · PATH_A_IMMUNE_FUSION
paper/figures/  (15 figures + FIGURES.md legend, incl. fig_capstone)      results/  (every metric as JSON)
```

## Honest non-claims
- No SOTA discrimination win anywhere (gate BLOCKED; ceiling proven six ways).
- The treatment and conformal positives are modest/coverage-only, fully caveated.
- IA18 generalization is directional (n=86, relapse-enriched), not a precise estimate.

License: MIT. Comparator code (`loss/partial_logrank.py`, `stats/weighted_concordance.py`) is vendored
from Ferle et al. (2026, npj Digital Medicine) with attribution.
