# Figure guide — what each figure shows, how to read it, what it means

> **This directory (`paper/figures/`) is the single consolidated bundle: all 14 figures + this
> legend in one place.** Each entry's `Source:`/`Script:` line points to where the figure is
> *generated* under `results/` (reproducible); the PNGs here are the curated submission copies,
> co-located with their explanations (section headers match the filenames in this folder).

Every figure below is generated from **real on-disk data** (no fabricated values) and every number
cited here traces to a `results/*.json`. Endpoint is **progression-free survival (PFS)** unless a
figure is explicitly the **overall-survival (OS)** secondary run. Reproduce with the script named
under each entry. Convention throughout: **higher risk score = worse prognosis**; survival curves are
Kaplan–Meier (KM); CV is patient-disjoint.

---

## A. Primary benchmark figures (`results/`)

### `fig_sota_benchmark.png` — the headline PFS benchmark + leakage audit
**Script:** `scripts/run_sota_figures.py` · **Source:** `results/sota_comparison.json` · PFS, N=769 / 323 events, 5-fold patient-disjoint CV.
- **Panel A (C-index forest + leakage callout).** Each dot is a model's cross-validated C-index (ability to rank who progresses sooner); the horizontal bar is its 95% bootstrap CI. Blue = honest published SOTA (gep70 0.624, sky92 0.620), grey = our models, red = the novel PH-free model. The dashed blue line marks the SOTA bar. The red callout box lists scores removed by the **leakage audit**. **How to read:** overlapping CIs = statistically comparable. **Key finding:** best honest model (Stacked) **0.644 [0.610, 0.678]** vs gep70 0.624 — *overlapping CIs = parity, NOT a SOTA win*. The novel model sits far left at 0.482 (chance).
- **Panel B (IBS / td-AUC bars).** Orange = Integrated Brier Score (calibration error, **lower is better**); blue = time-dependent AUC (**higher is better**). **What's happening:** every model clusters tightly (~0.18 IBS, ~0.65–0.68 td-AUC) — the whole panel hits the same ceiling.
- **Panel C (KM by first-line regimen).** PFS curves split by first-line drug class (VRd-like / PI-based / IMiD-based). Inset: *n_lines violates PH, Grambsch–Therneau p=1e-4*. **Meaning:** this previews Lane #2 — the non-PH signal is treatment-driven.
- **Honest takeaway:** static tumor-transcriptome→PFS is ceiling-bound (~0.62–0.64); the field's headline score is excluded for leakage; the way out is treatment dynamics (Lane #2) or a new modality (Path A).

### `fig_paper_main.png` — SECONDARY endpoint (overall survival)
**Script:** `scripts/paper_figures.py` · **Source:** `results/paper_metrics_real.json` · OS, GDC-open tier, N=781 / 161 deaths.
- **Panel A:** KM overall survival by Cox risk tertile — clean separation, **log-rank p=1.1e-22**.
- **Panel B:** C-index forest (Cox 0.737 … program_basin 0.698) — OS is *easier* than PFS, hence higher numbers; still parity (gate BLOCKED).
- **Panel C:** 2-year survival calibration — program_basin is best-calibrated (ECE 0.030).
- **Note:** OS is a *secondary, lower-event-count* view. The open GDC tier does not expose PFS; the mmSYGNAL substrate (above) is the primary PFS benchmark.

---

## B. Interpretability — "what drives the model's risk" (`results/interpretability/`)
**Script:** `scripts/interpret_pfs.py` · **Source:** `results/interpretability/interpretability.json` · GBS(prog+clin), OOF C-index 0.643. Narrative in `docs/RESULTS_INTERPRETABILITY.md`.

### `fig_risk_drivers.png` — what the model actually uses
Horizontal bars = each feature's **permutation importance** (drop in C-index when that feature is shuffled on a held-out fold; bigger bar = more the model relies on it). **Key finding:** **ISS is the top driver (≈0.033, ~2× any single program)**; the rest is a long tail of weakly-predictive transcriptional programs. **Meaning:** there is **no single "resistance program"** — discrimination is diffuse, consistent with the modality ceiling.

### `fig_tertile_km.png` — does the risk score stratify patients? (the clinical money figure)
KM PFS by model-risk tertile (Low/Mid/High). **Key numbers:** median PFS **47 / 30 / 24 months** (1436 / 914 / 715 days), **log-rank p=4.8e-12**. Side annotation = Fisher-tested enrichment in the high-risk tertile: **WHSC1 19.9% vs 10.5%**, **FGFR3 15.6% vs 7.2%** (both t(4;14)), **CCND1** (t(11;14)), **amp1q** — all canonical MM high-risk lesions; ISS mean climbs 1.51→2.11→2.33; **first-line drug class is flat across tertiles**. **Meaning:** the separation is **disease biology, not a treatment confound**.

### `fig_risk_tertile_heatmap.png` — which programs differ by risk
Heatmap of the top differential transcriptional programs (rows) across the three risk tertiles (columns), colored by mean program activity. **How to read:** rows that go dark→light left-to-right are programs that rise (or fall) monotonically with model risk — the transcriptional axes separating low- from high-risk patients.

### `fig_calibration.png` — are the predicted probabilities trustworthy?
Reliability diagram: predicted vs observed PFS-event fraction at 1- and 2-year horizons, with censoring-honest KM-binned observed risk; the diagonal = perfect calibration. **Key numbers:** **1-yr ECE 0.018, 2-yr ECE 0.034** — well-calibrated at clinically relevant horizons (points hug the diagonal).

### `fig_program_partial_dependence.png` — direction of effect
Partial-dependence curves: how predicted risk changes as each of the top-3 programs varies (others held fixed). **How to read:** an upward curve = higher program activity → higher predicted progression risk. Makes the model's input→risk direction explicit and inspectable.

---

## C. Lane #2 — treatment-conditioned, non-proportional-hazards (`results/lane2/`)
**Scripts:** `scripts/run_lane2_landmark.py` (v1), `scripts/run_lane2_v2.py` (v2). Cohort = mmSYGNAL ∩ open `treatments.tsv`, n=712 / 311 events. **Landmark design** = re-origin time at each landmark L and keep only patients still at-risk (event-free) at L — this is what makes it **immortal-time-safe** (avoids crediting the model for a covariate that only exists because the patient survived to accrue it).

### `fig_lane2_landmark.png` — v1, the first treatment result
Bars = forward time-dependent AUC at landmarks L=180/365/730 days, for three models: static Cox, time-varying Cox (+ treatment block Z(L)), and the PH-free treatment_basin. Error bars = fold std; the dashed line = chance (0.5). **What's happening:** time-varying edges static at L=180/365; **the basin collapses to 0.500 everywhere** (red, flat); signal decays to chance by 730 d.

### `fig_lane2_v2.png` — v2, the RIGOROUS nested-CV refinement (supersedes v1)
- **Left panel:** outer-fold forward td-AUC under **nested CV** (inner folds pick every hyperparameter; outer fold scored once) — static / enriched-time-varying / basin_v2.
- **Right panel:** the paired margin (enriched − static) with 95% CI per landmark; the green dashed line is v1's prior +0.0097 reference. **How to read:** a bar whose CI sits entirely above 0 is a real win. **Key finding:** CI-separated win **only at L=180 (Δ=+0.0145, CI [+0.0018, +0.0259])**; L=365 is non-significant (CI crosses 0) and L=730 reverses. **Meaning:** a **real but small and landmark-sensitive** treatment-driven non-PH gain — v1's L=365 win did *not* survive nesting; the basin_v2 (red) is partially un-collapsed but still loses to Cox. Honest, not overclaimed.

---

## D. Supplementary tangible figures (`results/figures/`)
**Script:** `scripts/make_more_figures.py` · **Source:** `results/figures/figures_manifest.json`.

### `fig_nonph_schoenfeld.png` — the non-PH "smoking gun"
Scatter = scaled **Schoenfeld residuals** for `n_lines` (one per event, x = event-time rank in months); red line = moving-average trend; dashed line at 0 = what proportional hazards *requires* (a flat trend). **What's happening:** the trend runs from ≈ −0.5 early to **positive later** — the hazard effect of treatment burden **changes over time**, i.e. PH is violated (**Grambsch–Therneau p=3.3e-5** for this Cox spec; consistent with Lane #2's p=1.2e-5). **Meaning:** this is the *visual proof* that the non-PH structure motivating Lane #2 is real, not a test artifact.

### `fig_tdauc_over_time.png` — where the signal lives in time
Lines = time-dependent AUC as a function of months-from-diagnosis, for gep70, sky92, and our GBS(prog+clin). **How to read:** all three start ~0.65–0.70 and **decay toward 0.5 and converge** by ~3 years. **Meaning:** the transcriptome→PFS signal is **early and shared** across models — quantitatively explaining why no static model separates from the others and why discrimination is hardest long-term.

### `fig_leakage_audit.png` — the audit as a standalone contribution
Horizontal bars = univariate C-index of each candidate score vs PFS; red = flagged leaky and **excluded**, blue = honest. **Key finding:** **GuanScore C-index = 1.000** (it re-encodes the outcome → leakage), risk_auc 0.846 (also mmSYGNAL-internal); the honest bar is **gep70 0.624 / sky92 0.620**. **Meaning:** quoting GuanScore as SOTA would be a circular 1.000 — this audit prevents exactly that failure.

### `fig_patient_manifold.png` — does risk track real biology on the data manifold?
**PHATE** 2-D embedding of all 769 patients from their 141-program profiles. **Left:** points colored by model PFS risk (blue→red). **Right:** amp1q+ patients (high-risk cytogenetics) in red. **How to read:** the risk gradient (left) and the amp1q+ cloud (right) **occupy the same region of the manifold** → the model's continuous risk co-localizes with a known high-risk lesion. **Meaning:** the learned risk is biologically structured, not noise.

### `fig_risk_stratification.png` — our model vs published signatures, head-to-head
Three KM panels — risk tertiles by gep70, by sky92, and by our GBS(prog+clin) — each with its own log-rank p. **How to read:** comparable vertical spread between Low/Mid/High across all three panels = comparable stratification. **Meaning:** our multimodal model separates patients *as well as* the established signatures (the honest "parity" claim, shown visually).

---

## One-paragraph reading order for a reviewer
Start with **fig_leakage_audit** (why the bar is 0.62, not 1.0) → **fig_sota_benchmark** (parity at the ceiling) → **fig_tdauc_over_time** (the signal is early & shared = a modality limit) → **fig_risk_drivers** + **fig_tertile_km** + **fig_patient_manifold** (the risk is interpretable & biology-anchored) → **fig_nonph_schoenfeld** (the non-PH structure is treatment-driven) → **fig_lane2_v2** (a small, honest, immortal-time-safe treatment-conditioned gain). That arc *is* the paper.
