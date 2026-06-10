# Lane #1 — Results narrative (PFS SOTA benchmark, mmSYGNAL substrate)

> **Provenance rule.** Every number below is copied from an on-disk results file and cited inline.
> Nothing here is computed in this document. Primary source: `results/sota_comparison.json`
> (mirror: `results/sota_table.md`). Secondary source: `results/paper_metrics_real.json`
> (mirror: `results/paper_table1.md`).

## 0. One-paragraph verdict

On the primary endpoint (IMWG **progression-free survival**, MMRF-CoMMpass IA12, mmSYGNAL
program-activity substrate, **N=769 / 323 events**, 5-fold patient-disjoint CV), our strongest
*honest* model reaches CV C-index **0.644 [0.610, 0.678]** — **comparable** to the published
transcriptomic SOTA bar (**gep70 0.624**, **sky92 0.620**), with **overlapping confidence intervals
(not CI-separated)**. A leakage audit removes the mmSYGNAL-internal **GuanScore** (in-sample C-index
= **1.000**) as a circular score. On baseline features **proportional hazards holds**
(Grambsch–Therneau **0/20** covariate violations), so the static-transcriptome ceiling is a property
of the **modality**, not the **method**; the non-PH structure lives in **treatment trajectory**
(Lane #2, n_lines p=**1e-4**). The governance gate `novel_model_beats_baselines` is **BLOCKED** by
design. This is an honest **parity** result — not a SOTA-beating claim.

---

## 1. Leakage audit (a contribution, not a footnote)

Source: `results/sota_comparison.json → leakage_audit`.

| Score | Kind | In-sample C-index | Leaky? | In SOTA bar? |
|---|---|---|---|---|
| gep70 | honest_published | 0.6242 | no | **YES (honest bar)** |
| sky92 | honest_published | 0.620 | no | **YES (honest bar)** |
| GuanScore | mmSYGNAL_internal | **1.000** | **YES** | **EXCLUDED** |
| risk_auc | mmSYGNAL_internal | 0.8462 | no (flagged) | EXCLUDED (internal) |

**The GuanScore=1.000 story.** GuanScore is mmSYGNAL's own internal composite. Evaluated in-sample
it reports a **perfect C-index of 1.000** — it does not predict the label, it *re-encodes* it. Any
benchmark that quietly admits GuanScore as a "model" would manufacture an impossible leaderboard
win. We flag it `is_leaky=true` and **exclude it from the SOTA bar**, along with the other
mmSYGNAL-internal score (risk_auc). The honest comparison bar is therefore the two genuinely
published, externally-defined transcriptomic scores: **gep70 (0.624)** and **sky92 (0.620)**. The
audit is run before any model enters Table 1; this is the contribution that keeps the benchmark
honest. (Note from the source file: *"SOTA bar = gep70/sky92 (honest). GuanScore/risk_auc excluded
as leaky."*)

---

## 2. Table 1 — primary benchmark (PFS)

Source: `results/sota_comparison.json → rows` (mirror `results/sota_table.md`).
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

**How to read this honestly.**
- Best honest model = **Stacked, C-index 0.644 [0.610, 0.678]**. Its CI **overlaps** gep70
  [0.589, 0.659] and sky92 [0.586, 0.655]. → **Parity / comparable discrimination, NOT a
  CI-separated win.** We do not claim a SOTA C-index.
- The prog+clin learners (Cox 0.632, RSF 0.631, GBS 0.643) all land in the **same ~0.62–0.64 band**
  as the published scores. Nothing breaks out of the band.
- **Paired per-fold tests** (`results/sota_comparison.json → paired_tests`) confirm the *novel*
  PH-free model is **worse**, not better:
  - `novel_vs_gep70_cindex` per fold = {−0.142, −0.193, −0.092, 0.000} → uniformly ≤ 0 (worse
    discrimination on every fold with signal).
  - `novel_vs_gep70_ibs` per fold = {+0.0161, +0.0083, +0.0233, 0.000} → positive throughout
    (worse survival-function calibration; lower IBS is better).
- The novel **ResistanceBasin-LR collapses to ~chance (C-index 0.482)** — it degenerates to a
  single basin on the baseline substrate. This is an **honest negative result**, predicted by §3.

---

## 3. Proportional hazards holds on baseline → the ceiling is a *modality* property

Source: `results/sota_comparison.json → non_ph_test`.

- Covariates tested: **20**.
- PH violations at p<0.05: **0**.
- Global interpretation: **"PH holds."**

Because proportional hazards holds on the baseline modality, a **PH-free / time-varying** model has
**no structural advantage** there — and the data agree: ResistanceBasin-LR collapses to chance
(§2). The crucial implication: the ~0.62–0.64 ceiling is shared by *every* method in the panel
(gep70, sky92, Cox, RSF, GBS, Stacked). **No method escapes it.** That uniformity is the signature
of a **modality limit** (static tumor-transcriptome signal on ~769 patients), not a method limit.
Throwing a fancier learner at the same modality cannot move the number — consistent with this
repo's parity-not-SOTA thesis.

---

## 4. Where the non-PH structure actually lives (the motivated extension)

Source: `docs/LANE2_TREATMENT_NONPH.md` (validated on open `treatments.tsv`).

The non-proportional-hazards structure that *would* justify a PH-free model is **not in the baseline
transcriptome** (PH holds there). It is in the **treatment trajectory**:

- **Number of lines of therapy violates PH decisively — Grambsch–Therneau χ²=14.7, p=1e-4.**
- Regimen class alone does **not** violate PH (the effect is the accumulation, not the label).
- Log-rank across 1st-line regimens: p=1.3e-3.

This is recoverable from **open data alone** (no VLAB/Zenodo gate). It motivates a
**treatment-conditioned, PH-free** hazard model where therapy's effect changes over time — the
regime static scores (mmSYGNAL/gep70/sky92, all PH) cannot follow.

**Critical honesty constraint (carried from Lane #2).** `n_lines` is partly a *consequence* of
progression; using it as a baseline feature is immortal-time bias / reverse causation. It MUST be a
**time-varying covariate** in a counting-process / landmark design (condition on lines accrued *by*
landmark t, predict forward). The pre-registered falsifier: does the time-varying model beat static
Cox on time-dependent AUC at post-baseline landmarks, **without** immortal-time inflation? Gated as
`treatment_nonph_beats_static` (BLOCKED until CI-separated and immortal-time-safe).

---

## 5. Access-gated extension — immune-microenvironment fusion (Path A)

Source: `docs/PATH_A_IMMUNE_FUSION.md`.

The only principled way to lift the static ceiling is to add a modality on a **different biological
axis**: the bone-marrow **immune microenvironment** (MMRF Immune Atlas, Nature Cancer 2025;
~16–18 patient-level features: cell-type proportions + UCell T-cell state scores). It is largely
orthogonal to tumor-intrinsic program activity.

**Honest access status:** the single-cell matrices are **request-gated on Zenodo**
(`10.5281/zenodo.14624955`, `access_right: "restricted"`), and the clinical PFS join is
**controlled via MMRF VLAB**. The loader fails closed (`FileNotFoundError` with the exact request
path) until access is granted. Pre-registered as a **CI-separated incremental-value** claim on the
immune subset (~300–340 patients, identical CV folds for both arms); routed through the claim gate.
**Scoped, not claimed** — no immune-fusion metric exists yet.

---

## 6. Secondary endpoint — overall survival (GDC-open)

Source: `results/paper_metrics_real.json` (mirror `results/paper_table1.md`).
Clearly labeled as **secondary**: this is a *separate, earlier* GDC-open OS run, **not** the PFS
primary benchmark. It has fewer events (161 deaths vs 323 PFS events).

N=781 patients, **161 deaths**, 5-fold patient-disjoint CV, train-fold median imputation.

| Model | C-index (95% CI) | IBS↓ | td-AUC↑ | ECE@1y | ECE@2y |
|---|---|---|---|---|---|
| cox_ph | 0.737 [0.700, 0.771] | 0.1288 | 0.7486 | 0.0141 | 0.0519 |
| cox_elasticnet | 0.727 [0.691, 0.762] | 0.1300 | 0.7404 | 0.0131 | 0.0408 |
| random_survival_forest | 0.721 [0.681, 0.757] | 0.1290 | 0.7457 | 0.0049 | 0.0347 |
| gradient_boosted_survival | 0.696 [0.657, 0.733] | 0.1521 | 0.7052 | 0.0468 | 0.0682 |
| **program_basin (novel)** | **0.698 [0.659, 0.740]** | 0.1445 | 0.7026 | 0.0263 | **0.0301** |

- The novel **program_basin reaches parity inside the baseline band** (best 2-yr calibration, ECE
  0.0301) but does **not** beat Cox.
- Governance (`results/paper_metrics_real.json → governance`): claim `novel_model_beats_baselines`,
  status **BLOCKED**, rule *"program_basin C-index CI-lower must exceed best baseline 0.737"*,
  `best_baseline_cindex` 0.7369. The gate honestly refuses the unearned "beats-baselines" claim.

> The higher absolute C-index on OS (~0.74) vs PFS (~0.62–0.64) reflects the **endpoint and feature
> block**, not a better method: OS uses the GDC-open 8-feature clinical+program block; PFS uses the
> mmSYGNAL transcriptomic substrate benchmarked against published scores. They are not directly
> comparable and OS is reported only as supporting evidence.

---

## 7. Honest verdict & governance status

- **Primary claim (PFS):** comparable discrimination to published transcriptomic SOTA
  (best honest 0.644 vs gep70 0.624 / sky92 0.620), **overlapping CIs → parity, not a win**.
- **Why the ceiling can't move on this modality:** PH holds on baseline (0/20), so all methods
  share the ~0.62–0.64 band — a modality limit.
- **Where the real signal is:** treatment trajectory (non-PH, p=1e-4) → treatment-conditioned model
  (Lane #2). Immune fusion is the access-gated next axis (Path A).
- **Novel PH-free model:** collapses to chance on baseline (C=0.482) — honest negative.
- **Governance:** `novel_model_beats_baselines` = **BLOCKED** (PFS paired tests show the novel model
  is worse; OS gate also BLOCKED, best_baseline 0.7369). Parity is the honest, submittable outcome.

**Bottom line:** the contribution is the **method + a leakage-audited, claim-gated open-data
benchmark**, reaching honest parity — explicitly **not** a SOTA C-index win.
