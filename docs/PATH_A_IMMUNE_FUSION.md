# Path A — Immune-Microenvironment Fusion for PFS

> **Status:** scoping only. No new metrics claimed. No data fabricated. The single-cell
> matrices needed to *produce* immune features are **not openly downloadable** (see §2);
> until they are obtained on the user's machine, the loader in
> `src/resistancemap/data/immune_atlas.py` raises `FileNotFoundError` with the exact
> request path. Everything below cites a real accession, URL, or in-repo source file.

## 0. Why Path A exists (the non-redundancy thesis)

The submittable core (Paths B/C/D) predicts IMWG progression-free survival (PFS;
`D_PFS` / `D_PFS_FLAG`, here `ttcpfs` / `censpfs`) on MMRF CoMMpass from **bulk-tumor**
mmSYGNAL program-activity features (141 programs × ~881 patients) plus clinical and
cytogenetics. The honest discrimination ceiling for tumor-transcriptome PFS models is
**C-index ≈ 0.62** (gep70 / SKY92 class). Repeated audits show our model *ties* that
ceiling — that is the nature of the data (low tumor-intrinsic signal, small N), not a
fixable bug. A bulk tumor model and a bulk tumor risk score are measuring the *same*
biological axis, so stacking them cannot move the ceiling.

The only principled way to break it is to add a modality that measures a **different**
biological axis. The bone-marrow **immune microenvironment** is that axis: it is largely
orthogonal to tumor-intrinsic program activity, and it is mechanistically front-and-center
in the CAR-T / bispecific era (T-cell exhaustion / dysfunction gates response to immune
redirection). The MMRF Immune Atlas (Nature Cancer, 2025) is the asset that operationalizes
it on the *same CoMMpass patients* we already model.

## 1. The asset: MMRF Immune Atlas

- **Paper:** Pilcher, Yao, Gonzalez-Kozlova, Pita-Juarez, et al. *"A single-cell atlas
  characterizes dysregulation of the bone marrow immune microenvironment associated with
  outcomes in multiple myeloma."* Nature Cancer (2025), DOI `10.1038/s43018-025-01072-4`.
  (Title verified from the in-repo cell-annotation supplementary note; full DOI string is
  the user's to confirm against the journal page — web access was unavailable in this
  sandbox.)
- **Code repo (cloned):** `https://github.com/theMMRF/MMRF_ImmuneAtlas`
  → local: `_external/MMRF_ImmuneAtlas/` (depth-1 clone).
- **Scale (verified from `processing_raw_data/construct_cell_metadata.R`):** ~345 patients,
  ~491 single-cell aliquots/visits ("Num Patients (All Atlas)", "Num Visits (All Atlas)").
  The locked discovery object the survival analysis runs on is **483 samples**
  (`SeuratObj_in_483_samples_..._Harmony_v1.rds`, per `MMRF_immune_dataset.R`).
- **Annotation depth:** 106 fine clusters across 5 compartments (Plasma, NKT [T + NK],
  Myeloid, B/Erythroid, Erythroid) — see `cell_annotation_dictionary/`.
- **Tissue:** bone-marrow aspirate, 10x scRNA-seq, CellBender ambient removal, Scrublet /
  DoubletFinder / Pegasus doublet calls, Harmony batch correction (PC25).

## 2. EXACT open-access data sources + access tier (verified live against Zenodo API)

| What | Accession / URL | Access tier (verified) | Format | Get-it command |
|------|-----------------|------------------------|--------|----------------|
| **Annotated single-cell Seurat object** (cell annotations, *no clinical metadata*) | Zenodo `10.5281/zenodo.14624955` → `https://zenodo.org/records/14624955` | **RESTRICTED.** Record license is CC-BY-4.0, but the API reports `access_right: "restricted"` and the file download 302-redirects to `/login/`. Files are **request-gated** (Zenodo login + grant), *not* one-click open. | `.rds` Seurat (`MMRF_ImmuneAtlas_Full_With_Corrected_Censored_Metadata.rds`, `COMBINED_VALIDATION_..._Censored_Metadata.rds`) | After being granted access on Zenodo, download via the record page; convert RDS→h5ad with the `ExportH5AD()` helper in `_external/MMRF_ImmuneAtlas/.../MMRF_immune_dataset.R`. |
| **Raw FASTQs + original Seurat object + clinical metadata** (needed for the PFS join) | **MMRF VLAB** `https://mmrfvirtuallab.org` (submit a request form) | **CONTROLLED.** Application + data-use agreement required. This is where survival/PFS labels live. | FASTQ / RDS | Submit VLAB request; see the paper's data-availability + requirements. |
| **CoMMpass clinical/PFS + bulk RNA (the join target)** | GDC project **MMRF-COMMPASS** (`phs000748`) — open tier. Bulk counts the atlas itself used: `COUNTS_Gene_Based_MMRF_CoMMpass_IA22_star_geneUnstranded_counts.tsv.gz` (IA22). | **OPEN.** No dbGaP credentials for harmonized clinical. | TSV / JSON | Already wired: `src/resistancemap/data/gdc_clinical.py` (clinical via GDC API) + `aws s3 cp --no-sign-request --recursive s3://gdc-mmrf-commpass-phs000748-2-open/ ...` for bulk. |

**Honest bottom line on access:** there is **no fully-open, one-click** copy of the
single-cell *matrices*. The README's phrase "public version of the Seurat object" is true
about the *license* but the **files are request-gated on Zenodo**, and the clinical
metadata required to attach PFS is **controlled (VLAB)**. Two viable paths:
1. **Atlas-direct (recommended for the breakthrough):** obtain the Zenodo object (with cell
   annotations) + clinical via VLAB, derive per-patient immune features, join to our
   CoMMpass cohort by `MMRF_xxxx`.
2. **Feature-table-only (lightest):** if the authors release (or a request yields) the
   per-sample cell-type **abundance table** + per-patient signature scores used in their
   Figure 7 survival models, we can skip the cell-level processing entirely and fuse those
   patient-level features directly. (These derived tables are smaller and may be obtainable
   without re-processing the full atlas — worth asking for explicitly.)

## 3. The immune feature set (≈20 features) and its biological justification

All features are **patient-level**, derived from the atlas's own annotation columns, and
mirror exactly what the atlas Figure-7 survival models use. Two families:

### 3a. Cell-type **proportions** (compositional; non-tumor compartments)

Computed as `table(sample_id, celltype) / row_total`, then aggregated to patient by mean
across that patient's baseline aliquots — the atlas does this with
`abundances <- table(imm_atlas_meta$sample_id, imm_atlas_meta$<celltype_col>)` and truncates
`sample_id` to 11 chars to recover `MMRF_xxxx` (`surv_models_and_abbundance.R:121-124`).
Use the **fine** label column `celltype_subclusters_label_transferring_Yizhe_v1` (or the
coarser `compartment`) collapsed to ~12 interpretable lineages:

| Feature (proportion of BM cells) | Why it plausibly carries PFS signal |
|---|---|
| `prop_CD8_T` | CD8 effector pool — substrate for immune control / redirection. |
| `prop_CD4_T` | helper compartment; CD4 dysfunction tracks progression. |
| `prop_Treg` | regulatory T cells — immunosuppressive; expansion → worse control. |
| `prop_NK` | NK cytotoxicity; depletion associated with progression. |
| `prop_NKT` | combined NK/T cytotoxic axis (atlas's "NKT" compartment). |
| `prop_CD14_Mono` | classical monocytes — atlas Fig 4 ties CD14 mono shifts to outcome. |
| `prop_CD16_Mono` | non-classical monocytes (inflammatory skew). |
| `prop_cDC` | conventional dendritic cells — antigen presentation capacity. |
| `prop_pDC` | plasmacytoid DC — type-I-IFN microenvironment. |
| `prop_B` | normal B compartment (immune competence, not tumor plasma). |
| `prop_Erythroid` | erythroid/progenitor expansion — marrow-replacement burden proxy. |
| `prop_HSC_Prog` | HSC/progenitor fraction — niche state. |

> Tumor plasma-cell fraction is **deliberately excluded** as a fusion feature: it is the
> tumor compartment and would re-introduce the redundant tumor-burden axis. (It can be a
> covariate to *adjust for purity*, but it is not part of the immune signal.)

### 3b. T-cell **state scores** (UCell signatures; the atlas's proven PFS markers)

The atlas's survival figures (Fig 4 L–O, Fig 6) score CD8/CD4 T cells with **UCell** against
literature gene sets, then average per patient (`group_by(public_id) |> summarise(mean(...))`)
and fit `coxph(Surv(ttcpfs, censpfs) ~ score + covariates)`
(`manuscript_figures/figure_4/Fig4L_M_N_O_Survival_Curves_for_CD3_Signatures.Rmd`). Reuse the
**same four**:

| State score (mean over a patient's T cells) | Gene-set source / definition (verbatim from repo) | PFS rationale |
|---|---|---|
| `tcell_naive` | Chu et al. pan-cancer T-cell atlas, Tables S4+S6 (col 1) | naïveté = preserved immune potential. |
| `tcell_cytotoxic` | Chu et al. Tables S4+S6 (col 5) | effector cytotoxicity = tumor control. |
| `tcell_exhaustion` | Chu et al. Tables S4+S6 (col 3) | exhaustion = loss of control → progression. |
| `tcell_dysfunctional` | custom marker set `B3GAT1+, ZEB2+, KLRG1+, KLRK1+, TIGIT+, PDCD1+, LAG3+, CTLA4+, HAVCR2+, TOX+, CD160+, CD27-, CD28-` | terminal dysfunction; gates CAR-T/bispecific response. |

> The Chu et al. gene-set tables ship as `pan-cancer_tcell_atlas_chu_et_al.xlsx` in the
> atlas authors' resource folder (not in this public repo); the custom dysfunctional set is
> reproduced literally above so it is reconstructable without that file.

### Optional compartment-pseudobulk scores (1–2 more, if cheap)

- `plasma_IFN_score` — type-I interferon signature in the **non-tumor** compartment
  (atlas Fig 3 links bulk↔scRNA IFN). Keep ≤2 to stay interpretable.

**Total: ~12 proportions + 4 T-state scores (+1–2 optional) = ~16–18 features.** All are
Seurat/scanpy-derivable from cell annotations alone (no re-clustering needed if the released
object carries `celltype_subclusters_label_transferring_Yizhe_v1` and `compartment`).

## 4. How features join to CoMMpass patient IDs

- The atlas stores `sample_id` like `MMRF_1817_1_BM` and `public_id` like `MMRF_1817`.
- Patient key = `public_id` (= `substr(sample_id, 1, 11)` → `MMRF_1817_1`, then strip the
  visit suffix to `MMRF_1817`). This is exactly the atlas's own join key
  (`construct_cell_metadata.R` indexes outcomes by `public_id`; `surv_models_and_abbundance.R`
  uses `substr(rownames(abundances), 1, 11)`).
- **Baseline only:** restrict to `VJ_INTERVAL == "Baseline"` / first visit per patient so the
  immune snapshot precedes the PFS clock (the atlas filters to baseline for its survival
  models). This avoids time-leakage.
- Join: `immune_features.index (MMRF_xxxx)` ⟕ our existing CoMMpass program/clinical frame on
  the same `MMRF_xxxx`. Expect ~**300–340** patients with immune data — a **subset** of the
  ~881 in the bulk cohort (see risk §6).

## 5. Fusion architecture

**Late fusion (recommended, primary).** Compute the ~16 immune features once, then
**concatenate** them to the existing feature block `[141 programs ⊕ clinical ⊕
cytogenetics]` and feed the union into the *same* survival learner already in
`src/resistancemap/survival/baselines.py` (Cox / EN-Cox / RSF / GBS). No architecture change;
the immune block is just more columns. This matches the atlas's own design — they add one
immune feature at a time *on top of* clinical + risk
(`censpfs ~ IMWG_2024_HR + age + sex + ISS + ASCT + <immune_feature>`), which is the cleanest
incremental-value test.

**Early fusion (secondary / ablation).** Learn a small joint embedding of programs + immune
features before the hazard head. Tradeoff: early fusion can capture program×immune
interactions (e.g., a tumor program that only matters in an exhausted niche) but needs more
data and risks overfitting at N≈300 — so it is an *ablation*, not the headline.

**Pre-registered incremental-value claim (gate it):** the fusion model must beat the
**bulk-only** model on the **same patients** (identical CV folds, identical patients-with-
immune subset) by a bootstrap-CI-separated ΔC-index. Report ΔC-index with 95% bootstrap CI;
if the CI crosses 0, the claim is *not earned* (route through `governance/claim_gate.py`).

## 6. Non-redundancy argument

1. **Different compartment.** Immune features are computed on **non-plasma** cells; the 141
   mmSYGNAL programs are tumor-plasma-intrinsic. Different cells → different variance source.
2. **Empirical orthogonality test (cheap, do first).** Regress each immune feature on the 141
   programs (+ tumor purity); the **residual** variance is the genuinely new signal. Only
   features with substantial residual variance are kept. (This is a guardrail, not a result.)
3. **Mechanistic independence.** T-cell exhaustion/dysfunction is a property of the immune
   effector pool and predicts response to immune redirection — a mechanism bulk tumor RNA
   cannot see.

## 7. HONEST risk assessment — what could make Path A *not* beat 0.62

- **R1 — N collapse (biggest risk).** Only ~300–340 patients have single-cell immune data
  vs ~881 with bulk. Adding ~16 features to ~300 patients can *reduce* discrimination via
  variance even if the biology is real. Mitigation: late fusion + heavy regularization
  (EN-Cox), nested CV, and report on the immune subset *for both* arms.
- **R2 — Access is the gate, not the modeling.** The single-cell matrices are **request-
  gated (Zenodo)** and clinical PFS is **controlled (VLAB)**. If neither is granted, Path A
  cannot run at all here. This is a procurement task, not a code task.
- **R3 — Snapshot vs trajectory.** A single baseline immune snapshot may not capture the
  immune *dynamics* that actually gate relapse; cross-sectional immune state could be weakly
  prognostic for *long-horizon* PFS (Stone-minimax caveat already in CLAUDE.md §2).
- **R4 — Batch / site confounding.** The atlas needed Harmony + explicit `batch`/`site`
  covariates; immune proportions are sensitive to processing site. Any apparent lift must
  survive adjustment for `batch`/`site` (the atlas includes these in every model).
- **R5 — Residual redundancy.** If immune composition is itself downstream of tumor burden
  (e.g., plasma infiltration crowds out normal cells), the "immune" signal may partly re-encode
  tumor burden. The §6.2 residualization test must be passed *before* claiming non-redundancy.
- **R6 — Honest ceiling.** Even with everything right, the realistic prize is a **modest,
  CI-separated** ΔC-index (e.g., 0.62 → ~0.64–0.66), not a leaderboard leap. Path A's value is
  a *credible non-redundant-modality* result, consistent with this repo's parity-not-hype thesis.

## 8. Concrete next steps

1. Request the Zenodo object (14624955) **and** apply to MMRF VLAB for clinical metadata.
   Separately ask the authors for the **derived per-sample abundance table + per-patient
   signature scores** (Path 2 in §2 — much lighter).
2. Once the annotated object is on the user's machine, convert RDS→h5ad
   (`ExportH5AD()` helper) and run `derive_immune_features_from_anndata()` (this repo).
3. Residualize immune features on the 141 programs (§6.2); drop redundant ones.
4. Late-fuse; run the pre-registered incremental-value gate (§5) on the immune-subset cohort.
