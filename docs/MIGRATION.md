# Migration: v20 -> clean repo. What survives, what dies, what to do next.

## 0. Verdict — was v1–v20 useless?
**No.** Three layers survive; only the plumbing and the fabricated numbers die.
1. **Components** — several v20 modules are already correct and should be *ported as-is*.
2. **Knowledge** — the empirical lessons (below) are the spine of an honest paper.
3. **Concept** — PK-inverse observation + basin-escape hazard + claim-gate *is* the contribution.

Today's clean repo is a fresh re-implementation of the **ideas** (so it actually runs and has no
dead code). The plan is **merge**: clean spine + governance + honest framing, *plus* the
battle-tested v20 modules ported in. Not a throwaway of either side.

---

## 1. PORT from v20 -> clean repo  (verify each compiles when you move it)
| v20 file (per the v20 audit) | Status in v20 | Move to | How |
|---|---|---|---|
| `mortfm/survival/time_to_event_metrics.py` (C-index, IBS, KM strata) | correct | `survival/metrics.py` | merge: add IBS + time-dependent AUC to today's metrics |
| `mortfm/survival/competing_risk_head.py` (discrete-time hazard, censoring) | sound | `survival/competing_risk.py` | port as-is; it's the relapse/death/NRM head |
| `mortfm/survival/survival_calibration.py` (temperature scaling) | standard | `survival/calibration.py` | port; add ECE reporting |
| `mortfm/survival/baselines.py::ClinicalOnlyCox` | correct | `survival/baselines.py` | today's version is richer; keep today's |
| `data/splits.py` (patient-disjoint GroupKFold) | correct | `survival/splits.py` (new) | port; replace today's inline shuffle split |
| IMWG-PFS endpoint logic in `spark_jobs/mortfm_lakehouse.py` | the real Bug#2 fix | `data/gdc_open.py` | port the **definition** only (progression = PD or death; censor at last follow-up). Drop the Spark. |
| `longitudinal/censoring_policy.py`, `temporal_splitter.py` | valid | `data/` (when lab layer turns on) | port when Gateway access lands |
| `models/trajectory.py::ChromatinODE` (Sneppen-Ringrose) | high value, had NeuralJumpSDE double-negation bug | `models/basin_sde.py` | reference for a learned 2-state chromatin basin; fix the bug if you port |
| `mortfm/trajectory/graph_energy_sde.py::WaddingtonPotential` | high value | `models/basin_sde.py` | today's basin_sde is the clean reimplementation; lift any basin-geometry specifics you want |

## 2. PORT LATER (genuinely novel but was DEAD/unverified in v20)
Keep these as reference; wire only if a real result needs them.
- `models/identifiable_ode.py` (Lyapunov certificates, causal drug intervention) — highest novelty, never in pipeline.
- `models/causal_fusion.py` (SCM layers, do-calculus).
- `models/disentangled_vae.py` (biological-anchor dims).

## 3. DROP (do not port)
- The 40-task **Airflow DAG** + agent orchestrator + zero-trust hash chain (ceremony; replaced by `pipeline.py`).
- The L0–L5 **mock inference path** that truncated proteomics to 128-D and skipped L1–L4.
- The broken `GradientReversalLayer` (nn.Module.backward no-op) and un-stabilised `SinkhornOT`.
- `~66%` dead `data/` (duplicate `mmrf_loader`, island `geo_*` loaders) and the Phase-3 exports in `models/__init__`.
- Every **AUROC/AUPRC/MAE with no provenance** (0.89/0.84 etc.) — these are the fabricated numbers that sink submissions.
- AgentOps/optimizer/scheduler/MCP-connector stubs (`NotImplementedError`).

## 4. KNOWLEDGE LEDGER — the v1–v20 lessons that shape the paper
1. **Parity ceiling.** On ~700–900 patients the model **ties** predict-mean / Ridge / mmSYGNAL. Frame as *parity + interpretability*, never "beats SOTA".
2. **Endpoint.** Use **IMWG PFS** (progression or death), not time-to-second-line (TT2L conflates clinical decisions).
3. **Event gate.** Count **observed events** (`event==1`), not all supervised rows.
4. **RSF must be censoring-aware** (`sksurv.RandomSurvivalForest`), never `RandomForestRegressor` on event time.
5. **Data access.** Gateway labs are gated + no-redistribution + 30-day notice -> **open tier only** for a today-submission.
6. **Fabricated metrics sink it** -> the **claim-gate** exists for exactly this.
7. **Patient-disjoint CV** is mandatory (leakage = instant reject).
8. **Identifiability bound.** Per-patient long-horizon forecasting is information-limited at small N -> don't claim it.
9. **Generality bound.** MM proteasome biology did **not** transfer to AML -> bound any "pan-heme" claim.
10. **Signal attribution.** Most survival signal came from the **mmSYGNAL** feature, not the deep net -> decompose contributions honestly in an ablation.

## 5. DAG changes (40 -> 7)
Replace the 40-task Airflow DAG with the linear stage list below (already runnable via `pipeline.py`).
If you still want Airflow for scheduling/retries, it's a **7-task** DAG, one task per stage:
```
data_validate -> data_prep(features+program_scores) -> split(patient-disjoint)
  -> baselines -> model(program-score + basin) -> evaluate(C-index/IBS/ECE) -> governance(claim-gate)
```
No agent orchestration, no hash-chain, no 33-stage sprawl.

---

## 6. FULL FILE MANIFEST for the new repo
### Already built (runs today)
```
README.md  LICENSE  pyproject.toml  .gitignore  STATUS.md  configs/default.yaml
docs/CONTRIBUTION.md  docs/MIGRATION.md
literature/PROTOCOL.md  literature/search_strings.md  literature/references.csv
paper/SKELETON.md
src/resistancemap/__init__.py  pipeline.py
src/resistancemap/data/{__init__,synthetic,gdc_open}.py
src/resistancemap/survival/{__init__,baselines,metrics}.py
src/resistancemap/models/{__init__,pk_observation,basin_sde}.py
src/resistancemap/governance/{__init__,claim_gate}.py
scripts/run_baselines.py  tests/test_modules.py
results/{baseline_smoketest,governance_report}.json
```
### To ADD (the gap between "foundation" and "full submission")
```
src/resistancemap/data/gdc_clinical.py     # GDC API pull of open clinical (vital_status, days_*, ISS, age, sex)
src/resistancemap/data/gene_sets.py        # named program gene-sets (proliferation, mmSYGNAL-style) -> program_scores
src/resistancemap/survival/splits.py       # ported patient-disjoint GroupKFold (k-fold, not single split)
src/resistancemap/survival/competing_risk.py  # ported discrete-time hazard head
src/resistancemap/survival/calibration.py  # ported temperature scaling + ECE
src/resistancemap/models/program_basin.py  # the novel model: program scores -> risk tilt -> basin hazard
scripts/download_gdc_open.sh               # one-command S3 + GDC clinical pull
scripts/run_all.sh                         # data_validate..governance in order
scripts/build_literature.py               # runs PubMed/biorxiv searches -> fills references.csv + PRISMA counts
tests/test_baselines.py  tests/test_data.py
CITATION.cff  environment.yml (or uv.lock)
```

## 7. Ordered next steps
0. **Rotate** the leaked GitHub token + 2 API keys (do this first).
1. **Finish the GDC parser** (`gdc_clinical.py` + `gene_sets.py`) so the survival frame builds from open data in one command. *(I can do this next.)*
2. **Run real baselines** on GDC-open -> your real numbers-to-beat table.
3. **Port** competing-risk head + calibration; add **ECE**.
4. **Train** the program-score + basin model; only now may `novel_model_beats_baselines` even be tested (and it may stay BLOCKED — that's fine, parity is the honest claim).
5. **Populate** `literature/references.csv` via real searches; fill PRISMA counts.
6. **Fill the [RUN] slots** in `paper/SKELETON.md` with step-2/4 outputs.
7. **(When Gateway access lands)** wire the PK/lab trajectory layer + mechanism decomposition.

---

## 8. Single-cell layer additions (if you take the scRNA route)
Keep the folder structure flat & simple (scNotebooks / Variant-Prediction style). Add:
```
src/resistancemap/data/scrna_qc.py     # SoupX/Scrublet-style QC + scanpy normalize (port r3)
src/resistancemap/models/scvi_latent.py# scVI latent (optional [omics] extra)
src/resistancemap/viz/phate.py         # PHATE / T-PHATE manifold figures
notebooks/01_qc.ipynb 02_latent.ipynb 03_trajectory.ipynb  # scNotebooks-style, runnable teaching path
comparators/MIOFLOW.md                 # how to run MIOFlow 2.0 as the trajectory comparator
```
These are **optional**: the open-data survival core (clinical + RNA program scores) stands alone.
Add the scRNA layer only if you go single-cell with MM-SCATLAS / GEO data.
