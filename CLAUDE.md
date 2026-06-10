# CLAUDE.md — ResistanceMap (open-data edition)

> **Read this first.** This is the project memory + operating contract for any agent (Claude Code)
> editing this repo on the user's machine. It summarizes the research, the honest reframing, the
> current verified state, and the rules that keep this work publishable. Section 7 is binding.

## 0. One-line thesis
Interpretable, **censoring-honest** forecasting of multiple-myeloma (MM) progression on open-access
data, where the contribution is the *method* — PK-inverse observation + basin-escape hazard +
pre-registered claim-gating — demonstrated at honest **parity** with strong baselines, not a
leaderboard win.

## 1. The shift (why this repo exists)
This project ran ~20 iterations (v1–v20) that chased "beats SOTA": it accumulated fabricated /
unprovenanced metrics (e.g. AUROC 0.89 with no traceable run), a 40-task Airflow DAG of ceremony,
dead code, a mock inference path that skipped the model entirely, and "200-agent" churn. Repeated
audits kept finding the same thing: on ~700–900 patients the model **ties** predict-the-mean /
Ridge / mmSYGNAL. That is the *nature of the data* (low signal ceiling, small N), not a fixable bug.

The pivot to *meaningful* research was to stop optimizing a number that cannot move, and instead:
- reframe the problem honestly — **survival**, not classification; **IMWG-PFS** endpoint;
- make the **method** the contribution (measurement inversion + a mechanistic, interpretable hazard);
- **gate every claim** by pre-registered falsification (refuse unearned claims by construction);
- use **only open-access data**, and label every number as run-produced or a placeholder.

**Agent rule:** do not regress to the old failure mode. If a change would (re)introduce fabricated
metrics, a "beats-SOTA" claim, the 40-task DAG, agent theater, or a mock model path — do not make it.

## 2. Research framing (full detail: docs/FORMULATION.md)
- **Task type:** right-censored **time-to-event (survival)**. Target = the pair `(T, δ)` (observed
  time, event indicator) under IMWG-PFS. Landmark binary classification (progression-by-horizon
  6/12/18 mo) is a *secondary* reporting view only — never the training target.
- **Features:** clinical (age, sex, ISS, cytogenetics) + RNA **program scores** + **treatment
  regimen** (required: RRMM standard-of-care is shifting CAR-T/bispecifics earlier, so the hazard is
  treatment-dependent and labels drift over calendar time).
- **Core equation:** relapse = first-passage escape from a sensitive basin under Langevin dynamics;
  hazard ≈ Kramers rate, `log λ(x) ≈ const − ΔU(x)/D` — an interpretable physical analog of the Cox
  linear predictor `βᵀx`.
- **Assumptions/theorems:** proportional hazards (baseline only), non-informative censoring,
  one-compartment PK, manifold hypothesis, metastability (Kramers / Freidlin–Wentzell), optimal-
  transport snapshot→dynamics (Waddington-OT / MIOFlow), treatment-as-tilt, conformal coverage, and
  Stone's minimax bound (why we do NOT claim per-patient long-horizon forecasting).

## 3. Contribution (full detail: docs/CONTRIBUTION.md)
- **C1** PK-inverse observation — `models/pk_observation.py`
- **C2** Basin-escape hazard — `models/basin_sde.py`, `models/program_basin.py`
- **C3** Censoring-honest open-data benchmark — `survival/`
- **C4** Claim-gate governance — `governance/claim_gate.py`
Generality beyond MM: any indirect-biomarker, censored forecasting problem (CKD, heart failure,
transplant rejection). Closest SOTA anchor: **MIOFlow 2.0** (Krishnaswamy Lab, Yale; arXiv 2603.22564).

## 4. Repo map
```
src/resistancemap/
  data/      synthetic.py · gdc_clinical.py (REAL open loader) · gene_sets.py (MM_PROGRAMS) · scrna_qc.py (opt)
  survival/  baselines.py (Cox/EN-Cox/RSF/GBS, feature-inferred) · splits.py (patient k-fold) ·
             metrics.py (C-index/IBS/td-AUC/bootstrap) · competing_risk.py · calibration.py (ECE/conformal)
  models/    pk_observation.py · basin_sde.py · program_basin.py (NOVEL) · scvi_latent.py (opt)
  viz/       phate.py (opt)      governance/ claim_gate.py      pipeline.py (stage runner = DAG replacement)
scripts/     download_gdc_open.sh · build_commpass_dataset.py · run_all.py · run_baselines.py
docs/        CONTRIBUTION · FORMULATION · RESEARCH_LANDSCAPE · MIGRATION · CLAUDE.md (this file)
literature/  PROTOCOL (SPAR-4-SLR) · references.csv (18) · PRISMA.md       paper/ SKELETON.md
comparators/ MIOFLOW.md       tests/ test_modules·test_data·test_survival·test_models       results/
```
- **Reused, not rewritten:** the `scripts/` are thin glue over `data/`, `survival/`, `models/`, `governance/`, `pipeline.py`.
- **Optional (single-cell route; run on a GPU box):** `scrna_qc.py`, `scvi_latent.py`, `viz/phate.py`, MIOFlow.
- **Redundant — safe to delete:** `data/gdc_open.py` (old stub loader, superseded by `gdc_clinical.py` + `build_commpass_dataset.py`).

## 5. Current verified state (SYNTHETIC; labelled — no biological meaning)
- 5-fold patient-disjoint CV: `cox_ph` 0.678, `discrete_time_hazard` 0.677; IBS ~0.11–0.12; td-AUC ~0.76; ECE ~0.07–0.08.
- `run_all` 3-fold: cox 0.677 / EN-Cox 0.675 / RSF 0.664 / GBS 0.645 / **program_basin (NOVEL) 0.637**.
- PK-inverse recovery R²=1.000; Kramers ↔ Monte-Carlo consistent; risk → hazard monotone.
- Governance: PK + mechanistic gates GRANTED; **`novel_model_beats_baselines` BLOCKED** (correct — this is parity).
- All `.py` compile; all 4 test suites green; literature = 18 papers + PRISMA (165 identified → 18 included).
- **PENDING (user's machine):** real GDC-open run; scVI/MIOFlow (need torch); the Gateway lab/PK layer.

## 6. How to run
```
pip install -e .                                  # core; add .[omics,deep] for scVI/torch
python scripts/run_all.py                         # synthetic pipeline: validate -> CV -> governance
bash scripts/download_gdc_open.sh                 # REAL: GDC-open RNA-seq + clinical (needs aws cli + requests)
python scripts/build_commpass_dataset.py          # -> data/processed/commpass.csv
python scripts/run_all.py --data data/processed/commpass.csv   # REAL run
pytest tests/ -q                                  # or: PYTHONPATH=src python tests/test_*.py
```

## 7. GUARDRAILS (binding — do not violate when editing)
1. **No fabricated metrics.** Every number comes from a run in this repo or is labelled a placeholder. Never hardcode results into docs, README, or code.
2. **"Beats baselines" stays BLOCKED** until `results/cv_governance.json` shows it GRANTED on REAL data (novel-model CV C-index CI-lower > best baseline). Parity is an acceptable, honest outcome — do not force a win.
3. **Survival discipline:** IMWG-PFS endpoint (progression OR death); count *observed events*, not rows; patient-disjoint CV; censoring-aware metrics. Never use time-to-second-line as the target.
4. **Open data only** for anything submittable. MMRF Researcher Gateway labs are gated (no redistribution; 30-day pre-publication notice) — keep the PK/lab layer OFF by default.
5. **Do not reintroduce** the 40-task Airflow DAG, agent-orchestration theater, the mock inference path, or dead experimental modules. The pipeline is `pipeline.py` (a small, legible stage runner).
6. **cell-line ≠ patient; IC50 ≠ clinical resistance.** Synthetic data carries no biological weight by construction; never report synthetic numbers as findings.
7. **Reuse, don't reinvent.** Extend existing modules; prefer open libraries (scvi-tools, scikit-survival, lifelines, phate, MIOFlow) over in-house rewrites.
8. **Report to TRIPOD+AI** standard; keep `STATUS.md` and the claim-gate honest and current.

## 8. Path to a publishable paper (ordered milestones)
1. **Real GDC-open run** → real baseline table + program-score model (`scripts/run_all.py --data ...`).
2. Report CV **C-index / IBS / time-dependent AUC** + **calibration (ECE)** + conformal coverage on real data.
3. Run the **governance gate on real data**; record GRANTED/BLOCKED honestly (parity is fine).
4. **Fill the `[RUN]` slots** in `paper/SKELETON.md` from `results/*.json`.
5. **Finalize the SLR** (`literature/PROTOCOL.md` + `references.csv` + `PRISMA.md`) for Related Work.
6. *(Stretch)* **single-cell route**: scVI latent + MIOFlow 2.0 comparator + PHATE/T-PHATE figures.
7. *(Stretch)* **external-cohort generalization** check; treatment-conditioned + per-cytogenetic-stratum reporting.

**Framing for the paper:** a *method + honest open-data benchmark* contribution (measurement inversion +
mechanistic interpretable hazard + calibration + pre-registered falsification), reaching parity through a
legible route — explicitly NOT a SOTA-beating claim. Target a computational-oncology / ML-for-health venue
or workshop.

## 9. See also
`docs/CONTRIBUTION.md` · `docs/FORMULATION.md` · `docs/RESEARCH_LANDSCAPE.md` · `docs/MIGRATION.md` ·
`paper/SKELETON.md` · `literature/PROTOCOL.md` · `STATUS.md`
