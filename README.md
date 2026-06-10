# ResistanceMap (open-data edition)

Interpretable, **censoring-honest** forecasting of multiple-myeloma progression — built only on
open-access data and pre-registered falsification.

> **Status:** clean rewrite. Runnable baselines + novel methods (synthetic smoke test passes).
> Real numbers come from the GDC-open CoMMpass run on your machine. **No fabricated metrics** —
> every number is produced by code in this repo or labelled a placeholder.

## What's novel (honest, parity-not-leaderboard)
1. **PK-inverse observation** — treat a serum biomarker as the half-life-blurred *shadow* of a
   latent production signal and **deconvolve** it (`models/pk_observation.py`).
2. **Basin-escape hazard** — relapse as first-passage in a double-well quasi-potential; the
   Kramers rate is an interpretable, covariate-driven hazard (`models/basin_sde.py`).
3. **Censoring-honest open-data benchmark** — Cox / EN-Cox / RSF / GBS, patient-disjoint CV,
   bootstrap CIs (`survival/`).
4. **Claim-gate governance** — pre-registered falsification that refuses unearned claims by
   construction (`governance/claim_gate.py`).

See `docs/CONTRIBUTION.md` for the full research design, math, and data plan.

## Data
| Path | Status | Use |
|---|---|---|
| GDC/AWS open CoMMpass (`gdc-mmrf-commpass-phs000748-2-open`) | open now | **primary** (expression + clinical outcomes) |
| MMRF Researcher Gateway labs | gated, no redistribution | optional PK/lab layer (off by default) |
| electricsheepafrica synthetic (CC-BY) | published | demo/portability only — **no scientific claims** |

## Quickstart
```bash
pip install -e .            # core deps
python scripts/run_baselines.py            # synthetic smoke test
python scripts/run_baselines.py --data /path/to/commpass.parquet   # real run
python tests/test_modules.py               # PK + basin + claim-gate checks
```

## Layout
```
src/resistancemap/{data,survival,models,governance}/   pipeline.py
scripts/   literature/   docs/   paper/   results/   tests/
```
License: MIT.
