# MIOFlow 2.0 — trajectory-inference comparator

## What it is

**MIOFlow** (Manifold Interpolating Optimal-transport Flows) is a generative,
dynamics-learning method from the **Krishnaswamy Lab (Yale)** for **inferring
continuous population dynamics from *static* snapshots** of high-dimensional data
(e.g. single-cell measurements taken at a few discrete timepoints, where each cell is
destroyed at measurement and so cannot be tracked over time).

**MIOFlow 2.0** (arXiv:2603.22564) combines three ingredients:

1. **Neural SDE / neural-ODE flow** — a learned drift (and diffusion) field that
   transports the population's probability mass continuously through a latent space.
2. **Optimal transport (OT)** — the flow is trained so that, when the empirical
   distribution at timepoint *t_i* is pushed forward to *t_{i+1}*, it matches the
   observed snapshot under an OT (Wasserstein) objective, with manifold-respecting
   regularisation so trajectories stay on the data manifold.
3. **PHATE latent geometry** — embeddings/geodesics from **PHATE** (`viz/phate.py`)
   define the manifold on which the flow is interpolated, so the learned dynamics
   follow the intrinsic geometry rather than straight Euclidean lines.

The output is an **inferred continuous-time stochastic dynamical system**: per-cell
trajectories, an interpolated population density at any intermediate time, and the
learned drift/diffusion (velocity) field.

## Inputs / outputs (as a comparator in this repo)

**Inputs**
- A **latent representation** of the data — produced here by
  `models/scvi_latent.scvi_latent(adata, ...)` (scVI, needs torch) or the
  numpy/sklearn fallback `models/scvi_latent.pca_latent(X, k=...)`, optionally
  further embedded with `viz/phate.phate_embed(X, ...)`.
- **Snapshot / timepoint labels** — an integer/float timepoint per sample assigning
  each observation to a discrete collection time (the snapshots OT is matched across).

**Outputs**
- Inferred **population dynamics**: continuous-time trajectories, interpolated
  intermediate distributions, and the learned **drift + diffusion field** over the
  latent manifold.

## Install / run (on a torch-capable machine — NOT this environment)

MIOFlow depends on PyTorch and `torchsde`/`torchdiffeq`, which are intentionally
**not installed here**. On a GPU/torch machine:

```bash
# Reference implementation (Krishnaswamy Lab)
git clone https://github.com/KrishnaswamyLab/MIOFlow
cd MIOFlow
pip install -e .            # pulls torch, torchsde/torchdiffeq, POT, phate, scprep
```

Sketch of a run, wired to this repo's latent + snapshot labels:

```python
import numpy as np
from resistancemap.models.scvi_latent import pca_latent          # or scvi_latent(adata)
from resistancemap.viz.phate import phate_embed

# 1. latent space (use scVI on a torch box; pca_latent runs anywhere)
Z   = pca_latent(X, k=10)                 # (n_cells, 10)
emb = phate_embed(Z, n_components=5)       # PHATE manifold coords MIOFlow interpolates on

# 2. snapshot/timepoint label per cell (discrete collection times)
tp  = snapshot_timepoints                  # e.g. np.array([0,0,1,1,2,...])

# 3. fit the neural-SDE + OT flow on (emb, tp), then sample inferred dynamics
#    (MIOFlow API: build geodesic OT loss on PHATE geometry, train the SDE,
#     then integrate to get trajectories / interpolated densities / velocity field)
```

(The exact class/function names track the upstream repo; the contract is
*latent embedding + snapshot labels in, learned population dynamics out*.)

## Relation to our `basin_sde` Kramers hazard

Both describe a population evolving under a **stochastic differential equation**, but
at different levels of structure:

| | **MIOFlow 2.0** | **`basin_sde` (this repo)** |
|---|---|---|
| Dynamics | **General, learned** drift/diffusion field (neural SDE) | **Structured, fixed-form** overdamped Langevin in a double-well quasi-potential `U(z)=z⁴/4 − z²/2 + tilt·z` |
| Fit to | Static snapshots via optimal transport | Survival (time-to-event) data; tilt driven by covariates/program scores |
| Output | Full inferred trajectories + velocity field | A single **first-passage / Kramers escape hazard** (sensitive well → resistant well) |
| Interpretability | Flexible but black-box velocity field | Mechanistic: barrier height, curvatures, escape rate |

Conceptually, **`basin_sde` is a structured, survival-coupled special case** of the
general learned dynamics MIOFlow infers. MIOFlow learns *whatever* SDE best transports
the snapshots; `basin_sde` *posits* a specific bistable landscape and reads relapse
off as thermally-activated **basin escape**, with the barrier tilted by patient risk
(see `models/program_basin.py`: program scores → tilt → Kramers hazard). MIOFlow is
therefore the natural **trajectory-inference comparator**: it can validate, from
snapshot geometry, whether the population flow looks like escape from a sensitive
attractor — the qualitative picture `basin_sde` encodes by construction.
