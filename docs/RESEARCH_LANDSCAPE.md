# Research landscape (current, June 2026) + tool-fit + how your prior work plugs in

## 1. Forefront methods most relevant to YOUR idea
- **Krishnaswamy Lab (Yale) — MIOFlow / MIOFlow 2.0** (arXiv 2603.22564, Mar 2026; orig. *Nature
  Methods* 2023 / arXiv 2206.14928). Neural-SDE + optimal transport + PHATE-distance latent to infer
  **stochastic cellular dynamics from static snapshots**, now incl. spatial + growth-rate.
  **Most relevant SOTA anchor.** Position your basin-escape hazard as a structured, survival-coupled
  special case; use MIOFlow 2.0 as comparator/engine. Use **PHATE / T-PHATE** for latent geometry + figures.
  (Open source: github.com/KrishnaswamyLab.)
- **ScientistOne — Chain-of-Evidence** (scientist-one.github.io): Problem Investigator -> Discovery
  Engine -> Paper Writer + **Claim Verifier** that checks every claim against its evidence source.
  Your `governance/claim_gate.py` is a domain instance of this — cite it as the methodological frame
  for evidence-chained claims (this is the "be like that paper" you asked for, in your field).

## 2. MM single-cell drug-resistance frontier (related work / comparators)
- **DrugFormer** (PMC11516065): graph-augmented LLM predicting drug resistance **at single-cell
  level**; pseudotime trajectory reveals resistant states. Closest single-cell comparator.
- **CancerFoundation** (bioRxiv 2024.11.01.621087): scRNA foundation model for drug resistance.
- **WashU/Emory/Harvard + MMRF bone-marrow immune atlas** (*Nature Cancer*): immune microenvironment
  -> survival / relapse prediction.
- **PTPRG-driven stemness + treatment-failure signatures** (PMC12488599, 2025): malignant-PC
  subcluster enriched in resistance (E2F8/FOXM1/E2F1) -> concrete program-score features for you.
- Spatial multiomics MM platforms (NK-myeloma interactions) -> microenvironment features.

## 3. Training / best-practice to adopt
- **scNotebooks** (*Nature Genetics* 2026; Rojas-Hidalgo et al.; integrativebioinformatics.github.io/scNotebooks):
  adopt their QC -> normalize -> integrate -> annotate -> trajectory conventions for the scRNA layer.
- **MPOP bioinformatics tutorials** + Galaxy single-cell community: pipeline patterns for reproducibility.

## 4. Tool-fit verdicts (you said "if need be it" — here is the honest call)
| Tool | Verdict | Why |
|---|---|---|
| scVI / scanpy | **USE** | standard scRNA latent + QC (per scNotebooks); your r3 already does this |
| PHATE / T-PHATE | **USE** (viz + latent geometry) | interpretable manifold; ties to Krishnaswamy; lightweight |
| MIOFlow / MIOFlow 2.0 | **USE** (comparator/engine) | SOTA snapshot->dynamics; directly relevant |
| PLINK | **MOSTLY NO** | germline GWAS/SNP tool; our signal is somatic expression + clinical. Only justified for an **ancestry/fairness** sub-analysis, not the core |
| DrugFormer / CancerFoundation | **COMPARE** | single-cell resistance baselines if you take the scRNA route |

## 5. Your own repos -> where each plugs in
| Repo | Plugs into |
|---|---|
| MM-SCATLAS | scRNA atlas -> cell-state definitions + program scores |
| MyeloMemory | Sneppen-Ringrose bistability + reversibility head -> **basin_sde** parameterization (HIGH value) |
| r1 | clinical temporal features (M-protein / FLC kinetics, SLiM-CRAB) -> clinical encoder + PK anchoring |
| r2 | GSVA/ssGSEA + DANN harmonization -> program_scores + multi-site robustness |
| r3 | scRNA QC (SoupX/Scrublet/Harmony/scVI) -> data/scrna QC (aligns with scNotebooks) |
| r4 | proteomics preprocessing (ComBat/UniProt/kNN) -> pending proteomic layer |
| r5 | imaging/radiomics -> out-of-scope for the core model (optional auxiliary) |
| ResistanceMap (standalone) | prior pipeline -> mine the correct survival modules; superseded by the clean repo |

## 6. Could NOT verify
- **arXiv 2605.10196** does not resolve (a future / May-2026 ID; closest real hit `2405.10196` is
  unrelated QCD physics). Paste the **title** or the corrected ID and I'll fold it in.

## Sources
- MIOFlow 2.0 — https://arxiv.org/abs/2603.22564 ; MIOFlow — https://pmc.ncbi.nlm.nih.gov/articles/PMC10312391/ ; Krishnaswamy projects — https://krishnaswamylab.org/projects
- ScientistOne — https://scientist-one.github.io/
- DrugFormer — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11516065/ ; CancerFoundation — https://www.biorxiv.org/content/10.1101/2024.11.01.621087.full.pdf
- MMRF/WashU immune atlas — https://www.genengnews.com/topics/cancer/bone-marrow-immune-cell-map-boosts-survival-relapse-prediction-in-multiple-myeloma/
- PTPRG stemness / treatment failure — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12488599/
- scNotebooks — https://www.nature.com/articles/s41588-026-02584-0 ; site — https://integrativebioinformatics.github.io/scNotebooks/
