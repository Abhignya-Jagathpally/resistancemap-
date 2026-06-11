# Theory loop — applying control theory / dynamical systems to MM progression

Goal: a simple, mathematically-engineered, theory-grounded survival method (ICLR/ICML-style methods
contribution) that reaches parity with — or honestly beats — SOTA (gep70/sky92) on MM PFS, with the
claim-gate, not wishful tuning, deciding. Each entry: hypothesis + math, result, honest verdict, next.
**Discipline: a surprising positive is treated as guilty-until-proven-innocent; nothing is claimed
until adversarially verified.**

---

## Iteration 1 — first-order LTI treatment-state (controllability energy)  [UNVERIFIED]
**Hypothesis / math.** Disease burden as a first-order LTI system `dx/dt = −a·x + u(t)`, control
input `u(t)` = treatment-intensity impulses (drug-class count per line, from open `treatments.tsv`).
State at landmark L = impulse response / discounted controllability energy
`x(L;a) = Σ_{lines started ≤ L} intensity·exp(−a(L−t)/365)`. The time-constant `a` (system id) is
selected by **inner CV** (system-identification flavor). Targeted ICLR/ICML property: an
interpretable, identifiable latent state with a controllability meaning.
**Design.** Immortal-time-safe landmark (keep `duration>L`, re-origin), nested CV (inner picks
`a`,penalizer; outer scored once), forward td-AUC; SOTA reference = forward td-AUC of gep70/sky92.
**Result (`results/theory_loop/iter1.json`).** L=180: base Cox(ISS) td-AUC 0.592 → +LTI **0.663**
(Δ=+0.072, 95% CI [+0.017, +0.125]); SOTA ref gep70 0.632 / sky92 0.644; inner-selected a*≈2–4/yr.
L=365: **NaN (bug)**.
**Honest verdict: UNVERIFIED — DO NOT CLAIM.** Two red flags: (1) +0.072 from one treatment feature
**exceeds the proven ~0.62 static ceiling** → likely residual **immortal-time / reverse-causation
leakage** (early progressors intensify treatment earlier, even within the landmark window); (2) the
**weak base (ISS-only, 0.592)** inflates the apparent margin; (3) **L=365 grid bug** (forward
horizons below the landmark → negative forward time). A clean static ceiling does not jump 0.07 from
one covariate without leakage — extraordinary claims need adversarial proof.
**Next (Iteration 2 = adversarial verification of Iter 1 before any new idea):**
1. Fix the forward-horizon grid (horizons strictly > L) and re-run L=365/730.
2. **Negative-control feature**: replace the real LTI state with a degree/intensity-matched *shuffled*
   treatment input; if the "win" survives, it's leakage. Permutation test the LTI Δ.
3. **Strong-baseline gate**: does LTI add over Cox(ISS + gep70) (not just ISS)? Beating a weak base is
   not beating SOTA.
4. Inspect what `x(L;a)` correlates with (early events?) and whether high a* (fast decay) just flags
   recently-intensified = already-progressing patients.
Only if the effect survives all four does it become a candidate claim; otherwise record as leakage and
move to a genuinely different theory (Kalman state-space estimator or switched-system).

---

## Candidate mechanism backlog (prioritized, honest fit-triage) — added 2026-06-10
User asked to broaden beyond control theory: diffusion, RLVR, spatio-temporal, ego-centric, similar.
Triaged by (theory fit × open-data availability × ICLR/ICML relevance). The loop pulls from the top.

**Tier 1 — test on the OPEN substrate, high fit (do these after Iter-2 verification):**
- **Score-based / denoising DIFFUSION SDE on disease-state latents.** The existing basin-escape model
  IS overdamped Langevin `dz=−∇U(z)dt+σdW`; learn the score ∇log p_t(z) with treatment-conditioned
  drift; survival = first-passage of the diffusion into a resistance basin. Property: proper SDE /
  Fokker–Planck grounding, generative + interpretable. Open data (mmSYGNAL latents + treatments.tsv).
- **Neural CONTROLLED differential equation (Neural CDE) / signature features.** CDEs are control
  theory for irregular time series: `dz = f_θ(z) dX(t)` driven by the treatment path X(t)=u(t). Exactly
  the treatment-as-control formulation; path-signature features are a principled, fixed alternative.
  Property: universal approximation for path-functionals; handles irregular sampling. Open data.
- **Selective state-space (SSM / Mamba-style) hazard.** A modern, identifiable realization of the
  linear state-space framing (Iter-1) with input-dependent dynamics. Property: linear-time, stable SSM.

**Tier 2 — open data, moderate fit:**
- **Hawkes / self-exciting point process** for the progression↔treatment feedback (relapse begets
  line-switch begets relapse). Property: branching-process intensity; mechanistic for the non-PH.
- **Ego-centric patient-similarity graph survival.** kNN ego-graph on the 141-program manifold (reuse
  the PHATE embedding); graph-regularized / transductive Cox or a tiny GNN. Caveat: same features →
  ceiling-limited; the contribution is the transductive/manifold angle, not new signal.

**Tier 3 — needs data we do NOT have on the open substrate (DEFER, do not thrash):**
- **RL / RLVR (treatment-policy track, not the prediction metric).** Fits optimal dynamic-treatment-
  regime / counterfactual; but observational N~700, no exploration → OPE high-variance; "verifiable
  reward" = the censored outcome is the hard part. Scope as a SEPARATE counterfactual track with OPE
  safety gates (legacy repo has IQL/CQL/OPE), NOT as a PFS-discrimination claim.
- **Spatio-temporal.** Bulk transcriptome has no spatial axis. Requires immune-atlas scRNA (Zenodo/VLAB
  gated = Path A) or SegPC plasma-cell imaging (open Kaggle) — a separate imaging lane. Defer to when
  that data lands; do not fake a spatial model on bulk data.

**Selection rule for the loop:** prefer the Tier-1 mechanism with the cleanest mathematical property
that a nested-CV ablation rewards; never adopt a Tier-3 mechanism on data that cannot support it.

---

## Iteration 2 — adversarial verification of Iter-1 LTI  [Iter-1 REFUTED]
**Did:** fixed the forward-grid bug (horizons now days-AFTER-L), re-ran L=180/365/730 under nested CV,
added a STRONG-baseline gate (Cox(ISS+gep70)), a NEGATIVE-CONTROL (shuffle treatment→patient mapping,
20 reps), and outcome-correlation diagnostics. `results/theory_loop/iter2.json`.
**Result.**
- Weak base: L=180 ISS 0.591 → +x 0.642 (Δ+0.051, CI [+0.011,+0.089]) — adds over ISS-only.
- **STRONG base (the real test): L=180 ISS+gep70 0.659 → +x 0.673, Δ+0.014 CI [−0.005,+0.031] (crosses 0).**
  L=365 Δ−0.002 (ns); L=730 Δ−0.017 (CI [−0.039,−0.001], significantly WORSE).
- Negative control @L=180: real ISS+x 0.642 vs shuffled-treatment null mean 0.606 (max 0.660), p=0.050.
- Diagnostic: Spearman(x, forward-time) = +0.30 @L=180 (not a clean monotone risk feature).
**Verdict: Iter-1 win REFUTED.** The control-theoretic LTI feature does NOT CI-separate over a strong
baseline at any landmark (and is significantly worse at 730 d); only a marginal (p=0.05) sliver of real
signal survives the negative control. The Iter-1 +0.072 was an artifact of the grid bug + a weak
ISS-only base + gep70-redundant signal. Gate `beats_SOTA` stays **BLOCKED**. Honest negative result —
the discipline caught the false positive. **Switch theory.**
**Next (Iteration 3):** Tier-1 #1 — **score-based / denoising DIFFUSION SDE on disease-state latents**
(the basin/Langevin connection `dz=−∇U(z)dt+σdW`), survival = first-passage into a resistance basin,
treatment-conditioned drift. Evaluate honestly vs gep70/sky92 + Cox(ISS+gep70) with the same nested-CV,
immortal-time-safe, guilty-until-proven-innocent protocol.

---

## Iteration 3 — score-based diffusion SDE on program latents  [honest parity-below, BLOCKED]
**Did:** PCA(16) of the 141 programs → latent z; learned a potential U_θ(z) by denoising score matching
(score s_θ=−∇U_θ); risk = inner-CV-selected energy U_θ(z) or ‖∇U_θ‖. 5-fold patient CV.
`results/theory_loop/iter3.json`.
**Result:** diffusion-energy C-index **0.570 [0.537, 0.603]** — above chance but BELOW SOTA (gep70
0.624 / sky92 0.620); strong base ISS+gep70 0.653 → +diffusion 0.654 (**adds nothing**). Risk-type
"energy+" selected every fold (atypical/low-density transcriptional states = higher risk — mild,
interpretable, but not a SOTA-beater).
**Verdict:** honest parity-below; the density-geometry of the *static* program latents carries only
weak prognostic signal and does not break the ~0.62 ceiling — **as predicted for the static regime**.
Gate BLOCKED. This is the **fifth** independent confirmation of the static ceiling (Cox, RSF, GBS,
ResistanceBasin-LR, diffusion-energy). **Switch theory.**
**Next (Iteration 4):** Tier-1 #2 — **Neural Controlled Differential Equation (Neural CDE) / path
signature**, back in the treatment-conditioned regime (the only place Lane #2 found signal). Math:
latent path dz = f_θ(z) dX(t) driven by the treatment control path X(t)=u(t) (regimen/line/switch
events); hazard from z(L). Immortal-time-safe landmark; same guilty-until-proven-innocent gauntlet.

---

## Iteration 4 — Neural-CDE / path-SIGNATURE of the treatment control path  [honest negative, BLOCKED]
**Did:** built treatment control path X(t)=(t, cum_PI, cum_IMiD, cum_CD38) up to L; truncated signature
S(X)^{≤2}; tested whether signature (and specifically LEVEL-2 = treatment ORDERING) adds over the
strong baseline Cox(ISS+gep70). Immortal-time-safe landmark, nested CV. `results/theory_loop/iter4.json`.
**Result (L=180, n=648/270 ev — the powered landmark):** base 0.660 → +sig1 0.662 → +sig12 0.663.
Δsignature-over-base **+0.004 [−0.022, +0.032]** (no add); Δordering (level-2 over level-1)
**+0.002 [−0.021, +0.027]** — **treatment ORDERING is not prognostic** for PFS here. (L=365: 20-feature
signature Cox did not converge → NaN; robustness limitation, reported honestly.)
**Verdict:** honest negative — the CDE/path-signature framing, including its novel non-commutative
ordering content, adds nothing over Cox(ISS+gep70). Gate BLOCKED.

### Emerging meta-result (after 4 iterations + Lanes B/C/D)
No mechanism beats the strong baseline / published signatures: **static** (Cox, RSF, GBS, log-rank
basin, diffusion-energy) all ~0.62–0.66; **treatment-conditioned** (LTI = leak/refuted, time-varying
Cox = marginal/non-robust, CDE-signature = no add). The MM-PFS ceiling holds across control-theory,
diffusion, CDE/signature, and clustering method families. **Convergence criterion:** run ≤2 more
distinct families (selective SSM; then Hawkes OR ego-centric patient-graph); if they also tie, STOP
adding mechanisms and synthesize the *multi-mechanism ceiling* as the contribution (a rigorous,
method-family-spanning negative + the honest method scaffold + the access-gated immune-fusion path).
**Next (Iteration 5):** Tier-1 #3 — selective state-space (SSM) hazard with input-dependent dynamics.

---

## Iteration 5 — selective diagonal SSM over treatment events  [honest negative, BLOCKED]
**Did:** diagonal selective SSM `z_k=exp(−a·Δ_k)z_{k-1}+ι_k` (4 timescales + fast−slow velocity) over
irregular treatment events to landmark L; tested vs strong baseline Cox(ISS+gep70). Nested CV,
immortal-time-safe. `results/theory_loop/iter5.json`.
**Result:** L=180 base 0.660 → +SSM 0.671, **Δ+0.011 [−0.016, +0.040]** (no separation); L=365 base
0.619 → +SSM 0.603, Δ−0.015 [−0.035, +0.001] (null/worse). **SSM does NOT add over strong baseline.**
**Verdict:** honest negative — the ceiling holds across the state-space family too. Gate BLOCKED.

### Consolidated signature across treatment-conditioned mechanisms
LTI (Iter1, +0.014 over strong / refuted), time-varying Cox (Lane #2, +0.0145), CDE-signature (Iter4,
+0.004), SSM (Iter5, +0.011): **all give a small POSITIVE point estimate (~+0.01–0.015 td-AUC) at
L=180 that NEVER CI-separates, and go null/negative by L=365.** This is a precise, honest, ICLR-style
statement: a real-but-sub-significant treatment-driven non-PH signal of ~+0.01 td-AUC at 6 months,
below the detection threshold at N≈650 — a power limit, not a model limit. (Static families — Cox/RSF/
GBS/log-rank-basin/diffusion — tie at ~0.62–0.66.)
**Next (Iteration 6 — LAST mechanism before synthesis):** Tier-2 ego-centric patient-similarity GRAPH
survival (kNN ego-graph on the program manifold; transductive/graph-regularized risk) — a distinct
(graph) family to complete the method-family-spanning claim. Then SYNTHESIZE docs/THEORY_LOOP_SYNTHESIS.md.

---

## Iteration 6 — ego-centric patient-similarity graph survival  [honest negative, BLOCKED]
**Did:** kNN ego-graph on the 141-program manifold; transductive risk = 1−S_KM(365d) from out-of-fold
neighbors; inner-CV k. `results/theory_loop/iter6.json`.
**Result:** graph C-index **0.598 [0.564, 0.633]** < SOTA (gep70 0.624/sky92 0.620); strong base 0.653
→ +graph 0.654 (no add). k*=50 every fold. Honest negative — ceiling holds in the graph family too.

## CONVERGED — synthesis written (docs/THEORY_LOOP_SYNTHESIS.md)
Six theory families (control-theoretic LTI/SSM, diffusion-SDE, CDE/path-signature, log-rank clustering,
transductive graph) all tie at the ~0.62–0.66 ceiling; none beats Cox(ISS+gep70). Residual signal is a
power-limited ~+0.01–0.015 td-AUC treatment effect at 6 months (one CI-separated, landmark-sensitive
config). Per the pre-registered convergence criterion, the loop STOPS adding mechanisms (further ones
= thrashing) and the contribution is the rigorous method-family-spanning ceiling result + the reusable
theory-grounded harness + localization of residual signal to treatment dynamics and the access-gated
immune modality. See docs/THEORY_LOOP_SYNTHESIS.md.
