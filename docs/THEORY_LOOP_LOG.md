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
