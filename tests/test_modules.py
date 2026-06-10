"""Smoke + correctness checks for the novel modules. Proves they run and that the
math is self-consistent (PK inversion recovers signal; Kramers ~ Monte-Carlo;
risk monotonically raises the basin-escape hazard)."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from resistancemap.models.pk_observation import pk_forward, pk_inverse, HALF_LIFE_DAYS
from resistancemap.models.basin_sde import kramers_rate, first_passage_mc, escape_rate_for_risk
from resistancemap.governance.claim_gate import ClaimGate, Governance

print("=== PK-inverse recovery (IgG, t_half=21 d) ===")
t = np.linspace(0, 180, 180)
prod_true = 0.5 + 0.02 * t + 0.5 * np.sin(t / 20.0)
C = pk_forward(prod_true, t, HALF_LIFE_DAYS["IgG"])
prod_hat = pk_inverse(C, t, HALF_LIFE_DAYS["IgG"])
m = slice(3, -3)
r2 = 1 - np.var(prod_true[m] - prod_hat[m]) / np.var(prod_true[m])
print(f"recovery R^2 = {r2:.3f}")

print("\n=== Kramers vs Monte-Carlo first-passage (direction + order of magnitude) ===")
for tilt in (0.0, 0.15, 0.30):
    rk = kramers_rate(tilt=tilt, D=0.12)
    _, rmc = first_passage_mc(tilt=tilt, D=0.12, n=250, seed=1)
    print(f"tilt={tilt:.2f}  Kramers={rk:.4f}  MC={rmc:.4f}  ratio={rk/rmc:.2f}")

print("\n=== Risk -> hazard monotonicity ===")
rates = [escape_rate_for_risk(r) for r in (0.0, 0.15, 0.30)]
print("escape_rate_for_risk(0.0, 0.15, 0.30) =", [round(x, 4) for x in rates])
mono = rates[0] < rates[1] < rates[2]
print("monotonically increasing with risk:", mono)

print("\n=== Claim-gate demonstration ===")
res_path = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_smoketest.json")
best_baseline = max((r["cindex"] for r in json.load(open(res_path))["rows"]), default=0.0) if os.path.exists(res_path) else 0.0
NOVEL_MODEL_CI_LO = None  # no trained novel model yet -> this claim MUST stay BLOCKED
gov = Governance()
gov.add(ClaimGate("pk_inverse_recovers_signal", lambda: r2 > 0.9,
                  "PK deconvolution recovers production R^2>0.9"))
gov.add(ClaimGate("hazard_is_mechanistic_and_monotone", lambda: mono,
                  "higher patient risk -> higher basin-escape hazard"))
gov.add(ClaimGate("novel_model_beats_baselines",
                  lambda: NOVEL_MODEL_CI_LO is not None and NOVEL_MODEL_CI_LO > best_baseline,
                  f"novel-model C-index CI-lower must exceed best baseline ({best_baseline:.3f})"))
rep = gov.emit(os.path.join(os.path.dirname(__file__), "..", "results", "governance_report.json"))
for r in rep:
    print(f"  {r['status']:8s} {r['claim']}")
