"""Capstone figure — the whole body of work in one figure (all real numbers from results JSONs).
Panel A: discrimination is method-invariantly ceiling-bound (~0.62) on the static substrate.
Panel B: treatment-conditioned Δ td-AUC across theory families — a power-limited ~+0.01 signal.
Panel C: trustworthy uncertainty — conformal coverage valid where naive under-covers.
"""
import os, sys, json
sys.path.insert(0, "resistancemap/src")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "resistancemap/results/"
s = json.load(open(R + "sota_comparison.json"))
i3 = json.load(open(R + "theory_loop/iter3.json")); i6 = json.load(open(R + "theory_loop/iter6.json"))
i2 = json.load(open(R + "theory_loop/iter2.json"))["landmarks"]["180"]
i4 = json.load(open(R + "theory_loop/iter4.json"))["landmarks"]["180"]
i5 = json.load(open(R + "theory_loop/iter5.json"))["landmarks"]["180"]
l2 = json.load(open(R + "lane2/landmark_results_v2.json"))["q1_enriched_cox_vs_static"][0]
cf = json.load(open(R + "extval/iter3.json"))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

# Panel A — discrimination ceiling
rows = [(r["model"], r["cindex"], r["ci_lo"], r["ci_hi"]) for r in s["rows"]]
rows += [("diffusion-SDE", i3["diffusion_energy_cindex"], i3["diffusion_ci"][0], i3["diffusion_ci"][1]),
         ("patient-graph", i6["graph_cindex"], i6["graph_ci"][0], i6["graph_ci"][1])]
rows = sorted(rows, key=lambda x: x[1])
y = np.arange(len(rows))
for i, (nm, c, lo, hi) in enumerate(rows):
    col = "#2c7fb8" if "SOTA" in nm else ("#d7301f" if c < 0.5 else "#555")
    ax[0].plot([lo, hi], [i, i], color=col, lw=2); ax[0].plot(c, i, "o", color=col, ms=6)
ax[0].axvspan(0.620, 0.624, color="#2c7fb8", alpha=0.15)
ax[0].axvline(0.653, ls="--", color="#000", lw=1, label="strong base Cox(ISS+gep70)≈0.653")
ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows], fontsize=7)
ax[0].set_xlim(0.44, 0.82); ax[0].set_xlabel("PFS C-index (95% CI)")
ax[0].set_title("A  Discrimination is method-invariantly ceiling-bound\n(static substrate, 6 model classes)")
ax[0].legend(fontsize=7, loc="lower right")

# Panel B — treatment-conditioned Δ across families
B = [("LTI\n(control)", i2["delta_x_over_STRONG"]),
     ("time-varying Cox\n(Lane 2)", [l2["delta_mean"], l2["ci_lo"], l2["ci_hi"]]),
     ("CDE/signature", i4["delta_sig12_over_base"]),
     ("selective SSM", i5["delta_ssm_over_strong"])]
xs = np.arange(len(B))
for i, (nm, d) in enumerate(B):
    m, lo, hi = d
    col = "#2ca25f" if lo > 0 else "#999"
    ax[1].errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=8, capsize=4, lw=2)
ax[1].axhline(0, ls="--", color="0.4")
ax[1].set_xticks(xs); ax[1].set_xticklabels([b[0] for b in B], fontsize=8)
ax[1].set_ylabel("Δ forward td-AUC over strong baseline @180d")
ax[1].set_title("B  Treatment-conditioned signal is small & power-limited\n(~+0.01; only one CI-separates)")

# Panel C — trustworthy uncertainty (conformal)
labels = ["naive\nGaussian", "split-\nconformal"]
means = [cf["naive_gaussian_coverage"]["mean"], cf["marginal_conformal_coverage"]["mean"]]
cis = [cf["naive_gaussian_coverage"]["ci"], cf["marginal_conformal_coverage"]["ci"]]
cols = ["#d7301f", "#2ca25f"]
for i, (m, ci, c) in enumerate(zip(means, cis, cols)):
    ax[2].bar(i, m, 0.6, color=c, alpha=0.85)
    ax[2].errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]], fmt="none", ecolor="k", capsize=5, lw=1.5)
ax[2].axhline(0.90, ls="--", color="#000", lw=1.5, label="nominal 90%")
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(labels)
ax[2].set_ylim(0.5, 1.0); ax[2].set_ylabel("Empirical coverage (uncensored PFS)")
ax[2].set_title("C  Trustworthy uncertainty: conformal is valid,\nnaive under-covers")
ax[2].legend(fontsize=8, loc="lower left")

fig.suptitle("ResistanceMap (open data): discrimination is information-capped → contribution is honesty + calibrated uncertainty + generalization",
             fontsize=12, y=1.03)
fig.tight_layout()
os.makedirs("resistancemap/paper/figures", exist_ok=True)
fig.savefig("resistancemap/paper/figures/fig_capstone.png", dpi=200, bbox_inches="tight")
fig.savefig("resistancemap/paper/figures/fig_capstone.pdf", bbox_inches="tight")
print("[ok] wrote paper/figures/fig_capstone.png/.pdf")
print(f"  A: {len(rows)} methods, ceiling band 0.620-0.624; B: 4 treatment families; C: conformal {means[1]:.3f} vs naive {means[0]:.3f} (nominal 0.90)")
