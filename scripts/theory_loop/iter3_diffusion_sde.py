"""Theory loop — Iteration 3: score-based DIFFUSION SDE on disease-state latents.

HYPOTHESIS. Disease-state latent z (PCA of the 141 mmSYGNAL programs) follows overdamped Langevin
dz = -grad U_theta(z) dt + sigma dW. Learn the potential U_theta by DENOISING SCORE MATCHING
(score s_theta = -grad U_theta). The diffusion/metastability view says high-energy (low-density),
atypical states are less stable -> candidate risk = U_theta(z) energy, or ||grad U_theta(z)|| (local
instability). ICLR/ICML property: proper SDE / Fokker-Planck score-matching grounding.

HONEST PRIOR: on STATIC program latents this is the proven PH-holds, ~0.62-ceiling regime, so a
density-geometry score should TIE at best. The real question: does the learned energy carry ANY
prognostic signal, and does it add over the strong baseline Cox(ISS+gep70)? Same nested-CV, no claim
unless it CI-separates above gep70/sky92.
"""
from __future__ import annotations
import os, sys, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import torch, torch.nn as nn
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from resistancemap.data.mmsygnal import load_ia12
from resistancemap.survival.splits import patient_kfold

torch.manual_seed(0); np.random.seed(0)
LATENT = 16
SIGMAS = [0.1, 0.3, 1.0]


class Potential(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(self, z):  # scalar energy U(z)
        return self.net(z).squeeze(-1)

    def score(self, z):    # s = -grad U
        z = z.clone().requires_grad_(True)
        U = self.forward(z).sum()
        g, = torch.autograd.grad(U, z, create_graph=self.training)
        return -g


def train_dsm(Z, epochs=300, lr=1e-3):
    """Denoising score matching: for z~=z+sigma eps, target score = -eps/sigma (weighted)."""
    d = Z.shape[1]; net = Potential(d); opt = torch.optim.Adam(net.parameters(), lr=lr)
    Zt = torch.tensor(Z, dtype=torch.float32)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        sig = torch.tensor(np.random.choice(SIGMAS, size=(Zt.shape[0], 1)), dtype=torch.float32)
        eps = torch.randn_like(Zt)
        zt = Zt + sig * eps
        s = net.score(zt)
        target = -eps / sig
        loss = ((sig ** 2) * (s - target) ** 2).sum(1).mean()
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    return net


def energy_and_gradnorm(net, Z):
    Zt = torch.tensor(Z, dtype=torch.float32)
    with torch.no_grad():
        U = net(Zt).numpy()
    s = net.score(Zt).detach().numpy()
    gn = np.linalg.norm(s, axis=1)
    return U, gn


def pca_fit(Xtr, k):
    mu = Xtr.mean(0); Xc = Xtr - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k]


def pca_apply(X, mu, comps):
    z = (X - mu) @ comps.T
    return (z - z.mean(0)) / (z.std(0) + 1e-9)


def main():
    d = load_ia12(); df = d.df.copy().reset_index(drop=True)
    PROG = d.program_cols
    X = df[PROG].fillna(df[PROG].median()).values
    dur, ev = df["duration"].values, df["event"].values
    print(f"[iter3] N={len(df)} events={int(ev.sum())} latent={LATENT}")

    # OOF diffusion-energy risk (inner CV picks risk-type: energy vs gradnorm vs their sign)
    oof = np.full(len(df), np.nan); chosen = []
    for tr_i, te_i in patient_kfold(df, k=5, seed=0):
        Xtr, Xte = X[tr_i], X[te_i]
        mu, comps = pca_fit(Xtr, LATENT)
        Ztr, Zte = pca_apply(Xtr, mu, comps), pca_apply(Xte, mu, comps)
        net = train_dsm(Ztr, epochs=300)
        # inner CV: which risk-type best ranks PFS on train
        cand = {}
        Utr, GNtr = energy_and_gradnorm(net, Ztr)
        best, best_c = ("energy", +1), -1
        for name, vec in [("energy", Utr), ("gradnorm", GNtr)]:
            for sign in (+1, -1):
                c = concordance_index(dur[tr_i], -sign * vec, ev[tr_i])
                if c > best_c: best_c, best = c, (name, sign)
        chosen.append(f"{best[0]}{'+' if best[1] > 0 else '-'}")
        Ute, GNte = energy_and_gradnorm(net, Zte)
        vec = Ute if best[0] == "energy" else GNte
        oof[te_i] = best[1] * vec
    diff_c = concordance_index(dur, -oof, ev)

    # SOTA + strong baseline (static C-index, 5-fold patient CV)
    def cox_oof(feats):
        o = np.full(len(df), np.nan)
        for tr_i, te_i in patient_kfold(df, k=5, seed=0):
            tr = df.iloc[tr_i].copy(); te = df.iloc[te_i].copy()
            for f in feats: tr[f] = tr[f].fillna(tr[f].median()); te[f] = te[f].fillna(tr[f].median())
            try:
                cph = CoxPHFitter(penalizer=1.0).fit(tr[feats + ["duration", "event"]], "duration", "event")
                o[te_i] = cph.predict_partial_hazard(te[feats]).values.ravel()
            except Exception: pass
        return o
    gep = pd.to_numeric(df["gep70"]).values; sky = pd.to_numeric(df["sky92"]).values
    gep_c = concordance_index(dur, -gep, ev); sky_c = concordance_index(dur, -sky, ev)
    strong_oof = cox_oof(["ISS", "gep70"]); strong_c = concordance_index(dur, -strong_oof[~np.isnan(strong_oof)],
                                                                          ev[~np.isnan(strong_oof)]) if (~np.isnan(strong_oof)).any() else float("nan")
    # does diffusion energy ADD to strong baseline? Cox(ISS+gep70+energy)
    df["_diff"] = oof
    strongx_oof = cox_oof(["ISS", "gep70", "_diff"])
    ok = ~np.isnan(strongx_oof); strongx_c = concordance_index(dur[ok], -strongx_oof[ok], ev[ok])

    # bootstrap CI for diffusion-energy C-index
    rng = np.random.default_rng(1); bs = []
    for _ in range(1000):
        i = rng.integers(0, len(df), len(df))
        try: bs.append(concordance_index(dur[i], -oof[i], ev[i]))
        except Exception: pass
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    out = {"iteration": 3, "mechanism": "score-based diffusion SDE (DSM potential) on program latents",
           "latent_dim": LATENT, "chosen_risk_type_per_fold": chosen,
           "diffusion_energy_cindex": round(diff_c, 4), "diffusion_ci": [round(lo, 4), round(hi, 4)],
           "sota_gep70_cindex": round(gep_c, 4), "sota_sky92_cindex": round(sky_c, 4),
           "strong_baseline_ISS_gep70_cindex": round(strong_c, 4),
           "strong_plus_diffusion_cindex": round(strongx_c, 4),
           "diffusion_beats_sota": bool(lo > max(gep_c, sky_c)),
           "diffusion_adds_to_strong": bool(strongx_c > strong_c + 0.005)}
    near_chance = abs(diff_c - 0.5) < 0.03
    out["verdict"] = ("diffusion energy ~chance: density-geometry of program latents is NOT prognostic — honest negative; static regime as predicted"
                      if near_chance else
                      ("diffusion energy carries weak signal but does NOT beat SOTA (static ceiling) — honest tie/parity"
                       if not out["diffusion_beats_sota"] else
                       "diffusion energy nominally > SOTA — REQUIRES adversarial verification before any claim"))
    out["gate_beats_sota"] = "PASS-PENDING-VERIFICATION" if out["diffusion_beats_sota"] else "BLOCKED"
    os.makedirs("results/theory_loop", exist_ok=True)
    json.dump(out, open("results/theory_loop/iter3.json", "w"), indent=2)
    print(f"  diffusion-energy C-index = {diff_c:.3f} [{lo:.3f},{hi:.3f}] | SOTA gep70 {gep_c:.3f}/sky92 {sky_c:.3f}")
    print(f"  strong base ISS+gep70 {strong_c:.3f} -> +diffusion {strongx_c:.3f} (adds={out['diffusion_adds_to_strong']})")
    print(f"  risk-type per fold: {chosen}")
    print(f"  VERDICT: {out['verdict']}  | gate={out['gate_beats_sota']}")
    print("  wrote results/theory_loop/iter3.json")


if __name__ == "__main__":
    main()
