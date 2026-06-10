"""PK-inverse observation model: treat a serum biomarker as the half-life-blurred
*shadow* of a latent production signal, and invert the measurement physics.

One-compartment, first-order elimination (standard clinical PK):
    dC/dt = p(t) - ke * C(t),     ke = ln(2) / t_half
Forward operator H maps a latent production rate p(t) (driven by tumour burden)
to the observed serum concentration C(t). Inversion recovers p(t) from serial C:
    p_hat(t) = dC/dt + ke * C(t)
This is the core conceptual novelty: competitors regress on C (the blurred shadow);
we deconvolve to the production signal that is the actual disease state.

Half-lives (literature, serum): IgG ~21 d, IgA ~6 d, IgM ~5 d, free light chains
~2-6 h (renal-dependent), beta-2-microglobulin ~1-2 h, albumin ~19 d.
"""
from __future__ import annotations
import numpy as np

HALF_LIFE_DAYS = {
    "IgG": 21.0, "IgA": 6.0, "IgM": 5.0,
    "FLC_kappa": 0.17, "FLC_lambda": 0.17,   # ~4 h
    "beta2_microglobulin": 0.06,             # ~1.5 h
    "albumin": 19.0, "M_protein": 21.0,      # M-protein ~ IgG class default
}


def ke_from_half_life(t_half_days: float) -> float:
    return np.log(2.0) / float(t_half_days)


def pk_forward(production: np.ndarray, t: np.ndarray, t_half_days: float, c0: float = 0.0) -> np.ndarray:
    """Integrate dC/dt = p - ke*C (explicit Euler on the observation grid)."""
    ke = ke_from_half_life(t_half_days)
    C = np.empty_like(production, dtype=float); C[0] = c0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        C[i] = C[i - 1] + dt * (production[i - 1] - ke * C[i - 1])
    return C


def pk_inverse(concentration: np.ndarray, t: np.ndarray, t_half_days: float) -> np.ndarray:
    """Recover production p_hat = dC/dt + ke*C from serial concentrations."""
    ke = ke_from_half_life(t_half_days)
    dCdt = np.gradient(np.asarray(concentration, float), np.asarray(t, float))
    return dCdt + ke * np.asarray(concentration, float)
