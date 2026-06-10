"""Relapse as basin escape. The latent resistance state evolves under overdamped
Langevin dynamics in a double-well quasi-potential (sensitive well vs resistant well):
    dz = -U'(z) dt + sqrt(2D) dW,   U(z) = z^4/4 - z^2/2 + tilt * z
'tilt' encodes patient risk / drug pressure (covariate-driven), lowering the escape
barrier. Time-to-relapse = first-passage time from the sensitive well across the
barrier. The hazard follows the Kramers escape rate:
    r_Kramers = sqrt(U''(min) * |U''(barrier)|) / (2*pi) * exp(-dU / D)
giving an interpretable, mechanistic alternative to a black-box hazard head.
"""
from __future__ import annotations
import numpy as np


def dU(z: float, tilt: float = 0.0) -> float:
    return z ** 3 - z + tilt


def U(z, tilt: float = 0.0):
    return z ** 4 / 4.0 - z ** 2 / 2.0 + tilt * z


def barrier_and_curvatures(tilt: float = 0.0):
    """Return (dU_barrier, U''(min), U''(barrier)) for the sensitive (z<0) well."""
    roots = np.roots([1.0, 0.0, -1.0, tilt])           # z^3 - z + tilt = 0
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    if len(real) < 3:                                   # monostable: no barrier
        return None
    zmin_s, zbar, _zmin_r = real[0], real[1], real[2]
    dU_b = U(zbar, tilt) - U(zmin_s, tilt)
    return dU_b, 3 * zmin_s ** 2 - 1, 3 * zbar ** 2 - 1


def kramers_rate(tilt: float = 0.0, D: float = 0.1) -> float:
    bc = barrier_and_curvatures(tilt)
    if bc is None:
        return np.inf
    dU_b, u2_min, u2_bar = bc
    pref = np.sqrt(u2_min * abs(u2_bar)) / (2 * np.pi)
    return float(pref * np.exp(-dU_b / D))


def first_passage_mc(z0: float = -1.0, tilt: float = 0.0, D: float = 0.1,
                     dt: float = 0.01, t_max: float = 5000.0, n: int = 400, seed: int = 0):
    """Monte-Carlo mean first-passage time across the barrier (z crosses 0)."""
    rng = np.random.default_rng(seed)
    sig = np.sqrt(2 * D * dt)
    fpts = []
    for _ in range(n):
        z = z0; t = 0.0
        while t < t_max and z < 0.0:
            z += -dU(z, tilt) * dt + sig * rng.standard_normal()
            t += dt
        fpts.append(t)
    fpts = np.asarray(fpts)
    return float(fpts.mean()), 1.0 / float(fpts.mean())


def risk_to_tilt(risk: float) -> float:
    """Map patient risk / drug-pressure to landscape tilt. Positive risk shallows the
    sensitive (z<0) well, lowering its escape barrier -> faster relapse."""
    return -float(risk)


def escape_rate_for_risk(risk: float = 0.0, D: float = 0.12) -> float:
    """Interpretable, covariate-driven hazard surrogate."""
    return kramers_rate(tilt=risk_to_tilt(risk), D=D)
