"""Covariance kernel functions for stochastic processes."""

import numpy as np
import scipy.special as sps


def rl_fbm_cov(s: float, t: float, H: float) -> float:
    """Riemann-Liouville fractional Brownian motion covariance E[W^H_s W^H_t]."""
    if s == 0 or t == 0:
        return 0.0
    u = min(s, t)
    v = max(s, t)
    x = v / u
    # Hypergeometric 2F1(1/2-H, 1; 3/2+H; 1/x)
    hyper = sps.hyp2f1(0.5 - H, 1.0, 1.5 + H, 1.0 / x)
    G_val = (2.0 * H) / (0.5 + H) * ((1.0 / x) ** (0.5 - H)) * hyper
    return float((u ** (2.0 * H)) * G_val)


def std_fbm_cov(s: float, t: float, H: float) -> float:
    """Standard fractional Brownian motion covariance E[B^H_s B^H_t]."""
    return float(0.5 * (abs(s) ** (2 * H) + abs(t) ** (2 * H) - abs(s - t) ** (2 * H)))


def stn_fou_cov(
    s: float, t: float, H: float, lam: float = 1.0, sgm: float = 1.0
) -> float:
    """Stationary fractional Ornstein-Uhlenbeck process covariance E[Y^H_s Y^H_t]."""
    tau = abs(s - t)
    if H == 0.5:
        return float((sgm**2) / (2.0 * lam) * np.exp(-lam * tau))

    # General H case using mpmath's hyp1f2
    import mpmath as mm

    hpg = float(mm.hyp1f2(1.0, H + 0.5, H + 1.0, (lam**2) * (tau**2) / 4.0))
    gamma_val = sps.gamma(2.0 * H + 1.0)
    term1 = lam ** (-2.0 * H) * np.cosh(lam * tau)
    term2 = (tau ** (2.0 * H)) / gamma_val * hpg

    return float((sgm**2) * gamma_val / 2.0 * (term1 - term2))