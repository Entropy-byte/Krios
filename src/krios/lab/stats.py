"""Small statistics helpers shared by the labs."""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sps
from scipy.optimize import least_squares


def chi2_pvalue(chi2: float, dof: int) -> float:
    """P(chi^2_dof >= chi2); NaN when there are no degrees of freedom."""
    return float(sps.chi2.sf(chi2, dof)) if dof > 0 else float("nan")


def bic(chi2: float, k: int, n: int) -> float:
    return float(chi2 + k * math.log(max(n, 1)))


def weighted_line_fit(x, y, sigma):
    """y = a + b x with known sigmas. Returns (a, b, cov 2x2, chi2)."""
    x, y, s = (np.asarray(v, dtype=float) for v in (x, y, sigma))
    w = 1.0 / s ** 2
    A = np.stack([np.ones_like(x), x], axis=1)
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    beta = cov @ (A.T @ (w * y))
    chi2 = float(np.sum(w * (y - A @ beta) ** 2))
    return float(beta[0]), float(beta[1]), cov, chi2


def fit_least_squares(resid: Callable[[np.ndarray], np.ndarray], starts: Sequence[Sequence[float]],
                      bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None):
    """Best (params, chi2, covariance) over several starting points (residuals already / sigma)."""
    best = None
    for x0 in starts:
        kw = {"bounds": bounds} if bounds is not None else {}
        try:
            sol = least_squares(resid, np.asarray(x0, dtype=float), **kw)
        except (ValueError, FloatingPointError):
            continue
        chi2 = float(np.sum(sol.fun ** 2))
        if best is None or chi2 < best[1]:
            jac = sol.jac
            cov = np.linalg.pinv(jac.T @ jac)
            best = (sol.x, chi2, cov)
    if best is None:
        raise RuntimeError("model fit failed from every starting point")
    return best


def interval(samples, level: float = 0.68):
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    lo, med, hi = np.percentile(s, [50 * (1 - level), 50, 50 * (1 + level)])
    return float(lo), float(med), float(hi)
