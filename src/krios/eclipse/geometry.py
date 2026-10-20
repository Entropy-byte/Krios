"""Shared geometry helpers for eclipse computations."""
from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ..constants import C_KM_S


def rows_dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ellipse_radius(x, y, semi_x, semi_y):
    """Radial distance from the centre to the ellipse boundary in the direction (x, y)."""
    th = np.arctan2(y, x)
    return 1.0 / np.sqrt((np.cos(th) / semi_x) ** 2 + (np.sin(th) / semi_y) ** 2)


def roots_on_grid(fvec: Callable[[np.ndarray], np.ndarray], a: float, b: float,
                  step: float, xtol: float = 1e-10) -> List[float]:
    """All sign-change roots of a vectorised function on [a, b] (grid + Brent)."""
    n = max(int(math.ceil((b - a) / step)), 2)
    grid = np.linspace(a, b, n + 1)
    vals = fvec(grid)
    roots = []
    for i in range(n):
        v0, v1 = vals[i], vals[i + 1]
        if v0 == 0.0:
            roots.append(float(grid[i]))
        elif v0 * v1 < 0.0:
            roots.append(brentq(lambda t: float(fvec(np.array([t]))[0]),
                                grid[i], grid[i + 1], xtol=xtol))
    return roots


def circle_overlap_fraction(s: float, r: float, d: float) -> float:
    """Fraction of the area of circle radius s covered by circle radius r (centres d apart)."""
    if s <= 0:
        return 0.0
    if d >= s + r:
        return 0.0
    if d <= abs(s - r):
        return 1.0 if r >= s else (r * r) / (s * s)
    a1 = math.acos((d * d + s * s - r * r) / (2 * d * s))
    a2 = math.acos((d * d + r * r - s * s) / (2 * d * r))
    area = (s * s * a1 + r * r * a2
            - 0.5 * math.sqrt((-d + s + r) * (d + s - r) * (d - s + r) * (d + s + r)))
    return area / (math.pi * s * s)


def minimize_bounded(f: Callable[[float], float], lo: float, hi: float, xatol: float = 1e-9) -> float:
    """Bounded scalar minimisation of f(t) on [lo, hi].

    SciPy's tolerance is *relative* to |x|; Julian dates are ~2.4e6, so we optimise
    over the offset from the bracket centre instead.
    """
    mid = 0.5 * (lo + hi)
    res = minimize_scalar(lambda u: f(mid + u), bounds=(lo - mid, hi - mid),
                          method="bounded", options={"xatol": xatol})
    return mid + float(res.x)


def geocentric_with_rates(ephem, jd, h_s: float = 30.0):
    """Geocentric Sun/Moon vectors (km) and their time derivatives (km/s)."""
    jd = np.atleast_1d(np.asarray(jd, dtype=float))
    h = h_s / 86400.0
    s0, m0 = ephem.sun_moon_geocentric(jd)
    s1, m1 = ephem.sun_moon_geocentric(jd + h)
    s2, m2 = ephem.sun_moon_geocentric(jd - h)
    return s0, m0, (s1 - s2) / (2.0 * h_s), (m1 - m2) / (2.0 * h_s)


def sun_and_effective_moon(ephem, jd, mode: str, light_time: bool = True):
    """Geocentric Sun and the Moon position that governs the shadow geometry.

    Light needs ~1.28 s to travel between Earth and Moon while the Earth moves
    ~30 km/s, which displaces the shadow by ~37 km (the same 20.5" as stellar
    aberration).  NASA's predictions include it, and so do we.

    mode 'solar': light passes the Moon, then reaches the Earth 1.28 s later,
                  so use the Moon's *retarded* BCRS position:  M + (dS/dt - dM/dt) tau.
    mode 'lunar': light passes the Earth first, then reaches the Moon, so the
                  Earth's *retarded* position is used:           M - (dS/dt) tau.
    """
    if not light_time:
        s, m = ephem.sun_moon_geocentric(np.atleast_1d(np.asarray(jd, dtype=float)))
        return s, m
    s, m, sd, md = geocentric_with_rates(ephem, jd)
    tau = (np.linalg.norm(m, axis=1) / C_KM_S)[:, None]
    if mode == "solar":
        return s, m + (sd - md) * tau
    if mode == "lunar":
        return s, m - sd * tau
    raise ValueError(mode)
