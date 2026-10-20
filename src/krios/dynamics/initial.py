"""Initial conditions for N-body runs.

* ``system_from_horizons``: exact DE441 states from the JPL Horizons REST API (needs network
  or a warm cache) - this is the accurate route.
* ``system_from_analytic``: works offline.  Sun/Earth/Moon come from the analytic theories,
  the other planets from JPL's approximate Keplerian elements (Standish, valid 1800-2050,
  arc-minute accuracy).  Raw analytic *velocities* are too imprecise to propagate the Moon
  for months, so use :func:`fit_system` to adjust the Earth/Moon states to the theory over
  a fitting window (a genuine, if small, orbit-determination step).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .. import frames as fr
from .. import timescales as ts
from ..constants import AU_KM, SECONDS_PER_DAY
from ..ephemeris.analytic import AnalyticEphemeris
from .bodies import DEFAULT_BODIES, GM, HORIZONS_ID
from .model import ForceModel
from .system import System, integrate

# JPL "Keplerian elements for approximate positions of the major planets", Table 1
# (1800-2050).  Per body: [a, e, I, L, long.peri, long.node] values and per-century rates
# (AU, deg).  Reference: https://ssd.jpl.nasa.gov/planets/approx_pos.html
_STANDISH = {
    "mercury": ([0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
                [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]),
    "venus": ([0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
              [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]),
    "mars": ([1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
             [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]),
    "jupiter": ([5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]),
    "saturn": ([9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
               [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]),
    "uranus": ([19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
               [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]),
    "neptune": ([30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
                [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]),
}
_OBLIQUITY_J2000 = 84381.448 * fr.ARCSEC


def _kepler_E(M, e):
    E = M + e * math.sin(M)
    for _ in range(30):
        dE = (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
        E -= dE
        if abs(dE) < 1e-14:
            break
    return E


def planet_heliocentric(name: str, jd_tdb: float) -> np.ndarray:
    """Approximate heliocentric position (km), J2000 equatorial axes."""
    base, rate = _STANDISH[name]
    T = (jd_tdb - 2451545.0) / 36525.0
    a, e, inc, L, wbar, node = [b + r * T for b, r in zip(base, rate)]
    inc, node = math.radians(inc), math.radians(node)
    w = math.radians(wbar) - node
    M = math.remainder(math.radians(L - wbar), 2 * math.pi)
    E = _kepler_E(M, e)
    xp = a * (math.cos(E) - e)
    yp = a * math.sqrt(1 - e * e) * math.sin(E)
    cw, sw, co, so, ci, si = math.cos(w), math.sin(w), math.cos(node), math.sin(node), math.cos(inc), math.sin(inc)
    x = (cw * co - sw * so * ci) * xp + (-sw * co - cw * so * ci) * yp
    y = (cw * so + sw * co * ci) * xp + (-sw * so + cw * co * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    ecl = np.array([x, y, z]) * AU_KM
    ce, se = math.cos(_OBLIQUITY_J2000), math.sin(_OBLIQUITY_J2000)
    return np.array([ecl[0], ecl[1] * ce - ecl[2] * se, ecl[1] * se + ecl[2] * ce])


def _helio_positions(names: Sequence[str], jd_tdb: float, an: AnalyticEphemeris) -> Dict[str, np.ndarray]:
    jd_tt = jd_tdb - float(ts.tdb_minus_tt_array(jd_tdb))
    sun_g, moon_g = an.sun_moon_geocentric(np.array([jd_tt]))
    earth = -sun_g[0]
    out = {"sun": np.zeros(3), "earth": earth, "moon": earth + moon_g[0]}
    for n in names:
        if n in _STANDISH:
            out[n] = planet_heliocentric(n, jd_tdb)
    return out


def system_from_analytic(jd_tdb: float, bodies: Sequence[str] = DEFAULT_BODIES) -> System:
    an = AnalyticEphemeris()
    gm = np.array([GM[b] for b in bodies])
    h = {"earth": 60.0 / SECONDS_PER_DAY, "moon": 60.0 / SECONDS_PER_DAY}
    pos, vel = [], []
    for b in bodies:
        step = h.get(b, 3600.0 / SECONDS_PER_DAY)
        p0 = _helio_positions(bodies, jd_tdb, an)[b]
        p1 = _helio_positions(bodies, jd_tdb + step, an)[b]
        p2 = _helio_positions(bodies, jd_tdb - step, an)[b]
        pos.append(p0)
        vel.append((p1 - p2) / (2 * step * SECONDS_PER_DAY))
    pos, vel = np.array(pos), np.array(vel)
    m = gm / gm.sum()
    return System(list(bodies), gm, pos - m @ pos, vel - m @ vel, jd_tdb,
                  source="analytic (VSOP87D/ELP + Standish elements)")


def system_from_horizons(client, jd_tdb: float, bodies: Sequence[str] = DEFAULT_BODIES) -> System:
    """DE441 barycentric states via the Horizons REST API."""
    from ..ephemeris.horizons import SSB
    gm = np.array([GM[b] for b in bodies])
    pos, vel = [], []
    for b in bodies:
        p, v = client.state_at(HORIZONS_ID[b], SSB, jd_tdb)
        pos.append(p)
        vel.append(v)
    return System(list(bodies), gm, np.array(pos), np.array(vel), jd_tdb,
                  source="JPL Horizons (DE441) barycentric states")


def earth_pole_j2000(jd_tt: float):
    """Earth's mean spin axis (unit vector) in the J2000 equatorial frame."""
    T = np.atleast_1d(fr.centuries(jd_tt))
    p = fr.precession_matrix(T)[0]
    return tuple(p.T @ np.array([0.0, 0.0, 1.0]))


def make_force_model(system: System, kind: str = "newton", earth_j2: bool = False) -> ForceModel:
    if not earth_j2:
        return ForceModel(kind=kind)
    jd_tt = system.epoch_jd_tdb - float(ts.tdb_minus_tt_array(system.epoch_jd_tdb))
    return ForceModel(kind=kind, earth_j2=True, earth_index=system.index("earth"),
                      pole=earth_pole_j2000(jd_tt))


def fit_system(system: System, reference, window_days: float = 90.0, model: Optional[ForceModel] = None,
               sigma_moon_km: float = 10.0, sigma_sun_km: float = 400.0, sample_days: float = 1.0,
               max_nfev: int = 12, verbose: bool = False) -> System:
    """Adjust the Earth & Moon initial states so the integration follows ``reference``.

    ``reference`` is any Ephemeris (analytic theory, Horizons table...).  Residuals are the
    geocentric Moon and Sun vectors at ``sample_days`` spacing over ``window_days``.
    """
    model = model or ForceModel()
    ie, im, isun = system.index("earth"), system.index("moon"), system.index("sun")
    epoch = system.epoch_jd_tdb
    t_days = np.arange(0.0, window_days + 1e-9, sample_days)
    jd_tdb = epoch + t_days
    jd_tt = jd_tdb - ts.tdb_minus_tt_array(jd_tdb)
    ref_sun, ref_moon = reference.sun_moon_geocentric(jd_tt)

    def apply(x):
        s = System(system.names, system.gm, system.r.copy(), system.v.copy(), epoch, system.source)
        s.r[ie] += x[0:3]
        s.v[ie] += x[3:6] * 1e-6
        s.r[im] += x[6:9]
        s.v[im] += x[9:12] * 1e-6
        return s

    def resid(x):
        tr = integrate(apply(x), window_days + 0.01, model, rtol=1e-11)
        r, _ = tr.state(t_days * SECONDS_PER_DAY)
        moon = r[:, im] - r[:, ie]
        sun = r[:, isun] - r[:, ie]
        return np.concatenate([((moon - ref_moon) / sigma_moon_km).ravel(),
                               ((sun - ref_sun) / sigma_sun_km).ravel()])

    steps = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # km, mm/s

    def jac(x):
        f0 = resid(x)
        cols = []
        for k in range(len(x)):
            dx = np.zeros_like(x)
            dx[k] = steps[k]
            cols.append((resid(x + dx) - f0) / steps[k])
        return np.array(cols).T

    x0 = np.zeros(12)
    sol = least_squares(resid, x0, jac=jac, method="lm", max_nfev=max_nfev,
                        verbose=2 if verbose else 0)
    if verbose:
        rms = float(np.sqrt(np.mean(resid(sol.x)[: 3 * len(t_days)] ** 2))) * sigma_moon_km
        print(f"fit: Moon rms residual {rms:.2f} km after {sol.nfev} evaluations")
    fitted = apply(sol.x)
    fitted.source = system.source + f"; Earth/Moon fitted to reference over {window_days:g} d"
    return fitted
