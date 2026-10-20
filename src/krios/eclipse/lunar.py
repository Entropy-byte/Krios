"""Lunar eclipses: Earth's penumbral/umbral shadow at the Moon, contacts, magnitudes.

Shadow model (default, 'herald'): the Earth is an oblate spheroid enlarged by an
87 km opaque atmospheric layer (Herald & Sinnott 2014, as used by NASA and
EclipseWise).  Its silhouette seen along the shadow axis is an ellipse; the
umbral / penumbral cones are tangent to that ellipsoid and the solar disc.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .. import frames as fr
from ..constants import EARTH_EQ_RADIUS_KM, EARTH_FLATTENING
from ..ephemeris.base import Ephemeris
from .config import EclipseConfig
from .geometry import (ellipse_radius, minimize_bounded, roots_on_grid, rows_dot,
                       sun_and_effective_moon, unit)

A_E = EARTH_EQ_RADIUS_KM


@dataclass
class LunarState:
    jd_tt: np.ndarray
    xm: np.ndarray            # Moon offset from the shadow axis, km (east-west)
    ym: np.ndarray            # (north-south, +north)
    z: np.ndarray             # depth of the Moon behind the Earth's centre, km
    dist: np.ndarray          # distance of the Moon's centre from the axis, km
    r_umbra: np.ndarray       # umbral radius at the Moon, km, toward the Moon's direction
    r_penumbra: np.ndarray
    moon_radius: float        # km
    eps_deg: np.ndarray       # angular axis distance seen from Earth's centre

    def f_pen(self):          # zero at P1/P4
        return self.dist - self.r_penumbra - self.moon_radius

    def f_umb_ext(self):      # zero at U1/U4
        return self.dist - self.r_umbra - self.moon_radius

    def f_umb_int(self):      # zero at U2/U3
        return self.dist - self.r_umbra + self.moon_radius

    def umbral_magnitude(self):
        return (self.r_umbra + self.moon_radius - self.dist) / (2 * self.moon_radius)

    def penumbral_magnitude(self):
        return (self.r_penumbra + self.moon_radius - self.dist) / (2 * self.moon_radius)


def compute_lunar(ephem: Ephemeris, jd_tt, cfg: EclipseConfig) -> LunarState:
    jd = np.atleast_1d(np.asarray(jd_tt, dtype=float))
    sun, moon = sun_and_effective_moon(ephem, jd, "lunar", cfg.light_time)
    q = fr.j2000_to_true_of_date(jd)
    s_t, m_t = fr.apply(q, sun), fr.apply(q, moon)
    d_sun = np.linalg.norm(s_t, axis=1)
    u = -s_t / d_sun[:, None]                                  # shadow axis, away from Sun
    z = rows_dot(m_t, u)
    yhat = unit(np.array([0.0, 0.0, 1.0])[None, :] - u[:, 2:3] * u)
    xhat = np.cross(yhat, u)
    xm, ym = rows_dot(m_t, xhat), rows_dot(m_t, yhat)
    dist = np.hypot(xm, ym)
    decl = np.arcsin(u[:, 2])

    a, b = A_E, A_E * (1.0 - EARTH_FLATTENING)
    if cfg.shadow_model == "herald":
        ae, be = a + cfg.layer_km, b + cfg.layer_km
        semi_x = ae
        semi_y = np.sqrt(ae ** 2 * np.sin(decl) ** 2 + be ** 2 * np.cos(decl) ** 2)
        r_cone = ae
    elif cfg.shadow_model == "danjon":
        semi_x = semi_y = np.full_like(d_sun, cfg.danjon_factor * a)
        r_cone = cfg.danjon_factor * a
    elif cfg.shadow_model == "none":
        semi_x = a
        semi_y = np.sqrt(a ** 2 * np.sin(decl) ** 2 + b ** 2 * np.cos(decl) ** 2)
        r_cone = a
    else:
        raise ValueError(f"unknown shadow_model {cfg.shadow_model!r}")

    sin_u = (cfg.sun_radius_km - r_cone) / d_sun
    sin_p = (cfg.sun_radius_km + r_cone) / d_sun
    cos_u, cos_p = np.sqrt(1 - sin_u ** 2), np.sqrt(1 - sin_p ** 2)
    tan_u, tan_p = sin_u / cos_u, sin_p / cos_p
    ax_u, ay_u = semi_x / cos_u - z * tan_u, semi_y / cos_u - z * tan_u
    ax_p, ay_p = semi_x / cos_p + z * tan_p, semi_y / cos_p + z * tan_p
    r_u = ellipse_radius(xm, ym, ax_u, ay_u)
    r_p = ellipse_radius(xm, ym, ax_p, ay_p)
    eps = np.degrees(np.arctan2(dist, z))
    return LunarState(jd, xm, ym, z, dist, r_u, r_p, cfg.k_penumbra * A_E, eps)


@dataclass
class LunarEclipse:
    kind: str                         # Total / Partial / Penumbral
    jd_greatest_tt: float
    gamma: float                      # signed least distance of the Moon from the axis (Earth radii)
    umbral_magnitude: float
    penumbral_magnitude: float
    axis_distance_deg: float
    contacts_tt: Dict[str, float] = field(default_factory=dict)
    penumbral_duration_s: Optional[float] = None
    partial_duration_s: Optional[float] = None
    total_duration_s: Optional[float] = None
    ephemeris: str = ""

    def to_dict(self):
        return dict(self.__dict__)


def _state_at(ephem, cfg, jd):
    return compute_lunar(ephem, np.array([jd]), cfg)


def lunar_global(ephem: Ephemeris, cfg: EclipseConfig, jd_guess: float) -> Optional[LunarEclipse]:
    """Global circumstances of the lunar eclipse near ``jd_guess`` (TT), or None."""
    if hasattr(ephem, "prepare"):
        ephem.prepare(jd_guess - 0.6, jd_guess + 0.6)
    step = 5.0 / 1440.0
    grid = np.arange(jd_guess - 0.25, jd_guess + 0.25 + step, step)
    st = compute_lunar(ephem, grid, cfg)
    i = int(np.argmin(st.dist))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    jd_g = minimize_bounded(lambda t: float(_state_at(ephem, cfg, t).dist[0]), lo, hi)
    g = _state_at(ephem, cfg, jd_g)
    pen, umb = float(g.penumbral_magnitude()[0]), float(g.umbral_magnitude()[0])
    if pen <= 0.0:
        return None
    kind = "Total" if umb >= 1.0 else ("Partial" if umb > 0.0 else "Penumbral")
    gamma = math.copysign(float(g.dist[0]), float(g.ym[0])) / A_E

    contacts: Dict[str, float] = {"Greatest": jd_g}

    def roots(fn):
        return roots_on_grid(lambda t: fn(compute_lunar(ephem, np.atleast_1d(t), cfg)),
                             jd_g - 0.3, jd_g + 0.3, 3.0 / 1440.0)

    r = roots(lambda s: s.f_pen())
    if len(r) >= 2:
        contacts["P1"], contacts["P4"] = r[0], r[-1]
    if umb > -0.5:      # umbral contacts only meaningful if the Moon gets near the umbra
        r = roots(lambda s: s.f_umb_ext())
        if len(r) >= 2:
            contacts["U1"], contacts["U4"] = r[0], r[-1]
    if kind == "Total":
        r = roots(lambda s: s.f_umb_int())
        if len(r) >= 2:
            contacts["U2"], contacts["U3"] = r[0], r[-1]

    def span(a, b):
        return (contacts[b] - contacts[a]) * 86400.0 if a in contacts and b in contacts else None

    return LunarEclipse(kind=kind, jd_greatest_tt=jd_g, gamma=gamma, umbral_magnitude=umb,
                        penumbral_magnitude=pen, axis_distance_deg=float(g.eps_deg[0]),
                        contacts_tt=contacts, penumbral_duration_s=span("P1", "P4"),
                        partial_duration_s=span("U1", "U4"), total_duration_s=span("U2", "U3"),
                        ephemeris=ephem.describe())
