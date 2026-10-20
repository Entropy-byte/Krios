"""Solar eclipses: Besselian geometry, global circumstances, local circumstances.

Frame: the fundamental plane passes through the Earth's centre perpendicular to
the shadow axis (Moon -> Sun).  x points east, y north, z towards the Sun, in
units of the Earth's equatorial radius.  Everything is evaluated directly from
geometric Sun/Moon vectors, so no polynomial Besselian fits are needed.
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
from .geometry import (circle_overlap_fraction, ellipse_radius, minimize_bounded, roots_on_grid,
                       rows_dot, sun_and_effective_moon, unit)

A_E = EARTH_EQ_RADIUS_KM
_K_FLAT = 1.0 / (1.0 - EARTH_FLATTENING) ** 2 - 1.0


@dataclass
class Besselian:
    jd_tt: np.ndarray
    x: np.ndarray
    y: np.ndarray
    zc: np.ndarray          # Moon's z coordinate (Earth radii)
    d: np.ndarray           # declination of the shadow axis (rad)
    mu: np.ndarray          # Greenwich hour angle of the axis (rad)
    l1: np.ndarray
    l2: np.ndarray
    tan_f1: np.ndarray
    tan_f2: np.ndarray
    gast: np.ndarray
    xhat: np.ndarray
    yhat: np.ndarray
    zhat: np.ndarray
    sun_t: np.ndarray       # geocentric Sun, true equator of date, km
    moon_t: np.ndarray

    def dist(self):
        return np.hypot(self.x, self.y)

    def limb(self):
        return limb_radius(self.x, self.y, self.d)


def compute_besselian(ephem: Ephemeris, jd_tt, cfg: EclipseConfig) -> Besselian:
    jd = np.atleast_1d(np.asarray(jd_tt, dtype=float))
    sun, moon = sun_and_effective_moon(ephem, jd, "solar", cfg.light_time)
    q = fr.j2000_to_true_of_date(jd)
    s_t, m_t = fr.apply(q, sun), fr.apply(q, moon)
    dvec = s_t - m_t
    dist_sm = np.linalg.norm(dvec, axis=1)
    zhat = dvec / dist_sm[:, None]
    yhat = unit(np.array([0.0, 0.0, 1.0])[None, :] - zhat[:, 2:3] * zhat)
    xhat = np.cross(yhat, zhat)
    x = rows_dot(m_t, xhat) / A_E
    y = rows_dot(m_t, yhat) / A_E
    zc = rows_dot(m_t, zhat) / A_E
    sin_f1 = (cfg.sun_radius_km + cfg.k_penumbra * A_E) / dist_sm
    sin_f2 = (cfg.sun_radius_km - cfg.k_umbra * A_E) / dist_sm
    cos_f1, cos_f2 = np.sqrt(1 - sin_f1 ** 2), np.sqrt(1 - sin_f2 ** 2)
    tan_f1, tan_f2 = sin_f1 / cos_f1, sin_f2 / cos_f2
    l1 = zc * tan_f1 + cfg.k_penumbra / cos_f1
    l2 = zc * tan_f2 - cfg.k_umbra / cos_f2
    d = np.arcsin(zhat[:, 2])
    ra = np.arctan2(zhat[:, 1], zhat[:, 0])
    g = fr.gast(jd)
    mu = np.mod(g - ra, 2 * math.pi)
    return Besselian(jd, x, y, zc, d, mu, l1, l2, tan_f1, tan_f2, g, xhat, yhat, zhat, s_t, m_t)


def limb_radius(x, y, d):
    """Earth's silhouette radius (Earth radii) along the direction (x, y) of the plane."""
    rho_y = np.sqrt(np.sin(d) ** 2 + (1.0 - EARTH_FLATTENING) ** 2 * np.cos(d) ** 2)
    return ellipse_radius(x, y, 1.0, rho_y)


def axis_surface_zeta(x, y, d):
    """z (Earth radii) where the shadow axis meets the Earth's surface on the sunlit side.

    Returns nan if the axis misses the Earth.
    """
    cd, sd = math.cos(d), math.sin(d)
    qa = 1.0 + _K_FLAT * sd * sd
    qb = _K_FLAT * y * cd * sd
    qc = x * x + y * y + _K_FLAT * y * y * cd * cd - 1.0
    disc = qb * qb - qa * qc
    if disc < 0.0:
        return float("nan")
    return (-qb + math.sqrt(disc)) / qa


# ---------------------------------------------------------------- results
@dataclass
class SolarEclipse:
    kind: str                         # Total / Annular / Hybrid / Partial
    jd_greatest_tt: float
    gamma: float                      # signed least axis distance (Earth radii)
    magnitude: float
    obscuration: float
    central: bool
    central_duration_s: Optional[float]
    greatest_lat_deg: Optional[float]
    greatest_lon_deg: Optional[float]
    sun_altitude_deg: Optional[float]
    sun_azimuth_deg: Optional[float]
    l1: float
    l2: float
    contacts_tt: Dict[str, float] = field(default_factory=dict)
    ephemeris: str = ""
    non_central: bool = False

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class SolarLocal:
    lat_deg: float
    lon_deg: float
    height_m: float
    eclipse_occurs: bool
    kind: str = "None"
    contacts_tt: Dict[str, float] = field(default_factory=dict)
    altitude_deg: Dict[str, float] = field(default_factory=dict)
    azimuth_deg: Dict[str, float] = field(default_factory=dict)
    magnitude: float = 0.0
    obscuration: float = 0.0
    duration_s: Optional[float] = None
    sun_up_at_max: bool = False
    note: str = ""

    def to_dict(self):
        return dict(self.__dict__)


def _bes_at(ephem, cfg, jd):
    return compute_besselian(ephem, np.array([jd]), cfg)


def refine_greatest(ephem: Ephemeris, cfg: EclipseConfig, jd_guess: float,
                    half_window: float = 0.25) -> float:
    """Instant of least distance between the shadow axis and the Earth's centre."""
    step = 5.0 / 1440.0
    grid = np.arange(jd_guess - half_window, jd_guess + half_window + step, step)
    dist = compute_besselian(ephem, grid, cfg).dist()
    i = int(np.argmin(dist))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, len(grid) - 1)]
    return minimize_bounded(lambda t: float(_bes_at(ephem, cfg, t).dist()[0]), lo, hi)


def solar_global(ephem: Ephemeris, cfg: EclipseConfig, jd_guess: float) -> Optional[SolarEclipse]:
    """Global circumstances of the solar eclipse near ``jd_guess`` (TT), or None."""
    if hasattr(ephem, "prepare"):
        ephem.prepare(jd_guess - 0.6, jd_guess + 0.6)
    jd_g = refine_greatest(ephem, cfg, jd_guess)
    b = _bes_at(ephem, cfg, jd_g)
    x, y, d = float(b.x[0]), float(b.y[0]), float(b.d[0])
    l1, l2 = float(b.l1[0]), float(b.l2[0])
    dist = math.hypot(x, y)
    limb = float(limb_radius(x, y, d))
    if dist - limb >= l1:
        return None                                    # penumbra misses the Earth
    gamma = math.copysign(dist, y)
    central = dist < limb
    zeta = axis_surface_zeta(x, y, d) if central else float("nan")

    # ---- classification
    non_central = False
    if central:
        kind = _classify_central(ephem, cfg, jd_g)
    elif dist - limb < abs(l2):
        kind = "Total" if l2 < 0 else "Annular"        # umbra grazes the limb
        non_central = True
    else:
        kind = "Partial"

    contacts = _global_contacts(ephem, cfg, jd_g, central)

    # ---- greatest-eclipse position, magnitude, duration
    lat = lon = alt = az = None
    dur = None
    if central:
        L1 = l1 - zeta * float(b.tan_f1[0])
        L2 = l2 - zeta * float(b.tan_f2[0])
        mag = (L1 - L2) / (L1 + L2)          # Moon/Sun diameter ratio on the central line
        pt_in = (x * b.xhat[0] + y * b.yhat[0] + zeta * b.zhat[0]) * A_E
        lat_r, lon_r, _ = fr.ecef_to_geodetic(fr.inertial_to_ecef(pt_in, b.gast[0]))
        lat, lon = math.degrees(float(lat_r)), math.degrees(float(lon_r))
        loc = solar_local(ephem, cfg, lat, lon, 0.0, jd_g)
        dur = loc.duration_s
        alt = loc.altitude_deg.get("max")
        az = loc.azimuth_deg.get("max")
        obsc = loc.obscuration
    else:
        m = dist - limb
        mag = max((l1 - m) / (l1 + l2), 0.0)
        s, r = (l1 + l2) / 2.0, (l1 - l2) / 2.0
        obsc = circle_overlap_fraction(s, r, m)
        if kind in ("Total", "Annular"):
            mag = max(mag, 0.0)

    return SolarEclipse(kind=kind, jd_greatest_tt=jd_g, gamma=gamma, magnitude=mag,
                        obscuration=obsc, central=central, central_duration_s=dur,
                        greatest_lat_deg=lat, greatest_lon_deg=lon,
                        sun_altitude_deg=alt, sun_azimuth_deg=az, l1=l1, l2=l2,
                        contacts_tt=contacts, ephemeris=ephem.describe(),
                        non_central=non_central)


def _classify_central(ephem, cfg, jd_g) -> str:
    """Total / Annular / Hybrid, from the sign of the umbral radius along the path."""
    def limb_gap(t):
        b = compute_besselian(ephem, np.atleast_1d(t), cfg)
        return b.dist() - b.limb()

    ends = roots_on_grid(limb_gap, jd_g - 0.25, jd_g + 0.25, 4.0 / 1440.0)
    if len(ends) < 2:
        ends = [jd_g - 0.02, jd_g + 0.02]
    t_in, t_out = ends[0], ends[-1]
    signs = []
    for f in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
        t = t_in + f * (t_out - t_in)
        b = _bes_at(ephem, cfg, t)
        z = axis_surface_zeta(float(b.x[0]), float(b.y[0]), float(b.d[0]))
        if math.isnan(z):
            z = 0.0
        signs.append(float(b.l2[0]) - z * float(b.tan_f2[0]))
    if all(s < 0 for s in signs):
        return "Total"
    if all(s > 0 for s in signs):
        return "Annular"
    return "Hybrid"


def _global_contacts(ephem, cfg, jd_g, central) -> Dict[str, float]:
    """P1/P4 penumbral, U1-U4 umbral, C1/C2 central-line contacts with the Earth (TT)."""
    out: Dict[str, float] = {}

    def gap(t):
        b = compute_besselian(ephem, np.atleast_1d(t), cfg)
        return b.dist() - b.limb(), b.l1, np.abs(b.l2)

    specs = [("P", lambda g, l1, al2: g - l1),
             ("U", lambda g, l1, al2: g - al2),          # external umbral tangency
             ("V", lambda g, l1, al2: g + al2),          # internal umbral tangency
             ("C", lambda g, l1, al2: g)]
    for tag, fn in specs:
        if tag in ("V", "C") and not central:
            continue
        roots = roots_on_grid(lambda t: fn(*gap(t)), jd_g - 0.25, jd_g + 0.25, 3.0 / 1440.0)
        if len(roots) >= 2:
            first, last = roots[0], roots[-1]
            if tag == "P":
                out["P1"], out["P4"] = first, last
            elif tag == "U":
                out["U1"], out["U4"] = first, last
            elif tag == "V":
                out["U2"], out["U3"] = first, last
            else:
                out["C1"], out["C2"] = first, last
    return out


# ------------------------------------------------------------ local circumstances
def _topocentric_horizon(b: Besselian, lat: float, lon: float, obs_in: np.ndarray):
    """Sun altitude/azimuth (deg) for the observer, geodetic normal based."""
    sl, cl, sn, cn = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    zen_e = np.array([cl * cn, cl * sn, sl])
    east_e = np.array([-sn, cn, 0.0])
    north_e = np.array([-sl * cn, -sl * sn, cl])
    zen = fr.ecef_to_inertial(zen_e[None, :], b.gast)
    east = fr.ecef_to_inertial(east_e[None, :], b.gast)
    north = fr.ecef_to_inertial(north_e[None, :], b.gast)
    sun_dir = unit(b.sun_t - obs_in * A_E)
    alt = np.degrees(np.arcsin(np.clip(rows_dot(sun_dir, zen), -1, 1)))
    az = np.degrees(np.arctan2(rows_dot(sun_dir, east), rows_dot(sun_dir, north))) % 360.0
    return alt, az


def _local_series(ephem, cfg, lat, lon, height_m, jd):
    b = compute_besselian(ephem, jd, cfg)
    obs_ecef = fr.geodetic_to_ecef(lat, lon, height_m / 1000.0) / A_E
    obs_in = fr.ecef_to_inertial(np.broadcast_to(obs_ecef, (len(b.jd_tt), 3)), b.gast)
    xi, eta, zeta = rows_dot(obs_in, b.xhat), rows_dot(obs_in, b.yhat), rows_dot(obs_in, b.zhat)
    m = np.hypot(b.x - xi, b.y - eta)
    L1 = b.l1 - zeta * b.tan_f1
    L2 = b.l2 - zeta * b.tan_f2
    return b, obs_in, m, L1, L2, zeta


def solar_local(ephem: Ephemeris, cfg: EclipseConfig, lat_deg: float, lon_deg: float,
                height_m: float, jd_ref: float, half_window: float = 0.2) -> SolarLocal:
    """Local circumstances for an observer; ``jd_ref`` is near the global greatest eclipse (TT)."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    res = SolarLocal(lat_deg, lon_deg, height_m, False)
    if hasattr(ephem, "prepare"):
        ephem.prepare(jd_ref - half_window - 0.1, jd_ref + half_window + 0.1)

    def series(t):
        return _local_series(ephem, cfg, lat, lon, height_m, np.atleast_1d(t))

    def partial_gap(t):
        _, _, m, L1, _, _ = series(t)
        return m - L1

    grid = np.arange(jd_ref - half_window, jd_ref + half_window, 1.0 / 1440.0)
    _, _, m_g, L1_g, _, _ = series(grid)
    if not np.any(m_g - L1_g < 0.0):
        return res
    roots = roots_on_grid(partial_gap, grid[0], grid[-1], 1.0 / 1440.0)
    if len(roots) < 2:
        return res
    c1, c4 = roots[0], roots[-1]

    def m_of(t):
        return float(series(t)[2][0])

    t_max = minimize_bounded(m_of, c1, c4)

    # The Besselian geometry also "sees" observers on the night side; require the Sun to be
    # above the horizon at some moment between first and last contact.
    check = np.linspace(c1, c4, 41)
    bb, oi, _, _, _, _ = series(check)
    alt_check, _ = _topocentric_horizon(bb, lat, lon, oi)
    if float(alt_check.max()) <= -0.8333:
        res.note = "The Moon's penumbra reaches this point but the Sun is below the horizon throughout."
        return res

    contacts = {"C1": c1, "max": t_max, "C4": c4}
    b, obs_in, m, L1, L2, zeta = series(t_max)
    s_ = (L1[0] + L2[0]) / 2.0
    r_ = (L1[0] - L2[0]) / 2.0
    inside_inner = m[0] < abs(L2[0])
    res.magnitude = float((L1[0] - L2[0]) / (L1[0] + L2[0]) if inside_inner
                          else (L1[0] - m[0]) / (L1[0] + L2[0]))
    res.obscuration = float(circle_overlap_fraction(s_, r_, float(m[0])))

    def inner_gap(t):
        _, _, mm, _, LL2, _ = series(t)
        return mm - np.abs(LL2)

    kind = "Partial"
    if inner_gap(t_max)[0] < 0.0:
        inner = roots_on_grid(inner_gap, c1, c4, 0.5 / 1440.0)
        if len(inner) >= 2:
            contacts["C2"], contacts["C3"] = inner[0], inner[-1]
            res.duration_s = (inner[-1] - inner[0]) * 86400.0
            kind = "Total" if L2[0] < 0 else "Annular"
    res.kind = kind
    res.contacts_tt = contacts
    res.eclipse_occurs = True
    for key, t in contacts.items():
        bb, oi, _, _, _, _ = series(t)
        alt, az = _topocentric_horizon(bb, lat, lon, oi)
        res.altitude_deg[key] = float(alt[0])
        res.azimuth_deg[key] = float(az[0])
    res.sun_up_at_max = res.altitude_deg["max"] > -0.8333
    return res
