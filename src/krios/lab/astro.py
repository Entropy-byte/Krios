"""Astronomy helpers for the labs (built on the Krios frames and analytic ephemeris).

Accuracy: Sun ~1", Moon ~5" -- a hundred times better than any citizen measurement, so these
numbers act as the *independent prediction* the measurements are compared with.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .. import frames as fr
from .. import timescales as ts
from ..ephemeris import AnalyticEphemeris

_EPH = AnalyticEphemeris()
K_FE = 111.195          # km per degree of latitude on a 6371 km sphere (flat-map scale)


# ------------------------------------------------------------------- time
def parse_utc(value) -> float:
    """Julian date (UTC) from 'YYYY-MM-DD[ T]HH:MM[:SS][Z|+HH:MM]' or a number (JD)."""
    if isinstance(value, (int, float)):
        return float(value)
    t = str(value).strip()
    offset = 0.0
    m = re.search(r"([+-])(\d\d):?(\d\d)$", t)
    if m and ("T" in t or " " in t) and len(t) > 16:
        offset = (1 if m.group(1) == "+" else -1) * (int(m.group(2)) + int(m.group(3)) / 60.0) / 24.0
        t = t[: m.start()]
    t = t.rstrip("Zz").strip()
    return ts.parse_datetime(t) - offset


def jd_to_iso(jd_utc: float, decimals: int = 0) -> str:
    return ts.format_jd(jd_utc, decimals).replace(" ", "T") + "Z"


# ------------------------------------------------------------- refraction
def refraction_deg(h_true_deg):
    """Atmospheric refraction (degrees), Saemundsson, 1010 hPa / 10 C."""
    h = np.asarray(h_true_deg, dtype=float)
    with np.errstate(all="ignore"):
        r = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))
    return np.where(h > -1.0, r, 0.0) / 60.0


def apparent_alt(h_true_deg):
    return np.asarray(h_true_deg, dtype=float) + refraction_deg(h_true_deg)


def true_alt(h_app_deg):
    h_app = np.asarray(h_app_deg, dtype=float)
    h = h_app.copy()
    for _ in range(6):
        h = h_app - refraction_deg(h)
    return h


# ------------------------------------------------------ topocentric Sun/Moon
def _enu(lat, lon):
    sl, cl, sn, cn = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return (np.array([-sn, cn, 0.0]), np.array([-sl * cn, -sl * sn, cl]), np.array([cl * cn, cl * sn, sl]))


def topocentric(body: str, lat_deg: float, lon_deg: float, height_m: float, jd_utc) -> Dict[str, np.ndarray]:
    """Geometric + apparent altitude, azimuth, distance and inertial unit vector of the Sun/Moon."""
    jd_utc = np.atleast_1d(np.asarray(jd_utc, dtype=float))
    jd_tt = np.array([ts.utc_to_tt(j) for j in jd_utc])
    sun, moon = _EPH.sun_moon_geocentric(jd_tt)
    geo = sun if body == "sun" else moon
    q = fr.j2000_to_true_of_date(jd_tt)
    g = fr.gast(jd_tt)
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    obs_ecef = fr.geodetic_to_ecef(lat, lon, height_m / 1000.0)
    obs_in = fr.ecef_to_inertial(np.broadcast_to(obs_ecef, (len(jd_tt), 3)), g)
    geo_t = fr.apply(q, geo)
    topo = geo_t - obs_in
    dist = np.linalg.norm(topo, axis=1)
    ecef = fr.inertial_to_ecef(topo, g)
    e, n, u = _enu(lat, lon)
    alt = np.degrees(np.arcsin(np.clip(ecef @ u / dist, -1, 1)))
    az = np.degrees(np.arctan2(ecef @ e, ecef @ n)) % 360.0
    ra, dec, _ = fr.cartesian_to_ra_dec(geo_t)
    return {"alt_geom": alt, "alt_app": apparent_alt(alt), "az": az, "dist_km": dist,
            "unit": topo / dist[:, None], "ra": ra, "dec": dec, "gast": g, "jd_tt": jd_tt,
            "obs_inertial": obs_in}


def sun_declination_hour_angle(lon_deg: float, jd_utc) -> Dict[str, np.ndarray]:
    """Geocentric Sun declination, right ascension, local hour angle (radians) at a longitude."""
    t = topocentric("sun", 0.0, lon_deg, 0.0, jd_utc)
    H = np.mod(t["gast"] + math.radians(lon_deg) - t["ra"] + math.pi, 2 * math.pi) - math.pi
    return {"dec": t["dec"], "ra": t["ra"], "H": H, "gast": t["gast"]}


def subsolar_point(jd_utc) -> Dict[str, np.ndarray]:
    """Sub-solar latitude / longitude (degrees, east positive)."""
    t = topocentric("sun", 0.0, 0.0, 0.0, jd_utc)
    lon = np.degrees(np.mod(t["ra"] - t["gast"] + math.pi, 2 * math.pi) - math.pi)
    return {"lat": np.degrees(t["dec"]), "lon": lon}


def latitude_from_sun_altitude(alt_true_deg, dec_rad, hour_angle_rad, lat_hint_deg):
    """Latitude (deg) for a measured true Sun altitude, from sin h = sin(phi) sin(d) + cos(phi) cos(d) cos(H).

    Two solutions exist (mirror about the Sun's azimuth plane); the one nearest the hint is returned.
    """
    h = np.radians(np.asarray(alt_true_deg, dtype=float))
    A = np.sin(dec_rad)
    B = np.cos(dec_rad) * np.cos(hour_angle_rad)
    C = np.hypot(A, B)
    with np.errstate(all="ignore"):
        s = np.sin(h) / C
        theta = np.arctan2(B, A)
        a = np.arcsin(s)
        p1 = a - theta
        p2 = math.pi - a - theta
    wrap = lambda p: (p + math.pi) % (2 * math.pi) - math.pi
    p1, p2 = wrap(p1), wrap(p2)
    hint = math.radians(lat_hint_deg)
    pick = np.where(np.abs(p1 - hint) <= np.abs(p2 - hint), p1, p2)
    out = np.degrees(pick)
    return np.where((np.abs(s) <= 1.0) & (np.abs(pick) <= math.pi / 2 + 1e-9), out, np.nan)


def enu_to_inertial(alt_deg, az_deg, lat_deg, lon_deg, gast_rad):
    """Unit vector in the true-of-date inertial frame for a local (alt, az) direction."""
    alt, az = np.radians(np.asarray(alt_deg, dtype=float)), np.radians(np.asarray(az_deg, dtype=float))
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e, n, u = _enu(lat, lon)
    local = (np.sin(az) * np.cos(alt))[..., None] * e + (np.cos(az) * np.cos(alt))[..., None] * n \
        + np.sin(alt)[..., None] * u
    return fr.ecef_to_inertial(local, gast_rad)


def sun_transit(lat_deg: float, lon_deg: float, jd_utc_guess: float):
    """(jd_utc of local solar transit, maximum altitude deg) near a guess."""
    lo, hi = jd_utc_guess - 0.12, jd_utc_guess + 0.12
    grid = np.linspace(lo, hi, 49)
    alt = topocentric("sun", lat_deg, lon_deg, 0.0, grid)["alt_geom"]
    i = int(np.argmax(alt))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    mid = 0.5 * (a + b)
    res = minimize_scalar(lambda u: -float(topocentric("sun", lat_deg, lon_deg, 0.0, mid + u)["alt_geom"][0]),
                          bounds=(a - mid, b - mid), method="bounded", options={"xatol": 1e-9})
    t = mid + float(res.x)
    return t, float(topocentric("sun", lat_deg, lon_deg, 0.0, t)["alt_geom"][0])


def spherical_distance_deg(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def fe_xy(lat_deg, lon_deg):
    """Azimuthal-equidistant 'flat-Earth map' (north pole at the centre), kilometres."""
    r = (90.0 - np.asarray(lat_deg, dtype=float)) * K_FE
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return r * np.cos(lon), r * np.sin(lon)


# ------------------------------------------------------------------ Polaris
_POLARIS = {"ra": 37.954542, "dec": 89.264111, "pmra_mas": 44.48, "pmdec_mas": -11.85}   # J2000, Hipparcos/Gaia


def polaris_vector_j2000(jd_tt: float) -> np.ndarray:
    ra, dec = math.radians(_POLARIS["ra"]), math.radians(_POLARIS["dec"])
    v = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
    east = np.array([-math.sin(ra), math.cos(ra), 0.0])
    north = np.array([-math.sin(dec) * math.cos(ra), -math.sin(dec) * math.sin(ra), math.cos(dec)])
    yrs = (jd_tt - 2451545.0) / 365.25
    mas = math.pi / (180 * 3600 * 1000)
    v = v + (_POLARIS["pmra_mas"] * yrs * east + _POLARIS["pmdec_mas"] * yrs * north) * mas
    return v / np.linalg.norm(v)


def polaris_offset_deg(jd_utc: float) -> float:
    """Angular distance of Polaris from the true celestial pole of date."""
    jd_tt = ts.utc_to_tt(jd_utc)
    v = fr.apply(fr.j2000_to_true_of_date(jd_tt)[0], polaris_vector_j2000(jd_tt))
    return float(np.degrees(np.arccos(np.clip(v[2], -1, 1))))


def polaris_alt_az(lat_deg: float, lon_deg: float, jd_utc: float):
    jd_tt = ts.utc_to_tt(jd_utc)
    v = fr.apply(fr.j2000_to_true_of_date(jd_tt)[0], polaris_vector_j2000(jd_tt))
    ra, dec = math.atan2(v[1], v[0]), math.asin(v[2])
    H = float(fr.gast(jd_tt)[0]) + math.radians(lon_deg) - ra
    lat = math.radians(lat_deg)
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(H)
    alt = math.asin(sin_alt)
    az = math.atan2(-math.cos(dec) * math.sin(H),
                    math.sin(dec) * math.cos(lat) - math.cos(dec) * math.sin(lat) * math.cos(H))
    return math.degrees(alt), math.degrees(az) % 360.0
