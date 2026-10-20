"""Reference-frame helpers: rotations, precession, nutation, sidereal time, geodesy.

The base inertial frame of the package is the J2000 equatorial frame (aligned
with ICRF to ~20 mas). Eclipse geometry is done in the *true equator and
equinox of date* frame so that the Earth's rotation is a simple z-rotation.

Rotation matrices follow the SOFA "frame rotation" convention:
    rot_z(a) = [[ c, s, 0], [-s, c, 0], [0, 0, 1]].
All functions accept scalars or 1-D arrays of epochs (matrices get a leading axis).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .constants import (ARCSEC, DAYS_PER_CENTURY, EARTH_EQ_RADIUS_KM,
                        EARTH_FLATTENING, JD_J2000)
from . import timescales as ts


# ----------------------------------------------------------------- rotations
def _stack(rows):
    arr = np.array(rows)
    if arr.ndim > 2:
        arr = np.moveaxis(arr, (0, 1), (-2, -1))
    return arr


def rot_x(a):
    a = np.asarray(a, dtype=float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return _stack([[o, z, z], [z, c, s], [z, -s, c]])


def rot_y(a):
    a = np.asarray(a, dtype=float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return _stack([[c, z, -s], [z, o, z], [s, z, c]])


def rot_z(a):
    a = np.asarray(a, dtype=float)
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return _stack([[c, s, z], [-s, c, z], [z, z, o]])


def apply(matrix, vec):
    """Apply (…,3,3) matrices to (…,3) vectors."""
    return np.einsum("...ij,...j->...i", matrix, vec)


def centuries(jd_tt):
    return (np.asarray(jd_tt, dtype=float) - JD_J2000) / DAYS_PER_CENTURY


# ------------------------------------------------------------ obliquity etc.
def mean_obliquity(T):
    """Mean obliquity of the ecliptic (radians), Laskar 1986 (valid ±10000 y)."""
    U = np.asarray(T, dtype=float) / 100.0
    coeffs = [84381.448, -4680.93, -1.55, 1999.25, -51.38, -249.67,
              -39.05, 7.12, 27.87, 5.79, 2.45]
    eps = np.zeros_like(U)
    for k, c in enumerate(coeffs):
        eps = eps + c * U ** k
    return eps * ARCSEC


def precession_angles(T):
    """IAU 1976 (Lieske) zeta, z, theta in radians (J2000 -> date)."""
    T = np.asarray(T, dtype=float)
    zeta = (2306.2181 * T + 0.30188 * T ** 2 + 0.017998 * T ** 3) * ARCSEC
    z = (2306.2181 * T + 1.09468 * T ** 2 + 0.018203 * T ** 3) * ARCSEC
    theta = (2004.3109 * T - 0.42665 * T ** 2 - 0.041833 * T ** 3) * ARCSEC
    return zeta, z, theta


def precession_matrix(T):
    """J2000 mean equator/equinox -> mean equator/equinox of date."""
    zeta, z, theta = precession_angles(T)
    return rot_z(-z) @ rot_y(theta) @ rot_z(-zeta)


# IAU 1980 nutation, 20 largest terms: D, M, M', F, Omega | dpsi, dpsi_T | deps, deps_T
# (coefficients in units of 0.0001 arcsec)
_NUT = np.array([
    [0, 0, 0, 0, 1, -171996, -174.2, 92025, 8.9],
    [-2, 0, 0, 2, 2, -13187, -1.6, 5736, -3.1],
    [0, 0, 0, 2, 2, -2274, -0.2, 977, -0.5],
    [0, 0, 0, 0, 2, 2062, 0.2, -895, 0.5],
    [0, 1, 0, 0, 0, 1426, -3.4, 54, -0.1],
    [0, 0, 1, 0, 0, 712, 0.1, -7, 0.0],
    [-2, 1, 0, 2, 2, -517, 1.2, 224, -0.6],
    [0, 0, 0, 2, 1, -386, -0.4, 200, 0.0],
    [0, 0, 1, 2, 2, -301, 0.0, 129, -0.1],
    [-2, -1, 0, 2, 2, 217, -0.5, -95, 0.3],
    [-2, 0, 1, 0, 0, -158, 0.0, 0, 0.0],
    [-2, 0, 0, 2, 1, 129, 0.1, -70, 0.0],
    [0, 0, -1, 2, 2, 123, 0.0, -53, 0.0],
    [2, 0, 0, 0, 0, 63, 0.0, 0, 0.0],
    [0, 0, 1, 0, 1, 63, 0.1, -33, 0.0],
    [2, 0, -1, 2, 2, -59, 0.0, 26, 0.0],
    [0, 0, -1, 0, 1, -58, -0.1, 32, 0.0],
    [0, 0, 1, 2, 1, -51, 0.0, 27, 0.0],
    [-2, 0, 2, 0, 0, 48, 0.0, 0, 0.0],
    [0, 0, -2, 2, 1, 46, 0.0, -24, 0.0],
])


def nutation(T) -> Tuple[np.ndarray, np.ndarray]:
    """(delta_psi, delta_epsilon) in radians (truncated IAU 1980, ~5 mas)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    d2r = math.pi / 180.0
    D = (297.85036 + 445267.111480 * T - 0.0019142 * T ** 2 + T ** 3 / 189474.0) * d2r
    M = (357.52772 + 35999.050340 * T - 0.0001603 * T ** 2 - T ** 3 / 300000.0) * d2r
    Mp = (134.96298 + 477198.867398 * T + 0.0086972 * T ** 2 + T ** 3 / 56250.0) * d2r
    F = (93.27191 + 483202.017538 * T - 0.0036825 * T ** 2 + T ** 3 / 327270.0) * d2r
    Om = (125.04452 - 1934.136261 * T + 0.0020708 * T ** 2 + T ** 3 / 450000.0) * d2r
    arg = (np.outer(D, _NUT[:, 0]) + np.outer(M, _NUT[:, 1]) + np.outer(Mp, _NUT[:, 2])
           + np.outer(F, _NUT[:, 3]) + np.outer(Om, _NUT[:, 4]))
    dpsi = (np.sin(arg) * (_NUT[:, 5][None, :] + _NUT[:, 6][None, :] * T[:, None])).sum(axis=1)
    deps = (np.cos(arg) * (_NUT[:, 7][None, :] + _NUT[:, 8][None, :] * T[:, None])).sum(axis=1)
    return dpsi * 1e-4 * ARCSEC, deps * 1e-4 * ARCSEC


def nutation_matrix(T):
    """Mean -> true equator/equinox of date."""
    dpsi, deps = nutation(T)
    eps = mean_obliquity(np.atleast_1d(T))
    return rot_x(-(eps + deps)) @ rot_z(-dpsi) @ rot_x(eps)


def j2000_to_true_of_date(jd_tt):
    """Matrices taking J2000 equatorial vectors to the true equator/equinox of date."""
    T = np.atleast_1d(centuries(jd_tt))
    return nutation_matrix(T) @ precession_matrix(T)


# ------------------------------------------------------------- sidereal time
def ut1_array(jd_tt):
    jd_tt = np.atleast_1d(np.asarray(jd_tt, dtype=float))
    dt = np.array([ts.delta_t_for_jd(j) for j in jd_tt])
    return jd_tt - dt / 86400.0


def gmst(jd_ut1):
    """Greenwich mean sidereal time (radians, IAU 1982)."""
    jd = np.asarray(jd_ut1, dtype=float)
    d = jd - JD_J2000
    T = d / DAYS_PER_CENTURY
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * T ** 2 - T ** 3 / 38710000.0
    return np.radians(np.mod(deg, 360.0))


def gast(jd_tt):
    """Greenwich apparent sidereal time (radians) for TT epochs (UT1 via DeltaT)."""
    jd_tt = np.atleast_1d(np.asarray(jd_tt, dtype=float))
    T = centuries(jd_tt)
    dpsi, _ = nutation(T)
    eps = mean_obliquity(T)
    return np.mod(gmst(ut1_array(jd_tt)) + dpsi * np.cos(eps), 2.0 * math.pi)


# ----------------------------------------------------------------- ecliptic
def ecliptic_to_equatorial(vec, eps):
    """Rotate ecliptic-of-date cartesian vectors (…,3) to equatorial-of-date."""
    vec = np.asarray(vec, dtype=float)
    eps = np.asarray(eps, dtype=float)
    c, s = np.cos(eps), np.sin(eps)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([x, y * c - z * s, y * s + z * c], axis=-1)


def spherical_to_cartesian(lon, lat, r):
    cl, sl, cb, sb = np.cos(lon), np.sin(lon), np.cos(lat), np.sin(lat)
    return np.stack([r * cb * cl, r * cb * sl, r * sb], axis=-1)


def cartesian_to_ra_dec(vec):
    vec = np.asarray(vec, dtype=float)
    r = np.linalg.norm(vec, axis=-1)
    ra = np.mod(np.arctan2(vec[..., 1], vec[..., 0]), 2 * math.pi)
    dec = np.arcsin(vec[..., 2] / r)
    return ra, dec, r


# ------------------------------------------------------------------ geodesy
def geodetic_to_ecef(lat, lon, height_km=0.0, a=EARTH_EQ_RADIUS_KM, f=EARTH_FLATTENING):
    """ECEF cartesian (km) from geodetic latitude/longitude (radians, east +)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    e2 = f * (2.0 - f)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.stack([(n + height_km) * np.cos(lat) * np.cos(lon),
                     (n + height_km) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - e2) + height_km) * np.sin(lat)], axis=-1)


def ecef_to_geodetic(xyz, a=EARTH_EQ_RADIUS_KM, f=EARTH_FLATTENING):
    """(lat, lon, height_km) from ECEF km (iterative, converges in a few steps)."""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    e2 = f * (2.0 - f)
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(8):
        n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - e2 * n / (n + h)))
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return lat, lon, h


def ecef_to_inertial(ecef, gast_rad):
    """ECEF vectors -> true-of-date inertial vectors for given GAST (radians)."""
    ecef = np.asarray(ecef, dtype=float)
    c, s = np.cos(gast_rad), np.sin(gast_rad)
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    return np.stack([c * x - s * y, s * x + c * y, z + 0.0 * c], axis=-1)


def inertial_to_ecef(vec, gast_rad):
    vec = np.asarray(vec, dtype=float)
    c, s = np.cos(gast_rad), np.sin(gast_rad)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z + 0.0 * c], axis=-1)
