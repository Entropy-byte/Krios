"""Central line of a total/annular eclipse as GeoJSON (for maps)."""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .. import frames as fr
from .. import timescales as ts
from ..constants import EARTH_EQ_RADIUS_KM
from ..ephemeris.base import Ephemeris
from .config import EclipseConfig
from .solar import SolarEclipse, axis_surface_zeta, compute_besselian

A_E = EARTH_EQ_RADIUS_KM


def central_line(ephem: Ephemeris, cfg: EclipseConfig, ecl: SolarEclipse, step_s: float = 60.0,
                 sample_every_s: float = 300.0) -> Dict[str, Any]:
    """GeoJSON FeatureCollection: central line, greatest-eclipse point and timed sample points.

    The width is the umbral/antumbral cross-track width 2|L2| R_E (within ~1 % for a high Sun, larger near
    sunrise and sunset).  Lines crossing the antimeridian are split.
    """
    c1, c2 = ecl.contacts_tt.get("C1"), ecl.contacts_tt.get("C2")
    if not ecl.central or c1 is None or c2 is None:
        raise ValueError("this eclipse has no central line (partial or non-central eclipse)")
    if hasattr(ephem, "prepare"):
        ephem.prepare(c1 - 0.05, c2 + 0.05)
    def first_valid(t0: float, sign: int) -> float:
        """The axis grazes the limb at C1/C2, where the surface intersection is ill-defined: step inward until it exists."""
        for dt in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
            t = t0 + sign * dt / 86400.0
            bb = compute_besselian(ephem, np.array([t]), cfg)
            if not math.isnan(axis_surface_zeta(float(bb.x[0]), float(bb.y[0]), float(bb.d[0]))):
                return t
        return t0

    t_start, t_end = first_valid(c1, +1), first_valid(c2, -1)
    times = np.append(np.arange(t_start, t_end, step_s / 86400.0), t_end)
    b = compute_besselian(ephem, times, cfg)
    coords, samples = [], []
    next_sample = 0.0
    for k, t in enumerate(times):
        z = axis_surface_zeta(float(b.x[k]), float(b.y[k]), float(b.d[k]))
        if math.isnan(z):
            continue
        pos = (b.x[k] * b.xhat[k] + b.y[k] * b.yhat[k] + z * b.zhat[k]) * A_E
        lat, lon, _ = fr.ecef_to_geodetic(fr.inertial_to_ecef(pos, b.gast[k]))
        lat, lon = math.degrees(float(lat)), math.degrees(float(lon))
        coords.append([round(lon, 5), round(lat, 5)])
        if (t - t_start) * 86400.0 >= next_sample - 1e-6:
            next_sample += sample_every_s
            la, lo = math.radians(lat), math.radians(lon)
            n_in = fr.ecef_to_inertial(np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]), b.gast[k])
            alt = math.degrees(math.asin(float(np.clip(b.zhat[k] @ n_in, -1, 1))))
            L2 = float(b.l2[k]) - z * float(b.tan_f2[k])
            samples.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(lon, 5), round(lat, 5)]},
                            "properties": {"kind": "sample", "time_utc": ts.format_jd(ts.tt_to_utc(float(t)), 0).replace(" ", "T") + "Z",
                                           "path_width_km": round(2 * abs(L2) * A_E, 1), "sun_altitude_deg": round(alt, 1),
                                           "umbra_type": "total" if L2 < 0 else "annular"}})
    lines, cur = [], []
    for c in coords:
        if cur and abs(c[0] - cur[-1][0]) > 180:
            lines.append(cur)
            cur = []
        cur.append(c)
    if cur:
        lines.append(cur)
    geom = ({"type": "LineString", "coordinates": lines[0]} if len(lines) == 1
            else {"type": "MultiLineString", "coordinates": lines})
    feats: List[Dict[str, Any]] = [{"type": "Feature", "geometry": geom,
                                    "properties": {"kind": "central_line", "eclipse_type": ecl.kind,
                                                   "greatest_utc": ts.format_jd(ts.tt_to_utc(ecl.jd_greatest_tt), 1).replace(" ", "T") + "Z",
                                                   "ephemeris": ecl.ephemeris, "note": "approximate; see Krios docs"}}]
    if ecl.greatest_lat_deg is not None:
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(ecl.greatest_lon_deg, 5), round(ecl.greatest_lat_deg, 5)]},
                      "properties": {"kind": "greatest_eclipse", "time_utc": ts.format_jd(ts.tt_to_utc(ecl.jd_greatest_tt), 1).replace(" ", "T") + "Z",
                                     "magnitude": round(ecl.magnitude, 5), "gamma": round(ecl.gamma, 5),
                                     "duration_s": None if ecl.central_duration_s is None else round(ecl.central_duration_s, 1),
                                     "sun_altitude_deg": None if ecl.sun_altitude_deg is None else round(ecl.sun_altitude_deg, 1)}})
    return {"type": "FeatureCollection", "features": feats + samples}
