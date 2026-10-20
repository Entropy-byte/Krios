"""NASA/JPL Horizons REST client and an Ephemeris provider built on it.

API reference: https://ssd-api.jpl.nasa.gov/doc/horizons.html  (GET /api/horizons.api).

Design notes
------------
* One request = one (target, centre, time grid) cartesian state table
  (``EPHEM_TYPE=VECTORS``, ``VEC_TABLE=2`` -> position + velocity, km and km/s,
  ICRF-aligned equatorial axes, geometric/uncorrected, TDB time scale).
* Every response is cached on disk (sha256 of the request), so repeated runs are
  instant and work offline.  ``offline=True`` never touches the network.
* The network layer is a plain callable ``transport(url) -> str``; tests inject a
  fake one.  Nothing else in the package knows about HTTP.
* Be polite: Horizons is a free public service.  Prefer few, larger requests
  (this module asks for ~1 day of 5-minute states per eclipse, not per instant).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .. import timescales as ts
from .base import Ephemeris

HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
USER_AGENT = "krios/0.1 (+https://github.com/entropy-byte/krios; educational eclipse/orbit tool)"

# Horizons target ids
SUN, MOON, EARTH = "10", "301", "399"
GEOCENTER = "500@399"
SSB = "@0"                              # solar-system barycentre


class HorizonsError(RuntimeError):
    pass


@dataclass
class VectorTable:
    command: str
    center: str
    jd_tdb: np.ndarray                  # (N,)
    pos_km: np.ndarray                  # (N,3)
    vel_km_s: np.ndarray                # (N,3)


def default_cache_dir() -> Path:
    base = os.environ.get("KRIOS_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
        "krios")
    return Path(base) / "horizons"


def _urllib_transport(timeout: float) -> Callable[[str], str]:
    def get(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:          # Horizons puts details in the body
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise HorizonsError(f"HTTP {exc.code} from Horizons: {body}") from exc
        except urllib.error.URLError as exc:
            raise HorizonsError(f"cannot reach Horizons ({exc.reason}); "
                                "check your connection or use --offline with a warm cache") from exc
    return get


def _calendar_arg(jd: float) -> str:
    """Horizons start/stop string 'YYYY-MM-DD HH:MM' (minute resolution)."""
    y, mo, d, hh, mm, _ = ts.jd_to_calendar(jd)
    if y < 1:
        raise HorizonsError("Horizons calendar input needs years >= 1 (use TLIST for older dates)")
    return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}"


def parse_vectors(result_text: str, command: str = "", center: str = "") -> VectorTable:
    """Parse the text between $$SOE and $$EOE of a VECTORS/VEC_TABLE=2 CSV response."""
    if "$$SOE" not in result_text:
        snippet = " ".join(result_text.strip().split())[:400]
        raise HorizonsError(f"Horizons returned no data table: {snippet}")
    body = result_text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    jds, rows = [], []
    for line in body.strip().splitlines():
        parts = [p.strip() for p in line.strip().rstrip(",").split(",")]
        if len(parts) < 8:
            continue
        try:
            jd = float(parts[0])
            vals = [float(v) for v in parts[2:8]]
        except ValueError as exc:
            raise HorizonsError(f"cannot parse Horizons row: {line[:120]!r}") from exc
        jds.append(jd)
        rows.append(vals)
    if not rows:
        raise HorizonsError("Horizons table is empty")
    arr = np.array(rows)
    return VectorTable(command, center, np.array(jds), arr[:, :3], arr[:, 3:6])


def regularize_nodes(table: VectorTable, step_minutes: int) -> VectorTable:
    """Replace the printed node times by the exact requested grid.

    Horizons prints Julian dates with 9 decimals (8.6e-5 s resolution).  Using those rounded times
    to compute the node spacing would put a ~1e-6 relative error into interpolated *velocities*
    (cm/s for the Earth), so the exact grid  t_first + k * step  is rebuilt instead.  The grid
    start is rounded to the nearest second (requests are minute-aligned); if the printed times
    disagree with the grid by more than 0.1 s the table is left untouched.
    """
    n = len(table.jd_tdb)
    if n < 2:
        return table
    first = round(float(table.jd_tdb[0]) * 86400.0) / 86400.0
    grid = first + np.arange(n) * (step_minutes / 1440.0)
    if np.max(np.abs(grid - table.jd_tdb)) > 1.0e-6:
        return table
    return VectorTable(table.command, table.center, grid, table.pos_km, table.vel_km_s)


class HorizonsClient:
    def __init__(self, cache_dir: Optional[os.PathLike] = None, timeout: float = 60.0,
                 offline: bool = False, transport: Optional[Callable[[str], str]] = None,
                 base_url: str = HORIZONS_API, min_interval_s: float = 0.0):
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self.offline = offline
        self.base_url = base_url
        self.transport = transport or _urllib_transport(timeout)
        self.min_interval_s = min_interval_s
        self._last_request = 0.0
        self.requests_made = 0

    # -- low level -----------------------------------------------------
    def _key(self, params: Dict[str, str]) -> str:
        blob = json.dumps(sorted(params.items()), separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _url(self, params: Dict[str, str]) -> str:
        return self.base_url + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)

    def query(self, params: Dict[str, str]) -> str:
        """Run a Horizons query and return the ``result`` text (cached)."""
        params = {"format": "json", **params}
        path = self.cache_dir / (self._key(params) + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["result"]
        if self.offline:
            raise HorizonsError("offline mode and this request is not in the cache "
                                f"({path.name}); run once online to populate {self.cache_dir}")
        wait = self.min_interval_s - (time.time() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        raw = self.transport(self._url(params))
        self._last_request = time.time()
        self.requests_made += 1
        try:
            doc = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HorizonsError(f"non-JSON reply from Horizons: {raw[:200]!r}") from exc
        if "error" in doc:
            raise HorizonsError(f"Horizons error: {doc['error']}")
        result = doc.get("result", "")
        if "$$SOE" in result:                  # only cache successful tables
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"result": result, "params": params}), encoding="utf-8")
        return result

    # -- high level ----------------------------------------------------
    def vectors(self, command: str, center: str, jd_start_tdb: float, jd_stop_tdb: float,
                step_minutes: int = 5, ref_plane: str = "FRAME") -> VectorTable:
        """Cartesian states of ``command`` relative to ``center`` on a uniform TDB grid.

        ref_plane 'FRAME' = equatorial ICRF axes (what the rest of krios uses).
        """
        params = {
            "COMMAND": f"'{command}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'VECTORS'", "CENTER": f"'{center}'",
            "START_TIME": f"'{_calendar_arg(jd_start_tdb)}'",
            "STOP_TIME": f"'{_calendar_arg(jd_stop_tdb)}'",
            "STEP_SIZE": f"'{int(step_minutes)} min'",
            "TIME_TYPE": "'TDB'", "REF_SYSTEM": "'ICRF'", "REF_PLANE": f"'{ref_plane}'",
            "OUT_UNITS": "'KM-S'", "VEC_TABLE": "'2'", "VEC_CORR": "'NONE'",
            "CSV_FORMAT": "'YES'", "VEC_LABELS": "'NO'",
        }
        return regularize_nodes(parse_vectors(self.query(params), command, center), step_minutes)

    def state_at(self, command: str, center: str, jd_tdb: float) -> Tuple[np.ndarray, np.ndarray]:
        """Single-epoch (position km, velocity km/s) relative to ``center``.

        Horizons tables are minute-aligned, so the epoch is bracketed by 1-minute nodes and
        both position and velocity come from the cubic Hermite interpolant (its derivative
        gives the velocity; error ~ (omega*60 s)^3, i.e. far below 1e-9 relative).
        """
        t0 = float(np.floor(jd_tdb * 1440.0) / 1440.0)       # minute boundary at or before the epoch
        tab = self.vectors(command, center, t0, t0 + 2.0 / 1440.0, step_minutes=1)
        pos, vel = hermite_state(tab.jd_tdb, tab.pos_km, tab.vel_km_s, np.array([jd_tdb]))
        return pos[0], vel[0]


def hermite_state(t_nodes, pos, vel, t_query):
    """Cubic Hermite interpolation of position *and* velocity.

    Nodes carry positions (km) and velocities (km/s); times are Julian days.  The velocity
    returned is the exact derivative of the interpolating cubic.
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
    idx = np.clip(np.searchsorted(t_nodes, t_query, side="right") - 1, 0, len(t_nodes) - 2)
    t0, t1 = t_nodes[idx], t_nodes[idx + 1]
    h = ((t1 - t0) * 86400.0)[:, None]                  # seconds
    u = ((t_query - t0) / (t1 - t0))[:, None]
    p0, p1, v0, v1 = pos[idx], pos[idx + 1], vel[idx], vel[idx + 1]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    d00 = 6 * u ** 2 - 6 * u
    d10 = 3 * u ** 2 - 4 * u + 1
    d01 = -6 * u ** 2 + 6 * u
    d11 = 3 * u ** 2 - 2 * u
    position = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
    velocity = (d00 * p0 + d10 * h * v0 + d01 * p1 + d11 * h * v1) / h
    return position, velocity


def hermite_interpolate(t_nodes, pos, vel, t_query):
    """Position-only convenience wrapper around :func:`hermite_state`."""
    return hermite_state(t_nodes, pos, vel, t_query)[0]


class HorizonsEphemeris(Ephemeris):
    """Sun & Moon geocentric vectors from Horizons (JPL DE441), cached + interpolated."""
    name = "JPL Horizons (DE441) via REST API"
    moon_accuracy_arcsec = 0.001

    def __init__(self, client: Optional[HorizonsClient] = None, step_minutes: int = 5):
        self.client = client or HorizonsClient()
        self.step_minutes = step_minutes
        self._segments: List[Tuple[float, float, Dict[str, VectorTable]]] = []

    def prepare(self, jd0_tt: float, jd1_tt: float) -> None:
        """Download (or load from cache) Sun & Moon geocentric states covering [jd0, jd1] TT."""
        for a, b, _ in self._segments:
            if a <= jd0_tt and jd1_tt <= b:
                return
        step = self.step_minutes / 1440.0
        t0 = np.floor((jd0_tt - 2 * step) * 1440.0) / 1440.0
        t1 = np.ceil((jd1_tt + 2 * step) * 1440.0) / 1440.0
        tabs = {name: self.client.vectors(cmd, GEOCENTER, t0, t1, self.step_minutes)
                for name, cmd in (("sun", SUN), ("moon", MOON))}
        lo = float(tabs["sun"].jd_tdb[0]) + step
        hi = float(tabs["sun"].jd_tdb[-1]) - step
        self._segments.append((lo - 0.0, hi, tabs))

    def _segment_for(self, lo: float, hi: float):
        for a, b, tabs in self._segments:
            if a <= lo and hi <= b:
                return tabs
        self.prepare(lo - 0.05, hi + 0.05)
        return self._segment_for(lo, hi)

    def sun_moon_geocentric(self, jd_tt):
        scalar = np.ndim(jd_tt) == 0
        jd = np.atleast_1d(np.asarray(jd_tt, dtype=float))
        jd_tdb = jd + ts.tdb_minus_tt_array(jd)
        tabs = self._segment_for(float(jd_tdb.min()), float(jd_tdb.max()))
        out = []
        for name in ("sun", "moon"):
            t = tabs[name]
            out.append(hermite_interpolate(t.jd_tdb, t.pos_km, t.vel_km_s, jd_tdb))
        if scalar:
            return out[0][0], out[1][0]
        return out[0], out[1]
