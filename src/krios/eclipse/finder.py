"""Eclipse search: scan lunations with the analytic theory, refine with any provider."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import brentq

from ..constants import SYNODIC_MONTH_DAYS
from ..ephemeris.analytic import AnalyticEphemeris, moon_ecliptic, sun_ecliptic
from ..ephemeris.base import Ephemeris
from .config import EclipseConfig
from .lunar import LunarEclipse, lunar_global
from .solar import SolarEclipse, solar_global

_K0_JDE = 2451550.09766            # mean new moon of 2000 Jan 6 (Meeus ch. 49)
_LAT_LIMIT_DEG = 1.9               # |Moon latitude| above which no eclipse is possible


def _elongation(jd, target):
    lm, _, _ = moon_ecliptic(jd)
    ls, _, _ = sun_ecliptic(jd)
    diff = float(lm[0] - ls[0]) - target
    return (diff + math.pi) % (2 * math.pi) - math.pi


def _conjunction(jd_guess: float, target: float) -> float:
    return brentq(lambda t: _elongation(t, target), jd_guess - 2.0, jd_guess + 2.0, xtol=1e-9)


def candidate_syzygies(jd_start_tt: float, jd_end_tt: float):
    """Yield (kind, jd_tt) for every new ('solar') / full ('lunar') moon near the ecliptic."""
    k0 = int(math.floor((jd_start_tt - _K0_JDE) / SYNODIC_MONTH_DAYS)) - 1
    k1 = int(math.ceil((jd_end_tt - _K0_JDE) / SYNODIC_MONTH_DAYS)) + 1
    for k in range(k0, k1 + 1):
        for phase, kind, target in ((0.0, "solar", 0.0), (0.5, "lunar", math.pi)):
            guess = _K0_JDE + SYNODIC_MONTH_DAYS * (k + phase)
            if guess < jd_start_tt - 2.0 or guess > jd_end_tt + 2.0:
                continue
            t = _conjunction(guess, target)
            _, lat, _ = moon_ecliptic(t)
            if abs(math.degrees(float(lat[0]))) < _LAT_LIMIT_DEG:
                yield kind, t


def find_eclipses(jd_start_tt: float, jd_end_tt: float, kinds: Sequence[str] = ("solar", "lunar"),
                  ephemeris: Optional[Ephemeris] = None,
                  config: Optional[EclipseConfig] = None) -> List[Union[SolarEclipse, LunarEclipse]]:
    """All eclipses whose greatest phase lies in [start, end] (TT Julian dates)."""
    ephem = ephemeris or AnalyticEphemeris()
    cfg = config or EclipseConfig()
    found = []
    for kind, t in candidate_syzygies(jd_start_tt, jd_end_tt):
        if kind not in kinds:
            continue
        ecl = solar_global(ephem, cfg, t) if kind == "solar" else lunar_global(ephem, cfg, t)
        if ecl is not None and jd_start_tt <= ecl.jd_greatest_tt <= jd_end_tt:
            found.append(ecl)
    found.sort(key=lambda e: e.jd_greatest_tt)
    return found
