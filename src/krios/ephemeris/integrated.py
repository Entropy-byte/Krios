"""Ephemeris provider backed by this package's own N-body integration."""
from __future__ import annotations

import numpy as np

from .. import timescales as ts
from ..dynamics.system import Trajectory
from .base import Ephemeris


class IntegratedEphemeris(Ephemeris):
    """Sun/Moon geocentric vectors from a :class:`~krios.dynamics.Trajectory`."""

    def __init__(self, trajectory: Trajectory, label: str = ""):
        self.traj = trajectory
        self.name = f"own N-body integration: {trajectory.model.label()}" + (f" [{label}]" if label else "")
        sysm = trajectory.system
        self._i_sun, self._i_earth, self._i_moon = (sysm.index("sun"), sysm.index("earth"), sysm.index("moon"))
        self._t0 = sysm.epoch_jd_tdb
        self._t1 = self._t0 + trajectory.t_end_s / 86400.0

    @property
    def span_jd_tdb(self):
        return tuple(sorted((self._t0, self._t1)))

    def sun_moon_geocentric(self, jd_tt):
        scalar = np.ndim(jd_tt) == 0
        jd = np.atleast_1d(np.asarray(jd_tt, dtype=float))
        jd_tdb = jd + ts.tdb_minus_tt_array(jd)
        lo, hi = self.span_jd_tdb
        if jd_tdb.min() < lo - 0.01 or jd_tdb.max() > hi + 0.01:
            raise ValueError("requested time is outside the integrated span "
                             f"(JD TDB {lo:.3f} .. {hi:.3f})")
        r = self.traj.positions_at_jd_tdb(jd_tdb)
        sun = r[:, self._i_sun] - r[:, self._i_earth]
        moon = r[:, self._i_moon] - r[:, self._i_earth]
        if scalar:
            return sun[0], moon[0]
        return sun, moon
