"""N-body systems, integration (DOP853) and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from .. import timescales as ts
from ..constants import SECONDS_PER_DAY
from .bodies import GM
from .model import ForceModel, accel_newton, make_rhs


@dataclass
class System:
    names: List[str]
    gm: np.ndarray                      # km^3/s^2
    r: np.ndarray                       # (N,3) km, barycentric J2000 equatorial
    v: np.ndarray                       # (N,3) km/s
    epoch_jd_tdb: float
    source: str = ""

    def index(self, name: str) -> int:
        return self.names.index(name)

    def barycentre(self):
        m = self.gm / self.gm.sum()
        return m @ self.r, m @ self.v

    def energy(self) -> float:
        """Newtonian total energy / G  (km^5/s^2 ... in GM units)."""
        ke = 0.5 * (self.gm * (self.v ** 2).sum(axis=1)).sum()
        d = self.r[None, :, :] - self.r[:, None, :]
        dist = np.sqrt((d ** 2).sum(-1))
        iu = np.triu_indices(len(self.gm), 1)
        pe = -(self.gm[:, None] * self.gm[None, :] / np.where(dist > 0, dist, np.inf))[iu].sum()
        return float(ke + pe)

    def angular_momentum(self) -> np.ndarray:
        return (self.gm[:, None] * np.cross(self.r, self.v)).sum(axis=0)

    def to_dict(self):
        return {"epoch_jd_tdb": self.epoch_jd_tdb, "frame": "ICRF-aligned J2000 equatorial, solar-system barycentre",
                "units": {"r": "km", "v": "km/s", "gm": "km^3/s^2"}, "source": self.source,
                "bodies": [{"name": n, "gm": float(g), "r": list(map(float, r)), "v": list(map(float, v))}
                           for n, g, r, v in zip(self.names, self.gm, self.r, self.v)]}

    @classmethod
    def from_dict(cls, d) -> "System":
        b = d["bodies"]
        return cls([x["name"] for x in b], np.array([x["gm"] if x.get("gm") else GM[x["name"]] for x in b], float),
                   np.array([x["r"] for x in b], float), np.array([x["v"] for x in b], float),
                   float(d["epoch_jd_tdb"]), d.get("source", ""))


@dataclass
class Trajectory:
    system: System
    model: ForceModel
    t_end_s: float
    sol: object                           # scipy OdeSolution
    nfev: int = 0

    def state(self, t_s) -> np.ndarray:
        """(r, v) arrays of shape (M, N, 3) for times in seconds from the epoch."""
        t = np.atleast_1d(np.asarray(t_s, dtype=float))
        y = self.sol(t)                                   # (6N, M)
        n = len(self.system.names)
        r = y[: 3 * n].T.reshape(len(t), n, 3)
        v = y[3 * n:].T.reshape(len(t), n, 3)
        return r, v

    def positions_at_jd_tdb(self, jd_tdb) -> np.ndarray:
        t = (np.asarray(jd_tdb, dtype=float) - self.system.epoch_jd_tdb) * SECONDS_PER_DAY
        return self.state(t)[0]

    def final_system(self) -> System:
        r, v = self.state(self.t_end_s)
        return System(self.system.names, self.system.gm, r[0], v[0],
                      self.system.epoch_jd_tdb + self.t_end_s / SECONDS_PER_DAY, self.system.source)


def integrate(system: System, days: float, model: Optional[ForceModel] = None,
              rtol: float = 1e-12, atol_km: float = 1e-5, atol_km_s: float = 1e-11) -> Trajectory:
    """Integrate ``days`` (negative = backwards) with DOP853 and dense output."""
    model = model or ForceModel()
    n = len(system.gm)
    y0 = np.concatenate([system.r.ravel(), system.v.ravel()])
    atol = np.concatenate([np.full(3 * n, atol_km), np.full(3 * n, atol_km_s)])
    t_end = days * SECONDS_PER_DAY
    sol = solve_ivp(make_rhs(system.gm, model), (0.0, t_end), y0, method="DOP853",
                    rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    return Trajectory(system, model, t_end, sol.sol, sol.nfev)
