"""Equations of motion: Newtonian N-body and 1PN (Einstein-Infeld-Hoffmann) N-body.

The 1PN model is the parametrised post-Newtonian N-body equation used for the
JPL planetary ephemerides (Standish & Williams; Moyer 2003), with GR values
beta = gamma = 1, in the solar-system barycentric frame, coordinate time TDB.
Optional Earth J2 (oblateness) acts on the Moon & others as a Newtonian term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constants import C_KM_S
from .bodies import EARTH_J2, EARTH_J2_RADIUS_KM


@dataclass(frozen=True)
class ForceModel:
    kind: str = "newton"                 # 'newton' | 'pn1'
    earth_j2: bool = False
    earth_index: Optional[int] = None
    pole: Optional[tuple] = None         # Earth's spin axis, J2000 equatorial unit vector
    c_km_s: float = C_KM_S

    def label(self) -> str:
        base = {"newton": "Newtonian N-body", "pn1": "1PN (EIH, GR) N-body"}[self.kind]
        return base + (" + Earth J2" if self.earth_j2 else "")


def _pairwise(r):
    d = r[None, :, :] - r[:, None, :]            # d[i, j] = r_j - r_i
    dist = np.sqrt((d ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return d, dist


def accel_newton(r, gm):
    d, dist = _pairwise(r)
    w = gm[None, :] / dist ** 3
    return np.einsum("ij,ijk->ik", w, d)


def accel_j2(r, gm, earth_index, pole, j2=EARTH_J2, radius=EARTH_J2_RADIUS_KM):
    """Acceleration from Earth's J2, with the equal-and-opposite reaction on the Earth."""
    p = np.asarray(pole, dtype=float)
    rho = r - r[earth_index]
    dist = np.linalg.norm(rho, axis=1)
    dist[earth_index] = np.inf
    z = rho @ p
    s2 = (z / dist) ** 2
    k = -1.5 * j2 * gm[earth_index] * radius ** 2 / dist ** 5
    a = k[:, None] * ((1.0 - 5.0 * s2)[:, None] * rho + 2.0 * z[:, None] * p[None, :])
    a[earth_index] = 0.0
    a_earth = -(gm[:, None] * a).sum(axis=0) / gm[earth_index]
    a[earth_index] = a_earth
    return a


def accel_pn1(r, v, gm, c=C_KM_S):
    """EIH equations (beta = gamma = 1) for all bodies. r km, v km/s, gm km^3/s^2."""
    c2 = c * c
    a_n = accel_newton(r, gm)
    d, dist = _pairwise(r)                        # d[i,j] = r_j - r_i
    inv_dist = 1.0 / dist
    pot = (gm[None, :] * inv_dist).sum(axis=1)    # U_i = sum_{l!=i} mu_l / r_il
    v2 = (v ** 2).sum(axis=1)
    vv = v @ v.T
    d_dot_vj = np.einsum("ijk,jk->ij", d, v)      # (r_j - r_i).v_j
    d_dot_vi = np.einsum("ijk,ik->ij", d, v)      # (r_j - r_i).v_i
    d_dot_aj = np.einsum("ijk,jk->ij", d, a_n)    # (r_j - r_i).a_j

    # bracket F_ij multiplying the Newtonian pair acceleration
    f = (1.0 - 4.0 * pot[:, None] / c2 - pot[None, :] / c2
         + v2[:, None] / c2 + 2.0 * v2[None, :] / c2 - 4.0 * vv / c2
         - 1.5 * (d_dot_vj * inv_dist) ** 2 / c2
         + 0.5 * d_dot_aj / c2)
    w = gm[None, :] / dist ** 3
    term1 = np.einsum("ij,ijk->ik", w * f, d)

    # velocity-dependent term: [(r_i - r_j).(4 v_i - 3 v_j)] (v_i - v_j)
    s = -(4.0 * d_dot_vi - 3.0 * d_dot_vj)
    dv = v[:, None, :] - v[None, :, :]
    term2 = np.einsum("ij,ijk->ik", w * s, dv) / c2

    # acceleration-dependent term (7/2) mu_j a_j / r_ij
    term3 = 3.5 * np.einsum("ij,jk->ik", gm[None, :] * inv_dist, a_n) / c2
    # the j = i entries vanish because inv_dist[i,i] = 0 and w[i,i] = 0
    return term1 + term2 + term3


def make_rhs(gm: np.ndarray, model: ForceModel):
    n = len(gm)
    if model.earth_j2 and (model.earth_index is None or model.pole is None):
        raise ValueError("earth_j2 needs earth_index and pole")

    def rhs(_t, y):
        r = y[: 3 * n].reshape(n, 3)
        v = y[3 * n:].reshape(n, 3)
        if model.kind == "newton":
            a = accel_newton(r, gm)
        elif model.kind == "pn1":
            a = accel_pn1(r, v, gm, model.c_km_s)
        else:
            raise ValueError(model.kind)
        if model.earth_j2:
            a = a + accel_j2(r, gm, model.earth_index, model.pole)
        return np.concatenate([v.ravel(), a.ravel()])

    return rhs
