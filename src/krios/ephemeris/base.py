"""Ephemeris provider interface.

Every provider answers one question: *where are the Sun and the Moon, seen from
the Earth's centre, at a given TT instant?*  Positions are geometric (no light
time / aberration), in km, in the J2000 equatorial frame (ICRF-aligned).
Eclipse code only depends on this interface, so a provider can be an analytic
theory, JPL Horizons data, or the package's own N-body integration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Ephemeris(ABC):
    name: str = "abstract"
    #: rough 1-sigma accuracy of the Moon's geocentric position, in arcseconds
    moon_accuracy_arcsec: float = float("nan")

    @abstractmethod
    def sun_moon_geocentric(self, jd_tt) -> Tuple[np.ndarray, np.ndarray]:
        """Return (sun, moon) geocentric vectors in km, shape (..., 3)."""

    def describe(self) -> str:
        return self.name
