"""Conventions used by the eclipse engine (defaults follow NASA / EclipseWise)."""
from __future__ import annotations

from dataclasses import dataclass

from ..constants import (K_PENUMBRA, K_UMBRA, SHADOW_LAYER_KM, SUN_RADIUS_KM)


@dataclass(frozen=True)
class EclipseConfig:
    """Tunable physical conventions.

    shadow_model (lunar eclipses):
        'herald'  Earth = oblate spheroid + 87 km occulting layer (Herald & Sinnott,
                  used by NASA/EclipseWise since 2014).  Default.
        'danjon'  spherical Earth enlarged by 2 % (older convention).
        'none'    bare oblate spheroid, no atmosphere.
    """
    k_penumbra: float = K_PENUMBRA
    k_umbra: float = K_UMBRA
    sun_radius_km: float = SUN_RADIUS_KM
    shadow_model: str = "herald"
    layer_km: float = SHADOW_LAYER_KM
    danjon_factor: float = 1.02
    light_time: bool = True      # shadow displacement from light-travel time / aberration
