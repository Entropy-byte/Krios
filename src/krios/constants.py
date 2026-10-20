"""Physical and astronomical constants used across krios.

Units: kilometres, seconds, radians unless the name says otherwise.
"""
from __future__ import annotations

import math

C_KM_S = 299_792.458                 # speed of light
AU_KM = 149_597_870.7                # IAU 2012
SECONDS_PER_DAY = 86_400.0
JD_J2000 = 2_451_545.0
DAYS_PER_CENTURY = 36_525.0
ARCSEC = math.pi / (180.0 * 3600.0)  # radians per arcsecond
TT_MINUS_TAI = 32.184                # seconds

# Earth reference ellipsoid (WGS-84)
EARTH_EQ_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_POLAR_RADIUS_KM = EARTH_EQ_RADIUS_KM * (1.0 - EARTH_FLATTENING)

# Eclipse-prediction conventions (match NASA / EclipseWise, F. Espenak)
SUN_RADIUS_KM = 696_000.0
K_PENUMBRA = 0.2725076               # lunar radius / Earth equatorial radius (IAU 1982)
K_UMBRA = 0.2722810                  # smaller value used for umbral contacts
MOON_RADIUS_KM = K_PENUMBRA * EARTH_EQ_RADIUS_KM
# Herald & Sinnott (2014): effective height of Earth's occulting atmospheric layer
SHADOW_LAYER_KM = 87.0
# Danjon rule (older convention): uniform enlargement of Earth's shadow
DANJON_ENLARGEMENT = 1.02

# Mean synodic month
SYNODIC_MONTH_DAYS = 29.530588861
