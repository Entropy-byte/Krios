"""Gravitational parameters (JPL DE440, km^3/s^2) and Horizons identifiers."""
from __future__ import annotations

GM = {
    "sun": 132712440041.279419,
    "mercury": 22031.868551,
    "venus": 324858.592000,
    "earth": 398600.435507,
    "moon": 4902.800118,
    "mars": 42828.375816,        # Mars system (planet + Phobos + Deimos)
    "jupiter": 126712764.100000,  # Jupiter system
    "saturn": 37940584.841800,    # Saturn system
    "uranus": 5794556.400000,     # Uranus system
    "neptune": 6836527.100580,    # Neptune system
}

# Horizons targets (ids 1-9 are barycentres of planetary systems)
HORIZONS_ID = {
    "sun": "10", "mercury": "199", "venus": "299", "earth": "399", "moon": "301",
    "mars": "4", "jupiter": "5", "saturn": "6", "uranus": "7", "neptune": "8",
}

DEFAULT_BODIES = ["sun", "mercury", "venus", "earth", "moon", "mars",
                  "jupiter", "saturn", "uranus", "neptune"]

EARTH_J2 = 1.08262668e-3
EARTH_J2_RADIUS_KM = 6378.137
