import math
import unittest

import numpy as np

from krios.constants import AU_KM
from krios.ephemeris import AnalyticEphemeris
from krios.ephemeris import analytic as an


class TestAnalyticTheories(unittest.TestCase):
    def test_moon_meeus_example_47a(self):
        lon, lat, dist = an.moon_ecliptic(2448724.5)       # 1992 Apr 12, 0h TD
        self.assertAlmostEqual(math.degrees(lon[0]), 133.162655, places=5)
        self.assertAlmostEqual(math.degrees(lat[0]), -3.229126, places=5)
        self.assertAlmostEqual(dist[0], 368409.7, places=0)

    def test_earth_vsop_meeus_example_25b(self):
        L, B, R = an.earth_heliocentric(2448908.5)         # 1992 Oct 13, 0h TD
        self.assertAlmostEqual(math.degrees(L[0]), 19.907372, places=5)
        self.assertAlmostEqual(R[0], 0.99760775, places=7)

    def test_vectors_have_sensible_geometry(self):
        e = AnalyticEphemeris()
        jd = np.linspace(2461000, 2462000, 200)
        sun, moon = e.sun_moon_geocentric(jd)
        rs, rm = np.linalg.norm(sun, axis=1), np.linalg.norm(moon, axis=1)
        self.assertTrue(np.all((rs > 0.98 * AU_KM) & (rs < 1.02 * AU_KM)))
        self.assertTrue(np.all((rm > 356000) & (rm < 407000)))
        # scalar input returns single vectors
        s1, m1 = e.sun_moon_geocentric(2461000.5)
        self.assertEqual(s1.shape, (3,))
        self.assertEqual(m1.shape, (3,))

    def test_sun_is_near_the_equinox_direction_in_march(self):
        s, _ = AnalyticEphemeris().sun_moon_geocentric(2461120.0)    # ~ 2027-03-20 12:00
        ra_dec = np.degrees(np.arctan2(s[1], s[0])) % 360
        self.assertTrue(abs(((ra_dec + 180) % 360) - 180) < 2.0)      # geocentric Sun near RA 0 (J2000 axes)


if __name__ == "__main__":
    unittest.main()
