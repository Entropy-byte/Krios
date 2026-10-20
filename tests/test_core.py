import math
import unittest

import numpy as np

from krios import frames as fr
from krios import timescales as ts


class TestTime(unittest.TestCase):
    def test_julian_dates_meeus(self):
        self.assertAlmostEqual(ts.calendar_to_jd(1957, 10, 4.81), 2436116.31, places=6)
        self.assertEqual(ts.calendar_to_jd(2000, 1, 1, 12), 2451545.0)
        self.assertEqual(ts.calendar_to_jd(333, 1, 27, 12), 1842713.0)       # Julian calendar

    def test_calendar_roundtrip(self):
        for jd in (2451545.0, 2461619.921266, 2299160.5, 2299161.0, 1842713.0):
            y, mo, d, hh, mm, ss = ts.jd_to_calendar(jd)
            back = ts.calendar_to_jd(y, mo, d, hh, mm, ss)
            self.assertAlmostEqual(back, jd, places=6)

    def test_parse_and_format(self):
        jd = ts.parse_datetime("2027-08-02 10:07:50.2")
        self.assertEqual(ts.format_jd(jd, 1), "2027-08-02 10:07:50.2")
        self.assertEqual(ts.format_jd(ts.parse_datetime("2027"), 0), "2027-01-01 00:00:00")

    def test_leap_seconds_and_tt(self):
        d = ts.calendar_to_jd(2026, 10, 9)
        self.assertAlmostEqual((ts.utc_to_tt(d) - d) * 86400, 69.184, places=3)
        d = ts.calendar_to_jd(2000, 1, 1)
        self.assertAlmostEqual((ts.utc_to_tt(d) - d) * 86400, 64.184, places=3)
        self.assertIsNone(ts.tai_minus_utc(ts.calendar_to_jd(1970, 1, 1)))
        for jd in (2461619.9, 2451545.0, 2457000.2):
            self.assertAlmostEqual(ts.tt_to_utc(ts.utc_to_tt(jd)), jd, places=9)

    def test_delta_t(self):
        self.assertAlmostEqual(ts.delta_t_seconds(2000.0), 63.86, places=2)
        self.assertAlmostEqual(ts.delta_t_seconds(2020.0), 69.36, places=2)
        self.assertGreater(ts.delta_t_seconds(2040.0), ts.delta_t_seconds(2026.0))
        ts.set_delta_t_override(70.0)
        try:
            self.assertEqual(ts.delta_t_seconds(1900.0), 70.0)
        finally:
            ts.set_delta_t_override(None)

    def test_tdb_minus_tt_is_millisecond_level(self):
        for jd in np.linspace(2451545, 2452000, 7):
            self.assertLess(abs(ts.tdb_minus_tt_seconds(jd)), 0.002)


class TestFrames(unittest.TestCase):
    def test_precession_direction_and_size(self):
        p = fr.precession_matrix(np.array([1.0]))[0]
        ra, dec, _ = fr.cartesian_to_ra_dec(p @ np.array([1.0, 0, 0]))
        self.assertAlmostEqual(math.degrees(ra), 1.2817, places=3)    # 46.1"/yr in RA
        self.assertAlmostEqual(math.degrees(dec), 0.5566, places=3)   # 20.04"/yr in Dec

    def test_nutation_meeus_22a(self):
        T = fr.centuries(ts.calendar_to_jd(1987, 4, 10))
        dpsi, deps = fr.nutation(T)
        self.assertAlmostEqual(dpsi[0] / fr.ARCSEC, -3.788, delta=0.01)
        self.assertAlmostEqual(deps[0] / fr.ARCSEC, 9.443, delta=0.01)
        eps0 = math.degrees(fr.mean_obliquity(T)) * 3600
        self.assertAlmostEqual(eps0, 23 * 3600 + 26 * 60 + 27.407, delta=0.001)

    def test_gmst_meeus_12a(self):
        g = fr.gmst(ts.calendar_to_jd(1987, 4, 10))
        self.assertAlmostEqual(math.degrees(g) / 15, 13 + 10 / 60 + 46.3668 / 3600, places=6)

    def test_geodetic_roundtrip(self):
        lat, lon, h = math.radians(35.2), math.radians(-77.1), 0.25
        xyz = fr.geodetic_to_ecef(lat, lon, h)
        la, lo, hh = fr.ecef_to_geodetic(xyz)
        self.assertAlmostEqual(float(la), lat, places=11)
        self.assertAlmostEqual(float(lo), lon, places=11)
        self.assertAlmostEqual(float(hh), h, places=7)

    def test_rotation_matrices_are_orthonormal(self):
        for m in (fr.rot_x(0.3), fr.rot_y(-1.1), fr.rot_z(2.0), fr.precession_matrix(np.array([0.7]))[0],
                  fr.j2000_to_true_of_date(ts.calendar_to_jd(2027, 8, 2))[0]):
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(m), 1.0, places=13)


if __name__ == "__main__":
    unittest.main()
