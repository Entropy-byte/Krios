import json
import math
import os
import unittest

import numpy as np

from krios import timescales as ts
from krios.eclipse import (EclipseConfig, compute_besselian, find_eclipses, lunar_global,
                            solar_global, solar_local)
from krios.ephemeris import AnalyticEphemeris

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "nasa_catalog_2021_2030.json")) as _fh:
    CAT = json.load(_fh)
EPH = AnalyticEphemeris()
CFG = EclipseConfig()


def jd_td(date, hms, **_):
    y, m, d = map(int, date.split("-"))
    hh, mm, ss = hms.split(":")
    return ts.calendar_to_jd(y, m, d, int(hh), int(mm), float(ss))


class TestCatalog2021_2030(unittest.TestCase):
    """The analytic engine against the NASA/EclipseWise decade tables (TD of greatest eclipse)."""

    @classmethod
    def setUpClass(cls):
        found = find_eclipses(ts.calendar_to_jd(2021, 1, 1), ts.calendar_to_jd(2031, 1, 1), ephemeris=EPH, config=CFG)
        cls.solar = [e for e in found if hasattr(e, "central")]
        cls.lunar = [e for e in found if hasattr(e, "umbral_magnitude")]

    def test_counts(self):
        self.assertEqual(len(self.solar), len(CAT["solar"]))
        self.assertEqual(len(self.lunar), len(CAT["lunar"]))

    def test_solar_types_times_magnitudes_durations(self):
        for row, e in zip(CAT["solar"], self.solar):
            date, hms, kind, mag, dur = row
            with self.subTest(date=date):
                self.assertEqual(e.kind, kind)
                self.assertLess(abs(e.jd_greatest_tt - jd_td(date, hms)) * 86400, 30.0)
                self.assertAlmostEqual(e.magnitude, mag, delta=0.003)
                if dur is not None:
                    self.assertAlmostEqual(e.central_duration_s, dur, delta=2.0)

    def test_lunar_types_times_magnitudes(self):
        for row, e in zip(CAT["lunar"], self.lunar):
            date, hms, kind, mag = row
            with self.subTest(date=date):
                self.assertEqual(e.kind, kind)
                self.assertLess(abs(e.jd_greatest_tt - jd_td(date, hms)) * 86400, 30.0)
                if mag is not None:
                    self.assertAlmostEqual(e.umbral_magnitude, mag, delta=0.004)


class TestSolar2027Aug02(unittest.TestCase):
    """Detailed comparison with the published Besselian elements / circumstances of 2027 Aug 2."""

    ref = CAT["solar_2027_08_02"]

    def test_besselian_elements_at_t0(self):
        ts.set_delta_t_override(72.8)          # NASA's DeltaT for that page
        try:
            b = compute_besselian(EPH, ts.calendar_to_jd(2027, 8, 2, 10, 0, 0), CFG)
        finally:
            ts.set_delta_t_override(None)
        r = self.ref["besselian_t0_10h_td"]
        self.assertAlmostEqual(b.x[0], r["x"], delta=0.0007)
        self.assertAlmostEqual(b.y[0], r["y"], delta=0.0007)
        self.assertAlmostEqual(math.degrees(b.d[0]), r["d_deg"], delta=0.003)
        self.assertAlmostEqual(b.l1[0], r["l1"], delta=2e-5)
        self.assertAlmostEqual(b.l2[0], r["l2"], delta=2e-5)
        self.assertAlmostEqual(b.tan_f1[0], r["tan_f1"], delta=2e-7)
        self.assertAlmostEqual(b.tan_f2[0], r["tan_f2"], delta=2e-7)

    def test_global_circumstances(self):
        e = solar_global(EPH, CFG, jd_td("2027-08-02", "10:00:00"))
        r = self.ref
        self.assertEqual(e.kind, "Total")
        self.assertLess(abs(e.jd_greatest_tt - jd_td("2027-08-02", r["greatest_td"])) * 86400, 5.0)
        self.assertAlmostEqual(e.gamma, r["gamma"], delta=0.0003)
        self.assertAlmostEqual(e.magnitude, r["magnitude"], delta=0.0005)
        self.assertAlmostEqual(e.greatest_lat_deg, r["greatest_lat"], delta=0.02)
        self.assertAlmostEqual(e.greatest_lon_deg, r["greatest_lon"], delta=0.03)
        self.assertAlmostEqual(e.sun_altitude_deg, r["sun_alt"], delta=0.2)
        self.assertAlmostEqual(e.central_duration_s, r["central_duration_s"], delta=1.0)
        for k, v in r["contacts_td"].items():
            with self.subTest(contact=k):
                self.assertLess(abs(e.contacts_tt[k] - jd_td("2027-08-02", v)) * 86400, 5.0)

    def test_local_circumstances_inside_and_outside_totality(self):
        e = solar_global(EPH, CFG, jd_td("2027-08-02", "10:00:00"))
        inside = solar_local(EPH, CFG, 25.69, 32.64, 80.0, e.jd_greatest_tt)
        self.assertTrue(inside.eclipse_occurs)
        self.assertEqual(inside.kind, "Total")
        self.assertTrue(300 < inside.duration_s < 400)
        self.assertTrue(inside.sun_up_at_max)
        partial = solar_local(EPH, CFG, 40.0, 3.0, 0.0, e.jd_greatest_tt)
        self.assertEqual(partial.kind, "Partial")
        self.assertTrue(0.5 < partial.magnitude < 1.0)
        far = solar_local(EPH, CFG, -45.0, -170.0, 0.0, e.jd_greatest_tt)       # night side of the Earth
        self.assertFalse(far.eclipse_occurs)
        self.assertIn("below the horizon", far.note)
        nowhere = solar_local(EPH, CFG, -60.0, 100.0, 0.0, e.jd_greatest_tt)    # penumbra never gets there
        self.assertFalse(nowhere.eclipse_occurs)


class TestLunar2026Mar03(unittest.TestCase):
    ref = CAT["lunar_2026_03_03"]

    def test_contacts_magnitudes_durations(self):
        e = lunar_global(EPH, CFG, jd_td("2026-03-03", "11:35:00"))
        r = self.ref
        self.assertEqual(e.kind, "Total")
        self.assertAlmostEqual(e.umbral_magnitude, r["umbral_magnitude"], delta=0.0005)
        self.assertAlmostEqual(e.penumbral_magnitude, r["penumbral_magnitude"], delta=0.0005)
        self.assertAlmostEqual(e.gamma, r["gamma"], delta=0.0002)
        self.assertAlmostEqual(e.axis_distance_deg, r["axis_distance_deg"], delta=0.0003)
        for k, v in r["contacts_td"].items():
            with self.subTest(contact=k):
                self.assertLess(abs(e.contacts_tt[k] - jd_td("2026-03-03", v)) * 86400, 12.0)
        self.assertAlmostEqual(e.penumbral_duration_s, r["durations_s"]["penumbral"], delta=6)
        self.assertAlmostEqual(e.partial_duration_s, r["durations_s"]["partial"], delta=6)
        self.assertAlmostEqual(e.total_duration_s, r["durations_s"]["total"], delta=6)

    def test_shadow_models_are_distinguishable(self):
        jd = jd_td("2026-03-03", "11:35:00")
        herald = lunar_global(EPH, EclipseConfig(shadow_model="herald"), jd).umbral_magnitude
        danjon = lunar_global(EPH, EclipseConfig(shadow_model="danjon"), jd).umbral_magnitude
        bare = lunar_global(EPH, EclipseConfig(shadow_model="none"), jd).umbral_magnitude
        self.assertTrue(bare < herald < danjon)
        self.assertAlmostEqual(herald, self.ref["umbral_magnitude"], delta=0.001)
        self.assertGreater(abs(danjon - self.ref["umbral_magnitude"]), 0.01)

    def test_light_time_matters_at_the_20_arcsecond_level(self):
        jd = jd_td("2026-03-03", "11:35:00")
        with_lt = lunar_global(EPH, EclipseConfig(light_time=True), jd)
        without = lunar_global(EPH, EclipseConfig(light_time=False), jd)
        shift = abs(with_lt.jd_greatest_tt - without.jd_greatest_tt) * 86400
        self.assertTrue(10 < shift < 90, shift)


class TestNoEclipse(unittest.TestCase):
    def test_empty_ranges_and_invalid_model(self):
        # 2026-10-10 .. 2027-01-31 has no eclipse (NASA catalog lists none)
        self.assertEqual(find_eclipses(ts.calendar_to_jd(2026, 10, 10), ts.calendar_to_jd(2027, 1, 31)), [])
        with self.assertRaises(ValueError):
            lunar_global(EPH, EclipseConfig(shadow_model="bogus"), jd_td("2026-03-03", "11:35:00"))


if __name__ == "__main__":
    unittest.main()
