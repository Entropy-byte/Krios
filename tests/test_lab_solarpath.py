import unittest

from krios import timescales as ts
from krios.lab import astro
from krios.lab.solarpath import SolarPathLab, day_events


def run(cls, spec, **kw):
    return cls(spec).run(**kw)


class TestSolarPathLab(unittest.TestCase):
    def test_known_day_greenwich_midsummer(self):
        ev = day_events(51.4779, 0.0, 0.0, astro.parse_utc("2027-06-21"))
        rise, sset = ts.jd_to_calendar(ev["sunrise_jd"]), ts.jd_to_calendar(ev["sunset_jd"])
        self.assertEqual(rise[3], 3)
        self.assertTrue(41 <= rise[4] <= 45, rise)           # 03:42-03:44 UTC (04:43 BST)
        self.assertEqual(sset[3], 20)
        self.assertTrue(19 <= sset[4] <= 23, sset)           # ~20:21 UTC (21:21 BST)
        self.assertAlmostEqual(ev["noon_alt_deg"], 90 - 51.4779 + 23.44, delta=0.1)

    def test_equator_equinox_day_is_longer_than_12_hours(self):
        ev = day_events(0.0, 30.0, 0.0, astro.parse_utc("2027-03-20"))
        self.assertAlmostEqual((ev["sunset_jd"] - ev["sunrise_jd"]) * 24, 12.12, delta=0.05)    # refraction + semi-diameter

    def test_polar_day_and_night(self):
        self.assertEqual(day_events(75.0, 0.0, 0.0, astro.parse_utc("2027-06-21"))["polar"], "day")
        self.assertEqual(day_events(75.0, 0.0, 0.0, astro.parse_utc("2027-12-21"))["polar"], "night")
        self.assertEqual(day_events(-75.0, 0.0, 0.0, astro.parse_utc("2027-12-21"))["polar"], "day")

    def test_example_opposite_seasons_and_residuals(self):
        r = run(SolarPathLab, SolarPathLab.example())
        s = r.summary
        n_long = [v for k, v in s.items() if k.startswith("Greenwich") and "longest" in k][0]
        s_long = [v for k, v in s.items() if k.startswith("Sydney") and "longest" in k][0]
        self.assertIn("2027-06", n_long)
        self.assertIn("2027-12", s_long)
        self.assertGreater(r.models[0].p_value, 0.01)
        self.assertLess(abs(s["mean_z"]), 0.5)

    def test_a_wrong_clock_is_detected(self):
        spec = SolarPathLab.example()
        for o in spec["observations"]:
            o["observed_utc"] = astro.jd_to_iso(astro.parse_utc(o["observed_utc"]) + 900 / 86400.0, 0)      # +15 min
        r = run(SolarPathLab, spec)
        self.assertLess(r.models[0].p_value, 1e-6)


if __name__ == "__main__":
    unittest.main()
