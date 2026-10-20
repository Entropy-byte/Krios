import math
import unittest

import numpy as np

from krios.lab import astro
from krios.lab.parallax import ParallaxLab, triangulate


def run(cls, spec, **kw):
    return cls(spec).run(**kw)


class TestParallaxLab(unittest.TestCase):
    def test_triangulation_recovers_a_known_point(self):
        pts = np.array([[0.0, 0, 0], [6000.0, 0, 0], [0, 6000.0, 0]])
        target = np.array([1000.0, 2000.0, 380000.0])
        dirs = np.array([(target - p) / np.linalg.norm(target - p) for p in pts])
        np.testing.assert_allclose(triangulate(pts, dirs), target, rtol=1e-9)

    def test_example_distance_globe_vs_flat(self):
        r = run(ParallaxLab, ParallaxLab.example())
        g = r.tables["groups"][0]
        self.assertLess(abs(g["distance_km"] - g["ephemeris_distance_km"]), 3 * g["sigma_km"])
        self.assertLess(g["sigma_km"] / g["distance_km"], 0.08)
        self.assertGreater(g["lower_bound_3sigma_km"], 250000)
        m = {x.name: x for x in r.models}
        self.assertGreater(m["Globe triangulation consistency"].p_value, 0.01)
        self.assertLess(m["Flat-map triangulation consistency"].p_value, 1e-6)
        self.assertAlmostEqual(g["parallax_measured_deg"], g["parallax_expected_deg"], delta=0.1)
        self.assertTrue(any("5,000 km is" in v["en"] for v in r.verdict))
        again = run(ParallaxLab, ParallaxLab.example())
        self.assertEqual(again.tables["groups"][0]["sigma_km"], g["sigma_km"])       # seeded Monte Carlo

    def test_the_test_is_not_rigged_flat_world_sightings_favour_flat_geometry(self):
        """Moon 5000 km above a flat map: the lines of sight meet in the flat geometry, not in the globe geometry."""
        stations = [("N", 55.0, 0.0), ("M", 30.0, 90.0), ("S", 10.0, 180.0), ("W", 40.0, -90.0)]
        moon = np.array([6000.0, -2000.0, 5000.0])
        obs = []
        for name, la, lo in stations:
            x, y = astro.fe_xy(la, lo)
            r = math.hypot(x, y)
            pos, east, north = np.array([x, y, 0.0]), np.array([-y / r, x / r, 0.0]), np.array([-x / r, -y / r, 0.0])
            v = moon - pos
            alt = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
            az = math.degrees(math.atan2(v @ east, v @ north)) % 360
            obs.append({"station": name, "lat": la, "lon": lo, "time_utc": "2026-10-25T22:00:00Z",
                        "alt_deg": float(astro.apparent_alt(alt)), "az_deg": az, "sigma_deg": 0.05})
        r = run(ParallaxLab, {"settings": {"mc_samples": 200}, "observations": obs})
        m = {x.name: x for x in r.models}
        self.assertGreater(m["Flat-map triangulation consistency"].p_value, 0.01)
        self.assertLess(m["Globe triangulation consistency"].p_value, 1e-3)
        self.assertTrue(any("do not meet consistently even in globe" in v["en"] for v in r.verdict))

    def test_short_baseline_is_reported_as_useless(self):
        ex = ParallaxLab.example()
        near = [dict(ex["observations"][0]), dict(ex["observations"][0])]
        near[1]["station"], near[1]["lat"] = "Next door", near[0]["lat"] + 0.5
        t = astro.topocentric("moon", near[1]["lat"], near[1]["lon"], 0.0, astro.parse_utc(near[1]["time_utc"]))
        near[1]["alt_deg"], near[1]["az_deg"] = float(t["alt_app"][0]), float(t["az"][0])
        r = run(ParallaxLab, {"settings": {"mc_samples": 300}, "observations": near})
        self.assertTrue(any("baseline is too short" in v["en"] for v in r.verdict))

    def test_single_station_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            run(ParallaxLab, {"observations": [ParallaxLab.example()["observations"][0]]})


if __name__ == "__main__":
    unittest.main()
