import unittest

from krios import timescales as ts
from krios.eclipse import EclipseConfig, solar_global
from krios.eclipse.path import central_line
from krios.ephemeris import AnalyticEphemeris


class TestEclipsePath(unittest.TestCase):
    def test_central_line_2027(self):
        ephem, cfg = AnalyticEphemeris(), EclipseConfig()
        ecl = solar_global(ephem, cfg, ts.calendar_to_jd(2027, 8, 2, 10, 0, 0))
        gj = central_line(ephem, cfg, ecl, step_s=120)
        self.assertEqual(gj["type"], "FeatureCollection")
        kinds = [f["properties"]["kind"] for f in gj["features"]]
        self.assertEqual(kinds[:2], ["central_line", "greatest_eclipse"])
        line = gj["features"][0]["geometry"]["coordinates"]
        self.assertTrue(-46 < line[0][0] < -41 and 88 < line[-1][0] < 92, (line[0], line[-1]))   # sunrise limb (Atlantic) -> sunset limb (Indian Ocean)
        g = gj["features"][1]
        self.assertAlmostEqual(g["geometry"]["coordinates"][0], 33.18, delta=0.05)
        self.assertAlmostEqual(g["geometry"]["coordinates"][1], 25.50, delta=0.05)
        widths = [f["properties"]["path_width_km"] for f in gj["features"] if f["properties"]["kind"] == "sample"]
        self.assertTrue(250 < max(widths) < 265, max(widths))                    # NASA: 257.7 km at greatest eclipse
        # every vertex lies close to the great-eclipse latitude band of the real path (Atlantic .. Somalia)
        self.assertTrue(all(-40 < c[1] < 40 for c in line))

    def test_partial_eclipse_has_no_line(self):
        ephem, cfg = AnalyticEphemeris(), EclipseConfig()
        ecl = solar_global(ephem, cfg, ts.calendar_to_jd(2025, 3, 29, 10, 48, 0))
        with self.assertRaises(ValueError):
            central_line(ephem, cfg, ecl)


if __name__ == "__main__":
    unittest.main()
