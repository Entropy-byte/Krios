import math
import unittest

from krios.lab import astro
from krios.lab.shadow import ShadowLab


def run(cls, spec, **kw):
    return cls(spec).run(**kw)


class TestShadowLab(unittest.TestCase):
    def test_example_globe_fits_flat_fails_and_radius_is_recovered(self):
        r = run(ShadowLab, ShadowLab.example())
        g, f1, f2 = r.models
        self.assertGreater(g.p_value, 0.1)
        self.assertLess(f1.p_value, 1e-6)
        self.assertLess(f2.p_value, 1e-3)
        s = r.summary
        self.assertLess(abs(s["R_estimate_km"] - 6371.0), 3 * s["R_sigma_km"])
        self.assertLess(s["R_sigma_km"], 150)
        self.assertGreater(s["flat_implied_Sun_height_ratio"], 2.0)
        self.assertTrue(any("Every flat-map variant is rejected" in v["en"] for v in r.verdict))

    def test_the_test_is_not_rigged_flat_world_data_favours_the_flat_model(self):
        """Data generated from a FLAT world (Sun 5000 km above the sub-solar map point) must not be explained by the globe."""
        jd = astro.parse_utc("2026-10-12T11:10:00Z")
        sub = astro.subsolar_point(jd)
        xs, ys = astro.fe_xy(sub["lat"][0], sub["lon"][0])
        stations = []
        for i, (la, lo) in enumerate([(60, 10), (45, 10), (30, 10), (15, 10), (0, 10), (20, 60), (35, -40)]):
            x, y = astro.fe_xy(la, lo)
            h = math.degrees(math.atan2(5000.0, math.hypot(x - xs, y - ys)))
            stations.append({"name": f"F{i}", "lat": la, "lon": lo, "time_utc": "2026-10-12T11:10:00Z", "stick": 1.0,
                             "shadow": round(1.0 / math.tan(math.radians(h)), 4), "sigma_stick": 0.001, "sigma_shadow": 0.002})
        r = run(ShadowLab, {"settings": {"sigma_floor_deg": 0.05}, "stations": stations})
        g, f1, f2 = r.models
        self.assertLess(g.p_value, 1e-6)                    # globe prediction fails on flat-world data
        self.assertGreater(f1.p_value, 0.01)                # fitted flat model fits
        self.assertAlmostEqual(f1.params["H_km"], 5000.0, delta=150)
        self.assertTrue(any("fits better than the globe" in v["en"] for v in r.verdict))

    def test_two_stations_cannot_discriminate(self):
        spec = ShadowLab.example()
        spec["stations"], spec["baselines"] = spec["stations"][:2], spec["baselines"][:1]
        r = run(ShadowLab, spec)
        self.assertTrue(any("fewer than 3 stations" in v["en"] for v in r.verdict))
        self.assertTrue(r.models[-1].dof <= 0 or math.isnan(r.models[-1].p_value))

    def test_missing_measurement_is_an_error_but_plan_mode_is_fine(self):
        spec = {"stations": [{"name": "A", "lat": 10.0, "lon": 10.0, "time_utc": "2026-10-12T11:00:00Z"}]}
        with self.assertRaises(ValueError):
            run(ShadowLab, spec)
        self.assertTrue(run(ShadowLab, spec, predict_only=True).tables["predictions"][0]["sun_alt_deg"] > 0)


if __name__ == "__main__":
    unittest.main()
