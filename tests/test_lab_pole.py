import unittest

from krios.lab.pole import PoleLab, pixel_altitude


def run(cls, spec, **kw):
    return cls(spec).run(**kw)


class TestPoleLab(unittest.TestCase):
    def test_pixel_altitude_geometry(self):
        cam = {"elevation_deg": 30.0, "vfov_deg": 40.0, "image_width_px": 6000, "image_height_px": 4000}
        self.assertAlmostEqual(pixel_altitude(3000, 2000, cam), 30.0, places=9)
        self.assertGreater(pixel_altitude(3000, 1000, cam), 30.0)           # above the centre of the frame = higher
        self.assertAlmostEqual(pixel_altitude(3000, 0, cam), 50.0, places=6)  # top edge = elevation + vfov/2

    def test_example_across_seeds(self):
        for seed in (11, 12, 13, 14):
            r = run(PoleLab, PoleLab.example(seed))
            for m in r.tables["measurements"]:
                self.assertLess(abs(m["residual_deg"]), 3 * m["sigma_deg"], (seed, m["site"]))
                self.assertTrue(m["rotation_ok"], (seed, m["site"]))
            self.assertAlmostEqual(r.summary["rate_over_sidereal"], 1.0, delta=0.02)
            self.assertLess(abs(r.summary["fit_slope"] - 1.0), 3 * r.summary["fit_slope_sigma"])
            self.assertLess(r.models[2].p_value, 1e-6)                    # single-pole flat model fails
            self.assertTrue(any("CLOCKWISE" in v["en"] for v in r.verdict))

    def test_predictions_do_not_need_images(self):
        r = run(PoleLab, {"sites": [{"name": "X", "lat": -33.9, "lon": 151.2, "time_utc": "2026-10-12T20:00:00Z"}]}, predict_only=True)
        p = r.tables["predictions"][0]
        self.assertEqual((p["expected_pole"], p["expected_rotation"]), ("south", "cw"))
        self.assertAlmostEqual(p["pole_alt_geometric_deg"], 33.9)

    def test_manual_altitude_and_wrong_rotation_is_flagged(self):
        spec = {"sites": [{"name": "A", "lat": 40.0, "pole_altitude_deg": 40.2, "sigma_deg": 0.5, "rotation": "cw"},
                          {"name": "B", "lat": 20.0, "pole_altitude_deg": 19.9, "sigma_deg": 0.5, "rotation": "ccw"},
                          {"name": "C", "lat": 60.0, "pole_altitude_deg": 60.3, "sigma_deg": 0.5, "rotation": "ccw"}]}
        r = run(PoleLab, spec)
        self.assertFalse(r.tables["measurements"][0]["rotation_ok"])
        self.assertTrue(any("rotation sense different" in v["en"] for v in r.verdict))


if __name__ == "__main__":
    unittest.main()
