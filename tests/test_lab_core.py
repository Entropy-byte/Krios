import json
import os
import tempfile
import unittest

from krios.lab import astro, check_plan, load_spec, make_plan
from krios.lab.core import spec_hash
from krios.lab.shadow import ShadowLab


def run(cls, spec, **kw):
    return cls(spec).run(**kw)


class TestFramework(unittest.TestCase):
    def test_reports_are_bilingual_and_carry_provenance(self):
        res = run(ShadowLab, ShadowLab.example())
        en, ar = res.to_markdown("en"), res.to_markdown("ar")
        self.assertIn("SYNTHETIC DATA", en)
        self.assertIn("بيانات اصطناعية", ar)
        self.assertIn("الخلاصة", ar)
        self.assertEqual(res.provenance["krios_version"], __import__("krios").__version__)
        self.assertEqual(len(res.provenance["input_sha256"]), 64)
        js = json.loads(res.to_json())
        self.assertEqual(js["experiment"], "shadow")
        self.assertIn("comparison", js["tables"])

    def test_results_are_reproducible(self):
        a, b = run(ShadowLab, ShadowLab.example()), run(ShadowLab, ShadowLab.example())
        self.assertEqual(a.prediction_sha256, b.prediction_sha256)
        self.assertEqual(a.summary["R_estimate_km"], b.summary["R_estimate_km"])       # seeded Monte Carlo

    def test_csv_loader_and_table_export(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "stations.csv")
            with open(p, "w") as f:
                f.write("# comment\nname,lat,lon,time_utc,stick,shadow\nA,45.0,10.0,2026-10-12T11:10:00Z,1.0,1.35\nB,10.0,10.0,2026-10-12T11:10:00Z,1.0,0.3\n")
            spec = load_spec(p, "stations")
            self.assertEqual(len(spec["stations"]), 2)
            self.assertEqual(spec["stations"][0]["lat"], 45.0)
            res = run(ShadowLab, spec)
            res.write(csv_dir=d)
            self.assertTrue(os.path.exists(os.path.join(d, "shadow_comparison.csv")))

    def test_plan_before_measurement_and_verification(self):
        spec = ShadowLab.example()
        planning = {**spec, "stations": [{k: v for k, v in s.items() if k != "shadow"} for s in spec["stations"]]}
        plan_res = run(ShadowLab, planning, predict_only=True)
        plan = make_plan(plan_res)
        full = run(ShadowLab, spec)
        self.assertEqual(plan["prediction_sha256"], full.prediction_sha256)       # predictions never used the measurements
        self.assertIn("IDENTICAL", check_plan(full, plan))
        moved = json.loads(json.dumps(spec))
        moved["stations"][0]["lat"] += 0.5
        self.assertIn("DIFFER", check_plan(run(ShadowLab, moved), plan))
        self.assertNotEqual(spec_hash(spec), spec_hash(moved))


class TestAstroHelpers(unittest.TestCase):
    def test_latitude_from_sun_altitude_round_trip(self):
        jd = astro.parse_utc("2026-10-12T11:10:00Z")
        for lat in (-40.0, -5.0, 20.0, 55.0):
            t = astro.topocentric("sun", lat, 10.0, 0.0, jd)
            sh = astro.sun_declination_hour_angle(10.0, jd)
            back = astro.latitude_from_sun_altitude(t["alt_geom"][0], sh["dec"][0], sh["H"][0], lat + 3.0)
            self.assertAlmostEqual(float(back), lat, places=2)      # solar parallax (8.8") limits the round trip

    def test_refraction_round_trip_and_size(self):
        self.assertAlmostEqual(float(astro.refraction_deg(0.0)) * 60, 28.8, delta=2.0)      # ~ 0.5 deg at the horizon (arcmin)
        self.assertLess(float(astro.refraction_deg(45.0)) * 60, 1.1)
        for h in (3.0, 20.0, 60.0):
            self.assertAlmostEqual(float(astro.true_alt(astro.apparent_alt(h))), h, places=4)

    def test_polaris_is_not_the_pole(self):
        off = astro.polaris_offset_deg(astro.parse_utc("2026-10-12T21:00:00Z"))
        self.assertTrue(0.55 < off < 0.7, off)
        alt, az = astro.polaris_alt_az(45.0, 10.0, astro.parse_utc("2026-10-12T21:00:00Z"))
        self.assertTrue(abs(alt - 45.0) < 0.75)

    def test_parse_utc_offsets(self):
        self.assertAlmostEqual(astro.parse_utc("2027-08-02T12:00:00+02:00"), astro.parse_utc("2027-08-02T10:00:00Z"), places=9)


if __name__ == "__main__":
    unittest.main()
