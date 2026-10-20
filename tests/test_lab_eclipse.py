import unittest

from krios.lab import astro
from krios.lab.eclipse_lab import EclipseLab


def run(cls, spec, **kw):
    return cls(spec).run(**kw)


class TestEclipseLab(unittest.TestCase):
    def test_solar_example_is_consistent_and_a_wrong_clock_is_not(self):
        r = run(EclipseLab, EclipseLab.example())
        self.assertGreater(r.models[0].p_value, 0.1)
        self.assertEqual(r.summary["type_here"], "Total")
        bad = EclipseLab.example()
        for o in bad["observations"]:
            o["time_utc"] = astro.jd_to_iso(astro.parse_utc(o["time_utc"]) + 120 / 86400.0, 1)
        self.assertLess(run(EclipseLab, bad).models[0].p_value, 0.01)

    def test_lunar_eclipse_has_observer_independent_contacts(self):
        r = run(EclipseLab, {"eclipse": {"date": "2026-03-03", "body": "lunar"}}, predict_only=True)
        events = [p["event"] for p in r.tables["predictions"]]
        self.assertEqual(events, ["P1", "U1", "U2", "Greatest", "U3", "U4", "P4"])
        g = [p for p in r.tables["predictions"] if p["event"] == "Greatest"][0]
        self.assertTrue(g["time_utc"].startswith("2026-03-03T11:33"))              # 11:34:52 TD - 69 s

    def test_no_eclipse_is_an_error(self):
        with self.assertRaises(ValueError):
            run(EclipseLab, {"eclipse": {"date": "2026-11-15", "body": "solar"}, "observer": {"lat": 0, "lon": 0}}, predict_only=True)


if __name__ == "__main__":
    unittest.main()
