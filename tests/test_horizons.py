import json
import tempfile
import unittest

import numpy as np

from krios import timescales as ts
from krios.eclipse import EclipseConfig, solar_global
from krios.ephemeris import AnalyticEphemeris
from krios.dynamics import ForceModel, integrate
from krios.dynamics.bodies import DEFAULT_BODIES, GM, HORIZONS_ID
from krios.dynamics.initial import system_from_horizons
from krios.ephemeris.horizons import (HorizonsClient, HorizonsEphemeris, HorizonsError,
                                       hermite_interpolate, hermite_state, parse_vectors)
from tests.fake_horizons import make_function_transport, make_transport

SAMPLE = """*** JPL Horizons header ***
$$SOE
2461619.250000000, A.D. 2027-Aug-02 18:00:00.0000, 1.0E+08, -2.0E+08, 3.0E+07, 4.5E+01, -5.5E+00, 6.5E-01,
2461619.250694444, A.D. 2027-Aug-02 18:01:00.0000, 1.1E+08, -2.1E+08, 3.1E+07, 4.6E+01, -5.6E+00, 6.6E-01,
$$EOE
*** end ***"""


class TestParser(unittest.TestCase):
    def test_parse_vectors(self):
        t = parse_vectors(SAMPLE, "10", "500@399")
        self.assertEqual(t.pos_km.shape, (2, 3))
        self.assertAlmostEqual(t.jd_tdb[1] - t.jd_tdb[0], 1 / 1440, places=7)
        self.assertEqual(t.pos_km[0, 1], -2.0e8)
        self.assertEqual(t.vel_km_s[1, 2], 0.66)

    def test_errors_are_explained(self):
        with self.assertRaises(HorizonsError) as cm:
            parse_vectors("No ephemeris for target 'xyz' prior to A.D. 1900-JAN-01")
        self.assertIn("No ephemeris", str(cm.exception))
        with self.assertRaises(HorizonsError):
            parse_vectors("$$SOE\nnot,a,row\n$$EOE")

    def test_hermite_is_exact_for_cubics(self):
        t = np.array([0.0, 0.5, 1.0])
        f = lambda x: 3 * x ** 3 - 2 * x ** 2 + x + 5
        df = lambda x: (9 * x ** 2 - 4 * x + 1) / 86400.0          # km/s with t in days
        pos = np.array([[f(x), 0, 0] for x in t])
        vel = np.array([[df(x), 0, 0] for x in t])
        q = np.array([0.1, 0.33, 0.9])
        np.testing.assert_allclose(hermite_interpolate(t, pos, vel, q)[:, 0], f(q), atol=1e-12)


class TestClientAndProvider(unittest.TestCase):
    def test_query_parameters_follow_the_api(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            c = HorizonsClient(cache_dir=d, transport=make_transport(calls))
            c.vectors("301", "500@399", 2461619.5, 2461619.6, 10)
        url = calls[0]
        self.assertTrue(url.startswith("https://ssd.jpl.nasa.gov/api/horizons.api?"))
        for needle in ("format=json", "COMMAND=%27301%27", "EPHEM_TYPE=%27VECTORS%27", "CENTER=%27500%40399%27",
                       "VEC_TABLE=%272%27", "CSV_FORMAT=%27YES%27", "REF_PLANE=%27FRAME%27", "OUT_UNITS=%27KM-S%27",
                       "STEP_SIZE=%2710%20min%27", "TIME_TYPE=%27TDB%27"):
            self.assertIn(needle, url)

    def test_cache_and_offline(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            c1 = HorizonsClient(cache_dir=d, transport=make_transport(calls))
            a = c1.vectors("10", "500@399", 2461619.5, 2461619.6, 10)
            b = c1.vectors("10", "500@399", 2461619.5, 2461619.6, 10)       # served from cache
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(a.pos_km, b.pos_km)
            c2 = HorizonsClient(cache_dir=d, offline=True)
            c2.vectors("10", "500@399", 2461619.5, 2461619.6, 10)
            with self.assertRaises(HorizonsError):
                c2.vectors("301", "500@399", 2461619.5, 2461619.6, 10)       # never fetched

    def test_server_error_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            c = HorizonsClient(cache_dir=d, transport=lambda url: json.dumps({"error": "bad command"}))
            with self.assertRaises(HorizonsError):
                c.vectors("301", "500@399", 2461619.5, 2461619.6, 10)

    def test_eclipse_through_the_horizons_pipeline_matches_direct_analytic(self):
        with tempfile.TemporaryDirectory() as d:
            client = HorizonsClient(cache_dir=d, transport=make_transport())
            he = HorizonsEphemeris(client, step_minutes=5)
            jd = ts.calendar_to_jd(2027, 8, 2, 10, 0, 0)
            via = solar_global(he, EclipseConfig(), jd)
            direct = solar_global(AnalyticEphemeris(), EclipseConfig(), jd)
            self.assertLess(abs(via.jd_greatest_tt - direct.jd_greatest_tt) * 86400, 0.05)
            self.assertAlmostEqual(via.gamma, direct.gamma, places=5)
            self.assertEqual(via.kind, direct.kind)
            self.assertGreater(client.requests_made, 0)
            # interpolation quality
            jds = jd + np.array([0.0, 0.0123, -0.07])
            s_h, m_h = he.sun_moon_geocentric(jds)
            s_a, m_a = AnalyticEphemeris().sun_moon_geocentric(jds)
            self.assertLess(np.abs(m_h - m_a).max(), 0.01)       # km
            self.assertLess(np.abs(s_h - s_a).max(), 0.05)


JD0 = 2461000.5


def linear_motion(command, center, jd):
    """x = 1e5 km + 1 km/s * t, exactly linear (regression case for state_at)."""
    t = (jd - JD0) * 86400.0
    pos = np.stack([1e5 + t, 0 * t, 0 * t], axis=1)
    vel = np.tile([1.0, 0.0, 0.0], (len(jd), 1))
    return pos, vel


# synthetic "solar system": body k on a circular orbit of radius r_k around the barycentre, ~10 km/s each,
# far enough apart (1e9 km) that mutual gravity is negligible over the short test integration
RADII = {b: 1.0e9 * (k + 1) for k, b in enumerate(DEFAULT_BODIES)}
OMEGA = {b: 1.0e-8 / (k + 1) for k, b in enumerate(DEFAULT_BODIES)}
ID_TO_BODY = {v: k for k, v in HORIZONS_ID.items()}


def circular_system(command, center, jd):
    b = ID_TO_BODY[command]
    t = (jd - JD0) * 86400.0
    r, w = RADII[b], OMEGA[b]
    pos = np.stack([r * np.cos(w * t), r * np.sin(w * t), 0.1 * r * np.sin(w * t)], axis=1)
    vel = np.stack([-r * w * np.sin(w * t), r * w * np.cos(w * t), 0.1 * r * w * np.cos(w * t)], axis=1)
    return pos, vel


class TestStateAtVelocity(unittest.TestCase):
    """Regression: state_at() used to return a ZERO velocity for epochs between minute nodes."""

    def test_hermite_state_derivative_is_exact_for_cubics(self):
        t = np.array([0.0, 0.5, 1.0])
        f = lambda x: 3 * x ** 3 - 2 * x ** 2 + x + 5
        df = lambda x: (9 * x ** 2 - 4 * x + 1) / 86400.0
        pos = np.array([[f(x), 0, 0] for x in t])
        vel = np.array([[df(x), 0, 0] for x in t])
        q = np.array([0.1, 0.33, 0.9])
        p, v = hermite_state(t, pos, vel, q)
        np.testing.assert_allclose(v[:, 0], df(q), rtol=1e-12)
        np.testing.assert_allclose(p[:, 0], f(q), atol=1e-12)

    def test_off_node_velocity_regression(self):
        with tempfile.TemporaryDirectory() as d:
            c = HorizonsClient(cache_dir=d, transport=make_function_transport(linear_motion))
            for offset_s in (0.0, 10.0, 30.0, 59.0, 59.99):
                jd = JD0 + 7 / 1440 + offset_s / 86400.0
                pos, vel = c.state_at("999", "@0", jd)
                np.testing.assert_allclose(vel, [1.0, 0.0, 0.0], atol=1e-9, err_msg=f"offset {offset_s}s")
                self.assertAlmostEqual(pos[0], 1e5 + (jd - JD0) * 86400.0, delta=1e-3)

    def test_curved_motion_velocity_is_accurate_between_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            c = HorizonsClient(cache_dir=d, transport=make_function_transport(circular_system))
            jd = JD0 + 3.0 + 17.3 / 86400.0
            for body in ("moon", "earth", "mercury"):
                pos, vel = c.state_at(HORIZONS_ID[body], "@0", jd)
                p_ref, v_ref = circular_system(HORIZONS_ID[body], "@0", np.array([jd]))
                self.assertLess(np.linalg.norm(vel - v_ref[0]) / np.linalg.norm(v_ref[0]), 1e-7, body)   # float64-JD floor ~3e-9
                self.assertLess(np.linalg.norm(pos - p_ref[0]), 1e-3, body)

    def test_system_from_horizons_has_velocities_and_integrates(self):
        with tempfile.TemporaryDirectory() as d:
            c = HorizonsClient(cache_dir=d, transport=make_function_transport(circular_system))
            # an epoch that is NOT on a whole minute, like every real UTC -> TDB conversion
            jd = ts.jd_tt_to_tdb(ts.utc_to_tt(JD0 + 2.0)) + 0.0
            self.assertGreater(abs(jd * 1440 - round(jd * 1440)), 1e-3)
            sysm = system_from_horizons(c, jd)
            for k, name in enumerate(sysm.names):
                _, v_ref = circular_system(HORIZONS_ID[name], "@0", np.array([jd]))
                self.assertGreater(np.linalg.norm(sysm.v[k]), 0.0, name)
                self.assertLess(np.linalg.norm(sysm.v[k] - v_ref[0]) / np.linalg.norm(v_ref[0]), 1e-7, name)
            # the state is usable: a short integration moves the bodies along their velocities
            tr = integrate(sysm, 0.01, ForceModel("newton"))
            moved = tr.final_system().r - sysm.r
            expected = sysm.v * 0.01 * 86400.0
            self.assertLess(np.abs(moved - expected).max(), 1e-3 * np.abs(expected).max())


    def test_rounded_table_times_do_not_leak_into_velocity(self):
        """Horizons prints JD with 9 decimals (8.6e-5 s); node spacing must come from the exact grid."""
        with tempfile.TemporaryDirectory() as d:
            c = HorizonsClient(cache_dir=d, transport=make_function_transport(circular_system))
            tab = c.vectors(HORIZONS_ID["earth"], "@0", JD0 + 3.0, JD0 + 3.0 + 5 / 1440, 1)
            grid = tab.jd_tdb[0] + np.arange(len(tab.jd_tdb)) * (1 / 1440)
            self.assertLess(np.abs(tab.jd_tdb - grid).max(), 1e-12)     # printed 9-decimal JDs would be off by ~5e-10


if __name__ == "__main__":
    unittest.main()
