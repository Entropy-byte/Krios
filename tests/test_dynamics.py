import math
import unittest

import numpy as np

from krios import timescales as ts
from krios.constants import C_KM_S
from krios.dynamics import ForceModel, System, accel_newton, accel_pn1, integrate
from krios.dynamics.bodies import GM
from krios.dynamics.initial import (fit_system, make_force_model, planet_heliocentric,
                                     system_from_analytic)
from krios.dynamics.model import accel_j2
from krios.ephemeris import AnalyticEphemeris
from krios.ephemeris.integrated import IntegratedEphemeris


def two_body(a, e, gm1, gm2, epoch=2451545.0):
    gm = np.array([gm1, gm2])
    mu = gm.sum()
    rp = a * (1 - e)
    vp = math.sqrt(mu * (1 + e) / rp)
    r_rel, v_rel = np.array([rp, 0, 0.0]), np.array([0, vp, 0.0])
    m1, m2 = gm[0] / mu, gm[1] / mu
    return System(["sun", "mercury"], gm, np.array([-m2 * r_rel, m1 * r_rel]),
                  np.array([-m2 * v_rel, m1 * v_rel]), epoch), mu


class TestNewtonian(unittest.TestCase):
    def test_kepler_period_and_closure(self):
        a, e = 149_597_870.7, 0.0167
        sysm, mu = two_body(a, e, GM["sun"], GM["earth"])
        period = 2 * math.pi * math.sqrt(a ** 3 / mu)
        tr = integrate(sysm, period / 86400.0, ForceModel("newton"))
        fin = tr.final_system()
        rel0 = sysm.r[1] - sysm.r[0]
        rel1 = fin.r[1] - fin.r[0]
        self.assertLess(np.linalg.norm(rel1 - rel0), 1.0)           # km after one full orbit
        self.assertAlmostEqual(fin.energy() / sysm.energy(), 1.0, places=11)

    def test_full_system_conservation(self):
        s = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        np.testing.assert_allclose(s.barycentre()[0], 0, atol=1e-6)
        tr = integrate(s, 200.0, ForceModel("newton"), rtol=1e-12)
        fin = tr.final_system()
        self.assertLess(abs(fin.energy() / s.energy() - 1), 1e-10)
        self.assertLess(np.linalg.norm(fin.angular_momentum() / s.angular_momentum() - 1), 1e-10)
        p0 = (s.gm[:, None] * s.v).sum(axis=0)
        p1 = (fin.gm[:, None] * fin.v).sum(axis=0)
        self.assertLess(np.linalg.norm(p1 - p0) / np.linalg.norm(s.gm[:, None] * s.v), 1e-9)

    def test_time_reversibility(self):
        s = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        fwd = integrate(s, 60.0, ForceModel("newton"), rtol=1e-12).final_system()
        back = integrate(fwd, -60.0, ForceModel("newton"), rtol=1e-12).final_system()
        self.assertLess(np.abs(back.r - s.r).max(), 0.05)           # km


class TestRelativity(unittest.TestCase):
    def test_pn_reduces_to_newton_for_infinite_c(self):
        s = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        a_n = accel_newton(s.r, s.gm)
        a_p = accel_pn1(s.r, s.v, s.gm, c=1e15)
        np.testing.assert_allclose(a_p, a_n, rtol=1e-12)

    def test_pn_correction_has_the_right_size(self):
        s = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        a_n = accel_newton(s.r, s.gm)
        a_p = accel_pn1(s.r, s.v, s.gm)
        i = s.index("mercury")
        rel = np.linalg.norm(a_p[i] - a_n[i]) / np.linalg.norm(a_n[i])
        self.assertTrue(1e-8 < rel < 1e-6, rel)                       # ~ (v/c)^2 ~ 2.6e-8..1e-7

    def test_mercury_perihelion_precession_matches_general_relativity(self):
        a, e = 57_909_050.0, 0.205630
        sysm, mu = two_body(a, e, GM["sun"], GM["mercury"])
        period = 2 * math.pi * math.sqrt(a ** 3 / mu)
        rates = {}
        for kind in ("newton", "pn1"):
            tr = integrate(sysm, 10 * 365.25, ForceModel(kind), rtol=1e-12)
            t = np.arange(0, tr.t_end_s, 43200.0)
            R, V = tr.state(t)
            rr, vv = R[:, 1] - R[:, 0], V[:, 1] - V[:, 0]
            h = np.cross(rr, vv)
            ev = np.cross(vv, h) / mu - rr / np.linalg.norm(rr, axis=1)[:, None]
            ang = np.unwrap(np.arctan2(ev[:, 1], ev[:, 0]))
            rates[kind] = np.polyfit(t, ang, 1)[0] * 86400 * 36525 * (180 / math.pi) * 3600
        gr = rates["pn1"] - rates["newton"]
        theory = 6 * math.pi * mu / (C_KM_S ** 2 * a * (1 - e ** 2)) / period * 86400 * 36525 * (180 / math.pi) * 3600
        self.assertAlmostEqual(theory, 42.98, delta=0.01)
        self.assertAlmostEqual(gr / theory, 1.0, delta=0.002)
        self.assertLess(abs(rates["newton"]), 0.05)


class TestJ2AndInitialConditions(unittest.TestCase):
    def test_j2_conserves_momentum_and_is_small(self):
        s = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        m = make_force_model(s, "newton", earth_j2=True)
        a = accel_j2(s.r, s.gm, m.earth_index, m.pole)
        np.testing.assert_allclose((s.gm[:, None] * a).sum(axis=0), 0, atol=1e-12)
        im = s.index("moon")
        rel = np.linalg.norm(a[im]) / np.linalg.norm(accel_newton(s.r, s.gm)[im])
        self.assertTrue(1e-8 < rel < 1e-5, rel)

    def test_standish_planets_are_where_they_should_be(self):
        jd = ts.calendar_to_jd(2027, 1, 1)
        for name, a_au, tol in (("mercury", 0.387, 0.21), ("venus", 0.723, 0.01), ("mars", 1.524, 0.15),
                                ("jupiter", 5.203, 0.3), ("saturn", 9.537, 0.55), ("neptune", 30.07, 0.3)):
            r = np.linalg.norm(planet_heliocentric(name, jd)) / 149_597_870.7
            self.assertTrue(abs(r - a_au) < tol * a_au + 0.01, (name, r))

    def test_fit_makes_the_integrated_moon_follow_the_theory(self):
        an = AnalyticEphemeris()
        s0 = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        model = make_force_model(s0, "newton", earth_j2=True)
        fit = fit_system(s0, an, window_days=30, model=model, max_nfev=15)
        days = np.arange(2.0, 80.0, 6.0)
        jd_tdb = s0.epoch_jd_tdb + days
        jd_tt = jd_tdb - ts.tdb_minus_tt_array(jd_tdb)
        errs = {}
        for label, sysm in (("raw", s0), ("fit", fit)):
            ie = IntegratedEphemeris(integrate(sysm, 85.0, model, rtol=1e-11))
            _, m = ie.sun_moon_geocentric(jd_tt)
            _, ma = an.sun_moon_geocentric(jd_tt)
            errs[label] = np.linalg.norm(m - ma, axis=1).max()
        self.assertGreater(errs["raw"], 200.0)           # raw analytic velocities drift quickly
        self.assertLess(errs["fit"], 40.0)               # a fitted state tracks the theory to ~10 km

    def test_integrated_ephemeris_rejects_times_outside_the_span(self):
        s0 = system_from_analytic(ts.calendar_to_jd(2027, 1, 1))
        ie = IntegratedEphemeris(integrate(s0, 10.0, ForceModel("newton")))
        jd = s0.epoch_jd_tdb + 100.0
        with self.assertRaises(ValueError):
            ie.sun_moon_geocentric(jd)


if __name__ == "__main__":
    unittest.main()
