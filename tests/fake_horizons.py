"""A fake Horizons server for offline tests: answers VECTORS queries from the analytic theory."""
import json
import urllib.parse

import numpy as np

from krios import timescales as ts
from krios.ephemeris import AnalyticEphemeris

_MON = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
_AN = AnalyticEphemeris()


def _parse(s):
    d, t = s.split()
    y, m, dd = map(int, d.split("-"))
    h, mi = map(int, t.split(":"))
    return ts.calendar_to_jd(y, m, dd, h, mi)


def make_transport(counter=None):
    def transport(url):
        if counter is not None:
            counter.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        g = lambda k: q[k][0].strip("'")
        cmd = g("COMMAND")
        if g("CENTER") != "500@399" or g("EPHEM_TYPE") != "VECTORS" or g("CSV_FORMAT") != "YES":
            return json.dumps({"error": "unsupported request in fake server"})
        a, b = _parse(g("START_TIME")), _parse(g("STOP_TIME"))
        step = int(g("STEP_SIZE").split()[0]) / 1440.0
        jd = a + np.arange(int(round((b - a) / step)) + 1) * step      # exact grid, like the real service
        jd_tt = jd - ts.tdb_minus_tt_array(jd)
        h = 30.0 / 86400.0
        s, m = _AN.sun_moon_geocentric(jd_tt)
        s1, m1 = _AN.sun_moon_geocentric(jd_tt + h)
        s2, m2 = _AN.sun_moon_geocentric(jd_tt - h)
        pos, vel = (s, (s1 - s2) / 60.0) if cmd == "10" else (m, (m1 - m2) / 60.0)
        lines = []
        for t, p, v in zip(jd, pos, vel):
            y, mo, d, hh, mi, ss = ts.jd_to_calendar(t)
            nums = ", ".join("%.15E" % x for x in list(p) + list(v))
            lines.append("%.9f, A.D. %04d-%s-%02d %02d:%02d:%07.4f, %s," % (t, y, _MON[mo - 1], d, hh, mi, ss, nums))
        text = "*** header ***\n$$SOE\n" + "\n".join(lines) + "\n$$EOE\n*** trailer ***"
        return json.dumps({"signature": {"source": "fake"}, "result": text})
    return transport


def _format(jd, pos, vel):
    lines = []
    for t, p, v in zip(jd, pos, vel):
        y, mo, d, hh, mi, ss = ts.jd_to_calendar(t)
        nums = ", ".join("%.15E" % x for x in list(p) + list(v))
        lines.append("%.9f, A.D. %04d-%s-%02d %02d:%02d:%07.4f, %s," % (t, y, _MON[mo - 1], d, hh, mi, ss, nums))
    return json.dumps({"signature": {"source": "fake"},
                       "result": "*** header ***\n$$SOE\n" + "\n".join(lines) + "\n$$EOE\n*** trailer ***"})


def make_function_transport(func, counter=None):
    """Fake server whose answer for (command, center, jd_tdb array) is ``func(...) -> (pos, vel)``."""
    def transport(url):
        if counter is not None:
            counter.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        g = lambda k: q[k][0].strip("'")
        a, b = _parse(g("START_TIME")), _parse(g("STOP_TIME"))
        step = int(g("STEP_SIZE").split()[0]) / 1440.0
        jd = a + np.arange(int(round((b - a) / step)) + 1) * step      # exact grid, like the real service
        pos, vel = func(g("COMMAND"), g("CENTER"), jd)
        return _format(jd, pos, vel)
    return transport
