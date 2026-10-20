"""Formatting helpers for the command-line output."""
from __future__ import annotations

import csv
import io
import json
import math
from typing import Dict, Iterable, List, Sequence

from . import timescales as ts


def utc_str(jd_tt: float, decimals: int = 0) -> str:
    jd = ts.tt_to_utc(jd_tt)
    return ts.format_jd(jd, decimals)


def tt_str(jd_tt: float, decimals: int = 1) -> str:
    return ts.format_jd(jd_tt, decimals)


def hms(seconds: float) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def latlon(lat, lon) -> str:
    if lat is None:
        return "-"
    return f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'} {abs(lon):.2f}{'E' if lon >= 0 else 'W'}"


def table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def to_csv(headers: Sequence[str], rows: Iterable[Sequence]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(headers)
    w.writerows(rows)
    return buf.getvalue()


def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def eclipse_record(e) -> Dict:
    """JSON-friendly dict for a SolarEclipse / LunarEclipse (times as TT and UTC strings)."""
    is_solar = hasattr(e, "central")
    d = jsonable(e.to_dict())
    d["body"] = "solar" if is_solar else "lunar"
    d["greatest_tt"] = tt_str(e.jd_greatest_tt, 2)
    d["greatest_utc"] = utc_str(e.jd_greatest_tt, 1)
    d["contacts"] = {k: {"tt": tt_str(v, 1), "utc": utc_str(v, 1)}
                     for k, v in sorted(e.contacts_tt.items(), key=lambda kv: kv[1])}
    d.pop("contacts_tt", None)
    d["jd_greatest_tt"] = e.jd_greatest_tt
    return d
