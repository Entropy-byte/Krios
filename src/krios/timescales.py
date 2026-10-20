"""Calendar, Julian date and time-scale conversions (UTC / TAI / TT / TDB / UT1).

Conventions
-----------
* All internal times are Julian dates (floats). Functions say which scale.
* UTC is only defined with leap seconds from 1972; before that we fall back
  to UT1 = TT - DeltaT and say so in the label returned by :func:`label_for`.
* DeltaT = TT - UT1 is *predicted* beyond the last measured year. The future
  value is uncertain by seconds; override with ``set_delta_t_override``.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Optional, Tuple

import numpy as np

from .constants import DAYS_PER_CENTURY, JD_J2000, SECONDS_PER_DAY, TT_MINUS_TAI


# --------------------------------------------------------------------------
# Calendar <-> Julian date (Meeus, Astronomical Algorithms, ch. 7)
# --------------------------------------------------------------------------
def calendar_to_jd(year: int, month: int, day: float, hour: float = 0.0,
                   minute: float = 0.0, second: float = 0.0,
                   gregorian: Optional[bool] = None) -> float:
    """Julian date from a calendar date. Julian calendar before 1582-10-15."""
    y, m = year, month
    if m <= 2:
        y -= 1
        m += 12
    d = day + (hour + (minute + second / 60.0) / 60.0) / 24.0
    if gregorian is None:
        gregorian = (year, month, int(math.floor(day))) >= (1582, 10, 15)
    if gregorian:
        a = math.floor(y / 100)
        b = 2 - a + math.floor(a / 4)
    else:
        b = 0
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + d + b - 1524.5)


def jd_to_calendar(jd: float) -> Tuple[int, int, int, int, int, float]:
    """(year, month, day, hour, minute, second) for a Julian date."""
    jd = jd + 0.5
    z = math.floor(jd)
    f = jd - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e)
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    secs = f * SECONDS_PER_DAY
    secs = round(secs, 3)
    if secs >= SECONDS_PER_DAY:          # rounding carried into next day
        return jd_to_calendar(z + 1 - 0.5)
    hh = int(secs // 3600)
    mm = int((secs - hh * 3600) // 60)
    ss = secs - hh * 3600 - mm * 60
    return int(year), int(month), int(day), hh, mm, ss


def parse_datetime(text: str) -> float:
    """Parse 'YYYY', 'YYYY-MM', 'YYYY-MM-DD' or 'YYYY-MM-DD[ T]HH:MM[:SS]' to a JD.

    The returned JD is in whatever scale the caller means (no conversion).
    A bare year means Jan 1 of that year, '-' years allowed (astronomical numbering).
    """
    t = text.strip().replace("T", " ")
    date_part, _, time_part = t.partition(" ")
    neg = date_part.startswith("-")
    pieces = date_part.lstrip("-").split("-")
    nums = [int(p) for p in pieces]
    year = -nums[0] if neg else nums[0]
    month = nums[1] if len(nums) > 1 else 1
    day = nums[2] if len(nums) > 2 else 1
    hh = mm = 0
    ss = 0.0
    if time_part:
        parts = time_part.split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        ss = float(parts[2]) if len(parts) > 2 else 0.0
    return calendar_to_jd(year, month, day, hh, mm, ss)


def format_jd(jd: float, decimals: int = 1) -> str:
    """'YYYY-MM-DD HH:MM:SS.s' for a Julian date (no scale label)."""
    y, mo, d, hh, mm, ss = jd_to_calendar(jd)
    width = 3 + decimals if decimals else 2
    sec = f"{ss:0{width}.{decimals}f}"
    if float(sec) >= 60.0:               # avoid 60.0 after rounding
        return format_jd(jd + 0.5 * 10 ** (-decimals) / SECONDS_PER_DAY, decimals)
    return f"{y:04d}-{mo:02d}-{d:02d} {hh:02d}:{mm:02d}:{sec}"


def jd_to_decimal_year(jd: float) -> float:
    return 2000.0 + (jd - JD_J2000) / 365.25


# --------------------------------------------------------------------------
# Leap seconds: TAI - UTC
# --------------------------------------------------------------------------
_LEAP_DATES = [
    (1972, 1, 10), (1972, 7, 11), (1973, 1, 12), (1974, 1, 13), (1975, 1, 14),
    (1976, 1, 15), (1977, 1, 16), (1978, 1, 17), (1979, 1, 18), (1980, 1, 19),
    (1981, 7, 20), (1982, 7, 21), (1983, 7, 22), (1985, 7, 23), (1988, 1, 24),
    (1990, 1, 25), (1991, 1, 26), (1992, 7, 27), (1993, 7, 28), (1994, 7, 29),
    (1996, 1, 30), (1997, 7, 31), (1999, 1, 32), (2006, 1, 33), (2009, 1, 34),
    (2012, 7, 35), (2015, 7, 36), (2017, 1, 37),
]
_LEAP_JD = [calendar_to_jd(y, m, 1) for y, m, _ in _LEAP_DATES]
_LEAP_VAL = [float(v) for _, _, v in _LEAP_DATES]
# IERS Bulletin C 72 (6 Jul 2026): no leap second at end of Dec 2026, TAI-UTC = 37 s.
LEAP_SECONDS_KNOWN_UNTIL = calendar_to_jd(2027, 1, 1)


def tai_minus_utc(jd_utc: float) -> Optional[float]:
    """TAI-UTC in seconds, or None before 1972 (no integer-second UTC)."""
    i = bisect_right(_LEAP_JD, jd_utc) - 1
    return None if i < 0 else _LEAP_VAL[i]


# --------------------------------------------------------------------------
# Delta T = TT - UT1
# --------------------------------------------------------------------------
_DT_TABLE_YEARS = np.arange(2005, 2026, dtype=float)
_DT_TABLE_VALS = np.array([
    64.69, 64.85, 65.15, 65.46, 65.78, 66.07, 66.32, 66.60, 66.91, 67.28, 67.64,
    68.10, 68.59, 68.97, 69.22, 69.36, 69.36, 69.29, 69.20, 69.18, 69.13])
_DT_LAST_YEAR = 2025.0
_DT_FUTURE_SLOPE = 0.25              # s/yr, rough prediction beyond the table
_delta_t_override: Optional[float] = None


def set_delta_t_override(seconds: Optional[float]) -> None:
    """Force a constant DeltaT (e.g. from IERS data). ``None`` restores the model."""
    global _delta_t_override
    _delta_t_override = seconds


def _espenak_poly(y: float) -> float:
    """Espenak & Meeus polynomial expressions for DeltaT (NASA), usable 1600-2150+."""
    if y < 1600:
        u = (y - 1820.0) / 100.0
        return -20.0 + 32.0 * u * u
    if y < 1700:
        t = y - 1600.0
        return 120.0 - 0.9808 * t - 0.01532 * t * t + t ** 3 / 7129.0
    if y < 1800:
        t = y - 1700.0
        return 8.83 + 0.1603 * t - 0.0059285 * t * t + 0.00013336 * t ** 3 - t ** 4 / 1174000.0
    if y < 1860:
        t = y - 1800.0
        return (13.72 - 0.332447 * t + 0.0068612 * t ** 2 + 0.0041116 * t ** 3
                - 0.00037436 * t ** 4 + 0.0000121272 * t ** 5
                - 0.0000001699 * t ** 6 + 0.000000000875 * t ** 7)
    if y < 1900:
        t = y - 1860.0
        return (7.62 + 0.5737 * t - 0.251754 * t ** 2 + 0.01680668 * t ** 3
                - 0.0004473624 * t ** 4 + t ** 5 / 233174.0)
    if y < 1920:
        t = y - 1900.0
        return (-2.79 + 1.494119 * t - 0.0598939 * t ** 2 + 0.0061966 * t ** 3
                - 0.000197 * t ** 4)
    if y < 1941:
        t = y - 1920.0
        return 21.20 + 0.84493 * t - 0.076100 * t ** 2 + 0.0020936 * t ** 3
    if y < 1961:
        t = y - 1950.0
        return 29.07 + 0.407 * t - t ** 2 / 233.0 + t ** 3 / 2547.0
    if y < 1986:
        t = y - 1975.0
        return 45.45 + 1.067 * t - t ** 2 / 260.0 - t ** 3 / 718.0
    if y < 2005:
        t = y - 2000.0
        return (63.86 + 0.3345 * t - 0.060374 * t ** 2 + 0.0017275 * t ** 3
                + 0.000651814 * t ** 4 + 0.00002373599 * t ** 5)
    if y < 2050:
        t = y - 2000.0
        return 62.92 + 0.32217 * t + 0.005589 * t * t
    if y < 2150:
        return -20.0 + 32.0 * ((y - 1820.0) / 100.0) ** 2 - 0.5628 * (2150.0 - y)
    u = (y - 1820.0) / 100.0
    return -20.0 + 32.0 * u * u


def delta_t_seconds(year: float) -> float:
    """DeltaT = TT - UT1 (s) for a decimal year (model + short-term prediction)."""
    if _delta_t_override is not None:
        return float(_delta_t_override)
    if year < 2005.0:
        return _espenak_poly(year)
    if year <= _DT_LAST_YEAR:
        return float(np.interp(year, _DT_TABLE_YEARS, _DT_TABLE_VALS))
    base = float(_DT_TABLE_VALS[-1]) + _DT_FUTURE_SLOPE * (year - _DT_LAST_YEAR)
    if year <= 2050.0:
        return base
    # beyond 2050: follow the long-term parabola's *increment*, keep continuity
    at2050 = float(_DT_TABLE_VALS[-1]) + _DT_FUTURE_SLOPE * (2050.0 - _DT_LAST_YEAR)
    return at2050 + (_espenak_poly(year) - _espenak_poly(2050.0))


def delta_t_for_jd(jd_tt: float) -> float:
    return delta_t_seconds(jd_to_decimal_year(jd_tt))


# --------------------------------------------------------------------------
# Scale conversions (scalars)
# --------------------------------------------------------------------------
def tt_to_ut1(jd_tt: float) -> float:
    return jd_tt - delta_t_for_jd(jd_tt) / SECONDS_PER_DAY


def ut1_to_tt(jd_ut1: float) -> float:
    jd = jd_ut1
    for _ in range(3):
        jd = jd_ut1 + delta_t_for_jd(jd) / SECONDS_PER_DAY
    return jd


def utc_to_tt(jd_utc: float) -> float:
    """UTC -> TT. Before 1972 treats the input as UT1."""
    d = tai_minus_utc(jd_utc)
    if d is None:
        return ut1_to_tt(jd_utc)
    return jd_utc + (d + TT_MINUS_TAI) / SECONDS_PER_DAY


def tt_to_utc(jd_tt: float) -> float:
    """TT -> UTC (UT1 before 1972 or beyond reliable leap-second knowledge)."""
    guess = jd_tt - (37.0 + TT_MINUS_TAI) / SECONDS_PER_DAY
    for _ in range(2):
        d = tai_minus_utc(guess)
        if d is None:
            return tt_to_ut1(jd_tt)
        guess = jd_tt - (d + TT_MINUS_TAI) / SECONDS_PER_DAY
    return guess


def utc_label(jd_tt: float) -> str:
    """'UTC' when atomic-time UTC applies, else 'UT1'."""
    d = tai_minus_utc(tt_to_utc(jd_tt))
    return "UT1" if d is None else "UTC"


def tdb_minus_tt_seconds(jd_tt: float) -> float:
    """TDB - TT (s), leading periodic terms (~1.7 ms amplitude)."""
    g = math.radians(357.53 + 0.98560028 * (jd_tt - JD_J2000))
    return 0.001657 * math.sin(g) + 0.000014 * math.sin(2.0 * g)


def jd_tt_to_tdb(jd_tt: float) -> float:
    return jd_tt + tdb_minus_tt_seconds(jd_tt) / SECONDS_PER_DAY


def jd_tdb_to_tt(jd_tdb: float) -> float:
    return jd_tdb - tdb_minus_tt_seconds(jd_tdb) / SECONDS_PER_DAY


def julian_centuries_tt(jd_tt):
    return (np.asarray(jd_tt, dtype=float) - JD_J2000) / DAYS_PER_CENTURY


def tdb_minus_tt_array(jd_tt):
    """Vectorised TDB - TT in days."""
    jd = np.asarray(jd_tt, dtype=float)
    g = np.radians(357.53 + 0.98560028 * (jd - JD_J2000))
    return (0.001657 * np.sin(g) + 0.000014 * np.sin(2.0 * g)) / SECONDS_PER_DAY
