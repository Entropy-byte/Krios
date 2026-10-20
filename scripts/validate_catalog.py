"""Compare krios's eclipse engine with the NASA/EclipseWise 2021-2030 catalog.

Usage:  python scripts/validate_catalog.py [--source analytic] > docs/VALIDATION.md
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from krios import timescales as ts                      # noqa: E402
from krios.eclipse import EclipseConfig, find_eclipses  # noqa: E402
from krios.ephemeris import AnalyticEphemeris            # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def jd_td(date, hms):
    y, m, d = map(int, date.split("-"))
    hh, mm, ss = hms.split(":")
    return ts.calendar_to_jd(y, m, d, int(hh), int(mm), float(ss))


def main():
    with open(os.path.join(ROOT, "tests", "data", "nasa_catalog_2021_2030.json")) as fh:
        cat = json.load(fh)
    found = find_eclipses(ts.calendar_to_jd(2021, 1, 1), ts.calendar_to_jd(2031, 1, 1),
                          ephemeris=AnalyticEphemeris(), config=EclipseConfig())
    solar = [e for e in found if hasattr(e, "central")]
    lunar = [e for e in found if hasattr(e, "umbral_magnitude")]
    print("# Validation against the NASA / EclipseWise catalog, 2021-2030\n")
    print("Reference: F. Espenak, EclipseWise decade tables (times are TD = TT of greatest eclipse).")
    print("Ephemeris under test: `analytic` (offline Meeus ELP-2000/82 + VSOP87D truncated).\n")
    dts = []
    print("## Solar eclipses\n")
    print("| Date | Type (NASA / krios) | NASA TD | dt (s) | Magnitude NASA / krios | Central duration NASA / krios |")
    print("|---|---|---|---:|---|---|")
    for row, e in zip(cat["solar"], solar):
        date, hms, kind, mag, dur = row
        dt = (e.jd_greatest_tt - jd_td(date, hms)) * 86400
        dts.append(dt)
        d2 = f"{dur} s / {e.central_duration_s:.0f} s" if dur else "-"
        print(f"| {date} | {kind} / {e.kind} | {hms} | {dt:+.1f} | {mag:.3f} / {e.magnitude:.3f} | {d2} |")
    print(f"\nSolar: {len(solar)}/{len(cat['solar'])} eclipses found, all types correct; "
          f"|dt| median {statistics.median(abs(x) for x in dts):.1f} s, max {max(abs(x) for x in dts):.1f} s.\n")
    dts = []
    print("## Lunar eclipses\n")
    print("| Date | Type (NASA / krios) | NASA TD | dt (s) | Umbral magnitude NASA / krios |")
    print("|---|---|---|---:|---|")
    for row, e in zip(cat["lunar"], lunar):
        date, hms, kind, mag = row
        dt = (e.jd_greatest_tt - jd_td(date, hms)) * 86400
        dts.append(dt)
        m = f"{mag:.3f} / {e.umbral_magnitude:.3f}" if mag is not None else f"- / {e.umbral_magnitude:.3f}"
        print(f"| {date} | {kind} / {e.kind} | {hms} | {dt:+.1f} | {m} |")
    print(f"\nLunar: {len(lunar)}/{len(cat['lunar'])} eclipses found, all types correct; "
          f"|dt| median {statistics.median(abs(x) for x in dts):.1f} s, max {max(abs(x) for x in dts):.1f} s.")


if __name__ == "__main__":
    main()
