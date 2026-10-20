"""Command-line interface:  ``krios <command> ...`` (also ``python -m krios``)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from typing import List, Optional

import numpy as np

from . import __version__
from . import timescales as ts
from . import report
from .constants import SECONDS_PER_DAY
from .eclipse import EclipseConfig, find_eclipses, solar_global, solar_local
from .ephemeris import AnalyticEphemeris


# ---------------------------------------------------------------- shared options
def _add_source_options(p: argparse.ArgumentParser):
    g = p.add_argument_group("ephemeris source")
    g.add_argument("--source", choices=["analytic", "horizons", "integrated"], default="analytic",
                   help="analytic: offline theory (default, ~10 s timing) | horizons: NASA JPL DE441 via REST "
                        "| integrated: this package's own N-body physics")
    g.add_argument("--offline", action="store_true", help="never use the network; read the Horizons cache only")
    g.add_argument("--cache-dir", default=None, help="Horizons cache directory (default ~/.cache/krios/horizons)")
    g.add_argument("--model", choices=["newton", "pn1"], default="pn1",
                   help="physics for --source integrated: newton or pn1 (GR corrections, default)")
    g.add_argument("--j2", action="store_true", help="include Earth's oblateness (J2) in integrated runs")
    g.add_argument("--init", default="analytic",
                   help="initial state for integrated runs: 'analytic', 'horizons', or a JSON file/URL")
    g.add_argument("--fit-days", type=float, default=120.0,
                   help="fit Earth/Moon initial state to the analytic theory over N days (analytic init only)")
    c = p.add_argument_group("conventions")
    c.add_argument("--shadow", choices=["herald", "danjon", "none"], default="herald",
                   help="Earth's shadow model for lunar eclipses (default herald = NASA/EclipseWise)")
    c.add_argument("--no-light-time", action="store_true", help="disable the light-time/aberration shadow shift")
    c.add_argument("--delta-t", type=float, default=None, help="force DeltaT = TT-UT1 in seconds")


def _config(args) -> EclipseConfig:
    return EclipseConfig(shadow_model=args.shadow, light_time=not args.no_light_time)


def _load_system_file(path_or_url: str):
    from .dynamics import System
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
    else:
        with open(path_or_url, encoding="utf-8") as f:
            data = json.load(f)
    return System.from_dict(data)


def _initial_system(args, epoch_jd_tdb, log):
    from .dynamics.initial import system_from_analytic, system_from_horizons
    from .ephemeris.horizons import HorizonsClient
    if args.init == "analytic":
        return system_from_analytic(epoch_jd_tdb), True
    if args.init == "horizons":
        log("fetching DE441 barycentric states from Horizons (cached after the first run)...")
        client = HorizonsClient(cache_dir=args.cache_dir, offline=args.offline)
        return system_from_horizons(client, epoch_jd_tdb), False
    return _load_system_file(args.init), False


def _build_integrated(args, jd_start_tt, jd_end_tt, log):
    from .dynamics import integrate
    from .dynamics.initial import fit_system, make_force_model
    from .ephemeris.integrated import IntegratedEphemeris
    margin = 5.0
    epoch = jd_start_tt - margin
    sysm, needs_fit = _initial_system(args, epoch, log)
    epoch = sysm.epoch_jd_tdb
    model = make_force_model(sysm, args.model, earth_j2=args.j2)
    if needs_fit and args.fit_days > 0:
        log(f"fitting Earth/Moon initial state to the analytic theory over {args.fit_days:g} d ({model.label()})...")
        sysm = fit_system(sysm, AnalyticEphemeris(), window_days=args.fit_days, model=model)
    days = (jd_end_tt - epoch) + margin
    log(f"integrating {days:.0f} days with {model.label()} ...")
    return IntegratedEphemeris(integrate(sysm, days, model, rtol=1e-11), label=sysm.source)


def build_ephemeris(args, jd_start_tt, jd_end_tt, log=lambda s: None):
    if args.source == "analytic":
        return AnalyticEphemeris()
    if args.source == "horizons":
        from .ephemeris.horizons import HorizonsClient, HorizonsEphemeris
        return HorizonsEphemeris(HorizonsClient(cache_dir=args.cache_dir, offline=args.offline))
    return _build_integrated(args, jd_start_tt, jd_end_tt, log)


def _log(msg):
    print(msg, file=sys.stderr)


def _range(args):
    start = ts.utc_to_tt(ts.parse_datetime(args.start))
    end = ts.utc_to_tt(ts.parse_datetime(args.end))
    if end <= start:
        raise SystemExit("error: --to must be after --from")
    return start, end


# ---------------------------------------------------------------- eclipse find
def cmd_eclipse_find(args) -> int:
    if args.delta_t is not None:
        ts.set_delta_t_override(args.delta_t)
    start, end = _range(args)
    kinds = ("solar", "lunar") if args.type == "all" else (args.type,)
    cfg = _config(args)
    ephem = build_ephemeris(args, start, end, _log)
    if args.source == "analytic" and (end - start) > 4000:
        _log("note: scanning a long range; this is fine but takes a while")
    found = find_eclipses(start, end, kinds, ephem, cfg)

    if args.format == "json":
        print(json.dumps({"ephemeris": ephem.describe(), "delta_t_s": ts.delta_t_for_jd((start + end) / 2),
                          "eclipses": [report.eclipse_record(e) for e in found]}, indent=2))
        return 0

    rows, headers = [], ["Date (UTC)", "Body", "Type", "Greatest (UTC)", "Mag", "Gamma", "Duration", "Position / detail"]
    for e in found:
        utc = report.utc_str(e.jd_greatest_tt, 0)
        if hasattr(e, "central"):
            dur = report.hms(e.central_duration_s) if e.central_duration_s else "-"
            pos = report.latlon(e.greatest_lat_deg, e.greatest_lon_deg)
            rows.append([utc[:10], "solar", e.kind, utc[11:] + " UTC", f"{e.magnitude:.3f}", f"{e.gamma:+.4f}", dur, pos])
        else:
            dur = report.hms(e.total_duration_s) if e.total_duration_s else (
                report.hms(e.partial_duration_s) if e.partial_duration_s else report.hms(e.penumbral_duration_s))
            detail = f"umbral {e.umbral_magnitude:+.3f}, penumbral {e.penumbral_magnitude:.3f}"
            mag = e.penumbral_magnitude if e.kind == "Penumbral" else e.umbral_magnitude
            rows.append([utc[:10], "lunar", e.kind, utc[11:] + " UTC", f"{mag:.3f}", f"{e.gamma:+.4f}", dur, detail])
    if args.format == "csv":
        print(report.to_csv(headers, rows), end="")
        return 0
    print(f"Ephemeris : {ephem.describe()}")
    print(f"Range     : {report.utc_str(start, 0)} .. {report.utc_str(end, 0)} (UTC)   DeltaT ~ {ts.delta_t_for_jd(start):.1f} s")
    print(f"Found     : {len(found)} eclipse(s)\n")
    if rows:
        print(report.table(headers, rows))
    if args.contacts:
        for e in found:
            print(f"\n{report.utc_str(e.jd_greatest_tt, 0)[:10]}  {'solar' if hasattr(e, 'central') else 'lunar'} {e.kind}")
            for k, v in sorted(e.contacts_tt.items(), key=lambda kv: kv[1]):
                print(f"   {k:<9}{report.utc_str(v, 1)} UTC   ({report.tt_str(v, 1)} TT)")
    print("\nMag: solar = Moon/Sun diameter ratio (central) or fraction of the Sun's diameter covered (partial);"
          " lunar = umbral magnitude (penumbral magnitude for penumbral eclipses). Duration: totality/annularity (solar);"
          " totality, else partial phase, else penumbral phase (lunar).")
    return 0


# ---------------------------------------------------------------- eclipse local
def cmd_eclipse_local(args) -> int:
    if args.delta_t is not None:
        ts.set_delta_t_override(args.delta_t)
    day = ts.parse_datetime(args.date)
    start, end = ts.utc_to_tt(day) - 1.0, ts.utc_to_tt(day) + 2.0
    cfg = _config(args)
    ephem = build_ephemeris(args, start - 5, end + 5, _log)
    sols = find_eclipses(start, end, ("solar",), ephem, cfg)
    if not sols:
        print(f"No solar eclipse found around {args.date}.", file=sys.stderr)
        return 2
    ecl = sols[0]
    loc = solar_local(ephem, cfg, args.lat, args.lon, args.height, ecl.jd_greatest_tt)
    if args.format == "json":
        d = report.jsonable(loc.to_dict())
        d["global"] = report.eclipse_record(ecl)
        d["contacts"] = {k: {"tt": report.tt_str(v, 2), "utc": report.utc_str(v, 2)} for k, v in loc.contacts_tt.items()}
        print(json.dumps(d, indent=2))
        return 0
    print(f"Solar eclipse of {report.utc_str(ecl.jd_greatest_tt, 0)[:10]} ({ecl.kind}, greatest eclipse "
          f"{report.utc_str(ecl.jd_greatest_tt, 0)[11:]} UTC)")
    print(f"Observer  : lat {args.lat:+.4f}  lon {args.lon:+.4f}  height {args.height:.0f} m")
    print(f"Ephemeris : {ephem.describe()}\n")
    if not loc.eclipse_occurs:
        print("No eclipse is visible from this location. " + (loc.note or "The Moon's penumbra does not reach it."))
        return 0
    names = {"C1": "Partial begins (C1)", "C2": "Total/annular begins (C2)", "max": "Maximum",
             "C3": "Total/annular ends (C3)", "C4": "Partial ends (C4)"}
    rows = []
    for k in ("C1", "C2", "max", "C3", "C4"):
        if k in loc.contacts_tt:
            alt = loc.altitude_deg[k]
            flag = "" if alt > -0.8333 else "  (Sun below horizon)"
            rows.append([names[k], report.utc_str(loc.contacts_tt[k], 1)[11:] + " UTC",
                         f"{alt:6.1f}", f"{loc.azimuth_deg[k]:6.1f}" + flag])
    print(report.table(["Event", "Time", "Sun alt (deg)", "Azimuth (deg)"], rows))
    print(f"\nType here     : {loc.kind}")
    print(f"Magnitude     : {loc.magnitude:.4f}")
    print(f"Obscuration   : {loc.obscuration * 100:.2f} %")
    if loc.duration_s:
        print(f"Duration      : {report.hms(loc.duration_s)} ({loc.duration_s:.1f} s)")
    print("\nNote: geometric altitudes (no refraction), no local terrain; times are UTC (leap seconds known to 2026).")
    return 0


# ---------------------------------------------------------------- orbit
def cmd_orbit(args) -> int:
    from .dynamics import integrate
    from .dynamics.initial import fit_system, make_force_model
    epoch_jd = ts.utc_to_tt(ts.parse_datetime(args.epoch))
    epoch_tdb = ts.jd_tt_to_tdb(epoch_jd)
    sysm, needs_fit = _initial_system(args, epoch_tdb, _log)
    model = make_force_model(sysm, args.model, earth_j2=args.j2)
    if needs_fit and args.fit_days > 0:
        _log(f"fitting Earth/Moon initial state to the analytic theory over {args.fit_days:g} d ...")
        sysm = fit_system(sysm, AnalyticEphemeris(), window_days=args.fit_days, model=model)
    _log(f"integrating {args.days:g} days with {model.label()} (DOP853, rtol 1e-12) ...")
    traj = integrate(sysm, args.days, model, rtol=1e-12)
    n = traj.nfev
    step = args.step * SECONDS_PER_DAY
    nsteps = int(abs(traj.t_end_s) // step)
    t = np.arange(0, nsteps + 1) * step * np.sign(traj.t_end_s)
    r, v = traj.state(t)
    names = [b.strip() for b in args.bodies.split(",")]
    center = args.center
    ci = None if center == "ssb" else sysm.index(center)
    rows = []
    for k, tk in enumerate(t):
        jd = sysm.epoch_jd_tdb + tk / SECONDS_PER_DAY
        for b in names:
            i = sysm.index(b)
            rr = r[k, i] - (0 if ci is None else r[k, ci])
            vv = v[k, i] - (0 if ci is None else v[k, ci])
            rows.append([f"{jd:.6f}", ts.format_jd(jd, 1), b, *[f"{x:.6f}" for x in rr], *[f"{x:.9f}" for x in vv]])
    headers = ["jd_tdb", "date_tdb", "body", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
    out = report.to_csv(headers, rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            f.write(out)
        _log(f"wrote {len(rows)} rows to {args.out}")
    else:
        print(out, end="")
    fin = traj.final_system()
    e0, e1 = sysm.energy(), fin.energy()
    _log(f"function evaluations: {n};  Newtonian energy drift: {abs(e1 / e0 - 1):.2e};  "
         f"angular momentum drift: {np.linalg.norm(fin.angular_momentum() / sysm.angular_momentum() - 1):.2e}"
         + ("  (energy is only conserved exactly by the Newtonian model)" if model.kind != "newton" else ""))
    return 0


# ---------------------------------------------------------------- compare / demos
def cmd_compare(args) -> int:
    from .dynamics import integrate
    from .dynamics.initial import fit_system, make_force_model
    epoch_jd = ts.utc_to_tt(ts.parse_datetime(args.epoch))
    sysm, needs_fit = _initial_system(args, ts.jd_tt_to_tdb(epoch_jd), _log)
    base_model = make_force_model(sysm, "newton", earth_j2=args.j2)
    if needs_fit and args.fit_days > 0:
        _log("fitting the common initial state with the Newtonian model ...")
        sysm = fit_system(sysm, AnalyticEphemeris(), window_days=args.fit_days, model=base_model)
    _log("integrating Newton and 1PN from the SAME initial state ...")
    tn = integrate(sysm, args.days, base_model, rtol=1e-12)
    tp = integrate(sysm, args.days, make_force_model(sysm, "pn1", earth_j2=args.j2), rtol=1e-12)
    sample = np.linspace(0, tn.t_end_s, args.samples + 1)[1:]
    rn, _ = tn.state(sample)
    rp, _ = tp.state(sample)
    ie, im = sysm.index("earth"), sysm.index("moon")
    body = args.body
    rows = []
    for k, tk in enumerate(sample):
        if body == "moon":
            diff = (rp[k, im] - rp[k, ie]) - (rn[k, im] - rn[k, ie])
            label = "Moon (geocentric)"
        else:
            i = sysm.index(body)
            diff = rp[k, i] - rn[k, i]
            label = f"{body} (barycentric)"
        rows.append([f"{tk / SECONDS_PER_DAY:8.1f}", f"{np.linalg.norm(diff):.4f}", f"{np.linalg.norm(diff) / 1000 * 1e3:.1f}"])
    print(f"Newton vs 1PN (GR) from identical initial conditions - position difference of {label}\n")
    print(report.table(["days", "|difference| km", "metres"], [[a, b, f"{float(b) * 1000:.0f}"] for a, b, _ in rows]))
    return 0


def cmd_demo_mercury(args) -> int:
    from .dynamics import ForceModel, System, integrate
    from .dynamics.bodies import GM
    from .constants import C_KM_S
    import math
    gm = np.array([GM["sun"], GM["mercury"]])
    mu = gm.sum()
    a, e = 57_909_050.0, 0.205630
    rp = a * (1 - e)
    vp = math.sqrt(mu * (1 + e) / rp)
    r_rel, v_rel = np.array([rp, 0, 0.0]), np.array([0, vp, 0.0])
    m1, m2 = gm[0] / mu, gm[1] / mu
    sysm = System(["sun", "mercury"], gm, np.array([-m2 * r_rel, m1 * r_rel]), np.array([-m2 * v_rel, m1 * v_rel]), 2451545.0)
    period = 2 * math.pi * math.sqrt(a ** 3 / mu)
    days = args.years * 365.25
    rates = {}
    for kind in ("newton", "pn1"):
        tr = integrate(sysm, days, ForceModel(kind=kind), rtol=1e-12)
        t = np.arange(0, tr.t_end_s, 43200.0)
        R, V = tr.state(t)
        rr, vv = R[:, 1] - R[:, 0], V[:, 1] - V[:, 0]
        h = np.cross(rr, vv)
        ev = np.cross(vv, h) / mu - rr / np.linalg.norm(rr, axis=1)[:, None]
        ang = np.unwrap(np.arctan2(ev[:, 1], ev[:, 0]))
        rates[kind] = np.polyfit(t, ang, 1)[0] * 86400 * 36525 * (180 / math.pi) * 3600
    gr = rates["pn1"] - rates["newton"]
    theory = 6 * math.pi * mu / (C_KM_S ** 2 * a * (1 - e ** 2)) / period * 86400 * 36525 * (180 / math.pi) * 3600
    print("Mercury's perihelion precession (Sun+Mercury, two bodies)")
    print(f"  Newtonian N-body : {rates['newton']:+.4f} arcsec/century (no precession without other planets)")
    print(f"  1PN (GR)         : {rates['pn1']:+.4f} arcsec/century")
    print(f"  GR contribution  : {gr:.4f} arcsec/century")
    print(f"  Theory 6*pi*GM/(c^2 a (1-e^2)) per orbit: {theory:.4f} arcsec/century   (observed anomaly ~43)")
    print(f"  agreement        : {abs(gr / theory - 1) * 100:.3f} %")
    return 0


# ---------------------------------------------------------------- horizons
def cmd_horizons_vectors(args) -> int:
    from .ephemeris.horizons import HorizonsClient
    client = HorizonsClient(cache_dir=args.cache_dir, offline=args.offline)
    a = ts.jd_tt_to_tdb(ts.utc_to_tt(ts.parse_datetime(args.start)))
    b = ts.jd_tt_to_tdb(ts.utc_to_tt(ts.parse_datetime(args.stop)))
    tab = client.vectors(args.target, args.center, a, b, args.step_minutes)
    headers = ["jd_tdb", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"]
    rows = [[f"{j:.9f}", *[f"{x:.9e}" for x in p], *[f"{x:.9e}" for x in v]]
            for j, p, v in zip(tab.jd_tdb, tab.pos_km, tab.vel_km_s)]
    out = report.to_csv(headers, rows)
    if args.out:
        open(args.out, "w", encoding="utf-8", newline="").write(out)
        _log(f"wrote {len(rows)} rows to {args.out}")
    else:
        print(out, end="")
    return 0


def cmd_state(args) -> int:
    epoch_jd = ts.utc_to_tt(ts.parse_datetime(args.epoch))
    sysm, _ = _initial_system(args, ts.jd_tt_to_tdb(epoch_jd), _log)
    text = json.dumps(sysm.to_dict(), indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        _log(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_info(args) -> int:
    from .constants import K_PENUMBRA, K_UMBRA, SHADOW_LAYER_KM, SUN_RADIUS_KM
    print(f"krios {__version__}")
    print("Ephemeris sources:")
    print("  analytic   Meeus ELP-2000/82 + VSOP87D truncated; offline; Moon ~10\" -> eclipse timing within ~20 s")
    print("  horizons   JPL DE441 through the Horizons REST API; cached; sub-second timing")
    print("  integrated this package's N-body engine (Newton / 1PN GR / +J2), fitted or from Horizons states")
    print("Conventions (NASA/EclipseWise): k = %.7f (penumbra) / %.7f (umbra), solar radius %.0f km, "
          "Herald-Sinnott shadow layer %.0f km" % (K_PENUMBRA, K_UMBRA, SUN_RADIUS_KM, SHADOW_LAYER_KM))
    print("Time: TT=UTC+69.184 s since 2017 (leap seconds known through 2026); DeltaT predicted beyond 2025.")
    return 0


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="krios", description="Orbits (Newton / 1PN GR), NASA Horizons data and eclipse prediction.")
    p.add_argument("--version", action="version", version=f"krios {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    # eclipse
    ep = sub.add_parser("eclipse", help="find eclipses / local circumstances").add_subparsers(dest="sub", required=True)
    f = ep.add_parser("find", help="list solar and lunar eclipses in a date range")
    f.add_argument("--from", dest="start", default="2026-10-10", help="start date (UTC), e.g. 2027-01-01")
    f.add_argument("--to", dest="end", default="2029-01-01", help="end date (UTC)")
    f.add_argument("--type", choices=["all", "solar", "lunar"], default="all")
    f.add_argument("--format", choices=["table", "json", "csv"], default="table")
    f.add_argument("--contacts", action="store_true", help="also print contact times (P1, U1, C1, ...)")
    _add_source_options(f)
    f.set_defaults(func=cmd_eclipse_find)

    l = ep.add_parser("local", help="local circumstances of a solar eclipse for an observer")
    l.add_argument("--date", required=True, help="UTC date of the eclipse, e.g. 2027-08-02")
    l.add_argument("--lat", type=float, required=True, help="geodetic latitude, degrees (north +)")
    l.add_argument("--lon", type=float, required=True, help="longitude, degrees (east +)")
    l.add_argument("--height", type=float, default=0.0, help="height above the ellipsoid, metres")
    l.add_argument("--format", choices=["table", "json"], default="table")
    _add_source_options(l)
    l.set_defaults(func=cmd_eclipse_local)

    # orbit
    o = sub.add_parser("orbit", help="integrate the solar system (Newton or 1PN) and export states")
    o.add_argument("--epoch", default="2027-01-01", help="initial epoch (UTC date/time)")
    o.add_argument("--days", type=float, default=365.0, help="duration in days (negative = backwards)")
    o.add_argument("--step", type=float, default=1.0, help="output step in days")
    o.add_argument("--bodies", default="moon,earth", help="comma list, e.g. moon,earth,mercury")
    o.add_argument("--center", default="earth", help="'ssb' or a body name (default earth)")
    o.add_argument("--out", default=None, help="CSV output file (default stdout)")
    _add_orbit_options(o)
    o.set_defaults(func=cmd_orbit)

    c = sub.add_parser("compare", help="Newton vs 1PN (GR) from the same initial state")
    c.add_argument("--epoch", default="2027-01-01")
    c.add_argument("--days", type=float, default=365.0)
    c.add_argument("--body", default="moon")
    c.add_argument("--samples", type=int, default=6)
    _add_orbit_options(c)
    c.set_defaults(func=cmd_compare)

    d = sub.add_parser("demo", help="physics demos").add_subparsers(dest="sub", required=True)
    m = d.add_parser("mercury", help="GR perihelion precession of Mercury (~43 arcsec/century)")
    m.add_argument("--years", type=float, default=25.0)
    m.set_defaults(func=cmd_demo_mercury)

    h = sub.add_parser("horizons", help="NASA JPL Horizons REST helpers").add_subparsers(dest="sub", required=True)
    v = h.add_parser("vectors", help="download state vectors (cached) as CSV")
    v.add_argument("--target", default="301", help="Horizons id, e.g. 301 Moon, 10 Sun, 399 Earth")
    v.add_argument("--center", default="500@399", help="e.g. 500@399 geocentre, @0 solar-system barycentre")
    v.add_argument("--start", required=True)
    v.add_argument("--stop", required=True)
    v.add_argument("--step-minutes", type=int, default=60)
    v.add_argument("--offline", action="store_true")
    v.add_argument("--cache-dir", default=None)
    v.add_argument("--out", default=None)
    v.set_defaults(func=cmd_horizons_vectors)

    s = sub.add_parser("state", help="export an initial N-body state as JSON (editable, re-loadable)")
    s.add_argument("--epoch", default="2027-01-01")
    s.add_argument("--out", default=None)
    _add_orbit_options(s)
    s.set_defaults(func=cmd_state)

    i = sub.add_parser("info", help="show conventions and accuracy notes")
    i.set_defaults(func=cmd_info)
    return p


def _add_orbit_options(p):
    p.add_argument("--model", choices=["newton", "pn1"], default="pn1", help="newton or pn1 (GR); default pn1")
    p.add_argument("--j2", action="store_true", help="include Earth's oblateness (J2)")
    p.add_argument("--init", default="analytic", help="'analytic' (offline), 'horizons' (DE441 states), or JSON file/URL")
    p.add_argument("--fit-days", type=float, default=120.0, help="fit Earth/Moon to the analytic theory over N days")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--cache-dir", default=None)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:                                       # friendly CLI errors
        from .ephemeris.horizons import HorizonsError
        if isinstance(exc, (HorizonsError, ValueError)):
            print(f"error: {exc}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
