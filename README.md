# Krios

[![tests](https://github.com/entropy-byte/krios/actions/workflows/tests.yml/badge.svg)](https://github.com/entropy-byte/krios/actions/workflows/tests.yml)

**Status: v0.1, first public release (alpha).**

**Krios** is a computational-science engine *and* an experiment lab, driven from the command line:

* **Engine** - N-body orbits with a choice of physics (**Newtonian**, or **Newtonian + general relativity**, 1PN
  Einstein-Infeld-Hoffmann; optional Earth oblateness), a **NASA JPL Horizons** REST client (cache + offline mode), and a
  **solar / lunar eclipse predictor** validated against the NASA/EclipseWise catalog (44 of 44 eclipses 2021-2030).
* **Labs** (`krios lab`) - reproducible experiments that anyone can run with their own measurements: you enter
  what you measured, Krios computes an **independent prediction** from the engine, compares them with real uncertainties,
  fits **fair alternative models** (a flat-map model is given *free parameters*, not set up as a straw man) and writes a
  **reproducible report** (Markdown / JSON / CSV, English or Arabic) with inputs, equations, assumptions and versions.
  Five labs ship: **shadow** (Sun altitude, Earth's radius - Eratosthenes), **pole** (celestial pole altitude and star-trail
  rotation), **parallax** (the Moon's distance by triangulation), **solarpath** (sunrise/sunset/day length in both
  hemispheres) and **eclipse** (contact times vs a pre-registered prediction).
* Everything is a normal Python package, so a GUI can be added later without touching the physics.

```
pip install -e .            # numpy and scipy only; Python 3.9+ (CI runs 3.9-3.12)
pip install -e ".[plots]"   # optional: matplotlib for --plot figures
python -m unittest discover -s tests -t .      # 90 tests, about 30 s
```

## Quick start

```bash
krios eclipse find --from 2026-10-10 --to 2028-12-31          # all eclipses in a range (offline)
krios eclipse find --from 2027 --to 2028 --type solar --contacts --format json
krios eclipse local --date 2027-08-02 --lat 25.69 --lon 32.64 # what an observer sees
krios demo mercury                                            # GR perihelion precession, ~43"/century
krios compare --days 360 --body moon                          # Newton vs GR, same initial state
krios orbit --epoch 2027-01-01 --days 365 --bodies moon,earth --model pn1 --out moon.csv
krios horizons vectors --target 301 --center 500@399 --start 2027-08-02 --stop 2027-08-03 --step-minutes 10
krios eclipse find --from 2027 --to 2028 --source horizons    # same engine, JPL DE441 positions
krios eclipse find --from 2027 --to 2028 --source integrated --model newton   # your own physics
```

## Labs: your measurements vs independent predictions

```bash
krios lab list                                  # shadow, pole, parallax, solarpath, eclipse
krios lab demo shadow                           # run on built-in SYNTHETIC data (always labelled as such)
krios lab demo shadow --lang ar --report r.md --plot r.png
krios lab template shadow --out my_data.json    # a data-entry sheet to replace with your numbers (CSV for tabular labs)
krios lab plan eclipse --data my_data.json --out plan.json     # PRE-REGISTER the predictions before the event
krios lab run  eclipse --data my_data.json --plan plan.json --report report.md --json result.json --csv-dir out/
krios eclipse path --date 2027-08-02 --out path.geojson        # central line of the total eclipse, for maps
```

Every lab follows the same four steps: **(1) your data** with places, times and uncertainties; **(2) predictions**
computed from the engine using *only* non-measurement inputs (`plan` proves it: the prediction hash is identical before
and after you add measurements); **(3) a quantitative comparison** (residuals, chi-square, p-values, Monte-Carlo
uncertainties, and model fits in which the alternative gets free parameters); **(4) a reproducible report**.
Verdicts are worded carefully: with too few stations the report says the data are *not discriminating*, and if the data
ever favoured a flat model it would say so (the test-suite generates flat-world data and checks exactly that).

| lab | you measure | Krios predicts | what the comparison shows |
|---|---|---|---|
| `shadow` | stick + shadow length, place, UTC time, 2-6+ stations (+ independent distances) | Sun altitude (0 free parameters) | globe fits; flat map with fitted Sun height/position does not; Eratosthenes radius from baselines; Sun height "implied by a flat Earth" changing from pair to pair |
| `pole` | star-trail arcs (+ camera tilt, field of view) or pole altitude, rotation sense, at several latitudes | pole altitude = latitude, ccw north / cw south, 15.04 deg/h, Polaris offset | slope 1.00 +/- 0.01, turning rate, a southern pole that a single northern pole cannot explain |
| `parallax` | Moon direction (alt/az or RA/Dec) from 2+ distant places at the same moment | Moon directions, distance, expected parallax | distance +/- Monte-Carlo error, lower bound, whether lines of sight meet in globe vs flat geometry |
| `solarpath` | sunrise / sunset / noon / altitude on chosen dates, places in both hemispheres | full-year Sun path, day length, polar day/night | residuals in seconds; opposite seasons |
| `eclipse` | contact times (C1-C4 or P1-P4) | contact times with the chosen ephemeris | residuals with observer + model uncertainty |

Details, equipment, procedures and the limits of each lab: [`docs/LABS.md`](docs/LABS.md).  Ready-made inputs, reports
and figures from the synthetic demos: `examples/`.

### Example output

```
Ephemeris : analytic (Meeus ELP-2000/82 + VSOP87D truncated)
Range     : 2026-10-10 00:00:00 .. 2028-12-31 00:00:00 (UTC)   DeltaT ~ 69.6 s
Found     : 9 eclipse(s)

Date (UTC)  Body   Type       Greatest (UTC)  Mag    Gamma    Duration  Position / detail
----------  -----  ---------  --------------  -----  -------  --------  ------------------------------
2027-02-06  solar  Annular    15:59:39 UTC    0.928  -0.2952  7m51s     31.30S 48.47W
2027-02-20  lunar  Penumbral  23:12:47 UTC    0.927  -1.0490  4h01m27s  umbral -0.057, penumbral 0.927
2027-07-18  lunar  Penumbral  16:02:58 UTC    0.003  -1.5758  17m19s    umbral -1.066, penumbral 0.003
2027-08-02  solar  Total      10:06:39 UTC    1.079  +0.1420  6m22s     25.50N 33.18E
2027-08-17  lunar  Penumbral  07:13:53 UTC    0.548  +1.2795  3h39m23s  umbral -0.523, penumbral 0.548
2028-01-12  lunar  Partial    04:13:09 UTC    0.068  +0.9818  56m52s    umbral +0.068, penumbral 1.048
2028-01-26  solar  Annular    15:07:53 UTC    0.921  +0.3902  10m27s    2.97N 51.58W
2028-07-06  lunar  Partial    18:19:42 UTC    0.391  -0.7903  2h22m15s  umbral +0.391, penumbral 1.428
2028-07-22  solar  Total      02:55:37 UTC    1.056  -0.6060  5m09s     15.61S 126.66E
```

```
Solar eclipse of 2027-08-02 (Total, greatest eclipse 10:06:39 UTC)
Observer  : lat +25.6900  lon +32.6400  height 80 m
Ephemeris : analytic (Meeus ELP-2000/82 + VSOP87D truncated)

Event                      Time            Sun alt (deg)  Azimuth (deg)
-------------------------  --------------  -------------  -------------
Partial begins (C1)        08:40:13.2 UTC    70.8          110.5
Total/annular begins (C2)  10:02:03.8 UTC    81.9          190.7
Maximum                    10:05:14.2 UTC    81.8          195.9
Total/annular ends (C3)    10:08:24.3 UTC    81.5          200.9
Partial ends (C4)          11:26:31.2 UTC    67.5          253.9

Type here     : Total
Magnitude     : 1.0790
Obscuration   : 100.00 %
Duration      : 6m20s (380.4 s)
```

```
Mercury's perihelion precession (Sun+Mercury, two bodies)
  Newtonian N-body : +0.0019 arcsec/century (no precession without other planets)
  1PN (GR)         : +42.9889 arcsec/century
  GR contribution  : 42.9870 arcsec/century
  Theory 6*pi*GM/(c^2 a (1-e^2)) per orbit: 42.9807 arcsec/century   (observed anomaly ~43)
  agreement        : 0.015 %
```

```
Newton vs 1PN (GR) from identical initial conditions - position difference of Moon (geocentric)

days      |difference| km  metres
--------  ---------------  ------
    90.0  0.8991           899
   180.0  2.0773           2077
   270.0  3.2959           3296
   360.0  3.9567           3957
```

## Ephemeris sources (`--source`)

| source | what it is | needs | timing accuracy (eclipses) |
|---|---|---|---|
| `analytic` (default) | Meeus' truncated ELP-2000/82 (Moon) + VSOP87D (Sun) | nothing, offline | solar <= 18 s, lunar <= 13 s vs NASA, 2021-2030 |
| `horizons` | JPL DE441 through the Horizons REST API, cached, Hermite-interpolated | network once, then offline | sub-second (data limited) |
| `integrated` | this package's N-body integration (Newton / 1PN / +J2) | analytic or Horizons initial state | about 10 s for 2027 (fitted to the analytic theory) |

All three plug into the same eclipse code through one interface (`krios.ephemeris.Ephemeris`: *where are the Sun and
Moon seen from the Earth's centre at time t?*).  That is also a benchmark: **the eclipse time error of a model is a
direct measure of how good that model's physics is.**

## Physics models (`--model`, `--j2`)

* `newton`: point-mass N-body (Sun, Mercury ... Neptune, Earth and Moon separate), solar-system barycentric frame.
* `pn1`: adds the 1PN terms (beta = gamma = 1).  Reproduces Mercury's 42.98 arcsec/century to 0.01 %.
  For identical initial conditions the Moon's geocentric position differs from Newton by ~4 km after one year
  and the Earth's barycentric position by ~60 km.  (GR is a small effect for eclipse timing, which is dominated by the
  Sun's Newtonian tidal field and the initial conditions.)
* `--j2`: Earth's oblateness acting on the Moon and others (Newtonian), with the reaction on the Earth.
* **Not implemented (honest scope):** solving Einstein's field equations *directly* is numerical relativity, a
  project of its own.  The plan is a separate `geodesic` mode for exact solutions (Schwarzschild, Kerr) where it makes
  sense (orbits around compact objects), see the roadmap.  Also not modelled: lunar figure/libration, Earth tides
  and the Moon's secular acceleration (all matter at the 10 km / year level for the Moon).

**Initial states.**  `--init horizons` uses exact DE441 states.  `--init analytic` (offline) builds a state from
the analytic theories and then *fits* the Earth/Moon state to the theory over `--fit-days` days: raw analytic
velocities are not accurate enough to propagate the Moon for months (error grows to ~2700 km in a year, measured),
the fitted state tracks the theory to ~10 km over a year.  `--init file.json|URL` loads any state in the JSON schema
written by `krios state`.

## Eclipse engine

* Geometry from first principles (no polynomial fits): Besselian fundamental-plane frame for solar eclipses,
  elliptical Earth-shadow model for lunar eclipses.
* Conventions follow NASA/EclipseWise: lunar radius k = 0.2725076 (penumbra) / 0.2722810 (umbra), Sun radius 696000 km,
  Earth's shadow from an oblate spheroid + 87 km atmospheric layer (Herald & Sinnott).  `--shadow danjon|none` give
  the older conventions.
* The shadow is displaced by ~37 km because the Earth moves ~30 km/s during the 1.28 s light travel Earth <-> Moon
  (same 20.5" as stellar aberration); omitting it changes eclipse times by ~40 s.  `--no-light-time` shows the effect.
* Time scales: results are given in TT (= NASA's "TD") and UTC.  TT-UTC = 69.184 s (leap seconds are known to the end
  of 2026, IERS Bulletin C 72).  DeltaT = TT-UT1 beyond 2025 is a *prediction*; use `--delta-t` to override.
  It only affects UT1-based quantities (where on Earth the shadow falls: ~0.5 km per second of DeltaT), not UTC clock times.

### Validation

`docs/VALIDATION.md` (generated by `scripts/validate_catalog.py`) compares all 44 eclipses of 2021-2030 with the
NASA/EclipseWise decade tables: **all found, all types right (including the hybrid of 2023 Apr 20)**, greatest-eclipse
time error <= 18 s (solar, median 4 s) and <= 13 s (lunar, median 6 s); solar magnitudes within 0.003, central
durations within 2 s.  Two eclipses were checked in detail:

* **2027 Aug 2 (total)**: Besselian x, y, l1, l2, tan f1, tan f2 at 10:00 TD match NASA's polynomial elements to
  1-3 parts in 10^4 or better; greatest eclipse within 1.8 s; gamma 0.14203 (NASA 0.14209); central duration 382.4 s
  (382.64 s); greatest-eclipse position 25.50N 33.19E (25.50N 33.18E); all eight global contacts within 2 s.
* **2026 Mar 3 (total lunar)**: umbral magnitude 1.15254 (NASA 1.15263), gamma -0.37652 (-0.37651), contacts within
  6.5 s, partial and total durations within 1-2 s.  The Herald-Sinnott shadow clearly beats Danjon (1.169) and a bare
  spheroid (1.127).

## Layout

```
src/krios/
  constants.py  timescales.py  frames.py      time scales, DeltaT, precession/nutation, sidereal time, geodesy
  ephemeris/    analytic.py  horizons.py  integrated.py  base.py
  dynamics/     model.py (Newton, EIH 1PN, J2)  system.py (DOP853 integrator)  initial.py (states + orbit fit)
  eclipse/      solar.py  lunar.py  finder.py  geometry.py  config.py  path.py (GeoJSON)
  lab/          core.py (results, reports, pre-registration)  astro.py  stats.py
                shadow.py  pole.py  parallax.py  solarpath.py  eclipse_lab.py
  cli.py  report.py
tests/          90 unit/integration tests, NASA reference data in tests/data
examples/       input templates (JSON/CSV), demo reports, figures, 2027 eclipse GeoJSON
docs/           VALIDATION.md (eclipse catalog), LABS.md
scripts/        validate_catalog.py
```

## Using NASA Horizons

The client follows the documented API (`https://ssd.jpl.nasa.gov/api/horizons.api`, `EPHEM_TYPE=VECTORS`, `VEC_TABLE=2`,
`REF_PLANE=FRAME`, `TIME_TYPE=TDB`, JSON reply, table between `$$SOE` and `$$EOE`).  The parser, cache, interpolation and the
whole pipeline are covered by tests that use a fake server reproducing the documented format; **a request against the live
service has not been verified yet**.  If you hit a mismatch please open an issue with the raw reply.  Try
`krios horizons vectors --start 2027-08-02 --stop 2027-08-03`.  Responses are cached in `~/.cache/krios/horizons`
(`--cache-dir`, `KRIOS_CACHE`); `--offline` never touches the network.  Horizons is a free public service: the eclipse
engine asks for about one day of 5-minute states per eclipse, not one request per instant.

## Known limitations

* Analytic ephemeris: Moon ~10" nominal (typically 2-5"), valid roughly 1600-2200; use `--source horizons` for precision.
* Local circumstances: geometric Sun altitude (no refraction), sea-level reference ellipsoid, no lunar limb profile
  (contact times can differ by a few seconds, central duration by up to ~1-2 s from limb-corrected predictions).
* UTC is only defined with leap seconds since 1972 (earlier times are reported as UT1).

## Roadmap

1. **v0.1 (this release)**: engine (time/frames, analytic ephemeris, eclipse engine, Horizons client, Newton / 1PN / J2 N-body),
   five labs (shadow, pole, parallax, solarpath, eclipse) with pre-registered predictions and bilingual reports, GeoJSON eclipse path.
2. **v0.2**: horizon-geometry lab (needs an honest refraction model: reports a range, not one number), Foucault pendulum lab
   (rate = Omega sin(latitude)), image-based star-trail centre detection, a lab that merges the reports of several observers.
3. Verify the live Horizons route; own-physics benchmark against DE441.
4. Earth-orientation data (IERS), lunar limb profile, exact eclipse path limits; `geodesic` mode (Schwarzschild/Kerr).
5. GUI on top of the same Python API.

## Contributing and sources

See [`CONTRIBUTING.md`](CONTRIBUTING.md).  Scientific sources, data credits and
conventions: [`docs/SOURCES.md`](docs/SOURCES.md).
