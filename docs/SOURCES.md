# Sources, data credits and conventions

Krios is independent software; it is not affiliated with or endorsed by NASA, JPL, or EclipseWise.

## Theories and algorithms
* **Moon** - truncated ELP-2000/82 lunar theory (M. Chapront-Touze and J. Chapront), as tabulated in J. Meeus, *Astronomical
  Algorithms*, 2nd ed., ch. 47.  **Sun / Earth** - truncated VSOP87D (P. Bretagnon and G. Francou), Meeus Appendix III.
  Both are checked against Meeus's worked examples 47.a and 25.b in `tests/test_analytic.py`.
* Precession (IAU 1976), nutation (IAU 1980, 20 largest terms), mean obliquity (Laskar 1986), sidereal time (IAU 1982),
  atmospheric refraction (Saemundsson): Meeus chs. 12, 16, 21, 22.
* Planet positions for offline N-body initial states: E. M. Standish, *Keplerian Elements for Approximate Positions of the Major
  Planets* (JPL), valid 1800-2050.
* General-relativistic N-body equations: the parametrised post-Newtonian (Einstein-Infeld-Hoffmann) formulation used for the
  JPL planetary ephemerides (T. D. Moyer 2003; E. M. Standish and J. G. Williams 2012), beta = gamma = 1.
* Gravitational parameters: JPL DE440 (Park et al. 2021).
* Time: leap seconds from IERS Bulletin C (TAI-UTC = 37 s, none announced for the end of 2026); DeltaT = TT-UT1 from the
  Espenak & Meeus polynomial expressions (NASA / EclipseWise) with a short table of recent values and a documented prediction.

## Eclipse conventions (matching NASA / EclipseWise, F. Espenak)
* Besselian fundamental-plane geometry for solar eclipses; lunar radius constants k = 0.2725076 (penumbra) and 0.2722810 (umbra);
  solar radius 696,000 km.
* Earth's shadow for lunar eclipses: oblate spheroid enlarged by an 87 km atmospheric layer (Herald & Sinnott), see the EclipseWise
  page "Enlargement of Earth's Shadows and Lunar Eclipses"; `--shadow danjon|none` select older conventions.
* The shadow displacement due to light travel time (~37 km, equal to the 20.5" aberration) is included, as in NASA's predictions.

## Data
* `tests/data/nasa_catalog_2021_2030.json` holds eclipse times, types and magnitudes transcribed from the EclipseWise / NASA GSFC
  decade tables and eclipse pages by F. Espenak.  They are used only as validation references (`docs/VALIDATION.md`).
* NASA JPL **Horizons** (https://ssd.jpl.nasa.gov/api/horizons.api, documentation at https://ssd-api.jpl.nasa.gov/doc/horizons.html) is a
  free public service: please keep requests few and cache results (Krios does).

## Further reading
* J. Meeus, *Astronomical Algorithms*, 2nd ed., Willmann-Bell (1998).
* *Explanatory Supplement to the Astronomical Almanac*, 3rd ed. (2013).
* F. Espenak and J. Meeus, *Five Millennium Canon of Solar/Lunar Eclipses*, NASA TP-2006-214141 / TP-2009-214172.
