# Changelog

## 0.1.0 - first public release

**Engine**
* Time scales (UTC/TAI/TT/TDB, leap seconds, predicted DeltaT), precession/nutation, sidereal time, WGS-84 geodesy.
* Offline analytic Sun and Moon (truncated VSOP87D and ELP-2000/82); JPL Horizons REST client with on-disk cache,
  offline mode and Hermite interpolation; an ephemeris provider backed by the package's own N-body integration.
* Newtonian and 1PN (Einstein-Infeld-Hoffmann, general relativity) N-body dynamics with optional Earth J2, DOP853
  integration, orbit fitting; Mercury's perihelion precession reproduced to 0.01 %.
* Solar and lunar eclipse engine (Besselian geometry, Herald-Sinnott Earth shadow, light-time shadow shift): global and local
  circumstances; 44 of 44 eclipses of 2021-2030 found with the right type, greatest-eclipse times within 18 s (solar) / 13 s (lunar)
  of the NASA/EclipseWise catalog; central line as GeoJSON.

**Labs** (`krios lab`): shadow (Eratosthenes, globe vs fitted flat-map models), celestial pole (star trails, spherical fit),
lunar parallax (triangulation, Monte-Carlo errors, globe vs flat geometry), solar path (sunrise/sunset/day length) and
eclipse (observed vs predicted contacts); pre-registered predictions (`plan`, `run --plan`), bilingual Markdown / JSON / CSV
reports with provenance, optional figures, labelled synthetic demos.

**CLI**: `eclipse find|local|path`, `orbit`, `compare`, `demo mercury`, `horizons vectors`, `state`, `lab list|template|plan|run|demo`, `info`.

**Known limitations**: the live Horizons service has not been exercised yet (a fake server covers the documented format);
the analytic Moon is ~10" nominal; DeltaT beyond 2025 is a prediction.
