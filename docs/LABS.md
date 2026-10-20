# Krios labs

A lab is a small, self-contained experiment: **your measurements -> independent predictions -> a quantitative comparison -> a
reproducible report.**  All labs share the same rules.

* **Predictions use only non-measurement inputs** (place, time, instrument settings).  `krios lab plan` freezes them and prints a SHA-256;
  `krios lab run --plan` later checks that the predictions did not change after you saw your data.  (It proves the
  software did not peek, not that *you* did not; publish the plan before measuring.)
* **Uncertainties are explicit**: every number carries a sigma, and statements are chi-square / Monte-Carlo based.
* **Alternatives are fair**: the flat-Earth comparison model gets free parameters (Sun height, Sun position, Polaris height)
  and is fitted by least squares from several starting points.  A model that fits only after tuning is reported with
  its parameters.  If your data cannot discriminate (too few or too close stations) the verdict says so.
* **Synthetic data are always labelled** (`metadata.synthetic: true`, a banner in every report).  The demos are *not* evidence.
* **Reports** (`--report`, `--json`, `--csv-dir`, `--lang en|ar`, `--plot`) include inputs, method, assumptions, versions and hashes.
* Inputs are JSON (complete) or CSV (the measurement table; `#` lines are comments).  `krios lab template <lab>` writes an example.

Prediction accuracy (Sun ~1", Moon ~5") is two orders of magnitude better than any citizen measurement, so the comparison is
limited by *your* instruments, which is what you want from an experiment.

---

## shadow - Sun altitude, Earth's radius (Eratosthenes), fair flat-map test

**Equipment**: a straight rod (1 m), a plumb line or level, a tape measure, a clock synchronised to UTC (GPS/NTP), GPS for
latitude/longitude.  For the radius: the distance between stations, measured *independently* (odometer, map, GPS track length).

**Procedure**: at an agreed UTC minute (several stations at once is best; any time works), measure the rod height and the shadow
length (to the middle of the blurred edge).  Use stations as far apart as possible, north-south being most informative; 4 or more
stations are needed to test the most generous flat model.

**Inputs** (`stations`): `name, lat, lon, time_utc, stick, shadow` (+ optional `height_m, sigma_stick, sigma_shadow`);
`baselines`: `a, b, km, sigma_km` (+ `source`; a source starting with "computed" is flagged as circular).
Settings: `sigma_floor_deg` (default 0.1, blur of the shadow edge), `seed`, `mc_samples`.

**What it computes**
1. Sun altitude `h = atan(stick/shadow)`; prediction from the ephemeris (rotating oblate Earth, standard refraction); residuals and z-scores.
2. Globe: 0 free parameters.  Flat: azimuthal-equidistant map centred on the north pole, Sun at height `H` over the sub-solar point
   (**F1**, 1 parameter) or over a free map point (**F2**, 3 parameters).  chi-square and p-values for each.
3. Eratosthenes: each station's latitude from its Sun altitude, the Sun's declination and the hour angle; the angle between adjacent
   stations divided into the measured distance gives R (Monte-Carlo sigma).  For the same pairs the Sun height a flat Earth would need.

**Reading the result**: *globe p-value high and both flat variants rejected* is the discriminating outcome.  *Both fit* means the
stations are too close or too few.  *Globe fails* means check clocks, longitudes, units and rod verticality first.
**Limits**: oblateness changes R by up to 0.3 % along a meridian; the 0.1 deg floor is a judgement call (it is in the report).

## pole - where the sky turns, and how fast

**Equipment**: a camera on a tripod, a phone inclinometer for the camera tilt, the lens' vertical field of view (EXIF / manufacturer),
a 1-3 hour exposure (or stacked trails).  A manual pole altitude (inclinometer) is also accepted.

**Inputs** (`sites`): `name, lat, lon, time_utc`, `visible_pole` (optional), then either `pole_altitude_deg + sigma_deg` or
`camera {elevation_deg, vfov_deg, image_width_px, image_height_px}` plus `trails [{points: [[x, y], ...] in TIME order,
exposure_s, sigma_px}]` (digitise 4-8 points along each of 2-5 trails with any image viewer).  Optional `rotation: ccw|cw`.

**What it computes**: pixels -> sky directions (level pinhole camera), then ONE rotation axis and an angular radius per arc fitted
**on the sphere** (a plain circle fit in the image is biased by the projection - about 0.4 deg in altitude and 7 % in rate when this was developed); pole altitude
with uncertainty (fit + tilt reading + field-of-view); rotation sense; turning rate vs 15.0411 deg/h; Polaris' separation from the true pole
(~0.65 deg now); across sites, a weighted fit of altitude vs |latitude| (prediction: slope 1, intercept 0) and the single-pole flat
model (Polaris at a fitted height over the centre of a polar map).

**Reading the result**: a southern site that sees clockwise rotation about a point in the *south* sky is something a single pole above
the northern centre does not provide.  Northern sites alone may leave the flat model alive (the report says so).
**Limits**: no roll, rectilinear lens, field of view known to ~1 %.

## parallax - the Moon's distance by triangulation

**Equipment**: a camera/theodolite, a way to get the Moon's direction to ~0.05 deg (alt/az from a calibrated phone app or, better, RA/Dec by
matching the stars in the photo with an astrometry tool), a synchronised clock; at least two observers thousands of km apart
(north-south is best), observing in the same minute.

**Inputs** (`observations`): `station, lat, lon, height_m, time_utc` and `alt_deg + az_deg` (apparent, true-north azimuth) **or**
`ra_deg + dec_deg` (+ `frame: J2000|of_date`), `sigma_deg`; optional `group`.  Settings: `group_tolerance_s`, `mc_samples`, `test_distances_km`.

**What it computes**: lines of sight in an inertial frame -> least-squares closest point -> distance from the Earth's centre with Monte-Carlo
68 % / 95 % intervals and a 99.7 % lower bound; measured vs expected parallax; measured vs predicted directions; and the **same
triangulation in two geometries** - globe (local verticals tilt) and a flat map (all verticals parallel) - scored by how consistently the
lines of sight meet (chi-square with 2n-3 degrees of freedom).  Hypothesis distances (default 5,000 and 50,000 km) are compared with the result.

**Reading the result**: a 5 % distance needs ~0.05 deg over ~5000 km; short baselines are reported as useless instead of giving a number.
**Limits**: refraction is removed with the standard model; time offsets inside a group are corrected with the predicted differential motion.

## solarpath - sunrise, sunset, day length

**Inputs**: `sites [{name, lat, lon, height_m}]`, `year` (sampled every `step_days`) and/or `dates`, `observations`:
`{site, kind: sunrise|sunset|noon|altitude, date, observed_utc, sigma_s}` (altitude: `time_utc, alt_deg, sigma_deg`).
**Computes**: full-year Sun path, sunrise/sunset (upper limb, -0.833 deg), solar transit and noon altitude, polar day/night, day length;
residuals in seconds with sigma = sqrt(observer^2 + model^2) (model default 90 s for horizon refraction/terrain).
**Reading the result**: places in the two hemispheres have their longest days on opposite solstices; a wrong clock shows as a constant offset.

## eclipse - contact times, observed vs predicted

**Inputs**: `eclipse {date, body: solar|lunar}`, `observer {lat, lon, height_m}` (solar), `observations [{event: C1|C2|max|C3|C4 or
P1|U1|U2|Greatest|U3|U4|P4, time_utc, sigma_s}]`; `--source horizons` for DE441 accuracy.
**Computes**: predicted contacts (UTC and TT), magnitude, obscuration, duration, Sun altitude; residual and z per contact; chi-square.
**Pre-register**: `krios lab plan eclipse ...` weeks ahead (the 2 August 2027 total eclipse crosses Spain, North Africa and the Arabian Peninsula);
`krios eclipse path --date 2027-08-02` writes the central line as GeoJSON.

---

## Writing a new lab

Subclass `krios.lab.core.Experiment`, set `name`, `rows_key`, `title`, implement `example()` (synthetic, seeded) and `run()`
(build the `predictions` table first from non-measurement inputs, then the comparison, models and verdict), decorate with `@register`.
Add a test that generates data from the *competing* model and checks that the lab does not misreport it.
