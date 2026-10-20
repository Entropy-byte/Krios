"""Celestial Pole Lab - where the sky turns, and how fast.

Measurement : (a) star-trail arcs digitised from a long exposure (pixel points in time order) with the camera's
              tilt and field of view, and/or (b) the pole's altitude read with an inclinometer; plus the
              rotation sense (ccw / cw) seen when facing the pole.
Prediction  : the visible celestial pole stands at altitude |latitude| (north pole in the north, south pole in
              the south); stars turn ccw about the north pole and cw about the south pole at 15.041 deg/h.
Comparison  : residuals per site; a fit across sites; the strongest single-pole flat model (Polaris at a fitted
              height above the centre of a polar map).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import astro
from .core import B, Experiment, LabResult, ModelFit, register
from .stats import chi2_pvalue, fit_least_squares, weighted_line_fit

SIDEREAL_RATE_DEG_H = 15.0410686


# ------------------------------------------------------------------ geometry
def pixel_altitude(cx: float, cy: float, cam: Dict[str, float]) -> float:
    """Altitude (deg) of image pixel (cx, cy) for a level pinhole camera looking at elevation e (no roll)."""
    W, H = float(cam["image_width_px"]), float(cam["image_height_px"])
    f = (H / 2.0) / math.tan(math.radians(float(cam["vfov_deg"])) / 2.0)
    a, b = (cx - W / 2.0) / f, (H / 2.0 - cy) / f
    e = math.radians(float(cam["elevation_deg"]))
    return math.degrees(math.asin((math.sin(e) + b * math.cos(e)) / math.sqrt(1 + a * a + b * b)))


def _arc_angles(points: np.ndarray, cx: float, cy: float) -> np.ndarray:
    return np.unwrap(np.arctan2(-(points[:, 1] - cy), points[:, 0] - cx))      # y up; ccw positive


def fit_concentric(arcs: List[np.ndarray], sigma_px: float):
    """Common centre (cx, cy) and one radius per arc.  Returns (cx, cy, radii, rms, cov_center)."""
    allp = np.vstack(arcs)
    # algebraic start from the first arc
    x, y = arcs[0][:, 0], arcs[0][:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    cx0, cy0 = sol[0], sol[1]
    m = len(arcs)

    def resid(p):
        c, out = p[:2], []
        for k, a in enumerate(arcs):
            out.append(np.hypot(a[:, 0] - c[0], a[:, 1] - c[1]) - p[2 + k])
        return np.concatenate(out)

    r0 = [float(np.mean(np.hypot(a[:, 0] - cx0, a[:, 1] - cy0))) for a in arcs]
    s = least_squares(resid, [cx0, cy0] + r0)
    rms = float(np.sqrt(np.mean(s.fun ** 2)))
    sig = max(sigma_px, rms)
    cov = sig ** 2 * np.linalg.pinv(s.jac.T @ s.jac)
    return float(s.x[0]), float(s.x[1]), s.x[2:], rms, cov[:2, :2]


def _pixel_dirs(points: np.ndarray, cam: Dict[str, float]) -> np.ndarray:
    """Unit vectors (x right, y forward-horizontal, z up) for pixels of a level pinhole camera."""
    W, H = float(cam["image_width_px"]), float(cam["image_height_px"])
    f = (H / 2.0) / math.tan(math.radians(float(cam["vfov_deg"])) / 2.0)
    a, b = (points[:, 0] - W / 2.0) / f, (H / 2.0 - points[:, 1]) / f
    e = math.radians(float(cam["elevation_deg"]))
    fwd, right, up = np.array([0, math.cos(e), math.sin(e)]), np.array([1.0, 0, 0]), np.array([0, -math.sin(e), math.cos(e)])
    d = fwd[None, :] + a[:, None] * right[None, :] + b[:, None] * up[None, :]
    return d / np.linalg.norm(d, axis=1)[:, None]


def _pole_vec(alt: float, az: float) -> np.ndarray:
    return np.array([math.cos(alt) * math.sin(az), math.cos(alt) * math.cos(az), math.sin(alt)])


def fit_pole_on_sphere(arcs: List[np.ndarray], cam: Dict[str, float], sigma_px: float):
    """Fit ONE rotation axis (the celestial pole) to small-circle arcs on the sky.

    Pixels are first turned into directions with the camera model, so the projection's distortion of
    circles into ellipses is handled exactly.  Returns dict(alt_deg, az_deg, radii_deg, rms_deg,
    sigma_alt_deg, sweeps_deg, sigma_sweep_deg)."""
    dirs = [_pixel_dirs(a, cam) for a in arcs]
    # start from the planar concentric fit
    cx, cy, _, _, _ = fit_concentric(arcs, sigma_px)
    d0 = _pixel_dirs(np.array([[cx, cy]]), cam)[0]
    alt0, az0 = math.asin(d0[2]), math.atan2(d0[0], d0[1])
    pv0 = _pole_vec(alt0, az0)
    r0 = [float(np.mean(np.arccos(np.clip(d @ pv0, -1, 1)))) for d in dirs]

    def resid(p):
        pv = _pole_vec(p[0], p[1])
        return np.concatenate([np.arccos(np.clip(d @ pv, -1, 1)) - p[2 + k] for k, d in enumerate(dirs)])

    sol = least_squares(resid, [alt0, az0] + r0)
    pv = _pole_vec(sol.x[0], sol.x[1])
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    f_px = (float(cam["image_height_px"]) / 2.0) / math.tan(math.radians(float(cam["vfov_deg"])) / 2.0)
    sig = max(sigma_px / f_px, rms)
    cov = sig ** 2 * np.linalg.pinv(sol.jac.T @ sol.jac)
    # swept angle about the axis, positive = rotation about +pole (clockwise for a viewer facing the pole)
    helper = np.array([0.0, 0.0, 1.0]) if abs(pv[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u1 = np.cross(pv, helper)
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(pv, u1)
    sweeps, sig_sw = [], []
    for k, d in enumerate(dirs):
        th = np.unwrap(np.arctan2(d @ u2, d @ u1))
        sweeps.append(math.degrees(th[-1] - th[0]))
        sig_sw.append(math.degrees(math.sqrt(2) * sig / max(math.sin(sol.x[2 + k]), 1e-3)))
    return {"alt_deg": math.degrees(sol.x[0]), "az_deg": math.degrees(sol.x[1]),
            "radii_deg": [math.degrees(r) for r in sol.x[2:]], "rms_deg": math.degrees(rms),
            "sigma_alt_deg": math.degrees(math.sqrt(cov[0, 0])), "sweeps_deg": sweeps, "sigma_sweep_deg": sig_sw}


@register
class PoleLab(Experiment):
    name = "pole"
    rows_key = "sites"
    title = B("Celestial Pole Lab: where the sky turns, and how fast",
              "مختبر القطب السماوي: أين تدور السماء وبأي سرعة")
    blurb = B("Star trails / pole altitude at several latitudes -> pole altitude = latitude, rotation sense, 15.04 deg/h.",
              "مسارات النجوم أو ارتفاع القطب في عدة خطوط عرض -> ارتفاع القطب = خط العرض، واتجاه الدوران، و15.04°/ساعة.")

    # ------------------------------------------------------------------ example
    @classmethod
    def example(cls, seed: int = 11) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        W, Hh, vfov = 6000, 4000, 46.0
        f = (Hh / 2) / math.tan(math.radians(vfov) / 2)
        sites = []
        for name, lat, lon in (("North-1", 52.0, 8.0), ("North-2", 36.0, 10.0), ("Tropics", 12.0, 100.0), ("South-1", -34.0, 151.0)):
            south = lat < 0
            alt_pole = abs(lat)
            e_reading = round(alt_pole - 6.0, 1)                       # what the user's inclinometer says
            e_true = e_reading + rng.normal(0, 0.2)
            up = np.array([0.0, 0.0, 1.0])
            hdir = np.array([0.0, -1.0, 0.0]) if south else np.array([0.0, 1.0, 0.0])      # x=east, y=north
            right = np.array([-1.0, 0.0, 0.0]) if south else np.array([1.0, 0.0, 0.0])
            et = math.radians(e_true)
            fwd = math.cos(et) * hdir + math.sin(et) * up
            cam_up = -math.sin(et) * hdir + math.cos(et) * up
            pole = math.cos(math.radians(alt_pole)) * hdir + math.sin(math.radians(alt_pole)) * up
            u1 = np.cross(pole, right)
            u1 /= np.linalg.norm(u1)
            u2 = np.cross(pole, u1)
            sign = +1.0 if south else -1.0                              # south: cw (+), north: ccw (-)
            exposure = 7200.0
            sweep = math.radians(SIDEREAL_RATE_DEG_H * exposure / 3600.0)
            trails = []
            for rho_deg, phase in ((8.0, 0.4), (16.0, 2.2), (27.0, 4.0), (35.0, 5.3)):
                pts = []
                for k in range(6):
                    th = phase + sign * sweep * k / 5.0
                    rho = math.radians(rho_deg)
                    d = math.cos(rho) * pole + math.sin(rho) * (math.cos(th) * u1 + math.sin(th) * u2)
                    if d @ fwd <= 0.05:
                        break
                    a, b = (d @ right) / (d @ fwd), (d @ cam_up) / (d @ fwd)
                    pts.append([round(W / 2 + a * f + rng.normal(0, 1.5), 1), round(Hh / 2 - b * f + rng.normal(0, 1.5), 1)])
                if len(pts) == 6 and all(0 <= p[0] <= W and 0 <= p[1] <= Hh for p in pts):
                    trails.append({"points": pts, "exposure_s": exposure, "sigma_px": 1.5})
            sites.append({"name": name, "lat": lat, "lon": lon, "time_utc": "2026-10-12T21:00:00Z",
                          "camera": {"elevation_deg": e_reading, "vfov_deg": vfov, "image_width_px": W, "image_height_px": Hh},
                          "trails": trails})
        return {"metadata": {"title": "Celestial Pole Lab demonstration", "synthetic": True, "seed": seed,
                             "note": "Synthetic star-trail arcs (2 h exposures) rendered by a pinhole-camera model."},
                "settings": {"sigma_elev_deg": 0.3, "sigma_fov_frac": 0.01}, "sites": sites}

    # ---------------------------------------------------------------------- run
    def run(self, predict_only: bool = False) -> LabResult:
        spec, st = self.spec, self.spec.get("settings", {})
        s_elev = float(st.get("sigma_elev_deg", 0.3))
        s_fov = float(st.get("sigma_fov_frac", 0.01))
        sites = spec["sites"]
        res = LabResult(self.name, self.title)
        preds, trails_tab, meas_rows = [], [], []
        for r in sites:
            lat = float(r["lat"])
            lon = float(r.get("lon", 0.0))
            expected_pole = "north" if lat >= 0 else "south"
            alt_true = abs(lat)
            row = {"site": r["name"], "lat": lat, "expected_pole": expected_pole,
                   "pole_alt_geometric_deg": alt_true, "pole_alt_apparent_deg": float(astro.apparent_alt(alt_true)),
                   "expected_rotation": "ccw" if lat >= 0 else "cw", "sidereal_rate_deg_h": SIDEREAL_RATE_DEG_H}
            if r.get("time_utc"):
                jd = astro.parse_utc(r["time_utc"])
                row["polaris_offset_from_pole_deg"] = astro.polaris_offset_deg(jd)
                pa, pz = astro.polaris_alt_az(lat, lon, jd)
                row["polaris_alt_deg"], row["polaris_az_deg"] = pa, pz
            preds.append(row)
        res.tables["predictions"] = preds
        res.method = [
            B("Each trail pixel becomes a sky direction through a level pinhole camera (elevation e, vertical field of view). One rotation axis "
              "(the pole) and one angular radius per arc are fitted on the sphere, so the projection's distortion of circles into ellipses is handled exactly.",
              "كل بكسل في المسار يتحول إلى اتجاه في السماء عبر كاميرا ثقبية مستوية (الارتفاع e ومجال الرؤية الرأسي). يُضبط محور دوران واحد (القطب) ونصف قطر زاوي لكل قوس على الكرة، فيُعالَج تشويه الإسقاط (دوائر تصبح قطوعًا ناقصة) بدقة."),
            B("Pole altitude = altitude of the fitted axis; its uncertainty combines the fit, the camera-tilt reading (~0.3 deg) and the field-of-view error.",
              "ارتفاع القطب = ارتفاع المحور المضبوط؛ وعدم اليقين يجمع خطأ المطابقة وقراءة ميل الكاميرا (~0.3°) وخطأ مجال الرؤية."),
            B("Rotation rate = swept angle about the centre / exposure time; the sidereal prediction is 15.0411 deg/h.",
              "معدل الدوران = الزاوية المقطوعة حول المركز ÷ زمن التعريض؛ والتنبؤ النجمي 15.0411°/ساعة."),
            B("Prediction for the pole altitude: |latitude| (+ standard refraction); Polaris is NOT the pole - it sits ~0.7 deg away, "
              "so it is predicted separately.", "التنبؤ بارتفاع القطب: |خط العرض| (مع الانكسار القياسي)؛ والنجم القطبي ليس القطب نفسه إذ يبعد ~0.7°، فيُتنبأ به على حدة.")]
        res.assumptions = [
            B("Camera level (no roll), rectilinear lens, field of view known to ~1 %, elevation read to ~0.3 deg.", "الكاميرا مستوية بلا ميلان جانبي وعدسة خطية ومجال الرؤية معلوم بدقة ~1% والارتفاع مقروء بدقة ~0.3°."),
            B("List the points of each trail in time order (first point = start of the exposure) so that the rotation sense is defined.", "رتّب نقاط كل مسار زمنيًا (أول نقطة = بداية التعريض) كي يتحدد اتجاه الدوران."),
            B("A short arc fixes its circle poorly; use several arcs and 1-3 hour exposures.", "القوس القصير يحدد دائرته بسوء؛ استخدم عدة أقواس وتعريضات 1-3 ساعات."),
            B("The flat comparison model is the strongest single-pole version: Polaris at a fitted height over the centre of a polar map.",
              "نموذج المقارنة المسطح هو أقوى صور القطب الواحد: النجم القطبي على ارتفاع مضبوط فوق مركز خريطة قطبية.")]
        if predict_only:
            res.summary = {"sites": len(sites), "mode": "prediction only (pre-registration)"}
            res.verdict = [B("Predictions generated before measuring. Save this plan.", "تم توليد التنبؤات قبل القياس. احفظ الخطة.")]
            return self._finish(res)

        # ---- measurements per site
        alt_m, alt_s, lat_l, names, south_l = [], [], [], [], []
        mismatch_rot, mismatch_pole = 0, 0
        rates = []
        for r, p in zip(sites, preds):
            lat = float(r["lat"])
            meas, sig, src, sense = None, None, "", r.get("rotation")
            pole_info = {}
            good = [t for t in r.get("trails", []) if len(t.get("points", [])) >= 3]
            arcs = [np.array(t["points"], dtype=float) for t in good]
            if arcs:
                sp = float(np.mean([float(t.get("sigma_px", 2.0)) for t in good]))
                if "camera" in r:
                    cam = dict(r["camera"])
                    fit = fit_pole_on_sphere(arcs, cam, sp)
                    cam_f = dict(cam, vfov_deg=cam["vfov_deg"] * (1 + s_fov))
                    d_fov = fit_pole_on_sphere(arcs, cam_f, sp)["alt_deg"] - fit["alt_deg"]
                    # camera elevation shifts the altitude one-for-one; its reading error is s_elev
                    meas = fit["alt_deg"]
                    sig = math.sqrt(fit["sigma_alt_deg"] ** 2 + s_elev ** 2 + d_fov ** 2)
                    src = "star trails + camera (sphere fit)"
                    sweeps, s_sw = fit["sweeps_deg"], fit["sigma_sweep_deg"]
                    sense_votes = [("cw" if sw > 0 else "ccw") for sw in sweeps]       # + = clockwise seen facing the pole
                    radii_out = fit["radii_deg"]
                    pole_info = {"pole_az_from_camera_axis_deg": fit["az_deg"], "fit_rms_deg": fit["rms_deg"]}
                else:
                    cx, cy, radii, rms, cov = fit_concentric(arcs, sp)
                    sweeps, s_sw, radii_out = [], [], list(radii)
                    for k, a in enumerate(arcs):
                        th = _arc_angles(a, cx, cy)
                        sweeps.append(-float(np.degrees(th[-1] - th[0])))              # same convention: + = clockwise
                        s_sw.append(math.degrees(math.sqrt(2) * sp / radii[k]))
                    sense_votes = [("cw" if sw > 0 else "ccw") for sw in sweeps]
                    pole_info = {"centre_x_px": cx, "centre_y_px": cy}
                    src = "star trails, no camera model (rotation only)"
                for k, t in enumerate(good):
                    ex = t.get("exposure_s")
                    rate = abs(sweeps[k]) / float(ex) * 3600.0 if ex else None
                    s_rate = s_sw[k] / float(ex) * 3600.0 if ex else None
                    if rate is not None:
                        rates.append((rate, s_rate))
                    trails_tab.append({"site": r["name"], "trail": k + 1,
                                       "radius_deg_or_px": radii_out[k], "sweep_deg": abs(sweeps[k]),
                                       "sense_seen": sense_votes[k], "rate_deg_h": rate, "rate_sigma_deg_h": s_rate,
                                       "rate_over_sidereal": None if rate is None else rate / SIDEREAL_RATE_DEG_H,
                                       **pole_info})
                if sense is None:
                    sense = "cw" if sum(1 for v_ in sense_votes if v_ == "cw") > len(sense_votes) / 2 else "ccw"
            if r.get("pole_altitude_deg") is not None:
                meas, sig, src = float(r["pole_altitude_deg"]), float(r.get("sigma_deg", 0.5)), "inclinometer / manual"
            row = {"site": r["name"], "lat": lat, "source": src or "none", "pole_alt_measured_deg": meas,
                   "sigma_deg": sig, "pole_alt_predicted_deg": p["pole_alt_apparent_deg"],
                   "residual_deg": None if meas is None else meas - p["pole_alt_apparent_deg"],
                   "z": None if meas is None else (meas - p["pole_alt_apparent_deg"]) / sig,
                   "rotation_seen": sense, "rotation_expected": p["expected_rotation"],
                   "rotation_ok": None if sense is None else sense == p["expected_rotation"]}
            if sense is not None and sense != p["expected_rotation"]:
                mismatch_rot += 1
            meas_rows.append(row)
            if meas is not None:
                alt_m.append(meas); alt_s.append(sig); lat_l.append(lat); names.append(r["name"]); south_l.append(lat < 0)
        res.tables["measurements"] = meas_rows
        if trails_tab:
            res.tables["trails"] = trails_tab

        # ---- models across sites
        s: Dict[str, Any] = {"sites": len(sites), "sites_with_altitude": len(alt_m)}
        models: List[ModelFit] = []
        if alt_m:
            am, asg, la = np.array(alt_m), np.array(alt_s), np.array(lat_l)
            ap = np.array([float(astro.apparent_alt(abs(x))) for x in la])
            chi_g = float(np.sum(((am - ap) / asg) ** 2))
            models.append(ModelFit("Globe: pole altitude = |latitude| (all sites)", 0, chi_g, len(am), chi2_pvalue(chi_g, len(am))))
            north = la >= 0
            if north.sum() >= 2:
                cg = float(np.sum((((am - ap) / asg) ** 2)[north]))
                models.append(ModelFit("Globe (northern sites only)", 0, cg, int(north.sum()), chi2_pvalue(cg, int(north.sum()))))
                d = np.array([astro.K_FE * (90.0 - x) for x in la[north]])
                rf = lambda p: (am[north] - np.degrees(np.arctan2(10 ** p[0], d))) / asg[north]
                pf, cf, _ = fit_least_squares(rf, [[v] for v in (2.5, 3, 3.5, 4, 4.5, 5)], ([1.0], [6.5]))
                dof = int(north.sum()) - 1
                models.append(ModelFit("Flat polar map, single Polaris at height H (northern sites only)", 1, cf, dof,
                                       chi2_pvalue(cf, dof), {"H_km": 10 ** float(pf[0])},
                                       "no southern pole exists in this model"))
            if len(am) >= 3:
                a0, b1, cov, _ = weighted_line_fit(np.abs(la), am, asg)
                s.update({"fit_slope": b1, "fit_slope_sigma": math.sqrt(cov[1, 1]), "fit_intercept_deg": a0,
                          "fit_intercept_sigma_deg": math.sqrt(cov[0, 0])})
            s["rms_residual_deg"] = float(np.sqrt(np.mean((am - ap) ** 2)))
        if rates:
            rv = np.array([r for r, _ in rates])
            s["rate_mean_deg_h"] = float(rv.mean())
            s["rate_over_sidereal"] = float(rv.mean() / SIDEREAL_RATE_DEG_H)
            s["rate_trails"] = len(rv)
        n_south_obs = sum(1 for r, p in zip(meas_rows, preds) if p["expected_pole"] == "south" and r["rotation_seen"] == "cw")
        s["southern_sites_with_cw_rotation"] = n_south_obs
        res.summary, res.models = s, models
        res.verdict = self._verdict(s, models, meas_rows, n_south_obs, mismatch_rot)
        return self._finish(res)

    @staticmethod
    def _verdict(s, models, rows, n_south, mismatch_rot) -> List[Dict[str, str]]:
        v: List[Dict[str, str]] = []
        if not models:
            return [B("No pole altitude could be measured: give a camera (tilt, field of view) with trails, or a manual altitude.",
                      "تعذّر قياس ارتفاع القطب: أدخل كاميرا (ميل ومجال رؤية) مع المسارات أو ارتفاعًا يدويًا.")]
        g = models[0]
        v.append(B(f"Pole altitude vs |latitude| over {s['sites_with_altitude']} site(s): chi2 = {g.chi2:.1f}, p = {g.p_value:.2g}, rms residual {s['rms_residual_deg']:.2f} deg"
                   + (f"; fitted slope {s['fit_slope']:.3f} +/- {s['fit_slope_sigma']:.3f} (prediction 1), intercept {s['fit_intercept_deg']:.2f} +/- {s['fit_intercept_sigma_deg']:.2f} deg (prediction 0)." if "fit_slope" in s else "."),
                   f"ارتفاع القطب مقابل |خط العرض| في {s['sites_with_altitude']} موقع: χ² = {g.chi2:.1f}، p = {g.p_value:.2g}، متوسط الخطأ {s['rms_residual_deg']:.2f}°"
                   + (f"؛ الميل المقاس {s['fit_slope']:.3f} ± {s['fit_slope_sigma']:.3f} (المتوقع 1) والقاطع {s['fit_intercept_deg']:.2f} ± {s['fit_intercept_sigma_deg']:.2f}° (المتوقع 0)." if "fit_slope" in s else ".")))
        flat = [m for m in models if m.name.startswith("Flat")]
        if flat and flat[0].dof > 0:
            f = flat[0]
            gn = [m for m in models if m.name.startswith("Globe (northern")][0]
            if f.p_value < 0.001 and gn.p_value >= 0.01:
                v.append(B(f"Even with its height fitted, the single-pole flat model fails on the northern sites (chi2 = {f.chi2:.1f}, p = {f.p_value:.2g}) while the globe prediction fits (p = {gn.p_value:.2g}).",
                           f"حتى بعد ضبط الارتفاع، يفشل نموذج القطب الواحد المسطح في المواقع الشمالية (χ² = {f.chi2:.1f}، p = {f.p_value:.2g}) بينما يطابق تنبؤ الكرة (p = {gn.p_value:.2g})."))
            elif f.p_value >= 0.01:
                v.append(B("The northern sites alone do not exclude a single-pole flat model with a fitted height: add a site at a very different latitude (and one in the southern hemisphere).",
                           "المواقع الشمالية وحدها لا تستبعد نموذج القطب الواحد المسطح بارتفاع مضبوط: أضف موقعًا بخط عرض مختلف جدًا وآخر في نصف الكرة الجنوبي."))
        if n_south:
            v.append(B(f"{n_south} southern site(s) saw the stars turn CLOCKWISE about a point in the SOUTH sky; a single pole above the northern centre predicts counter-clockwise turning about a northern point only.",
                       f"رصد {n_south} موقع جنوبي النجوم تدور مع عقارب الساعة حول نقطة في السماء الجنوبية؛ بينما يتنبأ نموذج القطب الواحد فوق المركز الشمالي بدوران عكس عقارب الساعة حول نقطة شمالية فقط."))
        if mismatch_rot:
            v.append(B(f"{mismatch_rot} site(s) reported a rotation sense different from the globe prediction: check the photo orientation.",
                       f"أبلغ {mismatch_rot} موقع عن اتجاه دوران يخالف تنبؤ الكرة: تحقق من اتجاه الصورة."))
        if "rate_over_sidereal" in s:
            v.append(B(f"Measured turning rate = {s['rate_mean_deg_h']:.2f} deg/h = {s['rate_over_sidereal']:.3f} x the sidereal rate (15.041 deg/h).",
                       f"معدل الدوران المقاس = {s['rate_mean_deg_h']:.2f}°/ساعة = {s['rate_over_sidereal']:.3f} × المعدل النجمي (15.041°/ساعة)."))
        return v

    def plot(self, result: LabResult, path: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        rows = [r for r in result.tables.get("measurements", []) if r["pole_alt_measured_deg"] is not None]
        if not rows:
            return False
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        xs = [abs(r["lat"]) for r in rows]
        ax.errorbar(xs, [r["pole_alt_measured_deg"] for r in rows], yerr=[r["sigma_deg"] for r in rows], fmt="ko", label="measured")
        ax.plot([0, 90], [0, 90], "g-", label="globe: altitude = |latitude|")
        ax.set_xlabel("|latitude| (deg)")
        ax.set_ylabel("altitude of the visible celestial pole (deg)")
        ax.set_title("Celestial Pole Lab" + (" (SYNTHETIC data)" if result.provenance.get("synthetic") else ""))
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return True
