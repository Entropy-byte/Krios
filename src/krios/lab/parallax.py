"""Lunar Parallax Lab - the Moon's distance by triangulation from several places.

Measurement : the Moon's direction (altitude + azimuth, or RA + Dec) seen from >= 2 places at the same moment.
Prediction  : topocentric Moon from the Krios ephemeris (directions, distance, expected parallax).
Comparison  : (a) measured vs predicted directions; (b) distance from the closest approach of the lines of
              sight, with Monte-Carlo uncertainty; (c) the same triangulation in two geometries - a globe
              (local verticals tilt from place to place) and a flat map (all verticals parallel) - and how
              consistently the lines of sight meet in each.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .. import frames as fr
from . import astro
from .core import B, Experiment, LabResult, ModelFit, register
from .stats import chi2_pvalue, interval


def triangulate(points: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Point closest (least squares) to the lines  p_i + s u_i."""
    I = np.eye(3)
    M = np.zeros((3, 3))
    b = np.zeros(3)
    for p, u in zip(points, dirs):
        P = I - np.outer(u, u)
        M += P
        b += P @ p
    return np.linalg.solve(M, b)


def line_misses(x: np.ndarray, points: np.ndarray, dirs: np.ndarray):
    rng_, miss = [], []
    for p, u in zip(points, dirs):
        v = x - p
        rng_.append(float(np.linalg.norm(v)))
        miss.append(float(np.linalg.norm(v - (v @ u) * u)))
    return np.array(rng_), np.array(miss)


def _perturb(u: np.ndarray, sigma_rad: float, rng) -> np.ndarray:
    helper = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(u, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(u, t1)
    v = u + rng.normal(0, sigma_rad) * t1 + rng.normal(0, sigma_rad) * t2
    return v / np.linalg.norm(v)


def _flat_frame(lat, lon):
    x, y = astro.fe_xy(lat, lon)
    r = math.hypot(x, y)
    if r < 1.0:
        return None
    north = np.array([-x / r, -y / r, 0.0])
    east = np.array([-y / r, x / r, 0.0])
    return np.array([x, y, 0.0]), east, north


@register
class ParallaxLab(Experiment):
    name = "parallax"
    rows_key = "observations"
    title = B("Lunar Parallax Lab: the Moon's distance by triangulation",
              "مختبر اختلاف المنظر القمري: بُعد القمر بالتثليث")
    blurb = B("Simultaneous Moon directions from distant places -> distance, parallax, globe vs flat triangulation.",
              "اتجاهات القمر المتزامنة من أماكن متباعدة -> المسافة واختلاف المنظر وتثليث الكرة مقابل المسطح.")

    # ------------------------------------------------------------------ example
    @classmethod
    def example(cls, seed: int = 5) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        stations = [("Oslo-ish", 60.0, 20.0), ("Cairo-ish", 30.0, 20.0), ("Equator", 0.0, 20.0),
                    ("Windhoek-ish", -22.0, 20.0), ("Cape-ish", -34.0, 20.0)]
        base = astro.parse_utc("2026-10-25T00:00:00Z")
        best, best_alt = None, -90
        for k in range(144):
            jd = base + k / 144.0
            alts = [float(astro.topocentric("moon", la, lo, 0.0, jd)["alt_app"][0]) for _, la, lo in stations]
            if min(alts) > best_alt:
                best, best_alt = jd, min(alts)
        sigma = 0.05
        obs = []
        for name, la, lo in stations:
            t = astro.topocentric("moon", la, lo, 0.0, best)
            obs.append({"station": name, "lat": la, "lon": lo, "height_m": 0.0, "time_utc": astro.jd_to_iso(best, 0),
                        "alt_deg": round(float(t["alt_app"][0]) + rng.normal(0, sigma), 3),
                        "az_deg": round(float(t["az"][0]) + rng.normal(0, sigma / max(math.cos(math.radians(t["alt_app"][0])), 0.2)), 3),
                        "sigma_deg": sigma})
        return {"metadata": {"title": "Lunar Parallax demonstration", "synthetic": True, "seed": seed,
                             "note": "Synthetic simultaneous Moon sightings from five places on the 20E meridian."},
                "settings": {"mc_samples": 3000, "seed": 12345, "group_tolerance_s": 30,
                             "test_distances_km": [5000.0, 50000.0]},
                "observations": obs}

    # ------------------------------------------------------------------ helpers
    def _prepare(self):
        obs = self.spec["observations"]
        rows = []
        for o in obs:
            jd = astro.parse_utc(o["time_utc"])
            rows.append({"o": o, "jd": jd, "lat": float(o["lat"]), "lon": float(o["lon"]),
                         "h": float(o.get("height_m", 0.0)), "name": str(o["station"]),
                         "sigma": float(o.get("sigma_deg", self.spec.get("settings", {}).get("sigma_deg", 0.1)))})
        tol = float(self.spec.get("settings", {}).get("group_tolerance_s", 30)) / 86400.0
        groups: Dict[str, List[int]] = {}
        explicit = any("group" in r["o"] for r in rows)
        if explicit:
            for i, r in enumerate(rows):
                groups.setdefault(str(r["o"].get("group", "g1")), []).append(i)
        else:
            order = sorted(range(len(rows)), key=lambda i: rows[i]["jd"])
            cur, last = [], None
            for i in order:
                if last is not None and rows[i]["jd"] - last > tol:
                    groups[f"g{len(groups) + 1}"] = cur
                    cur = []
                cur.append(i)
                last = rows[i]["jd"]
            if cur:
                groups[f"g{len(groups) + 1}"] = cur
        return rows, groups

    # ---------------------------------------------------------------------- run
    def run(self, predict_only: bool = False) -> LabResult:
        st = self.spec.get("settings", {})
        mc_n = int(st.get("mc_samples", 3000))
        rng = np.random.default_rng(int(st.get("seed", 12345)))
        tests = [float(x) for x in st.get("test_distances_km", [5000.0])]
        rows, groups = self._prepare()
        res = LabResult(self.name, self.title)

        # ---- independent predictions
        topo = [astro.topocentric("moon", r["lat"], r["lon"], r["h"], r["jd"]) for r in rows]
        preds = []
        for gname, idx in groups.items():
            exp_par = None
            if len(idx) >= 2:
                U = np.array([topo[i]["unit"][0] for i in idx])
                exp_par = max(math.degrees(math.acos(float(np.clip(U[a] @ U[b], -1, 1))))
                              for a in range(len(idx)) for b in range(a + 1, len(idx)))
            for i in idx:
                geo = topo[i]["unit"][0] * topo[i]["dist_km"][0] + topo[i]["obs_inertial"][0]
                preds.append({"group": gname, "station": rows[i]["name"], "time_utc": rows[i]["o"]["time_utc"],
                              "moon_alt_deg": float(topo[i]["alt_app"][0]), "moon_az_deg": float(topo[i]["az"][0]),
                              "topocentric_distance_km": float(topo[i]["dist_km"][0]),
                              "geocentric_distance_km": float(np.linalg.norm(geo)),
                              "expected_group_parallax_deg": exp_par})
        res.tables["predictions"] = preds
        res.method = [
            B("Each sighting becomes a unit vector in an inertial frame (local alt/az -> Earth-fixed -> inertial with sidereal time); "
              "the observer's position vector comes from latitude, longitude and height on the WGS-84 ellipsoid.",
              "كل رصد يتحول إلى متجه وحدة في إطار عطالي (ارتفاع/سمت محلي -> مرتبط بالأرض -> عطالي بالزمن النجمي)، وموضع الراصد من خط العرض والطول والارتفاع على المجسم WGS-84."),
            B("Moon position = least-squares closest point to all lines of sight; distance uncertainty from 3000 Monte-Carlo draws of the angular errors.",
              "موضع القمر = أقرب نقطة بالمربعات الصغرى لكل خطوط الرؤية؛ وعدم اليقين في المسافة من 3000 محاكاة مونت كارلو للأخطاء الزاوية."),
            B("Globe vs flat triangulation: globe = local verticals tilt with latitude/longitude; flat = a polar azimuthal-equidistant map with all verticals parallel. "
              "Statistic: miss distance of each line from the common point / (range x angular error).",
              "تثليث الكرة مقابل المسطح: الكرة = الأعمدة المحلية تميل بحسب خط العرض والطول؛ المسطح = خريطة قطبية متساوية المسافة بأعمدة متوازية. الإحصاء: مسافة ابتعاد كل خط عن النقطة المشتركة ÷ (المدى × الخطأ الزاوي)."),
            B("Time offsets inside a group are corrected with the predicted differential motion (a few arcseconds).",
              "تُصحَّح فروق الزمن داخل المجموعة بالحركة التفاضلية المتوقعة (بضع ثوانٍ قوسية).")]
        res.assumptions = [
            B("Directions are APPARENT (include refraction) when given as alt/az; Krios removes standard refraction. RA/Dec from star matching need no refraction correction.",
              "الاتجاهات ظاهرية (تتضمن الانكسار) عند إعطائها ارتفاعًا وسمتًا؛ ويزيل Krios الانكسار القياسي. أما RA/Dec من مطابقة النجوم فلا تحتاج تصحيح الانكسار."),
            B("Simultaneity within ~30 s and a clock good to ~1 s; azimuth measured from TRUE north (correct magnetic declination).",
              "تزامن خلال ~30 ثانية وساعة بدقة ~1 ثانية؛ والسمت من الشمال الحقيقي (صحّح الانحراف المغناطيسي)."),
            B("Accuracy is set by the angular error and the baseline: 0.05 deg over 5000 km gives roughly 5 % in distance. Short baselines cannot measure the Moon.",
              "الدقة يحددها الخطأ الزاوي وطول القاعدة: 0.05° على 5000 كم تعطي نحو 5% في المسافة. القواعد القصيرة لا تقيس القمر."),
            B("The flat map uses the common azimuthal-equidistant layout (111.195 km/deg); a flat-Earth advocate may supply another layout through station coordinates.",
              "الخريطة المسطحة تستخدم التخطيط المتساوي المسافة الشائع (111.195 كم/درجة)؛ ويمكن لأنصار الأرض المسطحة تقديم تخطيط آخر عبر إحداثيات المحطات.")]
        if predict_only:
            res.summary = {"observations": len(rows), "groups": len(groups), "mode": "prediction only (pre-registration)"}
            res.verdict = [B("Predictions generated before measuring. Save this plan.", "تم توليد التنبؤات قبل القياس. احفظ الخطة.")]
            return self._finish(res)

        # ---- measured directions (inertial, at the group reference time)
        cmp_rows, grp_rows, models, summary_all = [], [], [], {}
        chi_pred, n_pred = 0.0, 0
        flat_possible = True
        results_by_group = {}
        for gname, idx in groups.items():
            if len(idx) < 2:
                continue
            t_ref = float(np.mean([rows[i]["jd"] for i in idx]))
            pts, dirs, sig, pts_f, dirs_f, ranges_pred = [], [], [], [], [], []
            for i in idx:
                r, o = rows[i], rows[i]["o"]
                tp = topo[i]
                if "alt_deg" in o:
                    alt_true = float(astro.true_alt(float(o["alt_deg"])))
                    u = astro.enu_to_inertial(alt_true, float(o["az_deg"]), r["lat"], r["lon"], float(tp["gast"][0]))
                elif "ra_deg" in o:
                    jdtt = float(tp["jd_tt"][0])
                    v = np.array([math.cos(math.radians(o["dec_deg"])) * math.cos(math.radians(o["ra_deg"])),
                                  math.cos(math.radians(o["dec_deg"])) * math.sin(math.radians(o["ra_deg"])),
                                  math.sin(math.radians(o["dec_deg"]))])
                    u = v if str(o.get("frame", "J2000")).lower() == "of_date" else fr.apply(fr.j2000_to_true_of_date(jdtt)[0], v)
                else:
                    raise ValueError("each observation needs alt_deg+az_deg or ra_deg+dec_deg")
                u = np.asarray(u, dtype=float).reshape(3)
                u /= np.linalg.norm(u)
                # move the sighting to the group reference time with the *predicted* differential change
                if abs(r["jd"] - t_ref) > 0.05 / 86400:
                    tr = astro.topocentric("moon", r["lat"], r["lon"], r["h"], t_ref)
                    u = u + (tr["unit"][0] - tp["unit"][0])
                    u /= np.linalg.norm(u)
                    p_ref = tr["obs_inertial"][0]
                else:
                    p_ref = tp["obs_inertial"][0]
                pts.append(p_ref)
                dirs.append(u)
                sig.append(math.radians(r["sigma"]))
                sep = math.degrees(math.acos(float(np.clip(u @ topo[i]["unit"][0], -1, 1))))
                cmp_rows.append({"group": gname, "station": r["name"], "alt_measured_deg": o.get("alt_deg"),
                                 "az_measured_deg": o.get("az_deg"), "alt_predicted_deg": float(tp["alt_app"][0]),
                                 "az_predicted_deg": float(tp["az"][0]), "direction_error_deg": sep,
                                 "sigma_deg": r["sigma"], "z": sep / r["sigma"]})
                chi_pred += (sep / r["sigma"]) ** 2
                n_pred += 2
                # flat geometry (needs alt/az)
                if "alt_deg" in o:
                    fl = _flat_frame(r["lat"], r["lon"])
                    if fl is None:
                        flat_possible = False
                    else:
                        pos, east, north = fl
                        a_, z_ = math.radians(float(o["az_deg"])), math.radians(alt_true)
                        d = math.sin(a_) * math.cos(z_) * east + math.cos(a_) * math.cos(z_) * north + math.sin(z_) * np.array([0, 0, 1.0])
                        pts_f.append(pos)
                        dirs_f.append(d)
                else:
                    flat_possible = False
            pts, dirs, sig = np.array(pts), np.array(dirs), np.array(sig)
            x = triangulate(pts, dirs)
            rng_, miss = line_misses(x, pts, dirs)
            pulls = miss / (rng_ * sig)
            chi_g = float(np.sum(pulls ** 2))
            dof = 2 * len(idx) - 3
            # Monte Carlo
            Ds, Ps = [], []
            for _ in range(mc_n):
                d2 = np.array([_perturb(u, s_, rng) for u, s_ in zip(dirs, sig)])
                try:
                    xs = triangulate(pts, d2)
                except np.linalg.LinAlgError:
                    continue
                if np.linalg.norm(xs) > 0:
                    Ds.append(float(np.linalg.norm(xs)))
            Ds = np.array(Ds)
            D_lo, D_med, D_hi = interval(Ds)
            D_lo95, _, D_hi95 = interval(Ds, 0.95)
            D_lb = float(np.percentile(Ds, 0.135))
            # measured parallax: largest angle between two measured lines of sight
            par = max(math.degrees(math.acos(float(np.clip(dirs[a] @ dirs[b], -1, 1))))
                      for a in range(len(idx)) for b in range(a + 1, len(idx)))
            geo_pred = np.mean([np.linalg.norm(topo[i]["unit"][0] * topo[i]["dist_km"][0] + topo[i]["obs_inertial"][0]) for i in idx])
            exp_par = [p["expected_group_parallax_deg"] for p in preds if p["group"] == gname][0]
            gr = {"group": gname, "stations": len(idx), "distance_km": float(np.linalg.norm(x)), "distance_mc_median_km": D_med,
                  "sigma_km": 0.5 * (D_hi - D_lo), "ci95_low_km": D_lo95, "ci95_high_km": D_hi95,
                  "lower_bound_3sigma_km": D_lb, "ephemeris_distance_km": float(geo_pred),
                  "difference_in_sigma": (float(np.linalg.norm(x)) - float(geo_pred)) / max(0.5 * (D_hi - D_lo), 1e-9),
                  "parallax_measured_deg": par, "parallax_expected_deg": exp_par,
                  "globe_chi2": chi_g, "globe_dof": dof,
                  "moon_position_x_km": float(x[0]), "moon_position_y_km": float(x[1]), "moon_position_z_km": float(x[2])}
            if len(pts_f) == len(idx) and flat_possible:
                xf = triangulate(np.array(pts_f), np.array(dirs_f))
                rf, mf = line_misses(xf, np.array(pts_f), np.array(dirs_f))
                chi_f = float(np.sum((mf / (rf * sig)) ** 2))
                gr.update({"flat_chi2": chi_f, "flat_dof": dof, "flat_moon_height_km": float(xf[2]),
                           "flat_moon_horizontal_km": float(math.hypot(xf[0], xf[1]))})
            grp_rows.append(gr)
        res.tables["comparison"] = cmp_rows
        res.tables["groups"] = grp_rows
        if not grp_rows:
            raise ValueError("no group has two or more stations observing at the same moment")
        g0 = max(grp_rows, key=lambda g: g["stations"])
        models.append(ModelFit("Globe geometry: ephemeris directions (0 free parameters)", 0, chi_pred, n_pred, chi2_pvalue(chi_pred, n_pred),
                               note="all measured directions vs prediction"))
        models.append(ModelFit("Globe triangulation consistency", 3, g0["globe_chi2"], g0["globe_dof"], chi2_pvalue(g0["globe_chi2"], g0["globe_dof"]),
                               {"distance_km": g0["distance_km"]}, "lines of sight meet at one point"))
        if "flat_chi2" in g0:
            models.append(ModelFit("Flat-map triangulation consistency", 3, g0["flat_chi2"], g0["flat_dof"],
                                   chi2_pvalue(g0["flat_chi2"], g0["flat_dof"]),
                                   {"moon_height_km": g0["flat_moon_height_km"]}, "parallel verticals"))
        res.models = models
        tdist = [{"hypothesis_distance_km": D, "measured_minus_hypothesis_in_sigma": (g0["distance_km"] - D) / max(g0["sigma_km"], 1e-9)} for D in tests]
        res.tables["distance_hypotheses"] = tdist
        res.summary = {k: g0[k] for k in ("stations", "distance_km", "sigma_km", "ci95_low_km", "ci95_high_km",
                                          "lower_bound_3sigma_km", "ephemeris_distance_km", "difference_in_sigma",
                                          "parallax_measured_deg", "parallax_expected_deg")}
        res.summary["mean_direction_error_deg"] = float(np.mean([c["direction_error_deg"] for c in cmp_rows]))
        res.verdict = self._verdict(g0, models, tdist)
        return self._finish(res)

    @staticmethod
    def _verdict(g, models, tdist) -> List[Dict[str, str]]:
        v: List[Dict[str, str]] = []
        rel = g["sigma_km"] / g["distance_km"]
        v.append(B(f"Moon distance from {g['stations']} stations: {g['distance_km']:,.0f} +/- {g['sigma_km']:,.0f} km (95 %: {g['ci95_low_km']:,.0f} to {g['ci95_high_km']:,.0f}); "
                   f"ephemeris: {g['ephemeris_distance_km']:,.0f} km ({g['difference_in_sigma']:+.1f} sigma). Measured parallax {g['parallax_measured_deg']:.2f} deg, expected {g['parallax_expected_deg']:.2f} deg.",
                   f"بُعد القمر من {g['stations']} محطات: {g['distance_km']:,.0f} ± {g['sigma_km']:,.0f} كم (95%: من {g['ci95_low_km']:,.0f} إلى {g['ci95_high_km']:,.0f})؛ "
                   f"الإفيمريس: {g['ephemeris_distance_km']:,.0f} كم ({g['difference_in_sigma']:+.1f} σ). اختلاف المنظر المقاس {g['parallax_measured_deg']:.2f}° والمتوقع {g['parallax_expected_deg']:.2f}°."))
        if rel > 0.3:
            v.append(B(f"The relative uncertainty is {rel * 100:.0f} %: the baseline is too short (or the angles too coarse) for a useful distance. Use places thousands of km apart.",
                       f"عدم اليقين النسبي {rel * 100:.0f}%: القاعدة قصيرة جدًا (أو الزوايا خشنة) لمسافة مفيدة. استخدم أماكن تبعد آلاف الكيلومترات."))
        else:
            v.append(B(f"With 99.7 % confidence the Moon is farther than {g['lower_bound_3sigma_km']:,.0f} km.",
                       f"بثقة 99.7% القمر أبعد من {g['lower_bound_3sigma_km']:,.0f} كم."))
            for t in tdist:
                if abs(t["measured_minus_hypothesis_in_sigma"]) > 5:
                    v.append(B(f"A Moon at {t['hypothesis_distance_km']:,.0f} km is {abs(t['measured_minus_hypothesis_in_sigma']):.0f} sigma away from this measurement.",
                               f"قمر على بُعد {t['hypothesis_distance_km']:,.0f} كم يبعد {abs(t['measured_minus_hypothesis_in_sigma']):.0f} σ عن هذا القياس."))
        gm, tri = models[0], models[1]
        v.append(B(f"Measured directions agree with the ephemeris (chi2 = {gm.chi2:.1f}, dof {gm.dof}, p = {gm.p_value:.2g}).",
                   f"الاتجاهات المقاسة تتفق مع الإفيمريس (χ² = {gm.chi2:.1f}، درجات الحرية {gm.dof}، p = {gm.p_value:.2g})." ) if gm.p_value >= 0.01 else
                 B(f"Measured directions DISAGREE with the ephemeris (chi2 = {gm.chi2:.1f}, dof {gm.dof}, p = {gm.p_value:.2g}): check time, azimuth reference and coordinates.",
                   f"الاتجاهات المقاسة لا تتفق مع الإفيمريس (χ² = {gm.chi2:.1f}، p = {gm.p_value:.2g}): تحقق من الزمن ومرجع السمت والإحداثيات."))
        if len(models) > 2:
            fl = models[2]
            if tri.dof > 0 and fl.dof > 0:
                if fl.p_value < 0.001 <= tri.p_value:
                    v.append(B(f"The lines of sight meet consistently in globe geometry (chi2 = {tri.chi2:.1f}, p = {tri.p_value:.2g}) but NOT in flat geometry with parallel verticals (chi2 = {fl.chi2:.0f}, p = {fl.p_value:.2g}).",
                               f"تتلاقى خطوط الرؤية بانتظام في هندسة الكرة (χ² = {tri.chi2:.1f}، p = {tri.p_value:.2g}) ولا تتلاقى في الهندسة المسطحة ذات الأعمدة المتوازية (χ² = {fl.chi2:.0f}، p = {fl.p_value:.2g})."))
                elif fl.p_value >= 0.01 and tri.p_value >= 0.01:
                    v.append(B("Both geometries are consistent with these sightings: add stations farther apart (east-west and north-south) to discriminate.",
                               "كلتا الهندسيتين متسقتان مع هذه الأرصاد: أضف محطات أبعد (شرق-غرب وشمال-جنوب) للتمييز."))
                elif tri.p_value < 0.001:
                    v.append(B(f"The lines of sight do not meet consistently even in globe geometry (p = {tri.p_value:.2g}): check the data first.",
                               f"خطوط الرؤية لا تتلاقى بانتظام حتى في هندسة الكرة (p = {tri.p_value:.2g}): تحقق من البيانات أولًا."))
        return v

    def plot(self, result: LabResult, path: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        g = result.tables.get("groups")
        if not g:
            return False
        g0 = max(g, key=lambda r: r["stations"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.axvspan(g0["ci95_low_km"], g0["ci95_high_km"], color="tab:blue", alpha=.25, label="measured, 95 %")
        ax.axvline(g0["distance_km"], color="tab:blue")
        ax.axvline(g0["ephemeris_distance_km"], color="g", ls="--", label="ephemeris")
        for t in result.tables.get("distance_hypotheses", []):
            ax.axvline(t["hypothesis_distance_km"], color="r", ls=":", label=f"hypothesis {t['hypothesis_distance_km']:,.0f} km")
        ax.set_xscale("log")
        ax.set_xlabel("Moon distance from Earth's centre (km)")
        ax.set_yticks([])
        ax.set_title("Lunar Parallax Lab" + (" (SYNTHETIC data)" if result.provenance.get("synthetic") else ""))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return True
