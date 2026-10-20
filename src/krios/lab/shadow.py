"""Shadow Lab - Sun altitude from a vertical stick, Earth geometry (Eratosthenes) and a fair model test.

Measurement : stick height + shadow length, at a known place and UTC time, from >= 2 stations
              (the more spread out, the better; 4+ stations make the flat model testable).
Prediction  : Sun altitude from the Krios ephemeris (no free parameters, uses only lat/lon/time).
Comparison  : (a) residuals of measured vs predicted altitude;
              (b) the globe prediction vs a *flat map with a local Sun* whose height (and optionally
                  position) is FITTED to the data - i.e. the flat model gets free parameters;
              (c) Eratosthenes: the Earth's radius from independently measured baselines.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from . import astro
from .core import B, Experiment, LabResult, ModelFit, register
from .stats import bic, chi2_pvalue, fit_least_squares, interval


def _cot(deg):
    return 1.0 / np.tan(np.radians(deg))


@register
class ShadowLab(Experiment):
    name = "shadow"
    rows_key = "stations"
    title = B("Shadow Lab: Sun height, Earth geometry (Eratosthenes) and a fair flat-Earth test",
              "مختبر الظلال: ارتفاع الشمس وشكل الأرض (إراتوستينس) واختبار عادل لنموذج الأرض المسطحة")
    blurb = B("Vertical stick + shadow at several places -> Sun altitude, Earth radius, globe vs flat-map fit.",
              "عصا رأسية وظلّها في عدة أماكن -> ارتفاع الشمس ونصف قطر الأرض ومقارنة الكرة بنموذج مسطّح مضبوط.")

    # ------------------------------------------------------------------ example
    @classmethod
    def example(cls, seed: int = 7) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        jd = astro.parse_utc("2026-10-12T11:10:00Z")
        lats, lon, R = [60.0, 46.0, 32.0, 18.0, 4.0], 10.0, 6371.0
        stations, baselines = [], []
        for i, la in enumerate(lats):
            h = float(astro.topocentric("sun", la, lon, 0.0, jd)["alt_app"][0]) + rng.normal(0, 0.07)
            stations.append({"name": f"S{i + 1}", "lat": la, "lon": lon, "time_utc": "2026-10-12T11:10:00Z",
                             "stick": 1.000, "shadow": round(1.0 / math.tan(math.radians(h)) + rng.normal(0, 0.003), 3),
                             "sigma_stick": 0.002, "sigma_shadow": 0.005})
            if i:
                d = R * math.radians(lats[i - 1] - la) + rng.normal(0, 3.0)
                baselines.append({"a": f"S{i}", "b": f"S{i + 1}", "km": round(d, 1), "sigma_km": 3.0,
                                  "source": "odometer / road atlas (SYNTHETIC)"})
        return {"metadata": {"title": "Shadow Lab demonstration", "synthetic": True, "seed": seed,
                             "note": "Synthetic stations along the 10E meridian, one simultaneous reading."},
                "settings": {"sigma_floor_deg": 0.1, "refraction": True, "seed": 12345, "mc_samples": 3000},
                "stations": stations, "baselines": baselines}

    # ---------------------------------------------------------------------- run
    def run(self, predict_only: bool = False) -> LabResult:
        spec, st = self.spec, self.spec.get("settings", {})
        floor = float(st.get("sigma_floor_deg", 0.1))
        rng = np.random.default_rng(int(st.get("seed", 12345)))
        mc_n = int(st.get("mc_samples", 3000))
        rows = spec["stations"]
        n = len(rows)
        if n < 1:
            raise ValueError("the shadow lab needs at least one station")
        names = [str(r["name"]) for r in rows]
        lat = np.array([float(r["lat"]) for r in rows])
        lon = np.array([float(r["lon"]) for r in rows])
        height = np.array([float(r.get("height_m", 0.0)) for r in rows])
        jd = np.array([astro.parse_utc(r["time_utc"]) for r in rows])
        stick = np.array([float(r.get("stick", 1.0)) for r in rows])

        # ---- independent predictions (no measurement used)
        pred_alt, pred_geom, pred_az, noon_jd, noon_alt = [], [], [], [], []
        for i in range(n):
            t = astro.topocentric("sun", lat[i], lon[i], height[i], jd[i])
            pred_alt.append(float(t["alt_app"][0]))
            pred_geom.append(float(t["alt_geom"][0]))
            pred_az.append(float(t["az"][0]))
            tn, an_ = astro.sun_transit(lat[i], lon[i], jd[i] - (astro.sun_declination_hour_angle(lon[i], jd[i])["H"][0] / (2 * math.pi)))
            noon_jd.append(tn)
            noon_alt.append(an_)
        pred_alt = np.array(pred_alt)
        sub = astro.subsolar_point(jd)
        predictions = []
        for i in range(n):
            up = pred_alt[i] > 0
            predictions.append({"station": names[i], "time_utc": rows[i]["time_utc"], "lat": lat[i], "lon": lon[i],
                                "sun_alt_deg": pred_alt[i], "sun_az_deg": pred_az[i],
                                "shadow_per_stick": (1.0 / math.tan(math.radians(pred_alt[i]))) if up else None,
                                "shadow_for_stick": (stick[i] / math.tan(math.radians(pred_alt[i]))) if up else None,
                                "local_solar_noon_utc": astro.jd_to_iso(noon_jd[i]), "noon_alt_deg": noon_alt[i],
                                "subsolar_lat": float(sub["lat"][i]), "subsolar_lon": float(sub["lon"][i])})
        res = LabResult(self.name, self.title)
        res.tables["predictions"] = predictions
        res.method = [
            B("Sun altitude h = atan(stick / shadow); uncertainty from both lengths plus a floor for the blurred shadow edge "
              "(the Sun's disc is 0.53 deg wide).", "ارتفاع الشمس h = atan(العصا / الظل)، وعدم اليقين من الطولين مع حدّ أدنى لضبابية حافة الظل (قرص الشمس 0.53°)."),
            B("Prediction: topocentric Sun from the Krios analytic ephemeris (~1\"), oblate Earth, apparent altitude with standard refraction.",
              "التنبؤ: الشمس من إفيمريس Krios (~1″) على أرض مفلطحة مع الانكسار القياسي."),
            B("Flat-map model (the strongest version): azimuthal-equidistant map centred on the north pole, Sun at height H above "
              "the sub-solar point (F1: H free) or above a free point of the map (F2: H, dx, dy free); tan h = H / rho.",
              "نموذج الخريطة المسطحة (بأقوى صوره): خريطة متساوية المسافة مركزها القطب الشمالي والشمس على ارتفاع H فوق نقطة تحت الشمس (F1: H حرّ) أو فوق نقطة حرّة (F2: H وdx وdy حرّة)؛ tan h = H / rho."),
            B("Eratosthenes: latitude of each station from its Sun altitude, the Sun's declination and the hour angle; angle between "
              "stations / measured ground distance = radius R.", "إراتوستينس: خط عرض كل محطة من ارتفاع الشمس وميلها وزاويتها الساعية؛ والزاوية بين المحطتين ÷ المسافة الأرضية المقاسة = نصف القطر R.")]
        res.assumptions = [
            B("Clock error below ~10 s and longitudes accurate to ~0.01 deg (they set the hour angle).", "خطأ الساعة أقل من ~10 ثوانٍ والأطوال الجغرافية بدقة ~0.01°."),
            B("Level ground, stick truly vertical (plumb line); shadow tip read at the middle of the penumbra.", "أرض مستوية وعصا رأسية فعلًا (فتيل شاقول)، وقراءة طرف الظل عند منتصف شبه الظل."),
            B("Baselines must be measured independently (odometer, map, track length). If they are computed from the same "
              "coordinates the radius test is circular, and Krios will say so.", "يجب قياس المسافات الأرضية باستقلال (عدّاد، خريطة، مسار). وإن حُسبت من الإحداثيات نفسها فالاختبار دائري وسيذكر Krios ذلك."),
            B("The flat model is given free parameters on purpose, so it is not a straw man; a model that fits only by tuning "
              "is reported as such.", "منحتُ النموذج المسطح معاملات حرّة عمدًا حتى لا يكون «رجل قش»؛ والنموذج الذي ينجح بالضبط وحده يُذكر كذلك.")]
        if predict_only:
            res.summary = {"stations": n, "mode": "prediction only (pre-registration)"}
            res.verdict = [B("Predictions generated before measuring. Save this plan and measure at the stated times.",
                             "تم توليد التنبؤات قبل القياس. احفظ الخطة وقِس في الأوقات المذكورة.")]
            return self._finish(res)

        # ---- measurements
        if any("shadow" not in r or r["shadow"] is None for r in rows):
            raise ValueError("every station needs a 'shadow' length (or run in plan mode)")
        shadow = np.array([float(r["shadow"]) for r in rows])
        s_stick = np.array([float(r.get("sigma_stick", 0.002 * stick[i])) for i, r in enumerate(rows)])
        s_shadow = np.array([float(r.get("sigma_shadow", 0.01 * stick[i])) for i, r in enumerate(rows)])
        h_meas = np.degrees(np.arctan2(stick, shadow))
        s_stat = np.degrees(np.sqrt((shadow * s_stick) ** 2 + (stick * s_shadow) ** 2) / (stick ** 2 + shadow ** 2))
        sig = np.sqrt(s_stat ** 2 + floor ** 2)
        resid = h_meas - pred_alt
        z = resid / sig
        chi_g = float(np.sum(z ** 2))
        models = [ModelFit("Globe (Krios prediction)", 0, chi_g, n, chi2_pvalue(chi_g, n),
                           note="no free parameters")]

        # ---- flat-map model fits
        x, y = astro.fe_xy(lat, lon)
        xs, ys = astro.fe_xy(sub["lat"], sub["lon"])
        rho0 = np.hypot(x - xs, y - ys)
        f1_pred = f2_pred = None
        f1_fit = f2_fit = None
        if n >= 2:
            r1 = lambda p: (h_meas - np.degrees(np.arctan2(10 ** p[0], rho0))) / sig
            p1, c1, _ = fit_least_squares(r1, [[v] for v in (2.5, 3, 3.5, 4, 4.5, 5, 5.5)], ([1.0], [6.5]))
            f1_fit = (p1, c1)
            f1_pred = np.degrees(np.arctan2(10 ** p1[0], rho0))
            dof1 = n - 1
            models.append(ModelFit("Flat map, local Sun, H fitted (F1)", 1, c1, dof1, chi2_pvalue(c1, dof1),
                                   {"H_km": 10 ** float(p1[0])}, "Sun above the sub-solar point"))
            r2 = lambda p: (h_meas - np.degrees(np.arctan2(10 ** p[0], np.hypot(x - xs - p[1], y - ys - p[2])))) / sig
            starts = [[lh, dx, dy] for lh in (3.5, 4.0, 4.5) for dx in (-10000, 0, 10000) for dy in (-10000, 0, 10000)]
            p2, c2, _ = fit_least_squares(r2, starts, ([1.0, -30000, -30000], [6.5, 30000, 30000]))
            f2_fit = p2
            f2_pred = np.degrees(np.arctan2(10 ** p2[0], np.hypot(x - xs - p2[1], y - ys - p2[2])))
            dof2 = n - 3
            models.append(ModelFit("Flat map, local Sun, H + position fitted (F2)", 3, c2, dof2, chi2_pvalue(c2, dof2),
                                   {"H_km": 10 ** float(p2[0]), "dx_km": float(p2[1]), "dy_km": float(p2[2])},
                                   "most generous flat variant" if dof2 > 0 else "no degrees of freedom: untestable"))
        comp = []
        for i in range(n):
            comp.append({"station": names[i], "lat": lat[i], "lon": lon[i], "h_measured_deg": h_meas[i],
                         "sigma_deg": sig[i], "h_globe_deg": pred_alt[i], "residual_deg": resid[i], "z": z[i],
                         "h_flatF1_deg": None if f1_pred is None else f1_pred[i],
                         "h_flatF2_deg": None if f2_pred is None else f2_pred[i]})
        res.tables["comparison"] = comp

        # ---- Eratosthenes: radius from independent baselines (adjacent stations by latitude)
        pair_rows, R_list, H_list = [], [], []
        base = {frozenset((b["a"], b["b"])): b for b in spec.get("baselines", [])}
        order = np.argsort(lat)
        alt_true = astro.true_alt(h_meas)
        for a_, b_ in zip(order[:-1], order[1:]):
            bl = base.get(frozenset((names[a_], names[b_])))
            if bl is None:
                continue
            d, sd = float(bl["km"]), float(bl.get("sigma_km", 0.005 * float(bl["km"])))
            circular = str(bl.get("source", "")).lower().startswith("computed")
            sa = astro.sun_declination_hour_angle(lon[a_], jd[a_])
            sb = astro.sun_declination_hour_angle(lon[b_], jd[b_])
            da = rng.normal(alt_true[a_], sig[a_], mc_n)
            db = rng.normal(alt_true[b_], sig[b_], mc_n)
            pa = astro.latitude_from_sun_altitude(da, sa["dec"][0], sa["H"][0], lat[a_])
            pb = astro.latitude_from_sun_altitude(db, sb["dec"][0], sb["H"][0], lat[b_])
            ang = np.radians(astro.spherical_distance_deg(pa, lon[a_], pb, lon[b_]))
            dd = rng.normal(d, sd, mc_n)
            with np.errstate(all="ignore"):
                Rs = dd / ang
            lo, med, hi = interval(Rs[ang > 0])
            sR = 0.5 * (hi - lo)
            ha, hb = rng.normal(h_meas[a_], sig[a_], mc_n), rng.normal(h_meas[b_], sig[b_], mc_n)
            h_hi, h_lo = np.maximum(ha, hb), np.minimum(ha, hb)
            with np.errstate(all="ignore"):
                Hs = dd / (_cot(h_lo) - _cot(h_hi))
            Hs = Hs[(Hs > 0) & np.isfinite(Hs)]
            hl, hm, hh = interval(Hs) if len(Hs) > 50 else (float("nan"),) * 3
            row = {"pair": f"{names[a_]}-{names[b_]}", "baseline_km": d, "baseline_sigma_km": sd,
                   "angle_deg_from_sun": float(np.degrees(np.nanmedian(ang))),
                   "R_globe_km": med, "R_sigma_km": sR,
                   "H_flat_km": hm, "H_sigma_km": 0.5 * (hh - hl) if not math.isnan(hh) else None,
                   "baseline_independent": not circular}
            pair_rows.append(row)
            if sR > 0 and math.isfinite(med) and not circular and np.degrees(np.nanmedian(ang)) > 3 * np.degrees(np.nanstd(ang)):
                R_list.append((med, sR))
            if len(Hs) > 50 and math.isfinite(hm) and hm > 0:
                H_list.append((hm, 0.5 * (hh - hl)))
        if pair_rows:
            res.tables["pairs"] = pair_rows

        # ---- summary and verdict
        s: Dict[str, Any] = {"stations": n, "chi2_globe": chi_g, "p_globe": models[0].p_value,
                             "rms_residual_deg": float(np.sqrt(np.mean(resid ** 2))),
                             "max_abs_residual_deg": float(np.max(np.abs(resid)))}
        R_txt = None
        if R_list:
            w = np.array([1 / s_ ** 2 for _, s_ in R_list])
            Rv = np.array([r for r, _ in R_list])
            Rm = float(np.sum(w * Rv) / np.sum(w))
            sRm = float(1 / math.sqrt(np.sum(w)))
            chiR = float(np.sum(w * (Rv - Rm) ** 2))
            s.update({"R_estimate_km": Rm, "R_sigma_km": sRm, "circumference_km": 2 * math.pi * Rm,
                      "R_pairs_chi2": chiR, "R_pairs_dof": len(R_list) - 1,
                      "R_pairs_p": chi2_pvalue(chiR, len(R_list) - 1)})
            R_txt = (Rm, sRm)
        if len(H_list) >= 2:
            Hv = np.array([h for h, _ in H_list])
            s["flat_implied_Sun_height_min_km"] = float(Hv.min())
            s["flat_implied_Sun_height_max_km"] = float(Hv.max())
            s["flat_implied_Sun_height_ratio"] = float(Hv.max() / Hv.min())
        res.summary = s
        res.models = models
        res.verdict = self._verdict(n, models, R_txt, s)
        return self._finish(res)

    # ------------------------------------------------------------------ verdict
    @staticmethod
    def _verdict(n, models, R_txt, s) -> List[Dict[str, str]]:
        v: List[Dict[str, str]] = []
        pg = models[0].p_value
        flat = [m for m in models[1:] if m.dof > 0]
        flat_ok = [m for m in flat if m.p_value >= 0.01]
        flat_bad = [m for m in flat if m.p_value < 0.001]
        if n < 3:
            v.append(B("With fewer than 3 stations the flat model can match any pair of readings by tuning the Sun's height, so these data "
                       "cannot discriminate between shapes. Add stations far apart (>= 1000 km).",
                       "مع أقل من 3 محطات يستطيع النموذج المسطح مطابقة أي زوج من القراءات بضبط ارتفاع الشمس، فلا تميّز هذه البيانات بين الشكلين. أضف محطات متباعدة (≥ 1000 كم)."))
        if pg >= 0.01:
            v.append(B(f"The globe prediction, with NO free parameters, reproduces all {n} measured Sun altitudes "
                       f"(chi2 = {models[0].chi2:.1f}, p = {pg:.2g}; rms residual {s['rms_residual_deg']:.2f} deg).",
                       f"تنبؤ الكرة الأرضية، بلا أي معامل حرّ، يعيد إنتاج ارتفاعات الشمس المقاسة في {n} محطات (χ² = {models[0].chi2:.1f}، p = {pg:.2g}، متوسط الخطأ {s['rms_residual_deg']:.2f}°)."))
        elif pg < 0.001:
            v.append(B(f"The globe prediction does NOT fit (chi2 = {models[0].chi2:.1f}, p = {pg:.2g}). Check clock, units, latitude/longitude and that the stick was vertical before drawing any conclusion.",
                       f"تنبؤ الكرة لا يطابق القياسات (χ² = {models[0].chi2:.1f}، p = {pg:.2g}). تحقق من الساعة والوحدات وخطوط العرض/الطول ورأسية العصا قبل أي استنتاج."))
        else:
            v.append(B(f"The globe prediction is marginal (p = {pg:.3f}); more data are needed.", f"تنبؤ الكرة حدّي (p = {pg:.3f})؛ نحتاج بيانات أكثر."))
        if flat:
            best = min(flat, key=lambda m: m.chi2)
            if flat_ok and pg >= 0.01:
                v.append(B(f"The fitted flat-map model also fits (best: {best.name}, chi2 = {best.chi2:.1f}, p = {best.p_value:.2g}): these data are NOT discriminating. Use stations farther apart or at different times.",
                           f"النموذج المسطح بعد ضبط معاملاته يطابق القياسات أيضًا ({best.name}، χ² = {best.chi2:.1f}، p = {best.p_value:.2g}): هذه البيانات لا تميّز بين الشكلين. استخدم محطات أبعد أو أوقاتًا مختلفة."))
            elif len(flat_bad) == len(flat):
                v.append(B(f"Every flat-map variant is rejected even with its free parameters (best: {best.name}, chi2 = {best.chi2:.1f}, p = {best.p_value:.2g}).",
                           f"كل صور النموذج المسطح مرفوضة حتى مع معاملاتها الحرّة (الأفضل: {best.name}، χ² = {best.chi2:.1f}، p = {best.p_value:.2g})."))
            elif flat_ok and pg < 0.001:
                v.append(B(f"A fitted flat-map variant fits better than the globe prediction ({best.name}). Re-check the data before concluding anything.",
                           f"إحدى صور النموذج المسطح تطابق القياسات أفضل من تنبؤ الكرة ({best.name}). أعد فحص البيانات قبل أي استنتاج."))
        if pg > 0.99:
            v.append(B("Note: chi2 is far below its degrees of freedom, so the stated uncertainties are probably generous. That is safe, but it also weakens the test.",
                       "ملاحظة: قيمة χ² أقل بكثير من درجات الحرية، أي أن أخطاء القياس المعلنة سخيّة على الأرجح؛ هذا آمن لكنه يُضعف الاختبار."))
        if len(models) > 1 and all(m.dof <= 0 for m in models[1:]):
            v.append(B("The most general flat variant has no degrees of freedom with this few stations (untestable).", "أعمّ صور النموذج المسطح بلا درجات حرية بهذا العدد القليل من المحطات (غير قابل للاختبار)."))
        if R_txt:
            Rm, sR = R_txt
            v.append(B(f"Eratosthenes: R = {Rm:.0f} +/- {sR:.0f} km (circumference {2 * math.pi * Rm:,.0f} km) from independent baselines; reference mean radius 6371 km.",
                       f"إراتوستينس: R = {Rm:.0f} ± {sR:.0f} كم (المحيط {2 * math.pi * Rm:,.0f} كم) من مسافات مقاسة باستقلال؛ المرجع: متوسط نصف القطر 6371 كم."))
        if "flat_implied_Sun_height_ratio" in s and s["flat_implied_Sun_height_ratio"] > 1.5:
            v.append(B(f"The Sun height implied by a flat Earth changes from pair to pair by a factor of {s['flat_implied_Sun_height_ratio']:.1f} "
                       f"({s['flat_implied_Sun_height_min_km']:.0f} to {s['flat_implied_Sun_height_max_km']:.0f} km); one flat model needs one height.",
                       f"ارتفاع الشمس الذي تفرضه الأرض المسطحة يتغير من زوج لآخر بعامل {s['flat_implied_Sun_height_ratio']:.1f} "
                       f"(من {s['flat_implied_Sun_height_min_km']:.0f} إلى {s['flat_implied_Sun_height_max_km']:.0f} كم)؛ والنموذج المسطح الواحد يحتاج ارتفاعًا واحدًا."))
        return v

    # --------------------------------------------------------------------- plot
    def plot(self, result: LabResult, path: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        rows = result.tables.get("comparison")
        if not rows:
            return False
        lat = [r["lat"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(lat, [r["h_measured_deg"] for r in rows], yerr=[r["sigma_deg"] for r in rows], fmt="ko", label="measured")
        ax.plot(lat, [r["h_globe_deg"] for r in rows], "g-s", label="globe prediction (0 free parameters)")
        if rows[0].get("h_flatF1_deg") is not None:
            ax.plot(lat, [r["h_flatF1_deg"] for r in rows], "r--^", label="flat map, H fitted (F1)")
            ax.plot(lat, [r["h_flatF2_deg"] for r in rows], "m:v", label="flat map, H + position fitted (F2)")
        ax.set_xlabel("station latitude (deg)")
        ax.set_ylabel("Sun altitude (deg)")
        ax.set_title("Shadow Lab" + (" (SYNTHETIC data)" if result.provenance.get("synthetic") else ""))
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return True
