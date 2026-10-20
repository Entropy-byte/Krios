"""Eclipse Lab - observed contact times vs the Krios prediction.

Measurement : UTC times of the contacts you timed (solar: C1, C2, max, C3, C4 at your place; lunar: P1, U1, U2,
              Greatest, U3, U4, P4, valid anywhere the Moon is up).
Prediction  : Krios eclipse engine with the chosen ephemeris (analytic offline, or JPL Horizons).
Comparison  : residual in seconds with observer and model uncertainty, per contact.
Pre-registration is the point: generate the plan weeks before the eclipse (`krios lab plan`), keep its hash,
then compare after the event.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

from .. import timescales as ts
from ..eclipse import EclipseConfig, find_eclipses, solar_local
from ..ephemeris import AnalyticEphemeris
from . import astro
from .core import B, Experiment, LabResult, ModelFit, register
from .stats import chi2_pvalue

_SOLAR_ORDER = ["C1", "C2", "max", "C3", "C4"]
_LUNAR_ORDER = ["P1", "U1", "U2", "Greatest", "U3", "U4", "P4"]


@register
class EclipseLab(Experiment):
    name = "eclipse"
    rows_key = "observations"
    title = B("Eclipse Lab: contact times, observed vs predicted",
              "مختبر الكسوف: أوقات التماس، المرصودة مقابل المتوقعة")
    blurb = B("Time the contacts of a solar or lunar eclipse and compare them with a pre-registered prediction.",
              "سجّل أوقات تماسات كسوف أو خسوف وقارنها بتنبؤ مسجّل مسبقًا.")

    @classmethod
    def example(cls, seed: int = 21) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        cfg, ephem = EclipseConfig(), AnalyticEphemeris()
        jd = ts.calendar_to_jd(2027, 8, 2)
        ecl = find_eclipses(ts.utc_to_tt(jd), ts.utc_to_tt(jd) + 1.0, ("solar",), ephem, cfg)[0]
        obs_site = {"name": "Observer (demo)", "lat": 25.69, "lon": 32.64, "height_m": 80.0}
        loc = solar_local(ephem, cfg, obs_site["lat"], obs_site["lon"], obs_site["height_m"], ecl.jd_greatest_tt)
        obs = []
        for k in _SOLAR_ORDER:
            if k in loc.contacts_tt:
                t = ts.tt_to_utc(loc.contacts_tt[k]) + rng.normal(0, 3.0) / 86400.0
                obs.append({"event": k, "time_utc": astro.jd_to_iso(t, 1), "sigma_s": 3.0})
        return {"metadata": {"title": "Eclipse Lab demonstration", "synthetic": True, "seed": seed,
                             "note": "Synthetic contact times (3 s scatter) around the Krios prediction - NOT a real observation."},
                "settings": {"model_sigma_s": 20.0}, "eclipse": {"date": "2027-08-02", "body": "solar"},
                "observer": obs_site, "observations": obs}

    def run(self, predict_only: bool = False) -> LabResult:
        spec, st = self.spec, self.spec.get("settings", {})
        ephem = self.ephemeris or AnalyticEphemeris()
        model_sigma = float(st.get("model_sigma_s", 20.0 if isinstance(ephem, AnalyticEphemeris) else 3.0))
        cfg = EclipseConfig(shadow_model=st.get("shadow", "herald"), light_time=bool(st.get("light_time", True)))
        body = spec["eclipse"].get("body", "solar")
        jd = astro.parse_utc(spec["eclipse"]["date"])
        found = find_eclipses(ts.utc_to_tt(jd) - 0.5, ts.utc_to_tt(jd) + 1.5, (body,), ephem, cfg)
        if not found:
            raise ValueError(f"no {body} eclipse found around {spec['eclipse']['date']}")
        ecl = found[0]
        res = LabResult(self.name, self.title)
        preds: List[Dict[str, Any]] = []
        if body == "solar":
            ob = spec["observer"]
            loc = solar_local(ephem, cfg, float(ob["lat"]), float(ob["lon"]), float(ob.get("height_m", 0.0)), ecl.jd_greatest_tt)
            if not loc.eclipse_occurs:
                raise ValueError("no eclipse is visible from this observer: " + (loc.note or "the penumbra does not reach it"))
            for k in _SOLAR_ORDER:
                if k in loc.contacts_tt:
                    preds.append({"event": k, "time_utc": astro.jd_to_iso(ts.tt_to_utc(loc.contacts_tt[k]), 1),
                                  "time_tt": ts.format_jd(loc.contacts_tt[k], 1), "sun_alt_deg": loc.altitude_deg[k],
                                  "sun_az_deg": loc.azimuth_deg[k], "magnitude_at_max": loc.magnitude if k == "max" else None,
                                  "duration_s": loc.duration_s if k == "C2" else None})
            res.summary = {"eclipse": f"{ecl.kind} solar eclipse of {spec['eclipse']['date']}", "type_here": loc.kind,
                           "magnitude_here": loc.magnitude, "obscuration_here": loc.obscuration, "duration_here_s": loc.duration_s,
                           "global_greatest_utc": astro.jd_to_iso(ts.tt_to_utc(ecl.jd_greatest_tt), 1)}
        else:
            for k in _LUNAR_ORDER:
                if k in ecl.contacts_tt:
                    preds.append({"event": k, "time_utc": astro.jd_to_iso(ts.tt_to_utc(ecl.contacts_tt[k]), 1),
                                  "time_tt": ts.format_jd(ecl.contacts_tt[k], 1)})
            res.summary = {"eclipse": f"{ecl.kind} lunar eclipse of {spec['eclipse']['date']}",
                           "umbral_magnitude": ecl.umbral_magnitude, "penumbral_magnitude": ecl.penumbral_magnitude}
        res.tables["predictions"] = preds
        res.method = [
            B("Contact times from the Krios eclipse engine (Besselian geometry for solar, Herald-Sinnott shadow for lunar), including the light-time shadow shift; "
              "TT converted to UTC with the leap seconds known to the end of 2026.",
              "أوقات التماس من محرك Krios (هندسة بسّل للكسوف وظل Herald-Sinnott للخسوف) مع إزاحة الظل لزمن الضوء؛ ويُحوَّل TT إلى UTC بالثواني الكبيسة المعروفة حتى نهاية 2026."),
            B(f"Residual = observed - predicted; combined sigma = sqrt(observer^2 + model^2) with model = {model_sigma:.0f} s for {ephem.describe()}.",
              f"الفرق = المرصود − المتوقع؛ σ المركّب = √(المراقب² + النموذج²) بنموذج {model_sigma:.0f} ث لـ {ephem.describe()}.")]
        res.assumptions = [
            B("Time your contacts with a clock synchronised to UTC (GPS / NTP) and note the reading delay; second contact and third contact are sharper than first and fourth.",
              "اضبط ساعتك على UTC (GPS/NTP) وسجّل زمن رد الفعل؛ التماس الثاني والثالث أدقّ من الأول والرابع."),
            B("Local circumstances ignore the lunar limb profile (a few seconds) and terrain; ephemeris accuracy is stated in the provenance.",
              "الظروف المحلية تتجاهل شكل حافة القمر (بضع ثوانٍ) والتضاريس؛ ودقة الإفيمريس مذكورة في المصدر.")]
        if predict_only or not spec.get("observations"):
            res.verdict = [B("Predictions generated before the event. Save the plan file and its hash.", "تم توليد التنبؤات قبل الحدث. احفظ ملف الخطة وبصمته.")]
            return self._finish(res, ephem.describe())
        pmap = {p["event"]: p for p in preds}
        comp, zs = [], []
        for o in spec["observations"]:
            if o["event"] not in pmap:
                continue
            pj = astro.parse_utc(pmap[o["event"]]["time_utc"])
            r = (astro.parse_utc(o["time_utc"]) - pj) * 86400.0
            sg = math.sqrt(float(o.get("sigma_s", 5.0)) ** 2 + model_sigma ** 2)
            comp.append({"event": o["event"], "observed_utc": o["time_utc"], "predicted_utc": pmap[o["event"]]["time_utc"],
                         "residual_s": r, "sigma_s": sg, "z": r / sg})
            zs.append(r / sg)
        res.tables["comparison"] = comp
        z = np.array(zs)
        chi = float(np.sum(z ** 2))
        res.models = [ModelFit("Krios prediction (0 free parameters)", 0, chi, len(z), chi2_pvalue(chi, len(z)))]
        res.summary.update({"contacts_compared": len(z), "mean_residual_s": float(np.mean([c["residual_s"] for c in comp])) if comp else None})
        if len(z):
            p = res.models[0].p_value
            res.verdict = [B(f"{len(z)} contact(s) vs prediction: mean offset {np.mean([c['residual_s'] for c in comp]):+.1f} s, chi2 = {chi:.1f} (dof {len(z)}), p = {p:.2g}. "
                             + ("Consistent with the prediction." if p >= 0.01 else "Inconsistent: check your clock, location and event identification."),
                             f"{len(z)} تماسًا مقابل التنبؤ: متوسط الفرق {np.mean([c['residual_s'] for c in comp]):+.1f} ث، χ² = {chi:.1f}، p = {p:.2g}. "
                             + ("متسق مع التنبؤ." if p >= 0.01 else "غير متسق: تحقق من ساعتك وموقعك وتحديد الحدث."))]
        else:
            res.verdict = [B("No observed contacts matched the predicted events.", "لم تتطابق أي تماسات مرصودة مع الأحداث المتوقعة.")]
        return self._finish(res, ephem.describe())
