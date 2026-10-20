"""Solar Path Lab - sunrise, sunset, noon altitude and day length through the year.

Measurement : sunrise/sunset times, noon (or any) Sun altitudes, observed on chosen dates at one or more places.
Prediction  : Krios Sun (~1") + rotating oblate Earth + standard refraction; the Sun's upper limb touches the
              horizon at geometric centre altitude -0.833 deg.
Comparison  : residuals in seconds / degrees; and the seasonal pattern at places in both hemispheres
              (day length peaks on opposite solstices, polar day / night at high latitudes).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import astro
from .core import B, Experiment, LabResult, ModelFit, register
from .stats import chi2_pvalue

LEVEL_UPPER_LIMB = -0.8333       # refraction 34' + semi-diameter 16'


def _transit(lat, lon, h, guess):
    t = np.linspace(guess - 0.5, guess + 0.5, 289)
    alt = astro.topocentric("sun", lat, lon, h, t)["alt_geom"]
    i = int(np.argmax(alt))
    tt = np.linspace(t[i] - 0.007, t[i] + 0.007, 61)
    a2 = astro.topocentric("sun", lat, lon, h, tt)["alt_geom"]
    j = int(np.argmax(a2))
    lo, hi = max(j - 3, 0), min(j + 4, len(tt))
    c = np.polyfit(tt[lo:hi] - tt[j], a2[lo:hi], 2)
    dt = -c[1] / (2 * c[0]) if c[0] < 0 else 0.0
    return float(tt[j] + dt), float(np.polyval(c, dt)), t, alt


def day_events(lat: float, lon: float, h: float, jd0: float, level: float = LEVEL_UPPER_LIMB) -> Dict[str, Any]:
    """Events for the local solar day whose UTC calendar date starts at ``jd0`` (UTC midnight)."""
    guess = jd0 + 0.5 - lon / 360.0
    t_tr, a_tr, t, alt = _transit(lat, lon, h, guess)
    out: Dict[str, Any] = {"transit_jd": t_tr, "noon_alt_deg": a_tr, "sunrise_jd": None, "sunset_jd": None,
                           "sunrise_az": None, "sunset_az": None}
    s = alt - level
    out["polar"] = None
    if a_tr < level:
        out["polar"] = "night"
        return out
    anti = astro.topocentric("sun", lat, lon, h, t_tr + 0.5)["alt_geom"][0]
    if anti > level:
        out["polar"] = "day"
        return out

    def cross(k0, k1):
        tt = np.linspace(t[k0], t[k1], 41)
        aa = astro.topocentric("sun", lat, lon, h, tt)["alt_geom"] - level
        for q in range(len(tt) - 1):
            if aa[q] * aa[q + 1] <= 0 and aa[q] != aa[q + 1]:
                return float(tt[q] - aa[q] * (tt[q + 1] - tt[q]) / (aa[q + 1] - aa[q]))
        return None

    ups = [k for k in range(len(t) - 1) if s[k] < 0 <= s[k + 1] and t[k] < t_tr]
    downs = [k for k in range(len(t) - 1) if s[k] >= 0 > s[k + 1] and t[k] > t_tr]
    if ups:
        out["sunrise_jd"] = cross(ups[-1], ups[-1] + 1)
    if downs:
        out["sunset_jd"] = cross(downs[0], downs[0] + 1)
    for key, jd in (("sunrise_az", out["sunrise_jd"]), ("sunset_az", out["sunset_jd"])):
        if jd is not None:
            out[key] = float(astro.topocentric("sun", lat, lon, h, jd)["az"][0])
    return out


@register
class SolarPathLab(Experiment):
    name = "solarpath"
    rows_key = "observations"
    title = B("Solar Path Lab: sunrise, sunset and day length through the year",
              "مختبر مسار الشمس: الشروق والغروب وطول النهار على مدار السنة")
    blurb = B("Predicted Sun path for places in both hemispheres vs your observed sunrise/sunset/noon.",
              "مسار الشمس المتوقع لأماكن في نصفي الكرة مقابل شروقك وغروبك وظهيرتك المرصودة.")

    # ------------------------------------------------------------------ example
    @classmethod
    def example(cls, seed: int = 3) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        sites = [{"name": "Greenwich-like (north)", "lat": 51.48, "lon": 0.0, "height_m": 0.0},
                 {"name": "Sydney-like (south)", "lat": -33.87, "lon": 151.21, "height_m": 0.0}]
        obs = []
        for s in sites:
            for date in ("2027-03-20", "2027-06-21", "2027-09-23", "2027-12-21"):
                ev = day_events(s["lat"], s["lon"], 0.0, astro.parse_utc(date))
                for kind, key in (("sunrise", "sunrise_jd"), ("sunset", "sunset_jd")):
                    obs.append({"site": s["name"], "kind": kind, "date": date,
                                "observed_utc": astro.jd_to_iso(ev[key] + rng.normal(0, 45.0) / 86400.0, 0), "sigma_s": 60.0})
        return {"metadata": {"title": "Solar Path Lab demonstration", "synthetic": True, "seed": seed,
                             "note": "Synthetic sunrise/sunset readings (45 s scatter) at two places in opposite hemispheres."},
                "settings": {"step_days": 10, "model_sigma_s": 90.0},
                "sites": sites, "year": 2027, "observations": obs}

    # ---------------------------------------------------------------------- run
    def run(self, predict_only: bool = False) -> LabResult:
        st = self.spec.get("settings", {})
        sites = {s["name"]: s for s in self.spec["sites"]}
        step = int(st.get("step_days", 5))
        model_sigma = float(st.get("model_sigma_s", 90.0))
        dates: List[str] = list(self.spec.get("dates", []))
        if "year" in self.spec:
            y = int(self.spec["year"])
            jd = astro.parse_utc(f"{y}-01-01")
            n_days = int(astro.parse_utc(f"{y + 1}-01-01") - jd)
            dates += [astro.jd_to_iso(jd + k, 0)[:10] for k in range(0, n_days, step)]
        for o in self.spec.get("observations", []):
            if o.get("date") and o["date"] not in dates:
                dates.append(o["date"])
        dates = sorted(set(dates))
        res = LabResult(self.name, self.title)
        daily: List[Dict[str, Any]] = []
        cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for name, s in sites.items():
            for d in dates:
                ev = day_events(float(s["lat"]), float(s["lon"]), float(s.get("height_m", 0.0)), astro.parse_utc(d))
                cache[(name, d)] = ev
                day_len = None
                if ev["polar"] == "day":
                    day_len = 24.0
                elif ev["polar"] == "night":
                    day_len = 0.0
                elif ev["sunrise_jd"] and ev["sunset_jd"]:
                    day_len = (ev["sunset_jd"] - ev["sunrise_jd"]) * 24.0
                daily.append({"site": name, "date": d,
                              "sunrise_utc": None if ev["sunrise_jd"] is None else astro.jd_to_iso(ev["sunrise_jd"], 0),
                              "sunset_utc": None if ev["sunset_jd"] is None else astro.jd_to_iso(ev["sunset_jd"], 0),
                              "transit_utc": astro.jd_to_iso(ev["transit_jd"], 0), "noon_alt_deg": ev["noon_alt_deg"],
                              "sunrise_az_deg": ev["sunrise_az"], "sunset_az_deg": ev["sunset_az"],
                              "day_length_h": day_len, "polar": ev["polar"]})
        res.tables["predictions"] = daily
        res.method = [
            B("Sun altitude from the Krios ephemeris for the rotating oblate Earth; sunrise/sunset when the geometric altitude of the Sun's centre crosses -0.833 deg "
              "(34' refraction + 16' semi-diameter); local solar transit by a parabola through the altitude maximum.",
              "ارتفاع الشمس من إفيمريس Krios لأرض مفلطحة دوّارة؛ الشروق/الغروب عند عبور الارتفاع الهندسي لمركز الشمس القيمة −0.833° (انكسار 34′ + نصف قطر 16′)؛ والعبور بقطع مكافئ عبر قمة الارتفاع."),
            B("Residual = observed - predicted; combined uncertainty = sqrt(observer^2 + model^2), where the model term (default 90 s) reflects horizon refraction and terrain.",
              "الفرق = المرصود − المتوقع؛ عدم اليقين المركّب = √(المراقب² + النموذج²) وحدّ النموذج (افتراضيًا 90 ثانية) يمثل انكسار الأفق والتضاريس.")]
        res.assumptions = [
            B("Flat, unobstructed horizon at sea level; standard atmosphere. Hills or buildings shift sunrise/sunset by minutes.", "أفق مستوٍ بلا عوائق عند مستوى البحر وغلاف جوي قياسي. التلال والمباني تُزيح الشروق والغروب بدقائق."),
            B("Times in UTC (add the zone offset to local clock readings); longitudes east positive.", "الأزمنة بالتوقيت العالمي (اطرح فرق المنطقة الزمنية من ساعتك)؛ الطول شرقًا موجب."),
            B("Noon altitude is the maximum of the geometric altitude; apparent altitude adds <= 0.02 deg of refraction at high Sun.", "ارتفاع الظهيرة هو الأقصى الهندسي؛ والظاهري يضيف ≤ 0.02° من الانكسار عند علو الشمس.")]
        if predict_only or not self.spec.get("observations"):
            res.summary = self._summary(daily, sites)
            res.verdict = self._season_verdict(res.summary) + ([B("Predictions generated before observing.", "تم توليد التنبؤات قبل الرصد.")] if predict_only else [])
            return self._finish(res)

        comp, zs = [], []
        for o in self.spec["observations"]:
            site = sites[o["site"]]
            kind = o["kind"]
            key = (o["site"], o["date"]) if o.get("date") else None
            if key and key not in cache:
                cache[key] = day_events(float(site["lat"]), float(site["lon"]), float(site.get("height_m", 0.0)), astro.parse_utc(o["date"]))
            if kind in ("sunrise", "sunset", "noon"):
                ev = cache[key]
                pj = {"sunrise": ev["sunrise_jd"], "sunset": ev["sunset_jd"], "noon": ev["transit_jd"]}[kind]
                if pj is None:
                    continue
                obs_jd = astro.parse_utc(o["observed_utc"])
                r = (obs_jd - pj) * 86400.0
                sg = math.sqrt(float(o.get("sigma_s", 60.0)) ** 2 + (model_sigma if kind != "noon" else 10.0) ** 2)
                comp.append({"site": o["site"], "kind": kind, "date": o["date"], "observed_utc": o["observed_utc"],
                             "predicted_utc": astro.jd_to_iso(pj, 0), "residual_s": r, "sigma_s": sg, "z": r / sg})
                zs.append(r / sg)
            elif kind == "altitude":
                t = astro.topocentric("sun", float(site["lat"]), float(site["lon"]), float(site.get("height_m", 0.0)), astro.parse_utc(o["time_utc"]))
                pa = float(t["alt_app"][0])
                r = float(o["alt_deg"]) - pa
                sg = float(o.get("sigma_deg", 0.2))
                comp.append({"site": o["site"], "kind": "altitude", "date": o["time_utc"], "observed_utc": o["alt_deg"],
                             "predicted_utc": pa, "residual_s": r, "sigma_s": sg, "z": r / sg})
                zs.append(r / sg)
        res.tables["comparison"] = comp
        z = np.array(zs)
        chi = float(np.sum(z ** 2))
        res.models = [ModelFit("Krios prediction (0 free parameters)", 0, chi, len(z), chi2_pvalue(chi, len(z)))]
        s = self._summary(daily, sites)
        if len(z):
            s.update({"observations_compared": len(z), "mean_z": float(z.mean()), "rms_z": float(np.sqrt(np.mean(z ** 2))),
                      "max_abs_residual_s": float(max(abs(c["residual_s"]) for c in comp if c["kind"] != "altitude")) if any(c["kind"] != "altitude" for c in comp) else None})
        res.summary = s
        v = self._season_verdict(s)
        if len(z):
            p = res.models[0].p_value
            v.insert(0, B(f"{len(z)} observations vs prediction: chi2 = {chi:.1f} (dof {len(z)}), p = {p:.2g}; mean offset {np.mean([c['residual_s'] for c in comp if c['kind'] != 'altitude'] or [0]):+.0f} s.",
                          f"{len(z)} رصدًا مقابل التنبؤ: χ² = {chi:.1f} (درجات الحرية {len(z)})، p = {p:.2g}."))
        res.verdict = v
        return self._finish(res)

    @staticmethod
    def _summary(daily, sites) -> Dict[str, Any]:
        s: Dict[str, Any] = {"sites": len(sites)}
        for name in sites:
            rows = [r for r in daily if r["site"] == name and r["day_length_h"] is not None]
            if len(rows) > 3:
                mx = max(rows, key=lambda r: r["day_length_h"])
                mn = min(rows, key=lambda r: r["day_length_h"])
                s[f"{name}: longest day (h)"] = f"{mx['day_length_h']:.2f} on {mx['date']}"
                s[f"{name}: shortest day (h)"] = f"{mn['day_length_h']:.2f} on {mn['date']}"
                pol = [r for r in rows if r["polar"]]
                if pol:
                    s[f"{name}: polar {pol[0]['polar']} dates"] = len(pol)
        return s

    @staticmethod
    def _season_verdict(s) -> List[Dict[str, str]]:
        longest = {k: v for k, v in s.items() if "longest day" in k}
        v: List[Dict[str, str]] = []
        if len(longest) >= 2:
            items = list(longest.items())
            v.append(B("Seasons by place: " + "; ".join(f"{k.split(':')[0]} longest day {val}" for k, val in items) +
                       ". A globe with a tilted axis predicts opposite seasons in the two hemispheres.",
                       "الفصول بحسب المكان: " + "؛ ".join(f"{k.split(':')[0]} أطول نهار {val}" for k, val in items) +
                       ". كرة بمحور مائل تتنبأ بفصول متعاكسة في نصفي الكرة."))
        return v

    def plot(self, result: LabResult, path: str) -> bool:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        rows = [r for r in result.tables.get("predictions", []) if r["day_length_h"] is not None]
        if not rows:
            return False
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for name in sorted({r["site"] for r in rows}):
            rr = [r for r in rows if r["site"] == name]
            ax.plot(range(len(rr)), [r["day_length_h"] for r in rr], label=name)
        ax.set_xlabel("sample index through the year")
        ax.set_ylabel("day length (h)")
        ax.set_title("Solar Path Lab" + (" (SYNTHETIC data)" if result.provenance.get("synthetic") else ""))
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return True
