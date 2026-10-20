"""Lab framework: experiments, results, provenance, reports.

Every lab follows the same four steps:
  1. user data         - measurements, with places, times and uncertainties
  2. predictions       - computed from the Krios engine using ONLY non-measurement inputs
  3. comparison        - residuals, uncertainties, and model fits (fair, with free parameters)
  4. reproducible report - JSON / CSV / Markdown with inputs, equations, assumptions, versions

Text shown to people is bilingual: ``B(en, ar)``.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import numpy as np
import scipy

from .. import __version__
from .. import timescales as ts


def B(en: str, ar: str) -> Dict[str, str]:
    return {"en": en, "ar": ar}


EXPERIMENTS: Dict[str, Type["Experiment"]] = {}


def register(cls):
    EXPERIMENTS[cls.name] = cls
    return cls


@dataclass
class ModelFit:
    name: str
    n_params: int
    chi2: float
    dof: int
    p_value: float
    params: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class LabResult:
    experiment: str
    title: Dict[str, str]
    summary: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    models: List[ModelFit] = field(default_factory=list)
    verdict: List[Dict[str, str]] = field(default_factory=list)
    method: List[Dict[str, str]] = field(default_factory=list)
    assumptions: List[Dict[str, str]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    prediction_sha256: str = ""
    plan_check: Optional[str] = None

    # -- serialisation ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return _clean({"experiment": self.experiment, "title": self.title, "summary": self.summary,
                       "tables": self.tables, "models": [m.to_dict() for m in self.models],
                       "verdict": self.verdict, "method": self.method, "assumptions": self.assumptions,
                       "provenance": self.provenance, "prediction_sha256": self.prediction_sha256,
                       "plan_check": self.plan_check})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def table_csv(self, name: str) -> str:
        rows = self.tables[name]
        buf = io.StringIO()
        if rows:
            cols = list(rows[0].keys())
            w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(_clean(rows))
        return buf.getvalue()

    def write(self, markdown: Optional[str] = None, json_path: Optional[str] = None,
              csv_dir: Optional[str] = None, lang: str = "en") -> None:
        if markdown:
            with open(markdown, "w", encoding="utf-8") as f:
                f.write(self.to_markdown(lang))
        if json_path:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(self.to_json())
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            for name in self.tables:
                with open(os.path.join(csv_dir, f"{self.experiment}_{name}.csv"), "w", encoding="utf-8") as f:
                    f.write(self.table_csv(name))

    # -- markdown -----------------------------------------------------
    def to_markdown(self, lang: str = "en", max_rows: int = 60) -> str:
        T = _TITLES[lang]
        t = lambda b: b[lang] if isinstance(b, dict) else str(b)
        out = [f"# {t(self.title)}", ""]
        if self.provenance.get("synthetic"):
            out += [f"> **{T['synthetic']}**", ""]
        out += [f"## {T['verdict']}", ""] + [f"- {t(v)}" for v in self.verdict] + [""]
        if self.summary:
            out += [f"## {T['summary']}", "", "| | |", "|---|---|"]
            out += [f"| {k} | {_fmt(v)} |" for k, v in self.summary.items()] + [""]
        if self.models:
            out += [f"## {T['models']}", "", "| model | free params | chi2 | dof | p-value | note |", "|---|---:|---:|---:|---:|---|"]
            for m in self.models:
                out.append(f"| {m.name} | {m.n_params} | {_fmt(m.chi2)} | {m.dof} | {_fmt(m.p_value)} | {m.note} |")
            out.append("")
        for name, rows in self.tables.items():
            out += [f"## {T['table']}: {name}", ""]
            out += _md_table(rows, max_rows) + [""]
        out += [f"## {T['method']}", ""] + [f"- {t(v)}" for v in self.method] + [""]
        out += [f"## {T['assumptions']}", ""] + [f"- {t(v)}" for v in self.assumptions] + [""]
        out += [f"## {T['provenance']}", "", "```json", json.dumps(_clean(self.provenance), indent=2, ensure_ascii=False), "```"]
        if self.prediction_sha256:
            out += ["", f"{T['pred_hash']}: `{self.prediction_sha256}`"]
        if self.plan_check:
            out += ["", f"{T['plan']}: {self.plan_check}"]
        return "\n".join(out) + "\n"

    def brief(self, lang: str = "en") -> str:
        """Verdict + summary + model table (what the terminal prints)."""
        T = _TITLES[lang]
        t = lambda b: b[lang] if isinstance(b, dict) else str(b)
        out = [t(self.title), "=" * min(len(t(self.title)), 78)]
        if self.provenance.get("synthetic"):
            out.append(f"[{T['synthetic']}]")
        out += ["", T["verdict"] + ":"] + [f"  * {t(v)}" for v in self.verdict]
        if self.models:
            out += ["", T["models"] + ":"]
            for m in self.models:
                out.append(f"  - {m.name}: params={m.n_params}  chi2={_fmt(m.chi2)}  dof={m.dof}  p={_fmt(m.p_value)}")
        if self.plan_check:
            out += ["", f"{T['plan']}: {self.plan_check}"]
        return "\n".join(out) + "\n"

    def prediction_hash(self) -> str:
        rows = self.tables.get("predictions", [])
        blob = json.dumps(_round_rows(rows), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


_TITLES = {
    "en": {"verdict": "Verdict", "summary": "Summary", "models": "Model comparison", "table": "Table",
           "method": "Method", "assumptions": "Assumptions and limits", "provenance": "Provenance (for reproduction)",
           "synthetic": "SYNTHETIC DATA generated by Krios for demonstration. These are not real measurements.",
           "pred_hash": "SHA-256 of the independent predictions", "plan": "Pre-registered plan"},
    "ar": {"verdict": "الخلاصة", "summary": "الملخص", "models": "مقارنة النماذج", "table": "جدول",
           "method": "الطريقة", "assumptions": "الافتراضات والحدود", "provenance": "المصدر (لإعادة الإنتاج)",
           "synthetic": "بيانات اصطناعية ولّدها Krios للتوضيح فقط. ليست قياسات حقيقية.",
           "pred_hash": "بصمة SHA-256 للتنبؤات المستقلة", "plan": "الخطة المسجّلة مسبقًا"},
}


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "-"
        return f"{float(v):.6g}"
    return str(v)


def _md_table(rows: List[Dict[str, Any]], max_rows: int) -> List[str]:
    if not rows:
        return ["(empty)"]
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows[:max_rows]:
        out.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    if len(rows) > max_rows:
        out.append(f"| ... {len(rows) - max_rows} more rows (see the CSV export) |" + " |" * (len(cols) - 1))
    return out


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _round_rows(rows):
    def r(v):
        if isinstance(v, (float, np.floating)):
            f = float(v)
            return None if math.isnan(f) else round(f, 3)
        return v
    return [{k: r(v) for k, v in row.items()} for row in rows]


def spec_hash(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_clean(spec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def provenance(experiment: str, spec: Dict[str, Any], ephemeris: str) -> Dict[str, Any]:
    return {"krios_version": __version__, "experiment": experiment, "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__, "ephemeris": ephemeris,
            "delta_t_s": round(ts.delta_t_for_jd(2461000.5), 2), "input_sha256": spec_hash(spec),
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "synthetic": bool(spec.get("metadata", {}).get("synthetic", False)),
            "metadata": spec.get("metadata", {})}


def _coerce(v: str):
    s = v.strip()
    if s == "":
        return None
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return float(s)
    except ValueError:
        return s


def load_spec(path: str, rows_key: str) -> Dict[str, Any]:
    """Load a JSON spec, or a CSV table (rows go under ``rows_key``; '#' lines are comments)."""
    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        rows = [{k.strip(): _coerce(v or "") for k, v in r.items() if k} for r in csv.DictReader(lines)]
        rows = [{k: v for k, v in r.items() if v is not None} for r in rows]
        return {"metadata": {"source_file": os.path.basename(path)}, rows_key: rows}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Experiment:
    """Base class: subclasses set ``name`` / ``rows_key`` / ``title`` and implement run()."""
    name: str = ""
    rows_key: str = ""
    title: Dict[str, str] = B("", "")
    blurb: Dict[str, str] = B("", "")

    def __init__(self, spec: Dict[str, Any], ephemeris=None):
        self.spec = spec
        self.ephemeris = ephemeris

    @classmethod
    def example(cls) -> Dict[str, Any]:
        raise NotImplementedError

    def run(self, predict_only: bool = False) -> LabResult:
        raise NotImplementedError

    def plot(self, result: LabResult, path: str) -> bool:           # optional
        return False

    # helper for subclasses
    def _finish(self, res: LabResult, ephemeris_name: str = "analytic (Meeus ELP-2000/82 + VSOP87D)") -> LabResult:
        res.provenance = provenance(self.name, self.spec, ephemeris_name)
        res.prediction_sha256 = res.prediction_hash()
        return res


def check_plan(result: LabResult, plan: Dict[str, Any]) -> str:
    """Compare a run's predictions with a pre-registered plan file."""
    same = plan.get("prediction_sha256") == result.prediction_sha256
    when = plan.get("generated_at_utc", "?")
    return (f"plan generated {when} (Krios {plan.get('krios_version', '?')}): predictions "
            + ("IDENTICAL to the pre-registered ones" if same else "DIFFER from the pre-registered ones - the inputs or the engine changed"))


def make_plan(result: LabResult) -> Dict[str, Any]:
    return _clean({"krios_version": __version__, "experiment": result.experiment,
                   "generated_at_utc": result.provenance.get("generated_at_utc"),
                   "input_sha256": result.provenance.get("input_sha256"),
                   "prediction_sha256": result.prediction_sha256,
                   "predictions": result.tables.get("predictions", []),
                   "note": "Predictions computed from non-measurement inputs only. Keep this file (and its hash) "
                           "before taking measurements; pass it to `krios lab run --plan` afterwards."})
