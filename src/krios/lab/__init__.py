"""krios.lab - reproducible measurement-vs-prediction experiments."""
from .core import EXPERIMENTS, B, LabResult, ModelFit, check_plan, load_spec, make_plan

from . import shadow, pole, parallax, solarpath, eclipse_lab   # noqa: F401  (registers the experiments)

__all__ = ["EXPERIMENTS", "B", "LabResult", "ModelFit", "check_plan", "load_spec", "make_plan"]
