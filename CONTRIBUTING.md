# Contributing to Krios

Thanks for helping. A few rules keep the project trustworthy:

1. **Run the tests**: `pip install -e ".[plots]"` then `python -m unittest discover -s tests -t .` (about 30 s).
2. **Physics changes need a reference**: compare with a published value (NASA/EclipseWise catalog, JPL Horizons, a worked
   example from the literature) and add it to `tests/`. State the tolerance and why it is that tolerance.
3. **New lab** (see `docs/LABS.md`): subclass `krios.lab.core.Experiment`, register it, make `predictions` depend only on
   non-measurement inputs, give the alternative model free parameters, provide a seeded **synthetic** `example()`, write bilingual
   (`B(en, ar)`) verdicts, and add a test that generates data from the *competing* model and checks the lab does not misreport it.
4. **No hidden assumptions**: every assumption a result depends on belongs in the report's "Assumptions and limits".
5. Keep dependencies to numpy and scipy (matplotlib stays optional).
6. Be kind in issues and reviews; report data problems with the raw input attached.
