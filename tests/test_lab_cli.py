import contextlib
import io
import json
import os
import tempfile
import unittest

from krios.cli import main


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestLabCLI(unittest.TestCase):
    def test_list_and_demo_in_both_languages(self):
        code, out, _ = run("lab", "list")
        self.assertIn("parallax", out)
        code, out, _ = run("lab", "demo", "pole")
        self.assertIn("SYNTHETIC DATA", out)
        code, out, _ = run("lab", "demo", "shadow", "--lang", "ar")
        self.assertIn("الخلاصة", out)
        self.assertIn("إراتوستينس", out)

    def test_template_plan_run_roundtrip_with_preregistration(self):
        with tempfile.TemporaryDirectory() as d:
            tpl, plan, rep, js = (os.path.join(d, n) for n in ("shadow.json", "plan.json", "report.md", "result.json"))
            self.assertEqual(run("lab", "template", "shadow", "--out", tpl)[0], 0)
            with open(tpl, encoding="utf-8") as fh:
                spec = json.load(fh)
            self.assertTrue(spec["metadata"]["synthetic"])
            code, out, _ = run("lab", "plan", "shadow", "--data", tpl, "--out", plan)
            self.assertEqual(code, 0)
            self.assertIn("prediction SHA-256", out)
            code, out, _ = run("lab", "run", "shadow", "--data", tpl, "--plan", plan, "--report", rep, "--json", js,
                               "--csv-dir", d, "--plot", os.path.join(d, "fig.png"), "--lang", "en")
            self.assertEqual(code, 0)
            self.assertIn("IDENTICAL", out)
            with open(rep, encoding="utf-8") as fh:
                self.assertIn("## Verdict", fh.read())
            with open(js, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["experiment"], "shadow")
            self.assertTrue(os.path.exists(os.path.join(d, "shadow_pairs.csv")))
            try:
                import matplotlib  # noqa: F401
                self.assertTrue(os.path.getsize(os.path.join(d, "fig.png")) > 5000)
            except ImportError:
                pass

    def test_csv_template_and_errors(self):
        code, out, _ = run("lab", "template", "solarpath", "--format", "csv")
        self.assertEqual(code, 0)
        self.assertIn("# SYNTHETIC EXAMPLE", out)
        code, _, err = run("lab", "template", "pole", "--format", "csv")
        self.assertEqual(code, 1)
        self.assertIn("nested", err)
        code, _, err = run("lab", "demo", "nope")
        self.assertEqual(code, 1)
        self.assertIn("unknown lab", err)

    def test_eclipse_path_command(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "path.geojson")
            code, _, _ = run("eclipse", "path", "--date", "2027-08-02", "--step", "300", "--out", p)
            self.assertEqual(code, 0)
            with open(p, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["features"][1]["properties"]["kind"], "greatest_eclipse")


if __name__ == "__main__":
    unittest.main()
