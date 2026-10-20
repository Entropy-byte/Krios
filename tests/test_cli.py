import contextlib
import io
import json
import os
import tempfile
import unittest

from krios.cli import build_parser, main


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def test_eclipse_find_table(self):
        code, out, _ = run("eclipse", "find", "--from", "2027-08-01", "--to", "2027-08-05")
        self.assertEqual(code, 0)
        self.assertIn("Total", out)
        self.assertIn("2027-08-02", out)
        self.assertIn("10:06", out)

    def test_eclipse_find_json_and_csv(self):
        code, out, _ = run("eclipse", "find", "--from", "2027-02-01", "--to", "2027-03-01", "--format", "json")
        data = json.loads(out)
        self.assertEqual([e["body"] for e in data["eclipses"]], ["solar", "lunar"])
        self.assertEqual(data["eclipses"][0]["kind"], "Annular")
        code, out, _ = run("eclipse", "find", "--from", "2027-02-01", "--to", "2027-03-01", "--format", "csv")
        self.assertTrue(out.startswith("Date (UTC),Body,Type"))

    def test_eclipse_local(self):
        code, out, _ = run("eclipse", "local", "--date", "2027-08-02", "--lat", "25.69", "--lon", "32.64")
        self.assertEqual(code, 0)
        self.assertIn("Total/annular begins", out)
        code, out, err = run("eclipse", "local", "--date", "2027-09-15", "--lat", "10", "--lon", "10")
        self.assertEqual(code, 2)

    def test_orbit_csv_and_compare_use_the_dynamics_engine(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "moon.csv")
            code, _, err = run("orbit", "--epoch", "2027-01-01", "--days", "10", "--step", "5", "--bodies", "moon",
                               "--model", "newton", "--fit-days", "0", "--out", path)
            self.assertEqual(code, 0)
            with open(path) as fh:
                lines = fh.read().strip().splitlines()
            self.assertEqual(lines[0].split(",")[:3], ["jd_tdb", "date_tdb", "body"])
            self.assertEqual(len(lines), 1 + 3)                      # t = 0, 5, 10 d
        code, out, _ = run("demo", "mercury", "--years", "3")
        self.assertIn("GR contribution", out)

    def test_state_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            self.assertEqual(run("state", "--epoch", "2027-01-01", "--out", path)[0], 0)
            with open(path) as fh:
                data = json.load(fh)
            self.assertEqual(data["bodies"][0]["name"], "sun")
            code, out, _ = run("orbit", "--init", path, "--days", "2", "--step", "1", "--bodies", "moon", "--model", "newton")
            self.assertEqual(code, 0)
            self.assertIn(",moon,", out)

    def test_offline_horizons_without_cache_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            code, _, err = run("horizons", "vectors", "--start", "2027-08-02", "--stop", "2027-08-03",
                               "--offline", "--cache-dir", d)
            self.assertEqual(code, 1)
            self.assertIn("offline", err)

    def test_parser_has_all_commands(self):
        p = build_parser()
        for argv in (["info"], ["eclipse", "find"], ["demo", "mercury"], ["compare"], ["state"]):
            p.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
