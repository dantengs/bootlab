"""bootlab regress without a cluster: the check against the packaged bt steps, the
regression's inputs and its sbatch command: python -m unittest tests.test_regress."""
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"   # tools/make_test_fixture.py
sys.path.insert(0, str(ROOT / "src"))

from bootlab import cli, execute, inputs, regress  # noqa: E402

REF = json.loads(regress.REFERENCE.read_text())["steps"]


def shifted(v: str, rel: str) -> str:
    """v times (1 + rel), every digit kept."""
    d = Decimal(v)
    return str(regress.CTX.add(d, regress.CTX.multiply(d, Decimal(rel))))


class Check(unittest.TestCase):
    def setUp(self):
        self.run = Path(tempfile.mkdtemp())

    def write(self, steps):
        (self.run / "solves.jsonl").write_text("".join(json.dumps({**s, "verdict": "stepped"}) + "\n" for s in steps))

    def test_digits(self):
        self.assertEqual(regress.digits("1", "1.0000000001"), 10.0)
        self.assertEqual(regress.digits("-2.5e-3", "-2.5e-3"), "all")
        self.assertEqual(regress.digits(REF[0]["gap"], shifted(REF[0]["gap"], "1e-150")), 150.0)

    def test_the_reference_against_itself(self):
        self.write(REF)
        ok, lines, report = regress.check(self.run)
        self.assertTrue(ok, lines)
        self.assertEqual(report["least"], {k: "all" for k in ("x", "navigator", "ext_step", "gradient", "hessian")})
        self.assertTrue(lines[-1].startswith("PASS: 3 steps"))

    def test_digits_below_the_limit_fail(self):
        steps = json.loads(json.dumps(REF))
        steps[1]["gradient"][0] = shifted(steps[1]["gradient"][0], "1e-140")
        self.write(steps)
        ok, _, report = regress.check(self.run)
        self.assertTrue(ok)
        self.assertEqual(report["least"]["gradient"], 140.0)
        steps[1]["gradient"][0] = shifted(REF[1]["gradient"][0], "1e-120")
        steps[2]["reason"] = "found primal feasible solution"
        self.write(steps)
        ok, lines, report = regress.check(self.run)
        self.assertFalse(ok)
        self.assertEqual(report["problems"], ["step 2 gradient: 120.0",
                                              "step 3 reason: found primal feasible solution != " + REF[2]["reason"]])
        self.assertTrue(lines[-1].startswith("FAIL"))

    def test_a_missing_step_fails(self):
        self.write(REF[:2])
        ok, _, report = regress.check(self.run)
        self.assertFalse(ok)
        self.assertEqual(report["problems"], ["2 steps compared, the reference has 3"])


class Inputs(unittest.TestCase):
    def test_input_dir(self):
        d = Path(tempfile.mkdtemp()) / "in"
        run_toml = regress.input_dir(d, "lab", ["nodea-1"])
        text = run_toml.read_text()
        self.assertIn('name   = "regress-ising-l11"', text)
        self.assertIn("max_steps      = 3\n", text)
        p = inputs.load(run_toml)
        self.assertEqual(p.run["exec"], {"site": "lab"})
        self.assertEqual(p.run["resources"], {"nodes": ["nodea-1"]})
        self.assertEqual(p.run["skydive"]["start"], ["0.519", "1.4", "0.956"])    # bt's start (bt-steps.json)
        self.assertEqual(p.run["skydive"]["start"], REF[0]["x"])

    def test_submit_dry(self):
        """On the login node: the inputs under the site's runs, one node, the job runs `regress run`, 1 h."""
        runs = Path(tempfile.mkdtemp())
        site = dataclasses.replace(execute.load_site("lab", FIXTURES), runs=str(runs))
        sinfo = SimpleNamespace(stdout="nodea-1 idle\n")
        out = io.StringIO()
        with mock.patch.object(cli.execute, "load_site", return_value=site), \
                mock.patch.object(cli.subprocess, "run", return_value=sinfo) as run, contextlib.redirect_stdout(out):
            cli.main(["regress", "--site", "lab", "--dry"])
        self.assertEqual(run.call_count, 1)                      # sinfo only
        cmd = out.getvalue().splitlines()[-1]
        self.assertIn(" --time=01:00:00 ", cmd)
        self.assertIn(" --nodes=1 ", cmd)
        made = sorted(runs.glob("regress-input-*/run.toml"))
        self.assertEqual(len(made), 1)
        self.assertIn(f"-m bootlab.cli regress run {made[0]}", cmd.replace("'", ""))


if __name__ == "__main__":
    unittest.main()
