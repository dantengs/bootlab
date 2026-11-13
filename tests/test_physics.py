"""The physics layer against the production bt (r3.8): python -m unittest discover -s tests

bt's own reading and checks are the judge. tests/reference/physics/make_reference.py had bt read each case's
hand-written bt parameter file (bt-k.json) and the one bootlab makes from its TOML (bootlab-k.json), and kept
what bt printed (*.result.json). These tests need no cluster: they check that bootlab still makes the same
file, that bt read both alike, and that bootlab gives the errors and warnings bt gives.
"""
import io
import json
import os
import re
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "reference" / "physics"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REF))

from bootlab import inputs  # noqa: E402
from bootlab.numbers import dumps, frac  # noqa: E402
from make_reference import replace_spectrum  # noqa: E402

CASES = sorted(d for d in (REF / "cases").iterdir() if d.is_dir())


def result(path):
    return json.loads(path.read_text())


def points(case):
    return json.loads((case / "points.json").read_text())


def exact(text):
    """A JSON text with every number an exact Fraction."""
    return json.loads(text, parse_float=Fraction, parse_int=Fraction)


def bt_spectrum(stdout):
    out = exact(stdout)
    return (out["bound"]["model"] if "bound" in out else out["model"])["spectrum"]


def bt_spins(spec):
    """What bt prints for spins (Projects.BT.Spectrum, ToJSON Spins), with exact numbers."""
    s = inputs.read_spins(spec)
    if s[0] == "list":
        return [Fraction(l) for l in s[1]]
    if s[0] == "all":
        return "all"
    _, a, b, step = s
    return f"{a}-{b}" + (f":{step}" if step != 1 else "")


# ---------------------------------------------------------------- errors and warnings, from bt or from bootlab

ISSUES = [
    (r"gaps?\[(\d+)\]: there is no channel", "gap channel"),
    (r"isolated\[(\d+)\]: there is no channel", "isolated channel"),
    (r"gaps?\[(\d+)\]: \S+ has \w+ spins, not spin ([-\d, ]+)", "gap spin rule"),
    (r"isolated\[(\d+)\]: \S+ has \w+ spins, not spin ([-\d, ]+)", "isolated spin rule"),
    (r"gaps?\[(\d+)\]: no spin of", "gap no spin"),
    (r"(\S+) spin (\d+) has a gap in both|the gap of (\S+) spin (\d+) is set twice", "gap twice"),
    (r"isolated has (\S+) spin (\d+) at (\S+) twice", "isolated twice"),
    (r"gaps?\[(\d+)\](?:\.spins)?: spins must be", "gap bad spins"),
    (r"gaps?\[(\d+)\]: unknown key (\w+)", "gap unknown key"),
    (r"unknown parameter: model\.spectrum\.(\w+)|spectrum: unknown key (\w+)", "spectrum unknown key"),
    (r"gaps?\[(\d+)\]: the gap for \S+ needs exactly one", "gap needs one"),
    (r"(?:twistGap|default_above_unitarity) is negative", "twist negative"),
    (r"gaps?\[(\d+)\]: (?:aboveUnitarity|above_unitarity) is negative", "above negative"),
    (r"gaps?\[(\d+)\]: (?:gap|delta) (\S+) of (\S+) spin (\d+) is below the unitarity bound (\S+)", "gap below"),
    (r"gaps?\[(\d+)\]: spin ([\d, ]+) is not among", "gap outside"),
    (r"isolated\[(\d+)\]: delta (\S+) is below the unitarity bound (\S+)", "isolated below"),
    (r"isolated\[(\d+)\]: delta (\S+) is above where the continuum of (\S+) spin (\d+) starts, (\S+)",
     "isolated above"),
    (r"isolated\[(\d+)\]: spin (\d+) is not among", "isolated outside"),
]
HEADERS = re.compile(r"do not fit model|the inputs have errors")


def issues(lines):
    """Each error or warning as (kind, its entries, channels, spins and values): bt and bootlab word them a
    little differently (bt's Double against bootlab's exact decimal, gaps[0] against gap[0])."""
    out = []
    for line in lines:
        line = line.strip().rstrip(")")
        if not line or HEADERS.search(line):
            continue
        for pattern, kind in ISSUES:
            m = re.search(pattern, line)
            if m:
                groups = [g for g in m.groups() if g is not None]
                if kind == "gap twice":
                    groups += re.findall(r"gaps?\[(\d+)\]", line)
                out.append((kind, *(_norm(g) for g in groups)))
                break
        else:
            out.append(("not understood", line))
    return sorted(out, key=repr)


def _norm(g):
    """A value bt prints as a Double compares by its float; a list of spins by its numbers."""
    if re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", g):
        return float(g)
    if re.fullmatch(r"[-\d, ]+", g):
        return tuple(int(x) for x in g.split(","))
    return g


def bt_issues(stderr):
    return issues(l.removeprefix("warning: ") for l in stderr.splitlines())


def bootlab_issues(case, k):
    """bootlab's errors, or its warnings at point k."""
    try:
        p = inputs.load(case / "run.toml")
        inputs.resolve_spectrum(p, points(case)[k])
    except SystemExit as e:
        return issues(str(e).splitlines())
    return issues(p.warnings + inputs.spectrum_warnings(p, points(case)[k]))


class SpectrumAgainstBt(unittest.TestCase):
    def test_bootlab_spectrum_is_read_as_the_original(self):
        n = 0
        for case in CASES:
            for k, point in enumerate(points(case)):
                sent = case / f"bootlab-{k}.json"
                if not sent.exists():
                    continue
                n += 1
                with self.subTest(case=case.name, point=k):
                    p = inputs.load(case / "run.toml")
                    spectrum = inputs.resolve_spectrum(p, point)
                    # bootlab today makes exactly the file bt read
                    original = (case / f"bt-{k}.json").read_text()
                    self.assertEqual(replace_spectrum(original, dumps(spectrum, indent=1)), sent.read_text())
                    # bt read it as the hand-written file: the same output, warnings and exit code
                    ours, theirs = result(case / f"bootlab-{k}.result.json"), result(case / f"bt-{k}.result.json")
                    self.assertEqual(ours, theirs)
                    self.assertEqual(theirs["exit"], 0)
                    # and what bt read is bootlab's numbers, as exact rationals
                    read = bt_spectrum(theirs["stdout"])
                    self.assertEqual(read["twistGap"], Fraction(spectrum["twistGap"]))
                    self.assertEqual(len(read["gaps"]), len(spectrum["gaps"]))
                    for g, b in zip(spectrum["gaps"], read["gaps"]):
                        key = "gap" if "gap" in g else "aboveUnitarity"
                        self.assertEqual(b, {"channel": g["channel"], "spins": bt_spins(g["spins"]),
                                             key: Fraction(g[key])})
                    self.assertEqual(read["isolated"], [{"channel": i["channel"], "spin": i["spin"],
                                                         "delta": Fraction(i["delta"])} for i in spectrum["isolated"]])
        self.assertEqual(n, 4)       # the Ising cases' points

    def test_point_parameters(self):
        """A gap that refers to point parameters reaches bt as the exact rational at the point."""
        case = REF / "cases" / "point-ising-oe"
        want = [("3.8", "2.519", "1.2"),
                ("3.80000000000000000001", "2.51900000000000000001", "1.200000000000000000005"),
                ("3.82966", "2.518148972345678901234567890123456789", "1.208517305")]
        for k, (e0, s0, e2) in enumerate(want):
            read = bt_spectrum(result(case / f"bootlab-{k}.result.json")["stdout"])
            self.assertEqual([g.get("gap", g.get("aboveUnitarity")) for g in read["gaps"]],
                             [Fraction(e0), Fraction(s0), Fraction(e2)])
        p = inputs.load(case / "run.toml")
        self.assertEqual(p.coords, ["deltaSig", "deltaEps", "theta", "deltaEpsP"])
        self.assertEqual(p.model_coords, ["deltaSig", "deltaEps", "theta"])


class ChecksAgainstBt(unittest.TestCase):
    def test_errors_and_warnings(self):
        n = 0
        for case in CASES:
            for k in range(len(points(case))):
                with self.subTest(case=case.name, point=k):
                    bt = result(case / f"bt-{k}.result.json")
                    want = bt_issues(bt["stderr"])
                    self.assertNotIn("not understood", [w[0] for w in want])
                    self.assertEqual(bootlab_issues(case, k), want)
                    n += bool(want)
        self.assertEqual(n, 0)        # the Ising cases have no warnings

    def test_errors_stop_bootlab(self):
        for case in CASES:
            if case.name.startswith("err-"):
                with self.subTest(case=case.name):
                    self.assertEqual(result(case / "bt-0.result.json")["exit"], 1)
                    with self.assertRaises(SystemExit):
                        inputs.resolve_spectrum(inputs.load(case / "run.toml"), points(case)[0])


class CoordinatesAgainstBt(unittest.TestCase):
    def test_order(self):
        for f in sorted((REF / "coords").glob("*.result.json")):
            with self.subTest(model=f.name):
                m = re.search(r"needs \d+ coordinates: ([^\n)]*)", result(f)["stderr"])
                bt = m.group(1).split(", ")
                model = json.loads((REF / "models" / f.name.replace(".result.json", ".model.json")).read_text())
                # bt's feasibility point: the model's coordinates, then theta with an OPE angle
                self.assertEqual(inputs.model_coordinates(model), bt)


class Boundary(unittest.TestCase):
    """bootlab's own checks where its input is TOML: a value must keep the decimal written."""

    SPECTRUM = ('format = "bootlab-spectrum/1"\ndefault_above_unitarity = "0"\n\n[[gap]]\nchannel = "rep1+"\n'
                'spins   = [0]\ndelta   = "deltaEpsP"\n\n[[isolated]]\nchannel = "rep1+"\nspin    = 2\ndelta   = "3"\n')

    def _case(self, spectrum, objective='format = "bootlab-objective/1"\nkind = "navigator"\n'):
        src = REF / "cases" / "point-ising-oe"
        d = Path(tempfile.mkdtemp())
        (d / "run.toml").write_text((src / "run.toml").read_text().replace("../../models/", str(REF / "models").replace("\\", "/") + "/"))
        (d / "spectrum.toml").write_text(spectrum)
        (d / "objective.toml").write_text(objective)
        return d / "run.toml"

    def test_toml_float_is_an_error(self):
        spec = (src := self.SPECTRUM).replace('delta   = "3"', "delta = 3.0")
        self.assertNotEqual(spec, src)
        with self.assertRaises(SystemExit) as e:
            inputs.load(self._case(spec))
        self.assertIn("isolated[0]: write delta as a string", str(e.exception))
        spec = src.replace('default_above_unitarity = "0"', "default_above_unitarity = 0.51814897123456789")
        with self.assertRaises(SystemExit) as e:
            inputs.load(self._case(spec))
        self.assertIn("default_above_unitarity as a string", str(e.exception))
        p = inputs.load(self._case(src.replace('delta   = "3"', "delta = 3")))      # an integer is exact
        self.assertEqual(inputs.resolve_spectrum(p, {"deltaSig": "0.5", "deltaEps": "1.4", "theta": "0.9", "deltaEpsP": "3.8"})["isolated"][0]["delta"], "3")

    def test_sigma(self):
        spec = self.SPECTRUM
        obj = 'format = "bootlab-objective/1"\nkind = "navigator"\nsigma = { step = "1/5" }\n'
        p = inputs.load(self._case(spec, obj))
        self.assertEqual(p.model["objectives"]["navigator"]["sigma"], {"shift": "1/5", "step": "1/5", "width": "6"})
        with self.assertRaises(SystemExit) as e:
            inputs.load(self._case(spec, obj.replace('"1/5"', "0.2")))
        self.assertIn("sigma.step as a string", str(e.exception))


class RunNumbers(unittest.TestCase):
    """Every number of run.toml that reaches a program keeps the digits written."""
    EXAMPLE = ROOT / "examples" / "ising"

    def _run(self, edit):
        d = Path(tempfile.mkdtemp())
        for f in ("model.json", "spectrum.toml", "objective.toml"):
            (d / f).write_bytes((self.EXAMPLE / f).read_bytes())
        text = (self.EXAMPLE / "run.toml").read_text()
        new = edit(text)
        self.assertNotEqual(new, text)
        (d / "run.toml").write_text(new)
        return d / "run.toml"

    def _errors(self, edit):
        with self.assertRaises(SystemExit) as e:
            inputs.load(self._run(edit))
        return str(e.exception)

    def test_toml_float_is_an_error(self):
        e = self._errors(lambda t: t.replace('start          = ["0.519", "1.4", "0.956"]',
                                             'start = [0.51814897123456789, "1.4", 0.956]'))
        self.assertIn("[skydive] start = [0.5181489712345679, '1.4', 0.956] has a TOML float", e)
        self.assertIn("put each number in quotes, with all the digits as you wrote them", e)
        self.assertNotIn('"0.5181489712345679"', e)      # no number to copy that has lost digits
        e = self._errors(lambda t: t.replace('gap_upper_limit            = "0.1"', "gap_upper_limit = 0.1"))
        self.assertIn("[skydive] gap_upper_limit = 0.1 has a TOML float", e)
        e = self._errors(lambda t: t.replace("[solver]\n", "[solver]\ndualityGapThreshold = 1e-6\n")
                         .replace("[dynamical]\n", "[dynamical]\nbetaScanMin = 0.1\n"))
        self.assertIn("[solver] dualityGapThreshold", e)
        self.assertIn("[dynamical] betaScanMin", e)
        # booleans, integers and other strings stay as they are
        p = inputs.load(self._run(lambda t: t.replace("[dynamical]\n", "[dynamical]\nnavigatorWithLogDetX = true\n"
                                                                         "maxClimbingSteps = 2\n")
                                  .replace("[solver]\n", '[solver]\nwriteSolution = "x,y"\n')))
        self.assertEqual(p.run["solver"]["writeSolution"], "x,y")
        self.assertIs(p.run["dynamical"]["navigatorWithLogDetX"], True)

    def test_vectors(self):
        e = self._errors(lambda t: t.replace('box_min        = ["0.517", "1.2", "0.95"]', 'box_min = ["0.517", "1.2"]'))
        self.assertIn("box_min needs 3 entries", e)
        e = self._errors(lambda t: t.replace('hessian_guess  = ["1000000", "10000", "1000"]', 'hessian_guess = ["1e6", "x", "1"]'))
        self.assertIn("hessian_guess", e)
        e = self._errors(lambda t: t.replace('infinitesimal  = "1e-20"', 'infinitesimal = "tiny"'))
        self.assertIn("infinitesimal: 'tiny' is not a number", e)

    def test_infinitesimal(self):
        """docs/en/interfaces.md section 6: 1e-20 when left out; 1e-25 the smallest load takes (the generator's block keys keep
        about 31 digits of Delta12); simpleboot's 1e-40 rejected, with the reason."""
        from bootlab import skydive
        line = 'infinitesimal  = "1e-20"'
        p = inputs.load(self._run(lambda t: t.replace(line, "")))
        self.assertNotIn("infinitesimal", p.run["skydive"])
        r = skydive.Run.__new__(skydive.Run)
        r.p, r.opts = p, p.run["skydive"]
        x = skydive.start_state(p)["x"]
        self.assertEqual(frac(r.points(x)["plus_1"]["deltaEps"]) - frac(x["deltaEps"]), frac("1e-20"))
        p = inputs.load(self._run(lambda t: t.replace(line, 'infinitesimal = "1e-25"')))
        self.assertEqual(p.run["skydive"]["infinitesimal"], "1e-25")
        for small in ("1e-40", "9.99e-26"):
            e = self._errors(lambda t: t.replace(line, f'infinitesimal = "{small}"'))
            self.assertIn(f"infinitesimal {small} is below 1e-25", e)
            self.assertIn("about 31 digits of Delta12", e)
            self.assertIn("docs/en/interfaces.md 6", e)

    def test_gap_upper_limit_track(self):
        """docs/en/interfaces.md section 10: 0 < c <= 3 or "off"; a c not with gap_upper_limit_first_only = true; the floor kappa
        only while the limit tracks N. examples/ising (bt's settings) writes "off"."""
        off = 'gap_upper_limit_track      = "off"'
        first = 'gap_upper_limit_first_only = false'
        sub = lambda new: (lambda t: t.replace(off, new))
        self.assertIn(off, (self.EXAMPLE / "run.toml").read_text())
        p = inputs.load(self.EXAMPLE / "run.toml")
        self.assertIsNone(inputs.track_c(p.run["skydive"]))
        p = inputs.load(self._run(sub('gap_upper_limit_track = "3"')))
        self.assertEqual(inputs.track_c(p.run["skydive"]), "3")
        e = self._errors(sub('gap_upper_limit_track = "3.5"'))
        self.assertIn("gap_upper_limit_track 3.5 is not in (0, 3]", e)
        e = self._errors(sub('gap_upper_limit_track = "0"'))
        self.assertIn("gap_upper_limit_track 0 is not in (0, 3]", e)
        e = self._errors(lambda t: t.replace(off, 'gap_upper_limit_track = "1"').replace(first, 'gap_upper_limit_first_only = true'))
        self.assertIn("gap_upper_limit_track and gap_upper_limit_first_only = true are exclusive", e)
        e = self._errors(sub('gap_upper_limit_track = "big"'))
        self.assertIn("gap_upper_limit_track: 'big' is not a number", e)
        p = inputs.load(self._run(sub('gap_upper_limit_track = "1"\ngap_upper_limit_track_floor = "0.5"')))
        self.assertEqual(p.run["skydive"]["gap_upper_limit_track_floor"], "0.5")
        e = self._errors(sub('gap_upper_limit_track = "1"\ngap_upper_limit_track_floor = "x"'))
        self.assertIn("gap_upper_limit_track_floor: 'x' is not a number", e)
        e = self._errors(lambda t: t.replace(off, off + '\ngap_upper_limit_track_floor = "0.1"'))
        self.assertIn("gap_upper_limit_track_floor is set while the gap upper limit does not track N", e)

    def test_gap_upper_limit_track_default(self):
        """Left out, the limit tracks N with c = 1: with first_only false or left out, and the floor
        may be set alone; with first_only = true and no c the limits are first_only's."""
        off = 'gap_upper_limit_track      = "off"'
        first = 'gap_upper_limit_first_only = false'
        drop = lambda t: t.replace(off + "     # bt's fixed limit at every call; tracking N is the default (SPEC 6)\n", "")
        p = inputs.load(self._run(drop))
        self.assertNotIn("gap_upper_limit_track", p.run["skydive"])
        self.assertEqual(inputs.track_c(p.run["skydive"]), "1")
        p = inputs.load(self._run(lambda t: drop(t).replace(first + "\n", "")))
        self.assertEqual(inputs.track_c(p.run["skydive"]), "1")
        p = inputs.load(self._run(lambda t: drop(t).replace(first, first + '\ngap_upper_limit_track_floor = "0.5"')))
        self.assertEqual(p.run["skydive"]["gap_upper_limit_track_floor"], "0.5")
        p = inputs.load(self._run(lambda t: drop(t).replace(first, 'gap_upper_limit_first_only = true')))
        self.assertIsNone(inputs.track_c(p.run["skydive"]))
        e = self._errors(lambda t: drop(t).replace(first, 'gap_upper_limit_first_only = true\ngap_upper_limit_track_floor = "0.1"'))
        self.assertIn("gap_upper_limit_track_floor is set while the gap upper limit does not track N", e)

    def test_gap_threshold_tighten(self):
        """docs/en/interfaces.md section 10: true or false, left out in the examples (true)."""
        first = 'gap_upper_limit_first_only = false'
        self.assertNotIn("gap_threshold_tighten", inputs.load(self.EXAMPLE / "run.toml").run["skydive"])
        p = inputs.load(self._run(lambda t: t.replace(first, first + "\ngap_threshold_tighten = false")))
        self.assertIs(p.run["skydive"]["gap_threshold_tighten"], False)
        p = inputs.load(self._run(lambda t: t.replace(first, first + "\ngap_threshold_tighten = true")))
        self.assertIs(p.run["skydive"]["gap_threshold_tighten"], True)
        for bad in ('"off"', "0"):
            e = self._errors(lambda t: t.replace(first, first + f"\ngap_threshold_tighten = {bad}"))
            self.assertIn("gap_threshold_tighten = " + ("'off'" if bad == '"off"' else "0") + " is not true or false", e)

    def test_start_digits_reach_the_generator(self):
        from bootlab import pmp, skydive
        digits = "0.5181489712345678901234567890123456789012"
        p = inputs.load(self._run(lambda t: t.replace('start          = ["0.519",', f'start = ["{digits}",')))
        s = skydive.start_state(p)
        self.assertEqual(s["x"]["deltaSig"], digits)
        r = skydive.Run.__new__(skydive.Run)
        r.p, r.opts = p, p.run["skydive"]
        out = Path(tempfile.mkdtemp())
        text = pmp.write_request(p, out / "model.json", r.points(s["x"]), out, out / "blocks", 128).read_text()
        req = json.loads(text, parse_float=str)
        from decimal import Decimal, getcontext
        getcontext().prec = 100
        coords = {pt["name"]: pt["coords"] for pt in req["points"]}         # each number as written in the JSON
        self.assertEqual(coords["c"], [digits, "1.4", "0.956"])
        self.assertEqual(coords["plus_0"][0], str(Decimal(digits) + Decimal("1e-20")))   # infinitesimal 1e-20
        self.assertIn(f"{digits},", text)                                   # a bare JSON number, every digit


class RecordedPaths(unittest.TestCase):
    """The recorded bt files keep the paths bt read (here under a placeholder prefix) (the model files);
    they are the record, not an input. The other tests of this module pass with every open of such a path
    refused, so they pass on a machine without the cluster's file system."""

    def test_the_recorded_paths_are_not_read(self):
        recorded = {m for f in REF.rglob("*.json") for m in re.findall(r'"(/[^"\\]*\.model\.json)', f.read_text())}
        self.assertTrue(recorded)                                # the record has them, so the check means something
        refused = []
        real_open = io.open

        def guarded(file, *a, **k):
            name = os.fspath(file) if not isinstance(file, int) else ""
            if any(str(name).startswith(r.rsplit("/", 2)[0]) for r in recorded):
                refused.append(name)
                raise FileNotFoundError(name)
            return real_open(file, *a, **k)
        others = [c for c in (SpectrumAgainstBt, ChecksAgainstBt, CoordinatesAgainstBt, Boundary, RunNumbers)]
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in others)
        result = unittest.TestResult()
        with mock.patch("builtins.open", guarded), mock.patch("io.open", guarded):
            suite.run(result)
            by_the_tests = list(refused)
            with self.assertRaises(FileNotFoundError):          # the guard works: Path.read_text goes through it
                Path(sorted(recorded)[0]).read_text()
        self.assertEqual(by_the_tests, [])
        self.assertEqual(refused, [sorted(recorded)[0]])
        self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        self.assertGreater(result.testsRun, 10)


if __name__ == "__main__":
    unittest.main()
