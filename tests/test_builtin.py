"""Built-in models and the scan (docs/en/interfaces.md, section 19), without a cluster:
python -m unittest discover -s tests (from the repository root)."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import builtin, inputs, scan  # noqa: E402

EXAMPLE = ROOT / "examples" / "jjjj"


def variant(edit_files: dict):
    """A copy of examples/jjjj with files replaced or edited: name -> new text, or (old, new) to replace."""
    d = Path(tempfile.mkdtemp())
    for f in EXAMPLE.iterdir():
        shutil.copy(f, d / f.name)
    for name, change in edit_files.items():
        p = d / name
        if isinstance(change, tuple):
            text = p.read_text()
            assert change[0] in text, (name, change[0])
            p.write_text(text.replace(*change))
        else:
            p.write_text(change)
    return d / "run.toml"


class Load(unittest.TestCase):
    def test_example(self):
        p = inputs.load(EXAMPLE / "run.toml")
        self.assertEqual(p.coords, ["x"])
        self.assertEqual(p.model_coords, [])
        self.assertEqual(p.spins, list(range(41)))
        self.assertEqual(p.gap_spins, [[2]])

    def errors_of(self, edit):
        with self.assertRaises(SystemExit) as e:
            inputs.load(variant(edit))
        return str(e.exception)

    def test_errors(self):
        cases = [
            ({"spectrum.toml": ('channel = "P+"', 'channel = "rep1+"')}, "there is no channel 'rep1+'"),
            ({"objective.toml": ('kind      = "ope_bound"', 'kind = "navigator"')}, "kind 'navigator' is not among"),
            ({"objective.toml": ('["1", "x"]', '["1"]')}, "lambda needs 2 entries"),
            ({"objective.toml": ('direction = "upper"', '')}, "ope_bound needs direction"),
            ({"objective.toml": ('["1", "x"]', '[1.0, "x"]')}, "write lambda[0] as a string"),
            ({"run.toml": ('points = [["0"], ["-1"], ["1"]]', 'points = [[0.5]]')}, "points[0]"),
            ({"run.toml": ('points = [["0"], ["-1"], ["1"]]', 'points = [["0", "1"]]')}, "needs 1 values"),
            ({"run.toml": ('names = ["x"]', 'names = ["y"]')}, "coords.names is ['y']"),
            ({"run.toml": ("num_poles       = 20\n", "")}, "missing num_poles"),
            ({"run.toml": ("max_spin        = 40\n", "max_spin = 40\nblock_preset = \"hyperion\"\n")},
             "unknown block_preset"),
        ]
        for edit, msg in cases:
            with self.subTest(msg=msg):
                self.assertIn(msg, self.errors_of(edit))

    def test_scan_needs_builtin(self):
        """[scan] with a bt-model/2 model is an error."""
        d = Path(tempfile.mkdtemp())
        for f in (ROOT / "examples" / "ising").iterdir():
            shutil.copy(f, d / f.name)
        p = d / "run.toml"
        p.write_text(p.read_text() + '\n[scan]\npoints = [["0.518", "1.41", "0.97"]]\n')
        with self.assertRaises(SystemExit) as e:
            inputs.load(p)
        self.assertIn("[scan] is for built-in models", str(e.exception))


class Request(unittest.TestCase):
    def test_request(self):
        """The bound, upstream's SimpleBoundParams, each point's spectrum with resolved spins and its objective,
        numbers as bare exact JSON numbers."""
        p = inputs.load(variant({"spectrum.toml": ('delta   = "3.5"', 'delta   = "x + 4.5"')}))
        with tempfile.TemporaryDirectory() as d:
            text = builtin.write_request(p, {"p": {"x": "-1.0000000000000000000000000001"}}, Path(d), Path("/b"), 256).read_text()
        self.assertIn("-1.0000000000000000000000000001", text)
        self.assertIn('"gap": 3.4999999999999999999999999999\n', text)   # x + 4.5, exact, as a number
        req = json.loads(text)
        self.assertEqual(req["bound"], "JJJJ3d")
        self.assertEqual(req["numerics"], {"nmax": 6, "order": 60, "numPoles": 20, "maxSpin": 40, "precision": 768})
        self.assertEqual(req["cores"], 256)
        pt = req["points"][0]
        self.assertEqual(pt["name"], "p")
        self.assertEqual(pt["coords"], [])
        self.assertEqual(pt["spectrum"]["gaps"], [{"channel": "P+", "spins": [2], "gap": 3.5}])
        self.assertEqual(pt["spectrum"]["isolated"], [])
        self.assertEqual(pt["spectrum"]["twistGap"], 0)
        self.assertEqual(pt["objective"]["kind"], "ope_bound")
        self.assertEqual(pt["objective"]["direction"], "upper")
        self.assertEqual(len(pt["objective"]["lambda"]), 2)

    def test_feasibility(self):
        p = inputs.load(variant({"objective.toml": 'format = "bootlab-objective/1"\nkind = "feasibility"\n',
                                 "run.toml": ('names = ["x"]\n\n[scan]\npoints = [["0"], ["-1"], ["1"]]',
                                              'names = []\n\n[scan]\npoints = [[]]')}))
        self.assertEqual(p.coords, [])
        with tempfile.TemporaryDirectory() as d:
            req = json.loads(builtin.write_request(p, {"p": {}}, Path(d), Path("/b"), 8).read_text())
        self.assertEqual(req["points"][0]["objective"], {"kind": "feasibility"})


class GeneratorEnv(unittest.TestCase):
    def test_env(self):
        """The generator's own variables with {root} as the install root; the memory and no node-local paths."""
        env = builtin.generator_env({"env": {"HYP_SITE_DIR": "{root}/gen/current/site", "CUR_ROOT": "/c"}},
                                    1000, Path("/l"))
        self.assertEqual(env["HYP_SITE_DIR"], str(ROOT) + "/gen/current/site")
        self.assertEqual((env["CUR_ROOT"], env["CUR_MEM_MB"], env["CUR_LOCAL_PATHS"]), ("/c", "1000", ""))
        self.assertEqual(env["CUR_LOCAL_DIR"], str(Path("/l")))
        self.assertEqual(env["HYP_DATA_DIR"], str(Path("/l") / "hyp-data"))     # none given: under local_dir
        self.assertEqual(builtin.generator_env({"env": {"HYP_DATA_DIR": "/d"}}, 1, Path("/l"))["HYP_DATA_DIR"], "/d")
        self.assertNotIn("CUR_ROOT", builtin.generator_env({}, 1, Path("/l")).keys() - os.environ.keys())

    def test_container(self):
        """container = true: the generator runs in the site's image; left out, on the host."""
        from types import SimpleNamespace
        from unittest import mock
        site = SimpleNamespace(driver_reserve_mb=0, binds=["/b"], sif="/i.sif")
        node = SimpleNamespace(mem_mb=1000, cores=8, solo=lambda: None)
        for gen, image in (({"program": "/opt/g", "container": True}, True), ({"program": "/h/g", "env": {}}, False)):
            with mock.patch.object(builtin.execute, "start") as start:
                start.return_value.wait.return_value = 0
                builtin.generate(site, gen, node, Path("/r.json"), Path("/l"), Path("/log"))
            cmd = start.call_args[0][0]
            self.assertEqual("/i.sif" in cmd, image)
            self.assertEqual(cmd[-3:], [gen["program"], "pmp", str(Path("/r.json"))])

    def test_versions(self):
        """result.json's versions: the image's sha256; a host generator's own, none for one in the image (its
        path is not on the host)."""
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            sif, host = Path(d) / "i.sif", Path(d) / "gen"
            sif.write_bytes(b"image")
            host.write_bytes(b"generator")
            for gen in ({"program": "/opt/bootlab/gen/current/bin/generator", "container": True},
                        {"program": str(host)}):
                s = scan.Scan.__new__(scan.Scan)
                s.site, s.gen = SimpleNamespace(sif=str(sif)), gen
                v = s.versions()
                self.assertEqual(v["sif_sha256"], hashlib.sha256(b"image").hexdigest())
                if gen.get("container"):
                    self.assertEqual(v["generator_in"], "image")
                    self.assertNotIn("generator_sha256", v)
                else:
                    self.assertEqual(v["generator_sha256"], hashlib.sha256(b"generator").hexdigest())


class SolverArgs(unittest.TestCase):
    def test_merge(self):
        argv = ["--precision", "768", "--maxRuntime", "9223372036", "--dualityGapThreshold", "1.0e-30",
                "--writeSolution", "x,y", "--detectPrimalFeasibleJump", "--maxIterations", "500"]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "sdpb.args.json"
            f.write_text(json.dumps(argv))
            out = scan.solver_args(f, {"dualityGapThreshold": "1e-20", "detectPrimalFeasibleJump": False,
                                       "findDualFeasible": True}, 3600)
        self.assertEqual(out, ["--precision", "768", "--maxRuntime", "3600", "--dualityGapThreshold", "1e-20",
                               "--writeSolution", "x,y", "--maxIterations", "500", "--findDualFeasible"])

    def test_options(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "sdpb.args.json"
            f.write_text(json.dumps(["--precision", "768", "--detectPrimalFeasibleJump", "--writeSolution", "z"]))
            self.assertEqual(builtin.solver_options(f),
                             {"precision": "768", "detectPrimalFeasibleJump": True, "writeSolution": "z"})


SKYDIVE = EXAMPLE / "skydive.toml"
# what the generator writes for <JJJJ>: the bound's options, upstream's jjjj3dBound
BOUND_ARGS = ["--precision", "768", "--checkpointInterval", "3600", "--maxRuntime", "72000", "--dualityGapThreshold",
              "1.0e-30", "--primalErrorThreshold", "1.0e-30", "--dualErrorThreshold", "1.0e-30",
              "--initialMatrixScalePrimal", "1.0e20", "--initialMatrixScaleDual", "1.0e20",
              "--feasibleCenteringParameter", "0.1", "--infeasibleCenteringParameter", "0.3", "--stepLengthReduction",
              "0.7", "--maxComplementarity", "1.0e100", "--maxIterations", "1000", "--verbosity", "1",
              "--writeSolution", "z"]


def skydive_variant(edit: str, new: str = ""):
    """examples/jjjj with skydive.toml as its run.toml, one text replaced (edit -> new) or appended (edit "")."""
    text = SKYDIVE.read_text()
    return variant({"run.toml": text.replace(edit, new) if edit else text + new})


class SkydiveLoad(unittest.TestCase):
    def test_example(self):
        p = inputs.load(SKYDIVE)
        self.assertTrue(p.builtin)
        self.assertEqual((p.coords, p.solver_precision()), (["x"], 768))

    def test_errors(self):
        cases = [
            (skydive_variant("", '\n[scan]\npoints = [["0"]]\n'), "either [scan]"),
            (skydive_variant('[skydive]', '[skydive]\nfind_boundary_direction = ["1"]'), "no boundary at N = 0"),
            (skydive_variant('[skydive]', '[skydive]\nstop_on_negative = true'), "stop_on_negative"),
            (skydive_variant('[solver]', '[solver]\ndualityGapThreshold = "0.1"'), "not above [solver]"),
        ]
        for path, msg in cases:
            with self.subTest(msg=msg):
                with self.assertRaises(SystemExit) as e:
                    inputs.load(path)
                self.assertIn(msg, str(e.exception))

    def test_mixed_precision_request(self):
        """A mixed precision call below block_precision: the bound (blocks and solver) at the call's precision."""
        p = inputs.load(skydive_variant('[skydive]', '[skydive]\nmixed_precision = true'))
        with tempfile.TemporaryDirectory() as d:
            for bits, want in ((512, 512), (768, 768), (None, 768)):
                req = json.loads(builtin.write_request(p, {"c": {"x": "0.5"}}, Path(d), Path("/b"), 8, bits).read_text())
                self.assertEqual(req["numerics"]["precision"], want)

    def test_feasibility_has_no_objective(self):
        path = skydive_variant("")
        (path.parent / "objective.toml").write_text('format = "bootlab-objective/1"\nkind = "feasibility"\n')
        (path.parent / "run.toml").write_text(path.read_text().replace('names = ["x"]', "names = []")
                                              .replace('["0.5"]', "[]").replace('["-2"]', "[]").replace('["2"]', "[]"))
        with self.assertRaises(SystemExit) as e:
            inputs.load(path)
        self.assertIn("has none (use ope_bound)", str(e.exception))


class SkydiveStep(unittest.TestCase):
    """A built-in model's step with its programs replaced: the card's generator, the generator's pmp2sdp, the
    site's dynamical_sdp (or the generator's), and the bound's own solver options under [solver]."""

    def setUp(self):
        from types import SimpleNamespace
        self.tmp = Path(tempfile.mkdtemp())
        self.gen = {"program": str(self.tmp / "gen"), "pmp2sdp": "pmp2sdp_3.1.0", "sdpb": "sdpb_3.1.0"}
        (self.tmp / "gen").write_bytes(b"generator")
        self.site = SimpleNamespace(scratch=str(self.tmp / "scratch"), generator="/opt/bootlab/bin/generator",
                                    sif="s", pmp2sdp="pmp2sdp_3.0sky", dynamical_sdp="dynamical_sdp_3.0sky",
                                    driver_reserve_mb=0, bench_dir=str(self.tmp / "bench"),
                                    generators={"current": self.gen})
        self.dir = self.tmp / "run"
        self.dir.mkdir()

    def step(self, args=BOUND_ARGS, solver=None):
        from unittest import mock
        from bootlab import execute, skydive
        p = inputs.load(SKYDIVE)
        if solver:
            p.run["solver"] = solver
        r = builtin.Skydive(p, self.site, execute.Node("h-1", "h", 8, 100000), self.dir)
        seen = {}

        def generate(site, gen, node, request, local_dir, log, stall_s=None):
            seen["generator"] = gen["program"]
            seen["generated"] = seen.get("generated", 0) + 1
            seen["points"] = json.loads(request.read_text(), parse_float=str)["points"]
            for pt in seen["points"]:
                d = request.parent / pt["name"]
                d.mkdir()
                (d / "part.json").write_text("{}")
                (d / "pmp.nsv").write_text(f"{d / 'part.json'}\0")
                (d / "sdpb.args.json").write_text(json.dumps(args))
            log.parent.mkdir(parents=True, exist_ok=True)

        def build_sdps(site, node, lists, sdp_root, precision, log_dir, instances, procs, program=None):
            seen.setdefault("pmp2sdp", set()).add((program, precision))
            return ({n: sdp_root / n for n in lists},
                    [{"names": list(lists), "held_mb": 1000, "cgroup_peak_mb": 1000, "node_mb": 1000}])

        class Stop(Exception):
            pass

        def run_dynamical_sdp(site, node, procs, center, neighbors, out, ck, solver_, dynamical, state, log,
                              program=None):
            seen.update(solver=solver_, dynamical=dynamical, program=program, neighbors=len(neighbors))
            raise Stop

        with mock.patch.object(builtin, "generate", generate), \
                mock.patch.object(skydive.execute, "build_sdps", build_sdps), \
                mock.patch.object(skydive.execute, "run_dynamical_sdp", run_dynamical_sdp), \
                mock.patch.object(skydive.execute, "sdp_size", lambda sdp: (116, 61)), \
                mock.patch.object(skydive.execute, "job_seconds_left", lambda: 10 ** 6):
            seen["threshold"] = r.threshold_setting(0, {"x": "0.5"}, 768)    # step 1 made ahead
            with self.assertRaises(Stop):
                r.step(1, {"x": "0.5"}, {"gap_threshold": "1e-30", "precision": 768, "total_iterations": 0,
                                         "gap_upper_limit": "0.1", "box_min": ["-2"], "box_max": ["2"]}, 128)
            self.assertEqual(seen["generated"], 1)          # the PMPs made ahead are the step's
        return r, seen

    def test_programs_and_options(self):
        r, seen = self.step(solver={"maxIterations": 500})
        self.assertEqual(seen["generator"], self.gen["program"])
        self.assertEqual(seen["threshold"], "1.0e-30")      # the bound's
        self.assertEqual([pt["name"] for pt in seen["points"]], ["c", "plus_0"])
        self.assertEqual(seen["points"][1]["objective"]["lambda"][1], "0.50000000000000000001")
        self.assertEqual(seen["pmp2sdp"], {("pmp2sdp_3.1.0", 768)})
        self.assertEqual((seen["program"], seen["neighbors"]), ("dynamical_sdp_3.0sky", 1))
        s = seen["solver"]
        self.assertEqual((str(s["precision"]), s["feasibleCenteringParameter"], s["writeSolution"],
                          s["initialMatrixScaleDual"], s["maxIterations"], s["dualityGapThreshold"]),
                         ("768", "0.1", "z", "1.0e20", 500, "1e-30"))
        self.assertEqual(s["maxRuntime"], 10 ** 6 - 900)
        self.assertNotIn("procGranularity", s)          # bt's Skydive defaults are not used
        self.assertEqual(seen["dynamical"]["externalParamInfinitestimal"], "1e-20")
        self.assertEqual((self.dir / "steps" / "1" / "sdpb.args.json").read_text(), json.dumps(BOUND_ARGS))

    def test_parts_kept_between_steps(self):
        """Two steps write into the run's one PMP directory (pmp_dir): with a generator that skips the parts that
        exist, as upstream's scheduler does, the second step writes only the parts that depend on x (the objective's,
        of its two points), and its lists name the parts kept from the first."""
        from unittest import mock
        from bootlab import execute
        r = builtin.Skydive(inputs.load(SKYDIVE), self.site, execute.Node("h-1", "h", 8, 100000), self.dir)
        written = []

        def generate(site, gen, node, request, local_dir, log, stall_s=None):
            req = json.loads(request.read_text(), parse_float=str)
            parts = Path(req["out"]) / "json"
            parts.mkdir(parents=True, exist_ok=True)
            for pt in req["points"]:
                names = [f"constraint_{i}.json" for i in range(3)] + [f"objective_{pt['objective']['lambda'][1]}.json"]
                for name in names:
                    if not (parts / name).exists():
                        (parts / name).write_text("{}")
                        written.append(name)
                d = Path(req["out"]) / pt["name"]
                d.mkdir(exist_ok=True)
                (d / "pmp.nsv").write_text("".join(f"{parts / n}\0" for n in names))
                (d / "sdpb.args.json").write_text(json.dumps(BOUND_ARGS))
            log.parent.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(builtin, "generate", generate):
            r.make_pmps(1, {"x": "0.5"}, 768)
            self.assertEqual(len(written), 5)               # 3 constraints, 2 objectives (c, plus_0)
            written.clear()
            _, _, lists = r.make_pmps(2, {"x": "0.49"}, 768)
        self.assertEqual(sorted(written), ["objective_0.49.json", "objective_0.49000000000000000001.json"])
        self.assertEqual(lists["c"], self.tmp / "scratch" / "run" / "pmp" / "c" / "pmp.nsv")
        self.assertIn(str(r.pmp_dir / "json" / "constraint_0.json"), lists["c"].read_text())

    def test_generator_names_its_solver(self):
        self.gen["dynamical_sdp"] = "dynamical_sdp_3.1"
        self.assertEqual(self.step()[1]["program"], "dynamical_sdp_3.1")

    def test_precision_not_the_bounds(self):
        from bootlab import execute
        args = list(BOUND_ARGS)
        args[1] = "1024"
        with self.assertRaises(execute.StepFailed) as e:
            self.step(args)
        # made ahead by threshold_setting, raised at the step (the loop records it as a failed step)
        self.assertIn("solver precision 1024 is not 768", str(e.exception))

    def test_loop_ends_at_the_optimum(self):
        """Two steps with made-up dives: the gap threshold is the bound's (1.0e-30 as written: 1e-3 |N| is above
        it), the limit stays 0.1 (|N| about 4.4), and optimal with both objectives negative is the bound's
        optimum, not "allowed"."""
        from unittest import mock
        from bootlab import execute, skydive
        p = inputs.load(SKYDIVE)
        for name in ("model.json", "spectrum.toml", "objective.toml"):
            shutil.copy(EXAMPLE / name, self.dir / name)
        r = builtin.Skydive(p, self.site, execute.Node("h-1", "h", 8, 100000), self.dir)
        dives = [execute.Dive("stepped", "met criteria to update sdp input files", primal="-4.42", dual="-4.49",
                              gap="0.008", navigator="-4.435", ext_step=["-0.004"], gradient=["1.27"],
                              hessian=[["300"]], iterations=198, alpha_p="0.5", alpha_d="0.5", beta="0.3",
                              climbed=False, find_minimum=True),
                 execute.Dive("optimal", "found primal-dual optimal solution", primal="-4.4841", dual="-4.4841",
                              gap="1e-31", iterations=260)]
        seen = []

        def make_pmps(n, x, bits):
            r.bound_solver = builtin.solver_options_from(BOUND_ARGS)
            return None

        def step(n, x, st, procs):
            seen.append(dict(st))
            sdir = r.scratch / f"step_{n}"
            (sdir / "out").mkdir(parents=True)
            (sdir / "out" / "out.txt").write_text("out")
            nsv = sdir / "pmp" / "c" / "pmp.nsv"
            nsv.parent.mkdir(parents=True)
            (nsv.parent / "part.json").write_text("{}")
            nsv.write_text(f"{nsv.parent / 'part.json'}\0")
            (self.dir / "steps" / str(n)).mkdir(parents=True, exist_ok=True)
            return dives[n - 1], 128, {}, sdir, nsv

        s = skydive.start_state(p)
        with mock.patch.object(r, "make_pmps", make_pmps), mock.patch.object(r, "step", step):
            r.loop(s["x"], s["n"], s["bfgs"], s["boundary_on"], s["procs"])
        result = json.loads((self.dir / "result.json").read_text())
        self.assertEqual((result["status"], result["reason"], result["x"]), ("converged", "optimum", ["0.496"]))
        self.assertEqual([st["gap_threshold"] for st in seen], ["1.0e-30", "1.0e-30"])
        self.assertEqual([st["gap_upper_limit"] for st in seen], ["0.1", "0.1"])
        self.assertEqual(result["versions"]["pmp2sdp"], "pmp2sdp_3.1.0")
        self.assertEqual(result["versions"]["generator_sha256"], hashlib.sha256(b"generator").hexdigest())


if __name__ == "__main__":
    unittest.main()
