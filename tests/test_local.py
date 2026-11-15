"""Tests that need no cluster: python -m unittest discover -s tests (from the repository root)."""
import json
import re
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"   # tools/make_test_fixture.py
sys.path.insert(0, str(ROOT / "src"))

from bootlab import bench, execute, inputs, pmp, skydive  # noqa: E402
from bootlab.numbers import Expr, Num, decimal_text, dumps, frac  # noqa: E402

EXAMPLE = ROOT / "examples" / "ising" / "run.toml"
POINT = {"deltaSig": "0.519", "deltaEps": "1.4", "theta": "0.956"}


class Numbers(unittest.TestCase):
    def test_decimal(self):
        self.assertEqual(decimal_text(Fraction(1, 2)), "0.5")
        self.assertEqual(decimal_text(frac("0.518149") + frac("1e-40")), "0.5181490000000000000000000000000000000001")
        self.assertEqual(decimal_text(Fraction(-3, 40)), "-0.075")
        self.assertEqual(decimal_text(Fraction(7)), "7")
        with self.assertRaises(ValueError):
            decimal_text(Fraction(1, 3))

    def test_expr(self):
        e = Expr("deltaEps + 2*(1/5) - -deltaSig/2")
        self.assertEqual(e.names, ["deltaEps", "deltaSig"])
        self.assertEqual(e.value({"deltaEps": frac("1.4"), "deltaSig": frac("0.5")}), frac("2.05"))
        with self.assertRaises(ValueError):
            Expr("sqrt(2)")

    def test_dumps_keeps_digits(self):
        digits = "0.51814900000000000000000000000000000000000000000000001"
        text = dumps({"x": Num(digits), "s": "1.0"})
        self.assertIn(f'"x": {digits}', text)
        self.assertIn('"s": "1.0"', text)


class Inputs(unittest.TestCase):
    def test_load_example(self):
        p = inputs.load(EXAMPLE)
        self.assertEqual(p.model_coords, ["deltaSig", "deltaEps", "theta"])
        self.assertEqual(p.coords, p.model_coords)
        self.assertEqual(p.spins, list(range(21)) + [49, 52])
        self.assertEqual(p.solver_precision(), 785)

    def test_spectrum_at_point(self):
        p = inputs.load(EXAMPLE)
        s = inputs.resolve_spectrum(p, POINT)
        self.assertEqual(s["twistGap"], "0")
        self.assertEqual(s["gaps"], [{"channel": "rep1+", "spins": [0], "gap": "3"},
                                     {"channel": "rep2+", "spins": [0], "gap": "3"}])

    def _with_spectrum(self, text, coords=None):
        d = Path(tempfile.mkdtemp())
        for f in ("model.json", "objective.toml"):
            (d / f).write_bytes((EXAMPLE.parent / f).read_bytes())
        (d / "spectrum.toml").write_text(text)
        run = EXAMPLE.read_text()
        if coords:      # the extra coordinate deltaEpsP: its start, box and Hessian guess
            run = run.replace('names = ["deltaSig", "deltaEps", "theta"]', f"names = {json.dumps(coords)}")
            for key, extra in (("start", "3.8"), ("box_min", "3.5"), ("box_max", "4.2"), ("hessian_guess", "100")):
                run = re.sub(rf'^({key}\s*=\s*\[.*)\]', rf'\1, "{extra}"]', run, count=1, flags=re.M)
        (d / "run.toml").write_text(run)
        return d / "run.toml"

    def test_point_parameters(self):
        spec = ('format = "bootlab-spectrum/1"\n[[gap]]\nchannel = "rep1+"\nspins = [0]\ndelta = "deltaEpsP"\n'
                '[[gap]]\nchannel = "rep2+"\nspins = [0]\ndelta = "deltaSig + 2"\n')
        p = inputs.load(self._with_spectrum(spec, ["deltaSig", "deltaEps", "theta", "deltaEpsP"]))
        self.assertEqual(p.coords[-1], "deltaEpsP")
        s = inputs.resolve_spectrum(p, {**POINT, "deltaEpsP": "3.8"})
        self.assertEqual(s["gaps"][0]["gap"], "3.8")
        self.assertEqual(s["gaps"][1]["gap"], "2.519")

    def test_errors(self):
        bad = ('format = "bootlab-spectrum/1"\n[[gap]]\nchannel = "rep2-"\nspins = [0]\ndelta = "3"\n'
               '[[gap]]\nchannel = "nope"\nspins = [0]\ndelta = "3"\n')
        with self.assertRaises(SystemExit) as e:
            inputs.load(self._with_spectrum(bad))
        self.assertIn("rep2- has odd spins", str(e.exception))
        self.assertIn("no channel 'nope'", str(e.exception))
        twice = ('format = "bootlab-spectrum/1"\n[[gap]]\nchannel = "rep1+"\nspins = "all"\nabove_unitarity = "0.1"\n'
                 '[[gap]]\nchannel = "rep1+"\nspins = [0]\ndelta = "3"\n')
        with self.assertRaises(SystemExit) as e:
            inputs.load(self._with_spectrum(twice))
        self.assertIn("set twice", str(e.exception))
        third = 'format = "bootlab-spectrum/1"\n[[gap]]\nchannel = "rep1+"\nspins = [0]\ndelta = "1/3"\n'
        p = inputs.load(self._with_spectrum(third))
        with self.assertRaises(ValueError):
            inputs.resolve_spectrum(p, POINT)

    def test_gap_limit_above_threshold(self):
        """simpleboot's limit 1e-12 with bt's dualityGapThreshold 1e-6 would end the first call as optimal."""
        path = self._with_spectrum((EXAMPLE.parent / "spectrum.toml").read_text())
        run = path.read_text().replace('gap_upper_limit            = "0.1"', 'gap_upper_limit            = "1e-12"')
        path.write_text(run)
        with self.assertRaises(SystemExit) as e:
            inputs.load(path)
        self.assertIn("dualityGapThreshold", str(e.exception))
        path.write_text(run.replace("[solver]\n", '[solver]\ndualityGapThreshold = "1e-20"\n'))
        self.assertEqual(inputs.load(path).run["solver"]["dualityGapThreshold"], "1e-20")

    def test_run_dir_keeps_its_inputs(self):
        """The run directory's run.toml names the run's own copies, whatever paths the original used."""
        from types import SimpleNamespace
        from bootlab import cli
        p = inputs.load(ROOT / "examples" / "ising-l19" / "run.toml")
        d = cli._new_run_dir(p, SimpleNamespace(runs=tempfile.mkdtemp()))
        q = inputs.load(d / "run.toml")
        self.assertEqual(q.run["inputs"], {"model": "model.json", "spectrum": "spectrum.toml",
                                           "objective": "objective.toml"})
        self.assertEqual((q.numerics["lambda"], q.solver_precision()), (19, 785))

    def test_request(self):
        p = inputs.load(EXAMPLE)
        out = Path(tempfile.mkdtemp())
        r = skydive.Run.__new__(skydive.Run)
        r.p, r.opts = p, p.run["skydive"]
        pts = r.points(POINT)
        self.assertEqual(list(pts), ["c", "plus_0", "plus_1", "plus_2"])
        self.assertEqual(pts["plus_1"]["deltaEps"], "1.40000000000000000001")
        path = pmp.write_request(p, out / "model.json", pts, out, out / "blocks", 128)
        text = path.read_text()
        self.assertIn("1.40000000000000000001", text)       # a bare JSON number with all its digits
        req = json.loads(text)
        self.assertEqual(req["model"], {"name": "data", "file": str(out / "model.json"), "lambda": 11,
                                        "blockPreset": "simpleboot"})
        self.assertEqual(req["objective"], "navigator")
        self.assertEqual(len(req["points"]), 4)


class Bench(unittest.TestCase):
    def test_grid(self):
        self.assertTrue(all(bench.grid_ok(p) for p in (64, 96, 120, 126, 128, 256)))
        self.assertFalse(any(bench.grid_ok(p) for p in (127, 254, 122)))

    def test_choices(self):
        b = bench.load_bench(FIXTURES / "sites" / "lab" / "bench", "amd1")
        self.assertEqual(bench.solver_procs(b, 128, 854, 88)[0], 128)     # ising nmax=18
        self.assertEqual(bench.solver_procs(b, 128, 274, 46)[0], 128)     # ising nmax=10
        self.assertEqual(bench.solver_procs(b, 128, 608, 151)[0], 96)     # the L=11 model: 128 not measured there
        self.assertEqual(bench.solver_procs({"solver": []}, 256, 854, 88)[0], 256)

    def test_same_sdpb_version_first(self):
        """The nearest problem among the program's SDPB version; all rows when that version has none."""
        def row(problem, n, blocks, program, procs, s):
            return {"problem": problem, "N": n, "blocks": blocks, "program": program, "nodes": 1,
                    "procs_per_node": procs, "s_per_iter": s, "source": "test"}
        b = {"solver": [row("big", 2261, 223, "dynamical_sdp RV1B 3.0sky", 256, 20.5),
                        row("big", 2261, 223, "dynamical_sdp RV1B 3.0sky", 128, 30.0),
                        row("small", 116, 61, "sdpb 3.1.0", 128, 0.29), row("small", 116, 61, "sdpb 3.1.0", 256, 0.35)]}
        self.assertEqual(bench.solver_procs(b, 256, 274, 46)[0], 128)                               # nearest of all
        self.assertEqual(bench.solver_procs(b, 256, 274, 46, "dynamical_sdp_RV1B_3.0sky")[0], 256)
        self.assertEqual(bench.solver_procs(b, 256, 274, 46, "sdpb_3.1.0")[0], 128)
        self.assertEqual(bench.solver_procs(b, 256, 274, 46, "sdpb_3.0sky")[0], 256)                # same version
        self.assertEqual(bench.solver_procs(b, 256, 274, 46, "sdpb_9.9")[0], 128)                   # none: all rows

    # Node memory as Node.here() reads it, less site.toml's driver_reserve_mb.
    MEM = {"amd1": 253099 - 4096, "amd2": 376000 - 4096}

    def test_pmp2sdp_procs(self):
        self.assertEqual(bench.pmp2sdp_procs(256, 8), 31)
        self.assertEqual(bench.pmp2sdp_procs(128, 8), 16)
        self.assertEqual(bench.pmp2sdp_procs(128, 4), 31)

    def test_pmp2sdp_at_once(self):
        """k from one instance's measured peak (MB): the L=23 model's 8 points stay 8 at once on both node types
        (25.2 GiB at 16 processes; 28.6 at 32); the L=27 model takes fewer (52.3 GiB at 16;
        on amd2 31 processes not measured, 55.6 at 64 stands in); 0 when not one fits."""
        self.assertEqual(bench.pmp2sdp_at_once(self.MEM["amd1"], 25.2 * 1024), 8)
        self.assertGreaterEqual(bench.pmp2sdp_at_once(self.MEM["amd2"], 28.6 * 1024), 8)
        self.assertEqual(bench.pmp2sdp_at_once(self.MEM["amd1"], 52.3 * 1024), 4)
        self.assertEqual(bench.pmp2sdp_at_once(self.MEM["amd2"], 55.6 * 1024), 5)
        self.assertEqual(bench.pmp2sdp_at_once(40000, 52.3 * 1024), 0)


class SdpWaves(unittest.TestCase):
    """execute.build_sdps with made-up steps: waves of at most `instances`, each step's memory read from a
    made-up cgroup (memory.current, memory.stat, memory.peak, memory.max), and the watchdog."""

    MB = 2 ** 20

    def run_waves(self, count, instances, current_mb, file_mb=300, shmem_mb=100):
        from types import SimpleNamespace
        from unittest import mock
        tmp = Path(tempfile.mkdtemp())
        job = tmp / "job_42"
        job.mkdir()
        (job / "step_batch").mkdir()
        site = SimpleNamespace(pmp2sdp="pmp2sdp", rank_env="r", sif="s", binds=[], driver_reserve_mb=0)
        node = execute.Node("h-1", "h", 128, 10000)
        started, cancelled = [], []

        class Step:
            def __init__(self, polls):
                self.polls = polls

            def poll(self):
                self.polls -= 1
                return None if self.polls > 0 else 0

            def wait(self):
                return 0

        def start(cmd, log, env=None):
            d = job / f"step_{len(started)}" / "user"       # where Slurm puts --mem
            d.mkdir(parents=True)
            mem = next(int(str(c)[6:-1]) for c in cmd if str(c).startswith("--mem="))
            (d / "memory.current").write_text(str(current_mb * self.MB))
            (d / "memory.stat").write_text(f"anon 1\nfile {file_mb * self.MB}\nshmem {shmem_mb * self.MB}\n")
            (d / "memory.peak").write_text(str((current_mb + 50) * self.MB))
            (d / "memory.max").write_text(str(mem * self.MB))
            started.append((log.name, mem))
            return Step(3)

        def scancel(argv, **kw):
            cancelled.append(argv[1])

        pmps = {f"p{i}": tmp / f"p{i}.nsv" for i in range(count)}
        with mock.patch.object(execute, "job_cgroup", lambda: job), \
                mock.patch.object(execute, "node_used_mb", lambda: 500.0), \
                mock.patch.object(execute, "start", start), mock.patch.object(execute, "SAMPLE_SECONDS", 0), \
                mock.patch.object(execute.subprocess, "run", scancel), \
                mock.patch.dict("os.environ", {"SLURM_JOB_ID": "42"}):
            try:
                return execute.build_sdps(site, node, pmps, tmp / "sdp", 1024, tmp / "logs", instances, 16), \
                    started, cancelled
            except execute.StepFailed as e:
                return e, started, cancelled

    def test_job_cgroup(self):
        """The directory above the first step_*: Lab's job directory has no job id in its name."""
        self.assertEqual(execute.job_cgroup_of("0::/system.slice/slurmstepd.scope/s0000000000000/step_batch/user/task_0\n"),
                         Path("/sys/fs/cgroup/system.slice/slurmstepd.scope/s0000000000000"))
        self.assertEqual(execute.job_cgroup_of("0::/system.slice/slurmstepd.scope/job_42/step_batch/user/task_0"),
                         Path("/sys/fs/cgroup/system.slice/slurmstepd.scope/job_42"))

    def test_waves_and_held_memory(self):
        """3 SDPs 2 at a time: two waves, --mem 5000 MB a step in the first; held = current - file + shmem."""
        (sdps, waves), started, _ = self.run_waves(3, 2, 1200)
        self.assertEqual(list(sdps), ["p0", "p1", "p2"])
        self.assertEqual([w["names"] for w in waves], [["p0", "p1"], ["p2"]])
        self.assertEqual(started, [("pmp2sdp-p0", 5000), ("pmp2sdp-p1", 5000), ("pmp2sdp-p2", 5000)])
        self.assertEqual((waves[0]["held_mb"], waves[0]["cgroup_peak_mb"], waves[0]["node_mb"]), (1000, 1250, 0))

    def test_watchdog_cancels_the_wave(self):
        """A step holding more than 95% of its --mem: its whole wave is cancelled and the step fails."""
        e, started, cancelled = self.run_waves(2, 2, 5000, file_mb=10, shmem_mb=10)    # 5000 MB held of 5000
        self.assertIsInstance(e, execute.StepFailed)
        self.assertIn("held 5000 MB of its 5000 MB", str(e))
        self.assertEqual(sorted(cancelled), ["42.0", "42.1"])


class Outputs(unittest.TestCase):
    def test_read_dive(self):
        d = Path(tempfile.mkdtemp())
        (d / "out.txt").write_text('terminateReason = "met criteria to update sdp input files";\n'
                                   'primalObjective = -0.0015;\ndualObjective   = -0.0016;\ndualityGap      = 8e-7;\n')
        (d / "skydiving_out.txt").write_text("dualStepSize       = 1;\nprimalStepSize     = 0.5;\n"
                                             "NavigatorValue     = -2.5e-6;\nfindMinimumQ       = 1;\nbeta = 1;\n"
                                             "climbedQ           = 0;\nSolver runtime     = 3;\n")
        (d / "externalParamStep.txt").write_text("3 1\n1e-4\n-2e-4\n3e-3\n")
        (d / "gradient.txt").write_text("3 1\n1\n2\n3\n")
        (d / "hessBFGS.txt").write_text("3 3\n1\n0\n0\n0\n2\n0\n0\n0\n3\n")
        (d / "iterations.txt").write_text("42\n")
        v = execute.read_dive(d, 0)
        self.assertEqual((v.verdict, v.reason, v.navigator), ("stepped", "met criteria to update sdp input files", "-2.5e-6"))
        self.assertEqual(v.ext_step, ["1e-4", "-2e-4", "3e-3"])
        self.assertEqual(v.hessian, [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "3"]])
        self.assertEqual(v.iterations, 42)
        self.assertTrue(v.find_minimum)
        self.assertFalse(v.climbed)
        (d / "out.txt").write_text('terminateReason = "found primal-dual optimal solution";\n'
                                   'primalObjective = -1.1154e-11;\ndualObjective   = -1.1155e-11;\n'
                                   'dualityGap      = 8.07e-16;\n')
        (d / "skydiving_out.txt").write_text("NavigatorValue     = 0;\nfindMinimumQ       = 0;\nclimbedQ = 0;\n")
        v = execute.read_dive(d, 0)
        self.assertEqual((v.verdict, v.navigator, v.primal, v.dual), ("optimal", None, "-1.1154e-11", "-1.1155e-11"))
        (d / "out.txt").write_text('terminateReason = "maxIterations exceeded";\n')
        self.assertEqual(execute.read_dive(d, 0).reason, "unfinished: maxIterations exceeded")
        (d / "out.txt").unlink()
        self.assertEqual(execute.read_dive(d, 139).verdict, "aborted")

    def test_resume_state(self):
        p = inputs.load(EXAMPLE)
        d = Path(tempfile.mkdtemp())
        step1 = {"step": 1, "x": ["0.519", "1.4", "0.956"], "verdict": "stepped", "navigator": "1e-3",
                 "ext_step": ["1e-4", "-2e-4", "3e-3"], "gradient": ["1", "2", "3"],
                 "hessian": [["1", "0", "0"], ["0", "2", "0"], ["0", "0", "3"]], "iterations": 80,
                 "boundary_on": False, "procs": 128}
        step2 = {"step": 2, "x": ["0.5191", "1.3998", "0.959"], "verdict": "aborted",
                 "reason": "unfinished: maxIterations exceeded", "boundary_on": False, "procs": 128}
        (d / "solves.jsonl").write_text(json.dumps(step1) + "\n")
        s = skydive.resume_state(p, d, hot=True)
        self.assertEqual(s["x"], {"deltaSig": "0.5191", "deltaEps": "1.3998", "theta": "0.959"})
        self.assertEqual((s["n"], s["bfgs"]["total_iterations"], s["procs"]), (1, 80, 128))
        (d / "solves.jsonl").write_text(json.dumps(step1) + "\n" + json.dumps(step2) + "\n")
        s = skydive.resume_state(p, d, hot=True)
        self.assertEqual(s["x"]["deltaSig"], "0.5191")
        self.assertEqual(s["n"], 1)                 # step 2 is done again
        self.assertEqual(s["bfgs"]["prev_hessian"], ["1", "0", "0", "0", "2", "0", "0", "0", "3"])


if __name__ == "__main__":
    unittest.main()
