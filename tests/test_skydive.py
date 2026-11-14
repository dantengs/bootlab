"""The dynamical_sdp command line of a Skydive step against bt r3.8's, and the loop with made-up steps,
without a cluster: python -m unittest discover -s tests (from the repository root)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"   # tools/make_test_fixture.py
sys.path.insert(0, str(ROOT / "src"))

from bootlab import execute, inputs, skydive  # noqa: E402

EXAMPLE = ROOT / "examples" / "ising" / "run.toml"
LEFT_OUT = object()                # run_loop(track=LEFT_OUT): no gap_upper_limit_track in [skydive]

# What bt r3.8 passed to dynamical_sdp_RV1B_3.0sky at the first step of a reference run (ising-oe, Lambda 11,
# start (0.519, 1.4, 0.956)): SDPB's options as dynamical_sdp printed them, and the options bt's
# getDynamicalSdpArgs adds, with SkydiveBT's upstreamOptions. Paths and maxRuntime left out (bt: 72000 s).
BT_REFERENCE = {
    "precision": "785", "checkpointInterval": "3600", "dualityGapThreshold": "1e-6",
    "primalErrorThreshold": "1e-12", "dualErrorThreshold": "1e-12", "initialMatrixScalePrimal": "1e20",
    "initialMatrixScaleDual": "1e20", "feasibleCenteringParameter": "0.3", "infeasibleCenteringParameter": "0.3",
    "stepLengthReduction": "0.7", "maxComplementarity": "1e100", "maxIterations": "1000", "verbosity": "1",
    "procGranularity": "1", "writeSolution": "y,x",
    "numExternalParams": "3", "externalParamInfinitestimal": "1e-20", "centeringRThreshold": "1e-20",
    "dualityGapUpperLimit": "0.1", "betaScanMin": "0.1", "betaScanMax": "1.01", "betaScanStep": "0.1",
    "stepMinThreshold": "0.1", "stepMaxThreshold": "0.6", "primalDualObjWeight": "0.2", "maxClimbingSteps": "1",
    "betaClimbing": "2", "navigatorWithLogDetX": "False", "gradientWithLogDetX": "True",
    "finiteDualityGapTarget": "0", "totalIterationCount": "0", "BFGSPartialUpdate": "-1",
    "boundingBoxMax": ["0.530", "1.5", "0.99"], "boundingBoxMin": ["0.517", "1.2", "0.95"],
    "useExactHessian": "False", "prevHessianBFGS": ["1000000", "0", "0", "0", "10000", "0", "0", "0", "1000"],
}


def options(argv):
    """--flag value... -> {flag: value or [values]}."""
    out, key = {}, None
    for a in argv:
        if a.startswith("--"):
            key = a[2:]
            out[key] = []
        else:
            out[key].append(a)
    return {k: v[0] if len(v) == 1 else v for k, v in out.items()}


class FirstStep(unittest.TestCase):
    def test_command_line_is_bt_r38(self):
        p = inputs.load(EXAMPLE)
        o = p.run["skydive"]
        site = execute.load_site("lab", FIXTURES)
        node = execute.Node("nodea-1", "amd1", 128, 253089)
        state = {"total_iterations": 0, "gap_upper_limit": o["gap_upper_limit"], "boundary_direction": None,
                 "box_min": o["box_min"], "box_max": o["box_max"],
                 "hessian_guess": [o["hessian_guess"][i] if i == j else "0" for i in range(3) for j in range(3)]}
        solver = {"precision": p.solver_precision(), **skydive.solver_defaults((p.numerics["lambda"] + 1) // 2),
                  "maxRuntime": 3600, **p.run["solver"]}
        dynamical = {**skydive.DYNAMICAL_DEFAULTS, **p.run["dynamical"],
                     "externalParamInfinitestimal": o["infinitesimal"]}
        captured = {}

        class Done:
            def wait(self):
                return 0

        def start(cmd, log, env=None):
            captured["cmd"] = [str(c) for c in cmd]
            return Done()

        with mock.patch.object(execute, "start", start), mock.patch("pathlib.Path.write_text"):
            execute.run_dynamical_sdp(site, node, 128, Path("/c"), [Path("/p0"), Path("/p1"), Path("/p2")],
                                      Path("/o/out"), Path("/o/ck"), solver, dynamical, state, Path("/l/d"))
        cmd = captured["cmd"]
        got = options(cmd[cmd.index(site.dynamical_sdp) + 1:])
        for k in ("sdpDir", "newSdpDirs", "outDir", "checkpointDir", "maxRuntime"):
            got.pop(k)
        self.assertEqual(got, BT_REFERENCE)

    def test_solver_defaults_by_nmax(self):
        self.assertEqual(skydive.solver_defaults(6)["initialMatrixScalePrimal"], "1e20")
        self.assertEqual(skydive.solver_defaults(7)["checkpointInterval"], 1800)
        self.assertEqual(skydive.solver_defaults(14)["maxComplementarity"], "1e130")
        self.assertEqual(skydive.solver_defaults(18)["maxComplementarity"], "1e160")
        self.assertEqual(skydive.solver_defaults(22)["initialMatrixScaleDual"], "1e60")
        self.assertEqual(skydive.solver_defaults(22)["dualityGapThreshold"], "1e-6")


O5_L23 = ROOT / "tests" / "fixtures" / "runs" / "seven" / "run-l23.toml"   # 7 coordinates: 8 points a step
GIB = 1024                                                         # MB


class Pmp2sdpMemoryInStep(unittest.TestCase):
    """A run's first step converts the central SDP alone, measures it, and converts the others k at a time;
    later steps convert k at a time without measuring (docs/en/interfaces.md section 9). build_sdps is made up: it returns
    the held peak it is given, and the solver call stops the step."""

    def steps(self, node, held_mb, n_steps=2, sdp_mem=None):
        import dataclasses
        tmp = Path(tempfile.mkdtemp())
        site = dataclasses.replace(execute.load_site("lab", FIXTURES), scratch=str(tmp / "scratch"))
        run_dir = tmp / "run"
        run_dir.mkdir()
        p = inputs.load(O5_L23)
        r = skydive.Run(p, site, node, run_dir)
        r.sdp_mem = sdp_mem
        calls = []

        def generate(site_, node_, request, log):
            out = request.parent
            for point in json.loads(request.read_text())["points"]:
                d = out / point["name"]
                d.mkdir()
                (d / "part.json").write_text("{}")
                (d / "pmp.nsv").write_text(f"{d / 'part.json'}\0")

        def build_sdps(site_, node_, lists, sdp_root, precision, log_dir, instances, procs):
            calls.append((len(lists), instances, procs))
            return ({name: sdp_root / name for name in lists},
                    [{"names": list(lists), "held_mb": held_mb, "cgroup_peak_mb": held_mb + 2600, "node_mb": held_mb}])

        class Stop(Exception):
            pass

        def solver(*args, **kw):
            raise Stop

        x = dict(zip(p.coords, p.run["skydive"]["start"]))
        with mock.patch.object(skydive.pmp, "generate", generate), \
                mock.patch.object(skydive.execute, "build_sdps", build_sdps), \
                mock.patch.object(skydive.execute, "run_dynamical_sdp", solver), \
                mock.patch.object(skydive.execute, "job_seconds_left", lambda: 10 ** 6):
            for n in range(1, n_steps + 1):
                with self.assertRaises(Stop):
                    r.step(n, x, {"gap_threshold": "1e-30", "precision": p.solver_precision()}, 128)
        return calls, r, (run_dir / "driver.log").read_text()

    def test_l23_amd1(self):
        """the 7-coordinate model at L=23 at 25.2 GiB an instance (16 processes): first step 1 + 7, then 8 at once."""
        calls, r, log = self.steps(execute.Node("h-1", "amd1", 128, 253099), 25.2 * GIB)
        self.assertEqual(calls, [(1, 1, 16), (7, 7, 16), (8, 8, 16)])
        self.assertEqual((r.sdp_mem["k"], r.sdp_mem["measured_at_step"]), (8, 1))
        self.assertIn("to measure its memory", log)
        self.assertIn("k = 8", log)

    def test_l23_amd2(self):
        """the 7-coordinate model at L=23 at 28.6 GiB an instance: first step 1 + 7, then 8 at once, 31 processes each."""
        calls, r, _ = self.steps(execute.Node("h-1", "amd2", 256, 376000), 28.6 * GIB)
        self.assertEqual(calls, [(1, 1, 31), (7, 7, 31), (8, 8, 31)])

    def test_l27_in_rounds(self):
        """the 7-coordinate model at L=27 at 52.3 GiB (16 processes on amd1): k = 4, so 1 + 4 at a time, then 4 at a time."""
        calls, r, _ = self.steps(execute.Node("h-1", "amd1", 128, 253099), 52.3 * GIB)
        self.assertEqual(calls, [(1, 1, 16), (7, 4, 16), (8, 4, 16)])

    def test_not_one_fits(self):
        with self.assertRaises(execute.StepFailed):
            self.steps(execute.Node("h-1", "amd1", 128, 253099), 250 * GIB, n_steps=1)

    def test_resume_keeps_k_unless_the_node_type_changes(self):
        kept = {"node_type": "amd1", "procs": 16, "peak_mb": 25805, "k": 8, "measured_at_step": 1}
        calls, _, log = self.steps(execute.Node("h-1", "amd1", 128, 253099), 99 * GIB, n_steps=1, sdp_mem=kept)
        self.assertEqual(calls, [(8, 8, 16)])
        self.assertIn("measured at step 1", log)
        calls, r, _ = self.steps(execute.Node("h-1", "amd2", 256, 376000), 28.6 * GIB, n_steps=1, sdp_mem=kept)
        self.assertEqual(calls, [(1, 1, 31), (7, 7, 31)])
        self.assertEqual(r.sdp_mem["node_type"], "amd2")

    def test_resume_state_takes_it_from_solves(self):
        kept = {"node_type": "amd1", "procs": 16, "peak_mb": 25805, "k": 8, "measured_at_step": 1}
        d = Path(tempfile.mkdtemp())
        line = {"step": 1, "x": ["0.519", "1.4", "0.956"], "verdict": "stepped", "navigator": "-1e-3",
                "ext_step": ["1e-4", "0", "0"], "gradient": ["1", "0", "0"],
                "hessian": [["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"]], "iterations": 100,
                "boundary_on": False, "procs": 128, "pmp2sdp": kept}
        aborted = {"step": 2, "x": ["0.5191", "1.4", "0.956"], "verdict": "aborted", "boundary_on": False,
                   "procs": 128, "pmp2sdp": None}
        (d / "solves.jsonl").write_text(json.dumps(line) + "\n" + json.dumps(aborted) + "\n")
        self.assertEqual(skydive.resume_state(inputs.load(EXAMPLE), d, True)["pmp2sdp"], kept)


def stepped(n: int, navigator: str = None) -> execute.Dive:
    """A made-up step n that moves deltaSig by 1e-4."""
    return execute.Dive("stepped", "met criteria to update sdp input files", navigator=navigator or f"-{n}e-3",
                        primal="1", dual="-1", gap="2",
                        ext_step=["1e-4", "0", "0"], gradient=[str(n), "0", "0"],
                        hessian=[[str(n), "0", "0"], ["0", "1", "0"], ["0", "0", "1"]], iterations=100 * n)


def optimal(primal: str, dual: str) -> execute.Dive:
    """A made-up optimal dive: no N (docs/en/interfaces.md section 8)."""
    return execute.Dive("optimal", "found primal-dual optimal solution", primal=primal, dual=dual,
                        gap=str(abs(float(primal) - float(dual))), ext_step=["0", "0", "0"], iterations=1)


def jumped(n: int, cumulative: int) -> execute.Dive:
    """stepped(n) with alpha_d 0.05 and the cumulative iterations given."""
    d = stepped(n)
    d.alpha_d, d.iterations = "0.05", cumulative
    return d


class Loop(unittest.TestCase):
    """The loop's endings and records, and a resume, with made-up steps in place of the programs."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        gen = self.tmp / "generator"
        gen.write_bytes(b"gen")
        self.site = SimpleNamespace(scratch=str(self.tmp / "scratch"), generator=str(gen), sif="s", pmp2sdp="p",
                                    dynamical_sdp="d")
        self.node = execute.Node("h-1", "h", 4, 1000)
        self.dir = self.tmp / "run"
        self.dir.mkdir()
        for name in ("model.json", "spectrum.toml", "objective.toml"):
            (self.dir / name).write_text(name)

    def run_loop(self, dives, max_steps, state=None, first_only=False, track=None, kappa=None, tighten=None,
                 boundary=False, mixed=False, jumps=False):
        p = inputs.load(EXAMPLE)
        p.run["skydive"]["max_steps"] = max_steps
        if mixed:
            p.run["skydive"]["mixed_precision"] = True
        if jumps:
            p.run["skydive"]["large_steps"] = True
        if boundary:                    # finding the boundary from the first step
            p.run["skydive"]["find_boundary_direction"] = ["0", "1", "0"]
            p.run["skydive"]["find_boundary_when_allowed"] = False
        p.run["skydive"]["gap_upper_limit_first_only"] = first_only
        if tighten is not None:
            p.run["skydive"]["gap_threshold_tighten"] = tighten
        if track is LEFT_OUT:
            del p.run["skydive"]["gap_upper_limit_track"]
        elif track is not None:
            p.run["skydive"]["gap_upper_limit_track"] = track
        if kappa is not None:
            p.run["skydive"]["gap_upper_limit_track_floor"] = kappa
        r = skydive.Run(p, self.site, self.node, self.dir)
        seen = []

        def step(n, x, st, procs):
            seen.append((n, dict(x), dict(st)))
            (self.dir / "steps" / str(n)).mkdir(parents=True, exist_ok=True)
            if dives[n - 1] is None:
                raise execute.StepFailed("pmp2sdp exit 1")
            sdir = r.scratch / f"step_{n}"
            (sdir / "out").mkdir(parents=True)
            (sdir / "out" / "out.txt").write_text("out")
            part = sdir / "pmp" / "json" / "continuum_rep1+_l0_d3_mat_ab.json"
            part.parent.mkdir(parents=True)
            part.write_text("{}")
            nsv = sdir / "pmp" / "c" / "pmp.nsv"
            nsv.parent.mkdir()
            nsv.write_text(f"{part}\0")
            # the solver's checkpoint as dynamical_sdp leaves it (4 ranks), kept as step() keeps it
            r.ck.mkdir(parents=True, exist_ok=True)
            for rank in range(4):
                (r.ck / f"checkpoint_{n}_{rank}").write_text(f"after step {n}")
            (r.ck / "checkpoint.json").write_text(json.dumps({"current": n, "backup": n - 1}))
            (r.ck / "block_timings").write_text("1\n")
            r.keep_generation(4, st["precision"])
            return dives[n - 1], 4, {}, sdir, nsv

        s = state or skydive.start_state(p)
        with mock.patch.object(r, "step", step):
            r.loop(s["x"], s["n"], s["bfgs"], s["boundary_on"], s["procs"])
        return p, seen, json.loads((self.dir / "result.json").read_text())

    def assert_checkpoint(self, n):
        """checkpoint/ (the solver's, in the run directory) at step n's generation, kept for the last record
        alone: hard links to its 4 rank files, checkpoint.json, block_timings and layout.json (one node, 4
        processes); no kept/*.new left."""
        ck = self.dir / "checkpoint"
        self.assertEqual(execute.generation(ck), n)
        records = len((self.dir / "solves.jsonl").read_text().splitlines())
        k = execute.kept_for(ck, records)
        self.assertEqual([p.name for p in (ck / "kept").iterdir()], [k.name])
        self.assertEqual(sorted(f.name for f in k.iterdir()),
                         ["block_timings", "checkpoint.json"] + [f"checkpoint_{n}_{r}" for r in range(4)] + ["layout.json"])
        self.assertTrue(os.path.samefile(k / f"checkpoint_{n}_3", ck / f"checkpoint_{n}_3"))
        self.assertEqual(json.loads((k / "layout.json").read_text()),
                         {"nodes": ["h-1"], "node_type": "h", "ppn": 4, "precision": 785, "generation": n})

    def test_converged(self):
        """Optimal with primal and dual both negative: converged, allowed; N null, primal/dual/gap recorded."""
        _, seen, res = self.run_loop([stepped(1), stepped(2), optimal("-1.1e-11", "-1.2e-11")], 10)
        self.assertEqual((res["status"], res["reason"], res["steps"], res["navigator"]),
                         ("converged", "allowed", 3, None))
        self.assertEqual((res["primal"], res["dual"]), ("-1.1e-11", "-1.2e-11"))
        self.assertIsNotNone(res["gap"])
        self.assertEqual(res["x"], ["0.5192", "1.4", "0.956"])
        self.assertEqual(seen[2][2]["prev_hessian"], ["2", "0", "0", "0", "1", "0", "0", "0", "1"])
        self.assertEqual(seen[2][2]["total_iterations"], 200)
        labels = json.loads((self.dir / "endpoint" / "pmp" / "labels.json").read_text())
        self.assertEqual(labels[0]["channel"], "rep1+")
        self.assertTrue((self.dir / "endpoint" / "out" / "out.txt").exists())
        self.assertFalse((self.dir / "checkpoint").exists())
        self.assertFalse((self.tmp / "scratch" / "run").exists())
        self.assertEqual(len((self.dir / "solves.jsonl").read_text().splitlines()), 3)

    def test_excluded(self):
        """Optimal with both objectives positive: converged, excluded, with endpoint/."""
        _, _, res = self.run_loop([stepped(1), optimal("1.15e-6", "5.04e-7")], 10)
        self.assertEqual((res["status"], res["reason"], res["navigator"]), ("converged", "excluded", None))
        self.assertTrue((self.dir / "endpoint" / "pmp" / "labels.json").exists())
        self.assertFalse((self.dir / "checkpoint").exists())

    def test_mixed_precision_ends_at_full(self):
        """Mixed precision (Ising Lambda 11: 785 bits; gap upper limit 0.1): 512 bits from the start, not lowered
        at the made-up gap 2; an optimal call at 512 bits decides nothing: the same point at 785 bits, with the same
        BFGS state, decides."""
        _, seen, res = self.run_loop([stepped(1), stepped(2), optimal("-1.1e-11", "-1.2e-11"),
                                      optimal("-1.1e-11", "-1.2e-11")], 10, mixed=True)
        self.assertEqual([st["precision"] for _, _, st in seen], [512, 512, 512, 785])
        self.assertEqual(seen[3][1], seen[2][1])
        self.assertEqual(seen[3][2]["prev_hessian"], seen[2][2]["prev_hessian"])
        self.assertEqual((res["status"], res["reason"], res["steps"]), ("converged", "allowed", 3))
        lines = [json.loads(l) for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual([(l["step"], l["precision"], l["precision_next"]) for l in lines],
                         [(1, 512, 512), (2, 512, 512), (3, 512, 785), (3, 785, 785)])
        self.assertEqual((lines[0].get("first_call"), lines[1].get("first_call"), lines[1]["call_iterations"]),
                         (True, None, 100))
        self.assertIn("optimal at 512 bits", lines[2]["precision_note"])
        self.assertTrue((self.dir / "steps" / "3-512bits").is_dir())

    def test_mixed_precision_health_check(self):
        """findMinimumQ changing at 512 bits: full precision from the next call on, never lowered again."""
        d3 = stepped(3)
        d3.find_minimum = True
        _, seen, _ = self.run_loop([stepped(1), stepped(2), d3, stepped(4), stepped(5)], 5, mixed=True)
        self.assertEqual([st["precision"] for _, _, st in seen], [512, 512, 512, 785, 785])
        s = skydive.resume_state(inputs.load(EXAMPLE), self.dir, hot=True)
        self.assertEqual(s["bfgs"]["precision"], 785)

    def test_mixed_precision_off(self):
        """Without mixed_precision: the problem's precision at every call, no precision_next."""
        _, seen, _ = self.run_loop([stepped(1), stepped(2)], 2)
        self.assertEqual([st["precision"] for _, _, st in seen], [785, 785])
        lines = [json.loads(l) for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual([("precision_next" in l, l["precision"]) for l in lines], [(False, 785), (False, 785)])

    def test_large_steps(self):
        """Large steps (alpha 0.05, a straight path): step 2 sends the next point 8 steps ahead and 8 dx as the BFGS
        step; step 3 took 30 iterations after it (over 13.5): 1 step for 2 calls; then 8 again. The records keep
        the solver's step; a resume goes on 8 steps ahead of the last point."""
        d = [jumped(1, 100), jumped(2, 103), jumped(3, 133), jumped(4, 136), jumped(5, 139), jumped(6, 142)]
        _, seen, res = self.run_loop(d, 6, jumps=True)
        self.assertEqual([s[1]["deltaSig"] for s in seen], ["0.519", "0.5191", "0.5199", "0.52", "0.5201", "0.5209"])
        self.assertEqual(seen[2][2]["prev_step"], ["0.0008", "0", "0"])
        self.assertEqual(seen[3][2]["prev_step"], ["1e-4", "0", "0"])
        lines = [json.loads(l) for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual([l["jump"] for l in lines], [1, 8, 1, 1, 8, 8])
        self.assertIn("30 iterations after a step 8 times ahead", lines[2]["jump_note"])
        self.assertEqual([l["ext_step"][0] for l in lines], ["1e-4"] * 6)
        self.assertEqual(res["x"], ["0.5217", "1.4", "0.956"])
        s = skydive.resume_state(inputs.load(EXAMPLE), self.dir, hot=True)
        self.assertEqual((s["x"]["deltaSig"], s["bfgs"]["prev_step"][0]), ("0.5217", "0.0008"))

    def test_large_steps_mode_change_stops(self):
        d3 = jumped(3, 106)
        d3.find_minimum = True
        _, seen, res = self.run_loop([jumped(1, 100), jumped(2, 103), d3, jumped(4, 109)], 4, jumps=True)
        self.assertEqual(len(seen), 3)
        self.assertEqual((res["status"], res["reason"]),
                         ("stopped", "stopped: findMinimumQ changed after a step 8 times ahead"))
        self.assertEqual(res["x"], ["0.5199", "1.4", "0.956"])

    def test_large_steps_box(self):
        """8 steps ahead would leave the box (deltaSig <= 0.530): 1 step."""
        start = skydive.start_state(inputs.load(EXAMPLE))
        start["x"]["deltaSig"] = "0.5295"
        _, seen, _ = self.run_loop([jumped(1, 100), jumped(2, 103), jumped(3, 106)], 3, state=start, jumps=True)
        self.assertEqual([s[1]["deltaSig"] for s in seen], ["0.5295", "0.5296", "0.5297"])

    def test_large_steps_with_mixed_precision(self):
        """Both on: the 30 iterations after the large step do not send mixed precision to full (the large step
        has its own budget, jump.after_jump); 512 bits throughout."""
        d = [jumped(1, 100), jumped(2, 103), jumped(3, 133), jumped(4, 136), jumped(5, 139)]
        _, seen, _ = self.run_loop(d, 5, mixed=True, jumps=True)
        self.assertEqual([s[2]["precision"] for s in seen], [512] * 5)
        lines = [json.loads(l) for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual([l["jump"] for l in lines], [1, 8, 1, 1, 8])
        self.assertNotIn("precision_note", lines[2])

    def test_undecided_at_optimal(self):
        """Optimal with objectives of opposite signs: stopped, checkpoint/ kept as for unfinished, no endpoint/;
        a resume does the step again at the same point."""
        _, _, res = self.run_loop([stepped(1), optimal("2e-10", "-1e-9")], 10)
        self.assertEqual((res["status"], res["reason"], res["steps"]), ("stopped", "stopped: N undecided at optimal", 2))
        self.assertEqual((res["primal"], res["dual"], res["navigator"]), ("2e-10", "-1e-9", None))
        self.assert_checkpoint(2)
        self.assertFalse((self.dir / "endpoint").exists())
        s = skydive.resume_state(inputs.load(EXAMPLE), self.dir, hot=True)
        self.assertEqual((s["n"], s["x"]["deltaSig"]), (1, "0.5191"))

    def test_boundary_at_optimal(self):
        """Finding the boundary, optimal with objectives of opposite signs: converged on the boundary, N
        reported as 0 with n_bound = the gap (|N| <= gap), primal, dual, gap kept, endpoint/ kept, summary.md
        saying so."""
        from bootlab import cli
        _, seen, res = self.run_loop([stepped(1), optimal("5.806e-21", "-0.708e-21")], 10, boundary=True)
        self.assertTrue(seen[1][2]["boundary_direction"])
        self.assertEqual((res["status"], res["reason"], res["steps"], res["navigator"]),
                         ("converged", "boundary", 2, "0"))
        self.assertEqual((res["primal"], res["dual"]), ("5.806e-21", "-0.708e-21"))
        self.assertEqual(res["n_bound"], res["gap"])
        self.assertTrue((self.dir / "endpoint" / "pmp" / "labels.json").exists())
        self.assertFalse((self.dir / "checkpoint").exists())
        (self.dir / "run.toml").write_text(EXAMPLE.read_text())
        cli.write_summary(self.dir)
        self.assertIn(f"- result: converged: on the boundary, |N| <= {res['gap']}",
                      (self.dir / "summary.md").read_text())

    def test_boundary_same_signs_unchanged(self):
        """Finding the boundary, both objectives of one sign: converged, allowed or excluded as before."""
        _, _, res = self.run_loop([stepped(1), optimal("-1.1e-11", "-1.2e-11")], 10, boundary=True)
        self.assertEqual((res["status"], res["reason"], res["navigator"]), ("converged", "allowed", None))
        self.assertNotIn("n_bound", res)
        _, _, res = self.run_loop([stepped(1), optimal("1.15e-6", "5.04e-7")], 10, boundary=True)
        self.assertEqual((res["status"], res["reason"]), ("converged", "excluded"))

    def test_left_box(self):
        far = stepped(1)
        far.ext_step = ["1", "0", "0"]
        _, _, res = self.run_loop([far], 10)
        self.assertEqual((res["status"], res["reason"], res["x"]),
                         ("stopped", "stopped: left box", ["0.519", "1.4", "0.956"]))
        self.assertEqual((res["navigator"], res["primal"], res["dual"], res["gap"]), ("-1e-3", "1", "-1", "2"))

    def test_gap_threshold_rule(self):
        """max(1e-30, min(setting, 1e-3 |N_prev|)): the setting (examples/ising: bt's 1e-6, passed as written)
        at step 1 and while 1e-3 |N_prev| is not below it; then 1e-3 |N_prev|; then the floor."""
        self.assertEqual(skydive.gap_threshold("1e-6", None), "1e-6")
        self.assertEqual(skydive.gap_threshold("1e-6", "-0.06"), "1e-6")
        self.assertEqual(skydive.gap_threshold("1e-6", "1e-3"), "1e-6")
        self.assertEqual(skydive.gap_threshold("1e-6", "-5e-5"), "5.000000e-08")
        self.assertEqual(skydive.gap_threshold("1e-6", "1e-27"), "1e-30")
        self.assertEqual(skydive.gap_threshold("1e-6", "-1e-40"), "1e-30")
        dives = [stepped(1, "-0.06"), stepped(2, "5e-5"), stepped(3, "1e-40"), stepped(4, "2e-9")]
        _, seen, _ = self.run_loop(dives, 4)
        expected = ["1e-6", "1e-6", "5.000000e-08", "1e-30"]
        self.assertEqual([st["gap_threshold"] for _, _, st in seen], expected)
        recorded = [json.loads(l)["gap_threshold"] for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual(recorded, expected)

    def test_gap_threshold_on_resume(self):
        """A hot resume tightens from the last step's N; a cold resume's first call uses the setting."""
        dives = [stepped(1, "-0.06"), stepped(2, "5e-5"), stepped(3, "1e-40"), stepped(4, "2e-9")]
        _, _, _ = self.run_loop(dives, 2)
        p = inputs.load(EXAMPLE)
        lines = (self.dir / "solves.jsonl").read_text()
        _, hot, _ = self.run_loop(dives, 4, skydive.resume_state(p, self.dir, hot=True))
        self.assertEqual([(n, st["gap_threshold"]) for n, _, st in hot], [(3, "5.000000e-08"), (4, "1e-30")])
        (self.dir / "solves.jsonl").write_text(lines)
        _, cold, _ = self.run_loop(dives, 4, skydive.resume_state(p, self.dir, hot=False))
        self.assertEqual([(n, st["gap_threshold"]) for n, _, st in cold], [(3, "1e-6"), (4, "1e-30")])

    def test_gap_threshold_kept(self):
        """gap_threshold_tighten = false (docs/en/interfaces.md section 10): the setting at every step, on a hot resume too; written
        true it is the default's rule."""
        self.assertEqual(skydive.gap_threshold("1e-6", "-5e-5", False), "1e-6")
        self.assertEqual(skydive.gap_threshold("1e-6", "-1e-40", False), "1e-6")
        self.assertEqual(skydive.gap_threshold("1e-6", "-5e-5", True), "5.000000e-08")
        dives = [stepped(1, "-0.06"), stepped(2, "5e-5"), stepped(3, "1e-40"), stepped(4, "2e-9")]
        _, seen, _ = self.run_loop(dives, 2, tighten=False)
        self.assertEqual([st["gap_threshold"] for _, _, st in seen], ["1e-6", "1e-6"])
        p = inputs.load(EXAMPLE)
        _, hot, _ = self.run_loop(dives, 4, skydive.resume_state(p, self.dir, hot=True), tighten=False)
        self.assertEqual([(n, st["gap_threshold"]) for n, _, st in hot], [(3, "1e-6"), (4, "1e-6")])
        recorded = [json.loads(l)["gap_threshold"] for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual(recorded, ["1e-6"] * 4)

    def test_gap_threshold_kept_under_a_tracked_limit(self):
        """With the threshold kept, the tracked limit comes down to it once c |N_prev| and kappa |N_1| are below it:
        that call gets the limit as the rule gives it, and ends optimal in the solver (docs/en/interfaces.md section 10). The
        ending is the usual one for an optimal call."""
        dives = [stepped(1, "-0.06"), stepped(2, "-5e-9"), optimal("2e-10", "-1e-9")]
        _, seen, res = self.run_loop(dives, 10, track="1", kappa="1e-9", tighten=False)
        self.assertEqual([(st["gap_upper_limit"], st["gap_threshold"]) for _, _, st in seen],
                         [("0.1", "1e-6"), ("6.000000e-02", "1e-6"), ("5.000000e-09", "1e-6")])
        self.assertEqual((res["status"], res["reason"], res["steps"]), ("stopped", "stopped: N undecided at optimal", 3))
        _, seen, _ = self.run_loop(dives[:2] + [stepped(3)], 3, track="1", kappa="1e-9")      # tightened: 5e-12
        self.assertEqual((seen[2][2]["gap_upper_limit"], seen[2][2]["gap_threshold"]), ("5.000000e-09", "5.000000e-12"))

    def test_failed_then_resume(self):
        """A failed step ends the run (its line has no boundary_on); a resume does that step again from the
        step before (a resume once stopped at KeyError)."""
        _, _, res = self.run_loop([stepped(1), None], 10)
        self.assertEqual((res["status"], res["steps"]), ("failed", 1))
        self.assertEqual(res["reason"], "failed: pmp2sdp exit 1")
        self.assert_checkpoint(1)
        s = skydive.resume_state(inputs.load(EXAMPLE), self.dir, hot=True)
        self.assertEqual((s["n"], s["x"]["deltaSig"], s["boundary_on"]), (1, "0.5191", False))

    def test_max_steps_then_resume(self):
        """Stopped after 2 steps and resumed: steps 3 and 4 get what they get in a run that is not stopped."""
        dives = [stepped(1), stepped(2), stepped(3), stepped(4)]
        _, whole, _ = self.run_loop(dives, 4)
        (self.dir / "solves.jsonl").unlink()
        import shutil
        shutil.rmtree(self.dir / "checkpoint")
        _, first, res = self.run_loop(dives, 2)
        self.assertEqual((res["status"], res["steps"], res["navigator"]), ("unfinished", 2, None))
        self.assert_checkpoint(2)
        self.assertFalse((self.tmp / "scratch" / "run").exists())     # the checkpoint is the run directory's
        p = inputs.load(EXAMPLE)
        _, rest, res = self.run_loop(dives, 4, skydive.resume_state(p, self.dir, hot=True))
        self.assertEqual(res["steps"], 4)

        def drop(st):
            return {k: v for k, v in st.items() if k != "first_call"}
        self.assertEqual([(n, x, drop(st)) for n, x, st in first + rest], [(n, x, drop(st)) for n, x, st in whole])

    def test_gap_limit_on_resume(self):
        """gap_upper_limit_first_only: a new run passes the limit at its first call and 0 after it; a resume
        from checkpoint/ passes 0 from the start; a resume without one starts the solver from scratch and
        passes the limit at its first call again (docs/en/interfaces.md section 10)."""
        dives = [stepped(1), stepped(2), stepped(3), stepped(4)]
        _, first, _ = self.run_loop(dives, 2, first_only=True)
        self.assertEqual([st["gap_upper_limit"] for _, _, st in first], ["0.1", "0"])
        p = inputs.load(EXAMPLE)
        lines = (self.dir / "solves.jsonl").read_text()
        _, hot, _ = self.run_loop(dives, 4, skydive.resume_state(p, self.dir, hot=True), first_only=True)
        self.assertEqual([(n, st["gap_upper_limit"]) for n, _, st in hot], [(3, "0"), (4, "0")])
        (self.dir / "solves.jsonl").write_text(lines)
        _, cold, _ = self.run_loop(dives, 4, skydive.resume_state(p, self.dir, hot=False), first_only=True)
        self.assertEqual([(n, st["gap_upper_limit"]) for n, _, st in cold], [(3, "0.1"), (4, "0")])
        self.assertEqual(cold[0][2]["prev_step"], hot[0][2]["prev_step"])     # the BFGS state is kept either way


    def test_track_rule(self):
        """gap_upper_limit_track = c (docs/en/interfaces.md section 10): min(setting, max(c |N_prev|, kappa |N_1|, 1000 floor))."""
        f = skydive.tracked_gap_limit
        self.assertEqual(f("1e-12", "1", "0.1", None, None), "1e-12")                # no step behind: the setting
        self.assertEqual(f("1e-12", "1", "0.1", "-5e-14", "-5e-14"), "5.000000e-14") # step 2: |N_prev| = |N_1|
        self.assertEqual(f("1e-12", "2", "0.1", "-5e-14", "-5e-14"), "1.000000e-13") # times c
        self.assertEqual(f("1e-12", "1", "0.1", "4e-14", "5e-14"), "4.000000e-14")   # c |N_prev| > kappa |N_1|
        self.assertEqual(f("1e-12", "1", "0.1", "1e-16", "5e-14"), "5.000000e-15")   # near zero: kappa |N_1|
        self.assertEqual(f("1e-12", "1", "0.5", "1e-16", "5e-14"), "2.500000e-14")   # kappa as set
        self.assertEqual(f("1e-12", "1", "0.1", "1e-16", "-3e-14"), "3.000000e-15")  # dN no longer counts
        self.assertEqual(f("1e-12", "1", "0.1", "1e-40", "1e-40"), "1.000000e-27")   # the floor, 1000 x 1e-30
        self.assertEqual(f("1e-12", "1", "0", "1e-40", "5e-14"), "1.000000e-27")     # kappa 0: c |N_prev| and the floor
        self.assertEqual(f("1e-12", "3", "0.1", "-0.06", "0.02"), "1e-12")           # never above the setting
        self.assertEqual(f("1e-30", "1", "0.1", "1e-40", "1e-40"), "1e-30")          # a setting under the floor

    TRACKED = [stepped(1, "-0.06"), stepped(2, "5e-5"), stepped(3, "1e-40"), stepped(4, "2e-2"), stepped(5, "2e-9")]
    TRACKED_LIMITS = ["0.1", "6.000000e-02", "6.000000e-03", "6.000000e-03", "2.000000e-02"]

    def test_track_in_the_loop(self):
        """Step 1 the setting (examples/ising: 0.1), step 2 |N_1|, then kappa |N_1| = 0.006 while |N| is below it,
        c |N| again above; each step's limit is passed to the solver and written to solves.jsonl."""
        _, seen, _ = self.run_loop(self.TRACKED, 5, track="1")
        self.assertEqual([st["gap_upper_limit"] for _, _, st in seen], self.TRACKED_LIMITS)
        recorded = [json.loads(l)["gap_upper_limit"] for l in (self.dir / "solves.jsonl").read_text().splitlines()]
        self.assertEqual(recorded, self.TRACKED_LIMITS)

    def test_track_floor_set(self):
        """gap_upper_limit_track_floor = kappa in place of 0.1."""
        _, seen, _ = self.run_loop(self.TRACKED, 4, track="1", kappa="0.5")
        self.assertEqual([st["gap_upper_limit"] for _, _, st in seen],
                         ["0.1", "6.000000e-02", "3.000000e-02", "3.000000e-02"])

    def test_track_on_resume(self):
        """A resume takes N_1 from solves.jsonl, hot or cold: hot continues as if not stopped; a cold resume's
        first call gets the setting, the step after it follows N and N_1 again."""
        self.run_loop(self.TRACKED, 3, track="1")
        p = inputs.load(EXAMPLE)
        lines = (self.dir / "solves.jsonl").read_text()
        _, hot, _ = self.run_loop(self.TRACKED, 5, skydive.resume_state(p, self.dir, hot=True), track="1")
        self.assertEqual([(n, st["gap_upper_limit"]) for n, _, st in hot], [(4, "6.000000e-03"), (5, "2.000000e-02")])
        (self.dir / "solves.jsonl").write_text("\n".join(lines.splitlines()[:2]) + "\n")  # after step 2
        s = skydive.resume_state(p, self.dir, hot=False)
        self.assertEqual((s["bfgs"]["prev_navigator"], s["bfgs"]["first_navigator"]), ("5e-5", "-0.06"))
        _, cold, _ = self.run_loop(self.TRACKED, 4, s, track="1")
        self.assertEqual([(n, st["gap_upper_limit"]) for n, _, st in cold], [(3, "0.1"), (4, "6.000000e-03")])
        (self.dir / "solves.jsonl").write_text(lines.splitlines()[0] + "\n")     # after step 1
        s = skydive.resume_state(p, self.dir, hot=True)
        self.assertEqual((s["bfgs"]["prev_navigator"], s["bfgs"]["first_navigator"]), ("-0.06", "-0.06"))
        _, one, _ = self.run_loop(self.TRACKED, 2, s, track="1")
        self.assertEqual([(n, st["gap_upper_limit"]) for n, _, st in one], [(2, "6.000000e-02")])

    def test_track_off_is_bt(self):
        """"off" (examples/ising, bt's settings): bt's fixed limit at every call (first_only false), 0 after the
        first (first_only true)."""
        _, seen, _ = self.run_loop(self.TRACKED, 3)
        self.assertEqual([st["gap_upper_limit"] for _, _, st in seen], ["0.1", "0.1", "0.1"])
        _, seen, _ = self.run_loop(self.TRACKED, 3, first_only=True)
        self.assertEqual([st["gap_upper_limit"] for _, _, st in seen], ["0.1", "0", "0"])

    def test_track_left_out(self):
        """Left out: the limit tracks N with c = 1 and kappa 0.1, as track = "1" does; with
        first_only = true and no c, first_only's limits."""
        _, seen, _ = self.run_loop(self.TRACKED, 5, track=LEFT_OUT)
        self.assertEqual([st["gap_upper_limit"] for _, _, st in seen], self.TRACKED_LIMITS)
        _, seen, _ = self.run_loop(self.TRACKED, 3, first_only=True, track=LEFT_OUT)
        self.assertEqual([st["gap_upper_limit"] for _, _, st in seen], ["0.1", "0", "0"])


if __name__ == "__main__":
    unittest.main()
