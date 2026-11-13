"""bootlab bench and the probe of one partition, without a cluster: python -m unittest tests.test_bench."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import bench, benchrun, probe, sitegen  # noqa: E402

TABLE = """\
          time    mu     P-obj       D-obj      gap
----------------------------------------------------
1            0 1.4e+31  +6.10e+22   -3.24e+09   1.00
2            2 6.7e+30  +9.75e+22   -9.71e+08   1.00
3            3 3.2e+30  +1.56e+23   -2.91e+08   1.00
4            5 1.5e+30  +2.50e+23   -8.74e+07   1.00
5            6 7.4e+29  +3.99e+23   -2.62e+07   1.00
6            8 3.5e+29  +6.39e+23   -7.87e+06   1.00
7            9 1.7e+29  +1.02e+24   -2.36e+06   1.00
8           11 8.2e+28  +1.64e+24   -7.08e+05   1.00
9           12 3.9e+28  +2.62e+24   -2.12e+05   1.00
10          14 1.9e+28  +4.19e+24   -6.37e+04   1.00
11          15 9.0e+27  +6.70e+24   -1.91e+04   1.00
12          17 4.3e+27  +1.07e+25   -5.74e+03   1.00
"""


class Steady(unittest.TestCase):
    def test_from_the_fourth_iteration(self):
        # rows 4 (5 s) to 12 (17 s): 12 s over 8 iterations
        self.assertEqual(benchrun.steady_seconds(TABLE), 1.5)

    def test_too_few_rows(self):
        self.assertIsNone(benchrun.steady_seconds("\n".join(TABLE.splitlines()[:6])))
        self.assertIsNone(benchrun.steady_seconds("no table here"))

    def test_a_real_table(self):
        text = (ROOT / "tests" / "reference" / "local-mode" / "dynamical_sdp-resumed.out").read_text()
        s = benchrun.steady_seconds(text)
        self.assertTrue(s is not None and 0 < s < 5)

    def test_procs(self):
        for cores in (96, 128, 256, 127, 64):
            got = benchrun.procs_to_try(cores)
            self.assertTrue(set(got) <= {cores, cores // 2})
            self.assertTrue(all(bench.grid_ok(p) for p in got))
        self.assertEqual(benchrun.procs_to_try(128), [128, 64])
        self.assertNotIn(127, benchrun.procs_to_try(127))       # 127 x 1: a degenerate grid


class Rows(unittest.TestCase):
    ROW = {"problem": "ising-oe L=11 (bootlab bench)", "model": "ising-oe", "lambda": 11, "N": 274, "blocks": 46,
           "precision": 785, "program": "sdpb 3.0sky", "nodes": 1, "procs_per_node": 64, "s_per_iter": 0.5,
           "source": "bootlab bench 2026-11-13, 2 x CPU, 128 physical cores, 251 GB"}

    def test_a_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = benchrun.append_rows(Path(d) / "bench", "cpu128", 128, [self.ROW])
            data = json.loads(path.read_text())
            self.assertEqual((data["format"], data["node_type"], data["physical_cores"]),
                             ("bootlab-bench/1", "cpu128", 128))
            self.assertEqual(data["solver"], [self.ROW])
            # the layout reads it: the fastest measured count for this problem
            self.assertEqual(bench.solver_procs(bench.load_bench(Path(d) / "bench", "cpu128"), 128, 274, 46)[0], 64)

    def test_appended_to_an_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            old = {"format": "bootlab-bench/1", "node_type": "cpu128", "physical_cores": 128,
                   "solver": [{**self.ROW, "procs_per_node": 128, "s_per_iter": 0.7}], "sdp_build": [{"x": 1}]}
            (Path(d) / "cpu128.json").write_text(json.dumps(old))
            data = json.loads(benchrun.append_rows(Path(d), "cpu128", 128, [self.ROW]).read_text())
            self.assertEqual([r["procs_per_node"] for r in data["solver"]], [128, 64])
            self.assertEqual(data["sdp_build"], [{"x": 1}])


SINFO = """\
cpua-1|normal*|idle|512|2:128:2|376000|amd,cpua,mem384g
cpua-2|normal*|mixed|512|2:128:2|376000|amd,cpua,mem384g
cpub-1|normal*|idle|256|2:64:2|253000|amd,cpub,mem256g
"""


class OnePartition(unittest.TestCase):
    def plan(self, **kw):
        calls = []

        def run(cmd, **_):
            calls.append(cmd)
            if cmd[0] == "sinfo":
                return {"exit": 0, "out": SINFO, "err": ""}
            return {"exit": 0, "out": "", "err": ""}
        with tempfile.TemporaryDirectory() as d:
            sif = Path(d) / "i.sif"
            sif.write_bytes(b"x")
            args = SimpleNamespace(**{"shared": d, "out": str(Path(d) / "probe"), "sif": str(sif), "what": "node",
                                      "types": None, "python": None, "py_test": None, "submit": False, "local": [],
                                      "sdp": None, "sdp_precision": 785, "partition": None, "node": None, **kw})
            with mock.patch.object(probe, "run", run), mock.patch.object(probe, "apptainer_setup", lambda: ""):
                probe.cmd_plan(args)
            plan = json.loads((Path(d) / "probe" / "plan.json").read_text())
        return calls, plan

    def test_partition(self):
        calls, plan = self.plan(partition="p1")
        self.assertEqual(calls[0][-2:], ["-p", "p1"])
        self.assertEqual(plan["partition"], "p1")
        jobs = [t["sbatch"] for t in plan["types"].values() if t["sbatch"]]
        self.assertTrue(jobs and all(j[1] == "--partition=p1" for j in jobs))

    def test_node_busy_or_not(self):
        _, plan = self.plan(node="cpua-2")                       # mixed: not idle, the job waits for it
        self.assertEqual(list(plan["types"]), ["cpua"])
        self.assertEqual(plan["types"]["cpua"]["probe_hosts"], ["cpua-2"])
        self.assertIn("cpua-2", plan["types"]["cpua"]["sbatch"])
        with self.assertRaises(SystemExit):
            self.plan(node="nowhere-1")

    def test_site_of_the_partition(self):
        rec = {"host": "cpua-1", "job": "1", "nics": [], "ib": {"chosen": None}, "local": [],
               "shared": {"mountpoint": "/data"},
               "throughput": {"shared": {"bootlab": {"write": {"seconds": [1.0] * 3}, "read": {"seconds": [1.0] * 3}}}}}
        p = {"plan": {"partition": "p1", "apptainer_setup": ""}, "nodes": {"cpua-1": [rec]}, "pairs": [],
             "types": {"cpua": {"hosts": ["cpua-1", "cpua-2"], "physical_cores": 256, "mem_mb": 376000},
                       "cpub": {"hosts": ["cpub-1"], "physical_cores": 128, "mem_mb": 253000}}}
        site, _, _ = sitegen.generate(p, {}, set())
        self.assertEqual(list(site["node_types"]), ["cpua"])     # the probed type only
        self.assertEqual(site["slurm"], {"partition": "p1"})
        site, _, _ = sitegen.generate(p, {"slurm": {"partition": "p2", "time": "1:00:00"}}, set())
        self.assertEqual(site["slurm"], {"partition": "p2", "time": "1:00:00"})   # the hand file first
        del p["plan"]["partition"]
        site, _, _ = sitegen.generate(p, {}, set())
        self.assertEqual(sorted(site["node_types"]), ["cpua", "cpub"])            # no partition: as before


if __name__ == "__main__":
    unittest.main()
