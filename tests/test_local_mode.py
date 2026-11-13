"""Local mode (no Slurm) without a machine to run on: the launcher's choice, the physical
cores, the local commands, the watchdog's measure, the lock, the local site: python -m unittest tests.test_local_mode."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bootlab import cli, execute, probe, regress, sitegen, skydive  # noqa: E402

SITE = execute.Site(name="w", sif="/img/r5.sif", binds=["/data"], generator="/opt/bootlab/bin/generator",
                    scratch="shared", runs="/data/runs", pmp2sdp="pmp2sdp_3.0sky",
                    dynamical_sdp="dynamical_sdp_RV1B_3.0sky", driver_reserve_mb=4096, bench_dir=Path("."))


def fake_sys(root: Path, cores: dict):
    """cpu -> (package, core) under root/cpu<n>/topology."""
    for c, (pkg, core) in cores.items():
        t = root / f"cpu{c}" / "topology"
        t.mkdir(parents=True)
        (t / "physical_package_id").write_text(f"{pkg}\n")
        (t / "core_id").write_text(f"{core}\n")
        sib = sorted(o for o, k in cores.items() if k == (pkg, core))
        (t / "thread_siblings_list").write_text(",".join(map(str, sib)) + "\n")


class Launcher(unittest.TestCase):
    def test_choice(self):
        env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(execute.shutil, "which", return_value=None):
                self.assertEqual(execute.site_launcher(SITE)[0], "local")
            with mock.patch.object(execute.shutil, "which", return_value="/usr/bin/sbatch"):
                self.assertEqual(execute.site_launcher(SITE), ("slurm", "sbatch is on PATH (/usr/bin/sbatch)"))
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "7"}):
            self.assertEqual(execute.site_launcher(SITE)[0], "slurm")
            given = execute.Site(**{**SITE.__dict__, "launcher": "local"})
            self.assertEqual(execute.site_launcher(given)[0], "local")             # the site's word wins
        with self.assertRaises(SystemExit):
            execute.site_launcher(execute.Site(**{**SITE.__dict__, "launcher": "pbs"}))

    def test_bootlab_toml_gives_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "bootlab.toml").write_text('image = "/img/r5.sif"\nlauncher = "local"\n')
            self.assertEqual(execute.site_paths({}, root)["launcher"], "local")
            self.assertEqual(execute.site_paths({"launcher": "slurm"}, root)["launcher"], "slurm")


class Machine(unittest.TestCase):
    def test_physical_cpus(self):
        """Two cores of two threads each (cpu0 and 2 one core): the first thread of each."""
        with tempfile.TemporaryDirectory() as d:
            fake_sys(Path(d), {0: (0, 0), 1: (0, 1), 2: (0, 0), 3: (0, 1), 4: (1, 0), 5: (1, 0)})
            self.assertEqual(execute.physical_cpus({0, 1, 2, 3, 4, 5}, Path(d)), [0, 1, 4])
            self.assertEqual(execute.physical_cpus({2, 3}, Path(d)), [2, 3])         # the affinity's own threads

    def test_memory(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "meminfo"
            f.write_text("MemTotal:  65536000 kB\nMemAvailable:  40960000 kB\n")
            self.assertEqual(execute.local_mem_mb(None, f), 40000)
            self.assertEqual(execute.local_mem_mb(16, f), 16384)

    def test_env(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "7", "PMIX_RANK": "0", "UCX_TLS": "self,sm"}):
            env = execute.local_env()
        self.assertFalse([k for k in env if k.startswith(("SLURM_", "PMIX_"))])
        self.assertEqual((env["UCX_TLS"], env["OMPI_MCA_plm"]), ("self,sm", "isolated"))


class Commands(unittest.TestCase):
    def test_local_argv(self):
        self.assertEqual(execute.local_argv(SITE, ["pmp2sdp_3.0sky", "--input", Path("/p")], [0, 1, 4], True,
                                            Path("/l/x.rankfile")),
                         ["apptainer", "exec", "--bind", "/data", "/img/r5.sif", "mpirun", "-np", "3", "--rankfile",
                          "/l/x.rankfile", "--mca", "rmaps_rank_file_physical", "1", "--bind-to", "hwthread",
                          "pmp2sdp_3.0sky", "--input", "/p"])
        with tempfile.TemporaryDirectory() as d:
            rf = execute.write_rankfile(Path(d) / "a.rankfile", [0, 1, 4])
            host = execute.socket.gethostname().split(".")[0]
            self.assertEqual(rf.read_text(), f"rank 0={host} slot=0\nrank 1={host} slot=1\nrank 2={host} slot=4\n")
        self.assertEqual(execute.local_argv(SITE, ["/opt/bootlab/bin/generator", "pmp", "r.json"], [0, 1], False),
                         ["apptainer", "exec", "--bind", "/data", "/img/r5.sif", "taskset", "-c", "0,1",
                          "/opt/bootlab/bin/generator", "pmp", "r.json"])

    def test_launch_picks(self):
        node = execute.Node("w1", "local", 4, 64000, launcher="local", cpus=[0, 1, 2, 3])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(execute, "start", return_value="p") as start:
            log = Path(d) / "pmp2sdp-c"
            execute.launch(SITE, node, ["x"], 2, 1, 1000, True, log)
            cmd, _ = start.call_args[0]
            self.assertEqual(cmd[cmd.index("--rankfile") + 1], str(log.with_suffix(".rankfile")))
            slots = [l.split("slot=")[1] for l in log.with_suffix(".rankfile").read_text().splitlines()]
            self.assertEqual(slots, ["0", "1"])                                      # the first ntasks cores
            env = start.call_args[1]["env"]
            self.assertEqual(env["OMPI_MCA_plm"], "isolated")
            self.assertNotIn("SLURM_JOB_NODELIST", env)                              # MPI: no Slurm variable
            execute.launch(SITE, node, ["x"], 2, 1, 1000, True, log, cores=[2, 3])
            self.assertEqual([l.split("slot=")[1] for l in log.with_suffix(".rankfile").read_text().splitlines()],
                             ["2", "3"])
            execute.launch(SITE, node, ["gen"], 1, 4, 1000, False, log)
            self.assertEqual(start.call_args[1]["env"]["SLURM_JOB_NODELIST"], "w1")    # the generator's node list
            self.assertIn("0,1,2,3", start.call_args[0][0])
            slurm = execute.Node("n1", "t", 4, 64000)
            execute.launch(SITE, slurm, ["x"], 2, 1, 1000, True, Path("/l"))
            self.assertEqual(start.call_args[0][0], execute.srun_argv(SITE, ["x"], 2, 1, 1000, True))  # unchanged
            self.assertNotIn("env", start.call_args[1])


def fake_proc(root: Path, procs: dict):
    """pid -> (ppid, smaps_rollup kB {Pss_Anon, Pss_Shmem} or None, RssAnon kB, argv)."""
    for pid, (ppid, pss, rss, argv) in procs.items():
        d = root / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} (x y) S {ppid} 1 1\n")
        if pss:
            (d / "smaps_rollup").write_text("".join(f"{k}:  {v} kB\n" for k, v in pss.items()))
        (d / "status").write_text(f"Name: x\nRssAnon:\t{rss} kB\n")
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")


class Watchdog(unittest.TestCase):
    def test_tree_memory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fake_proc(root, {100: (1, {"Pss_Anon": 1000, "Pss_File": 50, "Pss_Shmem": 24}, 9999, ["mpirun"]),
                             101: (100, {"Pss_Anon": 2048, "Pss_Shmem": 0}, 9999, ["a"]),
                             102: (1, {"Pss_Anon": 5000, "Pss_Shmem": 0}, 0, ["other"])})
            self.assertEqual(sorted(execute.descendants(100, root)), [100, 101])
            self.assertEqual(execute.tree_memory_mb(100, root), (3.0, "pss"))       # 1000 + 24 + 2048 kB
            fake_proc(root, {104: (100, None, 0, ["ended"])})                    # a zombie: no rollup, no RssAnon
            (root / "104" / "status").write_text("Name: x\nState: Z (zombie)\n")
            self.assertEqual(execute.tree_memory_mb(100, root), (3.0, "pss"))       # passed, the measure still pss
            (root / "101" / "smaps_rollup").unlink()
            fake_proc(root, {103: (101, None, 1024, ["b"])})
            mb, measure = execute.tree_memory_mb(100, root)
            self.assertEqual(measure, "rss_anon")
            self.assertAlmostEqual(mb, (1024 + 9999 + 1024) / 1024)

    def test_solver_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fake_proc(root, {10: (1, None, 0, ["python3"]), 11: (10, None, 0, ["apptainer", "exec"]),
                             12: (11, None, 0, ["mpirun", "-np", "2"]),
                             13: (12, None, 0, ["dynamical_sdp_RV1B_3.0sky", "--sdpDir"]),
                             14: (12, None, 0, ["/opt/bt/bin/dynamical_sdp_RV1B_3.0sky"])})
            self.assertEqual(sorted(execute.solver_ranks(10, "dynamical_sdp_RV1B_3.0sky", root)), [13, 14])

    def test_build_sdps_local(self):
        """A wave on disjoint cores; over its share the wave is ended and the step fails."""
        node = execute.Node("w1", "local", 4, 8192 + 4096, launcher="local", cpus=[0, 1, 2, 3])
        started = []

        def fake_launch(site, n, argv, ntasks, cpus, mem, mpi, log, cores=None, wrap=None, **k):
            started.append((cores, mem))
            return subprocess.Popen(["sleep", "0.3"])
        with tempfile.TemporaryDirectory() as d, mock.patch.object(execute, "launch", fake_launch), \
                mock.patch.object(execute, "tree_memory_mb", return_value=(100.0, "pss")):
            sdps, waves = execute.build_sdps_local(SITE, node, {"a": "a.nsv", "b": "b.nsv"}, Path(d), 768,
                                                   Path(d) / "logs", 2, 2)
            self.assertEqual(started, [([0, 1], 4096), ([2, 3], 4096)])
            self.assertEqual((waves[0]["measure"], waves[0]["held_mb"], waves[0]["watchdog"]), ("pss", 100, True))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(execute, "launch", fake_launch), \
                mock.patch.object(execute, "tree_memory_mb", return_value=(8000.0, "pss")):
            with self.assertRaises(execute.StepFailed) as e:                 # over 95 % of its 8192 MB
                execute.build_sdps_local(SITE, node, {"a": "a.nsv"}, Path(d), 768, Path(d) / "logs", 1, 2)
            self.assertIn("a held 8000 MB of its 8192 MB (pss)", str(e.exception))
            self.assertIn("the wave was ended", str(e.exception))


class Lock(unittest.TestCase):
    def test_one_local_run(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d) / "runs"
            self.assertIsNone(execute.lock_holder(runs))
            try:
                execute.local_lock(runs, "run a.toml")
                self.assertIn("run a.toml", execute.lock_holder(runs))
                with self.assertRaises(SystemExit) as e:
                    execute.local_lock(runs, "run b.toml")
                self.assertIn("run a.toml", str(e.exception))
                run_dir = Path(d) / "r1"
                run_dir.mkdir()
                execute.write_driver_pid(run_dir)
                pid = json.loads((run_dir / "driver.pid").read_text())
                self.assertEqual((pid["pid"], pid["pgid"]), (os.getpid(), os.getpgid(0)))
            finally:
                execute.LOCK.close()
                execute.LOCK = None
            self.assertIsNone(execute.lock_holder(runs))


class ProbeAndSite(unittest.TestCase):
    def test_binding_check(self):
        threads = {0: {0, 4}, 1: {1, 5}, 4: {0, 4}, 5: {1, 5}}
        ok = probe.binding_check(["Cpus_allowed_list:\t0", "Cpus_allowed_list:\t1"], [0, 1], threads)
        self.assertTrue(ok["ok"])                                     # each rank on its core's first thread alone
        whole = probe.binding_check(["Cpus_allowed_list:\t0,4", "Cpus_allowed_list:\t1,5"], [0, 1], threads)
        self.assertFalse(whole["ok"])                                 # a whole core: its hyperthread too
        same = probe.binding_check(["Cpus_allowed_list:\t0", "Cpus_allowed_list:\t0"], [0, 1], threads)
        self.assertFalse(same["distinct_cores"])
        loose = probe.binding_check(["Cpus_allowed_list:\t0-5", "Cpus_allowed_list:\t1"], [0, 1], threads)
        self.assertFalse(loose["on_a_core_of_the_list"])
        self.assertEqual(probe.allowed_cpus("0-2,8"), {0, 1, 2, 8})

    def test_local_site(self):
        tp = {"bootlab": {"write": {"seconds": [0.2] * 3}, "read": {"seconds": [0.5] * 3}}}
        rec = {"host": "ws1", "job": "local", "physical_cpus": [0, 1, 2, 3], "memory_mb": {"MemTotal": 64000},
               "local": [{"path": "/scratch", "writable": True, "local_block_device": True, "free_gb": 500.0,
                          "block": {"device": "nvme0n1"}}],
               "throughput": {"/scratch": tp, "shared": {"bootlab": {"write": {"seconds": [2.0] * 3},
                                                                     "read": {"seconds": [0.5] * 3}}}},
               "shared": {"path": "/data/runs", "where": {"realpath": "/data/runs"}, "mountpoint": "/data"},
               "ib": {"chosen": None}, "nics": [],
               "mpi": {"ok": True, "full": [{"ok": True}] * 3, "halves": [{"ok": True}] * 3},
               "memory_guard": {"available": False, "why": "no systemd-run"}}
        p = {"plan": {"what": "local", "apptainer_setup": ""}, "nodes": {"ws1": [rec]}, "pairs": [],
             "types": {"local": {"hosts": ["ws1"], "physical_cores": 4, "mem_mb": 64000}}}
        site, notes, report = sitegen.generate_local(p, {})
        self.assertEqual((site["launcher"], site["memory_guard"], site["scratch"]), ("local", "rss", "/scratch/$USER-bootlab"))
        self.assertEqual(site["env"]["UCX_TLS"], "self,sm")
        self.assertEqual(site["mpi"], {"transport": "local"})
        self.assertEqual(site["node_types"], {"local": {"hosts": ["ws1"], "physical_cores": 4, "mem_gb": 62}})
        self.assertNotIn("multinode", site)
        self.assertTrue(any("mpirun of the image on every core: ok (3 of 3 full" in l for l in report))


class Cli(unittest.TestCase):
    def test_submit_dry_and_stop(self):
        with tempfile.TemporaryDirectory() as d:
            site = execute.Site(**{**SITE.__dict__, "runs": str(Path(d) / "runs"), "python": "/py/bin/python3"})
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.local_submit(site, "run", Path(d) / "a.toml", True, False, "no sbatch")
            text = out.getvalue()
            self.assertIn("# launcher local: no sbatch", text)
            self.assertIn("/py/bin/python3 -I -c", text)
            self.assertIn(f" run {Path(d) / 'a.toml'}", text)
            self.assertFalse(list((Path(d) / "runs").glob("local-*.log")))           # --dry: nothing started
            run_dir = Path(d) / "r"
            run_dir.mkdir()
            dead = subprocess.Popen(["true"], start_new_session=True)
            dead.wait()
            (run_dir / "driver.pid").write_text(json.dumps({"pid": dead.pid, "pgid": dead.pid,
                                                            "host": execute.socket.gethostname().split(".")[0]}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(["stop", str(run_dir)])
            self.assertIn("no driver runs", out.getvalue())


class Iterations(unittest.TestCase):
    """A call stopped (bin/bootlab stop: SIGTERM to the solver) and resumed from its checkpoint: its record has
    the iterations it made, so the cumulative counts are those of a run that was not stopped."""
    TABLE = ROOT / "tests" / "reference" / "local-mode" / "dynamical_sdp-resumed.out"

    def test_the_iteration_table(self):
        self.assertEqual(execute.call_iterations(self.TABLE), 110)         # the resumed call's, recorded
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.out"
            f.write_text("Loading binary checkpoint from : x\\nSaving checkpoint to : x\\n")
            self.assertIsNone(execute.call_iterations(f))                  # no table
            self.assertIsNone(execute.call_iterations(Path(d) / "none.out"))

    def test_record(self):
        stepped = execute.Dive("stepped", "met criteria to update sdp input files", iterations=144)
        self.assertEqual(skydive.record_iterations(stepped, 0), (144, None))
        stopped = execute.Dive("aborted", "unfinished: SIGTERM signal received", call_iterations=34)
        self.assertEqual(skydive.record_iterations(stopped, 0), (34, None))
        self.assertEqual(skydive.record_iterations(stopped, 147), (181, None))
        unknown = execute.Dive("aborted", "unfinished: SIGTERM signal received")
        n, why = skydive.record_iterations(unknown, 147)
        self.assertIsNone(n)
        self.assertIn("not counted", why)

    def test_resume(self):
        rec = lambda step, verdict, it, **k: {"step": step, "verdict": verdict, "iterations": it, **k}
        # local: step 1 stopped after 34 iterations, the resume counts from 34 (34 + 110 = 144, as uninterrupted)
        self.assertEqual(skydive.resume_iterations([rec(1, "aborted", 34)]), (34, None))
        # a cluster job cancelled: no aborted record, the last step's count (the rule unchanged)
        self.assertEqual(skydive.resume_iterations([rec(1, "stepped", 144), rec(2, "stepped", 147)]), (147, None))
        # a failed step's record has no count: the one before
        self.assertEqual(skydive.resume_iterations([rec(1, "stepped", 144), {"step": 2, "verdict": "aborted"}]),
                         (144, None))
        # unknown: passed, with a warning
        total, warning = skydive.resume_iterations([rec(1, "stepped", 144),
                                                    rec(2, "aborted", None, iterations_unknown="no table")])
        self.assertEqual(total, 144)
        self.assertIn("unknown (no table)", warning)
        # records written before this rule: an aborted call without a count, the stepped one before
        self.assertEqual(skydive.resume_iterations([rec(1, "stepped", 144), rec(2, "aborted", None)]), (144, None))

    def test_regress_of_a_stopped_run(self):
        steps = json.loads(regress.REFERENCE.read_text())["steps"]
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            stopped = {"step": 1, "verdict": "aborted", "reason": "unfinished: SIGTERM signal received", "iterations": 34}
            write = lambda lines: (run / "solves.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))
            write([stopped] + [{**s, "verdict": "stepped"} for s in steps])
            ok, lines, _ = regress.check(run)
            self.assertTrue(ok, lines)
            self.assertIn("step 1: a call ended (SIGTERM signal received) and the step was done again; "
                          "its iterations counted: 34", lines)
            lost = [{**s, "verdict": "stepped", "iterations": s["iterations"] - 34} for s in steps]
            write([{**stopped, "iterations": None, "iterations_unknown": "no table"}] + lost)
            ok, lines, report = regress.check(run)
            self.assertFalse(ok)
            self.assertTrue(all("has a call with its iterations unknown" in p for p in report["problems"]), report)


if __name__ == "__main__":
    unittest.main()
