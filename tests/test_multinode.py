"""The multinode parts that run without a cluster (docs/en/interfaces.md section 14): srun's command, a
node's memory, the checkpoint saved and placed on one node, and `bootlab submit --dry`."""
import contextlib
import dataclasses
import io
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

from bootlab import cli, execute, inputs, layout  # noqa: E402

SITE = execute.load_site("lab", FIXTURES)
L23 = ROOT / "tests" / "fixtures" / "runs" / "seven" / "run-l23.toml"
L27 = ROOT / "tests" / "fixtures" / "runs" / "seven" / "run-l27.toml"


class Srun(unittest.TestCase):
    def test_one_node_job_unchanged(self):
        """hosts None (a job of one node), [mpi] one_node = "srun": a rank a task, as before multinode."""
        site = dataclasses.replace(SITE, mpi={**SITE.mpi, "one_node": "srun"})
        self.assertEqual(execute.srun_argv(site, ["p"], 64, 1, 1000, mpi=True)[:5],
                         ["srun", "--exact", "--nodes=1", "--ntasks=64", "--cpus-per-task=1"])

    def test_one_node_mpirun(self):
        """One node (the default): one task with the ranks' cores, node_mpirun.sh, the image's mpirun."""
        cmd = execute.srun_argv(SITE, ["p", "--x", 1], 64, 1, 1000, mpi=True)
        self.assertEqual(cmd[:5], ["srun", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=64"])
        self.assertNotIn("--mpi=pmix", cmd)
        i = cmd.index("bash")
        self.assertEqual(cmd[i:i + 4], ["bash", str(execute.NODE_MPIRUN), "apptainer", "exec"])
        j = cmd.index(SITE.sif)
        self.assertEqual(cmd[j + 1:], ["mpirun", "-np", "64", "--map-by", "core", "--bind-to", "core",
                                       "p", "--x", "1"])
        n = execute.Node("nodeb-2", "amd2", 256, 376000, ["nodeb-2", "nodeb-4"])
        self.assertEqual(execute.srun_argv(SITE, ["p"], 64, 1, 1000, mpi=True, hosts=n.solo())[:6],
                         ["srun", "--exact", "--nodes=1", "-w", "nodeb-2", "--ntasks=1"])

    def test_node0_and_all(self):
        n = execute.Node("nodeb-2", "amd2", 256, 376000, ["nodeb-2", "nodeb-4"])
        site = dataclasses.replace(SITE, mpi={**SITE.mpi, "one_node": "srun"})
        self.assertEqual(execute.srun_argv(site, ["p"], 64, 1, 1000, mpi=True, hosts=n.solo())[:6],
                         ["srun", "--exact", "--nodes=1", "-w", "nodeb-2", "--ntasks=64"])
        self.assertEqual(execute.srun_argv(SITE, ["p"], 256, 1, 1000, mpi=True, hosts=n.spread())[:8],
                         ["srun", "--exact", "--nodes=2", "-w", "nodeb-2,nodeb-4", "--ntasks=512",
                          "--ntasks-per-node=256", "--cpus-per-task=1"])
        one = execute.Node("h-1", "h", 4, 1000)
        self.assertEqual((one.solo(), one.spread(), one.all_hosts, one.multi), (None, None, ["h-1"], False))

    def test_copy_report(self):
        """driver.log names how the copies reach each node: the transfer, the streams, the address."""
        import dataclasses
        n = execute.Node("nodeb-2", "amd2", 256, 376000, ["nodeb-2", "nodeb-4"])
        with mock.patch.object(execute, "ib_address", return_value="nfs-ib4"):
            self.assertEqual(execute.copy_report(SITE, n), [
                "copies to nodeb-4: transfer ssh, 8 tar streams to nfs-ib4 (IPoIB, ipoib_prefix 10.10.0.)"])
        srun = dataclasses.replace(SITE, multinode={**SITE.multinode, "transfer": "srun", "copy_network": "10.10.0."})
        with mock.patch.object(execute, "copy_address", return_value="nfs-ib4"):
            self.assertEqual(execute.copy_report(srun, n), [
                "copies to nodeb-4: transfer srun, 8 tar streams to receive.py at nfs-ib4 (copy_network 10.10.0.)"])
        shared = execute.Node("nodeb-2", "amd2", 256, 376000, ["nodeb-2", "nodeb-4"], shared=True)
        self.assertEqual(execute.copy_report(srun, shared), ["copies: none, the scratch is shared (every node sees node 0's files)"])
        self.assertEqual(execute.copy_report(srun, execute.Node("h-1", "h", 4, 1000)), [])

    def test_memory_per_node(self):
        self.assertEqual(execute.alloc_mem_mb("JobId=1 NumNodes=1 NumCPUs=256 AllocTRES=cpu=256,mem=253089M,node=1"),
                         253089)
        self.assertEqual(execute.alloc_mem_mb("JobId=2 NumNodes=2 AllocTRES=cpu=1024,mem=752000M,node=2"), 376000)
        self.assertEqual(execute.alloc_mem_mb("JobId=3 NumNodes=2 AllocTRES=cpu=1024,mem=734.38G,node=2"), 376002)


class Checkpoint(unittest.TestCase):
    """The solver's checkpoint/ in the run directory: a call's generation kept for its record, a resume set
    back to it whatever the solver saved after it (docs/en/interfaces.md section 11)."""

    LAY = {"nodes": ["nodea-1"], "node_type": "amd1", "ppn": 4}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.run_dir = self.tmp / "run"
        self.ck = self.run_dir / "checkpoint"
        self.ck.mkdir(parents=True)
        (self.ck / "block_timings").write_text("5\n")
        self.node = execute.Node("nodea-1", "amd1", 128, 1000)
        self.logged = []
        self.r = SimpleNamespace(site=SITE, node=self.node, ck=self.ck, log=self.logged.append)

    def save(self, g, ranks=4, partial=False):
        """What dynamical_sdp does for generation g (save_checkpoint.cxx): the backup's files removed, g's
        written, checkpoint.json renamed into place last (not when partial: the job killed while it wrote)."""
        meta = json.loads((self.ck / "checkpoint.json").read_text()) if (self.ck / "checkpoint.json").exists() else None
        if meta:
            for r in range(ranks):
                (self.ck / f"checkpoint_{meta['backup']}_{r}").unlink(missing_ok=True)
        for r in range(ranks // 2 if partial else ranks):
            (self.ck / f"checkpoint_{g}_{r}").write_text(f"{g} {r}")
        if partial:
            return
        (self.ck / "checkpoint_new.json").write_text(json.dumps({"current": g, "backup": g - 1}))
        os.replace(self.ck / "checkpoint_new.json", self.ck / "checkpoint.json")

    def record(self, n, keep=1):
        """keep = 1: the tests written when one generation was kept; KeepCheckpoints below for more."""
        with open(self.run_dir / "solves.jsonl", "a") as f:
            f.write(json.dumps({"step": n, "node": "nodea-1", "procs": 4}) + "\n")
        execute.prune_kept(self.ck, len(cli.records(self.run_dir)), keep)

    def test_kept_while_the_solver_saves_on(self):
        """Call 1 ends at generation 2, kept for record 1; call 2 saves 3 and 4 (generation 2's names gone)
        and is killed writing 5: the resume is back at generation 2, files and content."""
        self.save(1), self.save(2)
        self.assertEqual(execute.keep_generation(self.ck, 1, self.LAY)["generation"], 2)
        self.record(1)
        self.save(3), self.save(4), self.save(5, partial=True)
        self.assertFalse((self.ck / "checkpoint_2_0").exists())
        self.assertEqual(cli.resume_layout(self.run_dir, SITE), {**self.LAY, "generation": 2})
        self.assertTrue(cli.place_checkpoint(self.r, self.run_dir, 4))
        self.assertEqual(sorted(f.name for f in self.ck.iterdir()),
                         ["block_timings", "checkpoint.json"] + [f"checkpoint_2_{r}" for r in range(4)] + ["kept"])
        self.assertEqual(execute.generation(self.ck), 2)
        self.assertEqual((self.ck / "checkpoint_2_3").read_text(), "2 3")
        self.save(3)          # the resumed call saves over the killed one's names
        self.assertEqual((self.ck / "checkpoint_2_3").read_text(), "2 3")
        self.assertIn("generation 2 kept for record 1", self.logged[-1])

    def test_record_without_a_generation(self):
        """A failed step's record keeps none: the one before is the resume's; one kept but not recorded (the
        job killed before the record) is dropped; older ones go at the record."""
        self.save(1)
        execute.keep_generation(self.ck, 1, self.LAY), self.record(1)
        self.save(2)
        execute.keep_generation(self.ck, 2, self.LAY), self.record(2)
        self.assertEqual(sorted(p.name for p in (self.ck / "kept").iterdir()), ["2"])
        self.record(3)                                  # failed: nothing kept
        self.save(3)
        execute.keep_generation(self.ck, 4, self.LAY)  # killed before record 4
        (self.ck / "kept" / "5.new").mkdir()           # and a keep that did not finish
        self.assertEqual(execute.kept_for(self.ck, 3).name, "2")
        self.assertTrue(cli.place_checkpoint(self.r, self.run_dir, 4))
        self.assertEqual(execute.generation(self.ck), 2)
        self.assertEqual(sorted(p.name for p in (self.ck / "kept").iterdir()), ["2"])

    def test_checkpoint_from_before_kept(self):
        """A run's checkpoint/ from 80f63f6 (its last call's generation, layout.json or none): hot, kept for
        the last record."""
        self.save(6), self.save(7)
        self.record(1)
        self.assertTrue(cli.place_checkpoint(self.r, self.run_dir, 4))
        self.assertEqual(execute.generation(self.ck), 7)
        self.assertEqual(len(list(self.ck.glob("checkpoint_6_*"))), 0)
        self.assertEqual(json.loads((self.ck / "kept" / "1" / "layout.json").read_text()),
                         {**self.LAY, "generation": 7})

    def test_layout_changed_or_none(self):
        self.save(1)
        execute.keep_generation(self.ck, 1, self.LAY), self.record(1)
        self.r.node = execute.Node("nodeb-0", "amd2", 256, 1000)
        self.assertFalse(cli.place_checkpoint(self.r, self.run_dir, 4))     # another node type
        self.assertFalse(self.ck.exists())
        self.assertIn("layout changed", self.logged[-1])
        self.assertFalse(cli.place_checkpoint(self.r, self.run_dir, 4))     # none
        self.assertIsNone(cli.resume_layout(self.run_dir, SITE))

    def test_processes_changed(self):
        """Same nodes and node type, other processes a node (80f63f6's resume_path: ppn 96 for 128 -> cold):
        the kept generation has the other number of rank files, so cold."""
        self.save(1)
        execute.keep_generation(self.ck, 1, self.LAY), self.record(1)
        self.assertFalse(cli.place_checkpoint(self.r, self.run_dir, 2))
        self.assertFalse(self.ck.exists())
        self.assertIn("layout changed: the checkpoint's 1 x amd1 x 4 processes, now 1 x amd1 x 2", self.logged[-1])

    def generations_held(self) -> int:
        """Distinct rank-file data under checkpoint/ (kept/ included), in generations of 4 ranks."""
        inodes = {f.stat().st_ino for f in self.ck.rglob("checkpoint_*_*") if f.is_file()}
        return len(inodes) / 4

    def test_kept_does_not_grow(self):
        """8 calls, each saving twice (an interval save and the end's): one kept/ after each record, two only
        between a keep and its record; the data held never more than 3 generations (the solver's current and
        backup, and the kept one it has saved over), 2 after each record, however many calls."""
        g = 0
        for k in range(1, 9):
            for _ in range(2):
                g += 1
                self.save(g)
                self.assertLessEqual(self.generations_held(), 3)
            execute.keep_generation(self.ck, k, self.LAY)
            self.assertEqual(len(list((self.ck / "kept").iterdir())), 1 if k == 1 else 2)
            self.assertLessEqual(self.generations_held(), 3)
            self.record(k)
            self.assertEqual([p.name for p in (self.ck / "kept").iterdir()], [str(k)])
            self.assertEqual(self.generations_held(), 2)

    def test_resume_touches_rank_files_only(self):
        """Setting checkpoint/ back removes other generations' checkpoint_<g>_<rank> and nothing else."""
        self.save(1)
        execute.keep_generation(self.ck, 1, self.LAY), self.record(1)
        self.save(2), self.save(3)
        others = ["checkpoint_new.json", "checkpoint_2_0.bak", "checkpoint_notes.txt", "profiling.0"]
        for n in others:
            (self.ck / n).write_text("x")
        (self.ck / ".schur").mkdir()
        self.assertTrue(cli.place_checkpoint(self.r, self.run_dir, 4))
        names = {f.name for f in self.ck.iterdir()}
        self.assertTrue(set(others) | {".schur"} <= names)
        self.assertEqual(sorted(n for n in names if execute.RANK_FILE.fullmatch(n)), [f"checkpoint_1_{r}" for r in range(4)])

    def kept(self) -> list:
        return sorted(int(p.name) for p in (self.ck / "kept").iterdir())

    def test_keep_checkpoints(self):
        """keep = 2 (the default, inputs.KEEP_CHECKPOINTS) and 3: after record k the newest min(k, keep) kept/,
        the data held min(k, keep) + 1 generations after a record and never more than keep + 2."""
        self.assertEqual(inputs.KEEP_CHECKPOINTS, 2)
        for keep in (inputs.KEEP_CHECKPOINTS, 3):
            with self.subTest(keep=keep):
                self.setUp()
                g = 0
                for k in range(1, 7):
                    for _ in range(2):
                        g += 1
                        self.save(g)
                        self.assertLessEqual(self.generations_held(), keep + 2)
                    execute.keep_generation(self.ck, k, self.LAY)
                    self.assertLessEqual(self.generations_held(), keep + 2)
                    self.record(k, keep)
                    self.assertEqual(self.kept(), list(range(max(1, k - keep + 1), k + 1)))
                    self.assertEqual(self.generations_held(), min(k, keep) + 1)

    def test_resume_loads_the_newest_of_two(self):
        """Two kept: a resume loads the newest (kept_for), the older stays as it was; a step killed before its
        record drops its keep, as with one."""
        self.save(1)
        execute.keep_generation(self.ck, 1, self.LAY), self.record(1, 2)
        self.save(2)
        execute.keep_generation(self.ck, 2, self.LAY), self.record(2, 2)
        self.save(3)
        execute.keep_generation(self.ck, 3, self.LAY)          # killed before record 3
        self.assertTrue(cli.place_checkpoint(self.r, self.run_dir, 4))
        self.assertEqual(execute.generation(self.ck), 2)
        self.assertEqual(self.kept(), [1, 2])
        self.assertEqual((self.ck / "kept" / "1" / "checkpoint_1_0").read_text(), "1 0")

    def test_keep_checkpoints_checked(self):
        run = {"skydive": {}}
        self.assertFalse([e for e in inputs.check_run_numbers(run, []) if "keep_checkpoints" in e])
        for good in (1, 3):
            run["skydive"]["keep_checkpoints"] = good
            self.assertFalse([e for e in inputs.check_run_numbers(run, []) if "keep_checkpoints" in e])
        for bad in (0, -1, True, "2"):
            run["skydive"]["keep_checkpoints"] = bad
            self.assertTrue([e for e in inputs.check_run_numbers(run, []) if "keep_checkpoints" in e], bad)

    def test_hard_links_checked(self):
        execute.check_hard_links(self.tmp)                     # a local disk makes them
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["run"])
        with mock.patch("os.link", side_effect=OSError(1, "Operation not permitted")):
            with self.assertRaises(SystemExit) as e:
                execute.check_hard_links(self.tmp)
        self.assertIn("cannot make hard links", str(e.exception))
        self.assertEqual([p.name for p in self.tmp.iterdir()], ["run"])


class BlockTimings(unittest.TestCase):
    """[skydive] block_timings: an earlier run's file, by absolute path."""

    def load_with(self, value):
        import shutil
        d = Path(tempfile.mkdtemp())
        for f in ("model.json", "spectrum.toml", "objective.toml"):
            shutil.copy(ROOT / "examples" / "ising" / f, d / f)
        text = (ROOT / "examples" / "ising" / "run.toml").read_text()
        (d / "run.toml").write_text(text.replace("max_steps      = 300", f"max_steps      = 300\nblock_timings = {value}"))
        return inputs.load(d / "run.toml")

    def test_paths(self):
        f = Path(tempfile.mkdtemp()) / "block_timings"
        f.write_text("1\n")
        self.assertEqual(self.load_with(json.dumps(str(f))).run["skydive"]["block_timings"], str(f))
        for bad in ('"block_timings"', json.dumps(str(f) + ".missing"), "3"):
            with self.assertRaises(SystemExit):
                self.load_with(bad)


SINFO = "\n".join(f"{h} {s}" for h, s in [
    ("nodea-0", "idle"), ("nodea-1", "idle"), ("nodea-2", "idle"),
    ("nodeb-0", "allocated"), ("nodeb-1", "allocated"), ("nodeb-2", "idle"),
    ("nodeb-3", "allocated"), ("nodeb-4", "idle"), ("nodeb-4", "idle")]) + "\n"


class Submit(unittest.TestCase):
    def submit(self, run_toml, resources):
        p = inputs.load(run_toml)
        if resources is not None:
            p.run["resources"] = resources
        p.run["exec"] = {"site": "lab"}               # the examples name none (the installation's site)
        out = io.StringIO()
        fixture_site = execute.load_site
        with mock.patch.object(inputs, "load", return_value=p), \
                mock.patch.object(cli.execute, "load_site", lambda name, root=None: fixture_site(name, FIXTURES)), \
                mock.patch.object(cli.subprocess, "run", return_value=SimpleNamespace(stdout=SINFO)) as run, \
                contextlib.redirect_stdout(out):
            cli.main(["submit", "--dry", str(run_toml)])
        self.assertEqual(run.call_count, 1)          # sinfo only: --dry does not call sbatch
        return out.getvalue().splitlines()

    HEAD = "sbatch --job-name=bootlab --exclusive --hint=nomultithread --mem=0 --time=48:00:00"

    def test_every_example(self):
        """Every examples/*/run.toml through submit --dry: a run that had only been started inside a job (a built-in
        model's, whose card has no name: KeyError in layout.problem_key) shows here first."""
        tomls = sorted(ROOT.glob("examples/*/run.toml"))
        self.assertTrue(tomls)
        for t in tomls:
            with self.subTest(example=t.parent.name):
                self.assertTrue(any(l.startswith(self.HEAD) for l in self.submit(t, None)))

    def test_builtin_key(self):
        """A built-in model's bench key is its card's bound and Lambda, as current's bench rows name it."""
        p = inputs.load(ROOT / "examples" / "jjjj" / "run.toml")
        self.assertEqual(p.model_name, "JJJJ3d")
        self.assertEqual(layout.problem_key(p), ("JJJJ3d", p.numerics["lambda"]))
        q = inputs.load(ROOT / "examples" / "ising" / "run.toml")
        self.assertEqual(q.model_name, q.model["name"])

    def test_no_resources_one_node(self):
        """No [resources]: one node, where Slurm puts it; the job sources the site's setup.sh and starts the
        driver with the site's python, the run.toml given."""
        cmd = self.submit(L23, None)[-1]
        self.assertTrue(cmd.startswith(self.HEAD))
        self.assertIn(" --partition=normal --nodes=1 --wrap ", cmd)
        self.assertIn("setup.sh", cmd)
        self.assertIn(f"/shared/user/bootlab/python/bin/python3 -m bootlab.cli run {L23}", cmd.replace("'", ""))

    def test_auto_l23_one_l27_two(self):
        lines = self.submit(L23, {"nodes": "auto"})
        self.assertIn(" --nodes=1 --ntasks-per-node=256 --ntasks=256 -w nodeb-2 --wrap ", lines[-1])
        self.assertTrue(any("not 2: S(2) = 1.51" in l for l in lines))
        lines = self.submit(L27, {"nodes": "auto"})
        self.assertIn(" --nodes=2 --ntasks-per-node=256 --ntasks=512 -w nodeb-2,nodeb-4 --wrap ",
                      lines[-1])


class Portable(unittest.TestCase):
    """The parts that let bootlab run on another Slurm cluster."""

    def test_sites_load(self):
        """Every site loads; lab-generic uses nothing of bt's and no ssh, its bench is lab's."""
        from bootlab import layout
        for d in sorted((ROOT / "sites").iterdir()):
            if (d / "site.toml").exists():
                s = execute.load_site(d.name)
                if s.multinode:     # as the driver builds it (copy_network was once not a field)
                    layout.multinode_of({"multinode": s.multinode})
        g = execute.load_site("lab-generic", FIXTURES)
        text = (FIXTURES / "sites" / "lab-generic" / "site.toml").read_text() + \
            (FIXTURES / "sites" / "lab-generic" / "setup.sh").read_text()
        self.assertNotIn("/bt/", text)
        self.assertEqual((g.transfer, g.transport, g.rank_env), ("srun", "tcp-1g", str(execute.RANK_ENV)))
        self.assertEqual(g.bench_dir, (FIXTURES / "sites" / "lab" / "bench").resolve())
        c = execute.load_site("lab", FIXTURES)
        self.assertEqual(c.transfer, "ssh")
        self.assertIn(f"{FIXTURES / 'sites' / 'lab'}/known_hosts", " ".join(c.multinode["ssh"]))
        self.assertNotIn("{site}", c.multinode["remote_setup"])

    def test_node_type_by_host_pattern(self):
        from bootlab import layout
        self.assertEqual(layout.node_type_of("nodeb-2", SITE.node_types), "amd2")
        self.assertEqual(layout.node_type_of("nodez-0", SITE.node_types), "default")
        self.assertEqual(layout.node_type_of("cn017", {"fat": {"hosts": ["cn01[0-9]"]}}), "fat")

    def test_transport_of_multinode_rows(self):
        """Rows of more than one node count only with the site's transport (left out: ib); one node's always."""
        from bootlab import layout
        b = {"solver": [{"model": "m", "lambda": 23, "procs_per_node": 128, "program": "sdpb 3.0sky", "nodes": n,
                         "s_per_iter": t, **({"transport": tr} if tr else {})}
                        for n, t, tr in ((1, 50.0, None), (2, 30.0, None), (2, 45.0, "tcp-1g"))]}
        self.assertEqual(layout.solver_times(b, ("m", 23), 128), {1: 50.0, 2: 30.0})
        self.assertEqual(layout.solver_times(b, ("m", 23), 128, "tcp-1g"), {1: 50.0, 2: 45.0})
        self.assertEqual(layout.solver_times(b, ("m", 23), 128, "tcp-ipoib"), {1: 50.0})

    def test_srun_output_per_node(self):
        """transfer srun: one step on all the other nodes, each output line tagged with its node."""
        node = execute.Node("n0", "t", 4, 1000, ["n0", "n1", "n2"])
        site = SimpleNamespace(transfer="srun")
        ran = SimpleNamespace(returncode=0, stdout=b"n1\ta\nn2\tb\nn1\tc\nn2\t\n", stderr=b"")
        with mock.patch.object(execute.subprocess, "run", return_value=ran) as run:
            out = execute.on_hosts(site, node, "ls", ["n1", "n2"])
        self.assertEqual(out, {"n1": "a\nc\n", "n2": "b\n"})
        argv = run.call_args.args[0]
        self.assertEqual(argv[:8], ["srun", "--overlap", "--nodes=2", "--ntasks=2", "--ntasks-per-node=1",
                                    "-w", "n1,n2", "--cpu-bind=none"])
        self.assertEqual(run.call_args.kwargs["env"]["BOOTLAB_SCRIPT"], "ls")

    def test_shared_scratch_nothing_to_other_nodes(self):
        node = execute.Node("n0", "t", 4, 1000, ["n0", "n1"], shared=True)
        self.assertEqual(node.others, [])
        self.assertEqual(execute.Node("n0", "t", 4, 1000, ["n0", "n1"]).others, ["n1"])

    @unittest.skipIf(sys.platform == "win32", "the probe is a bash script")
    def test_choose_scratch(self):
        """Candidates in order: one not writable, then a writable one; or none writable, then shared."""
        import dataclasses
        tmp = Path(tempfile.mkdtemp())
        (tmp / "ro").mkdir()
        (tmp / "ro").chmod(0o500)
        node = execute.Node("h-1", "default", 4, 1000)
        run_dir = tmp / "runs" / "r-1"
        site = dataclasses.replace(SITE, scratch=[str(tmp / "ro" / "x"), str(tmp / "rw")])
        scratch, shared, notes = execute.choose_scratch(site, node, run_dir)
        if os.geteuid() != 0:   # root writes anywhere
            self.assertEqual((scratch, shared), (tmp / "rw" / "r-1", False))
            self.assertIn("not writable", notes[0])
        site = dataclasses.replace(SITE, scratch=["/proc/bootlab-nowhere", "shared"])
        scratch, shared, notes = execute.choose_scratch(site, node, run_dir)
        self.assertEqual((scratch, shared), (run_dir / "work", True))
        self.assertEqual(SITE.scratch, "/local/scratch/user-bootlab")    # one path: no probe
        self.assertEqual(execute.choose_scratch(SITE, node, run_dir)[:2],
                         (Path("/local/scratch/user-bootlab/r-1"), False))

    def test_binds_and_env(self):
        import dataclasses
        tmp = Path(tempfile.mkdtemp())
        site = dataclasses.replace(SITE, binds=["/shared"])
        self.assertEqual(execute.with_binds(site, [Path("/shared/user/bootlab/runs/r/work")]).binds, ["/shared"])
        self.assertEqual(execute.with_binds(site, [tmp / "a" / "r-1" / "blocks"]).binds, ["/shared", str(tmp)])
        with mock.patch.dict(execute.os.environ, {"UCX_NET_DEVICES": "x", "OMPI_MCA_plm": "y"}):
            execute.apply_env(SITE)
            self.assertNotIn("UCX_NET_DEVICES", execute.os.environ)
            self.assertEqual(execute.os.environ["UCX_TLS"], "rc,self,sm")
            self.assertIn("nodeb-2=mlx5_1:1", execute.os.environ["BOOTLAB_NET_DEVICES"])


if __name__ == "__main__":
    unittest.main()
